"""
Lo que hay en el MUNDO mientras se graba ([G]) o se reproduce ([I]) — fuentes del formato 3 (16-nov).

Cada fuente puede faltar (sin FR5, sin mano, sin táctiles, sin cámara): entonces se graba NaN y la
red lo sabe (bandera de los táctiles; el objeto es obligatorio o no según la receta). Ninguna fuente
bloquea: se leen desde el hilo de la física a 30 Hz.

  · brazo real : fr5.q_real_rad()                           (proceso fr5_servo, memoria compartida)
  · mano real  : LectorManoReal → OrcaHardwareWrapper.estado_real() en su propio hilo (~15 Hz)
  · táctiles   : guante.tactil.fuerzas()                    (puente_tactil.py, por UDP)
  · objetos    : FuenteObjetosCamara — la cámara de cabeza del modo autónomo (pelotas.py por color
                 + profundidad y herramientas.py con ArUco), lanzada aquí si la teleoperación no la
                 usa — o FuenteObjetosSim — los objetos de la mesa del taller [3] en MuJoCo.

El objeto «objetivo» de cada episodio es el que se puede coger más cercano a la palma la primera
vez que se ve (a menos de IA_OBJETO_RADIO_M); se sigue ese mismo durante todo el episodio.
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

from skill_shield.ia.episodios import (COLS_OBJ, DEDOS, EJES_TACTIL, elegir_objeto, mat_a_quat,
                                       objeto_desde_deteccion, tactil_array)

NAN = float("nan")
CANAL_OBJETOS_IA = "skill_ia_objetos"
FISICO_A_DEDO = {"pulgar": "thumb", "indice": "index", "corazon": "middle", "anular": "ring", "menique": "pinky"}


def tactil_por_dedo_fisico(f, mapa=None):
    """TactilReal.fuerzas() trae los nombres de SENSOR de orca_core, que no coinciden con el cableado
    de la mano; con ORCA_TACTIL_SENSOR_DE_DEDO ({dedo físico: sensor}) se pasa a dedos FÍSICOS
    (thumb = pulgar, index = índice…). Así los episodios no dependen de cómo esté cableada."""
    if f is None or not mapa:
        return f
    out = {}
    for fisico, sensor in mapa.items():
        d = FISICO_A_DEDO.get(fisico)
        if d is not None and sensor in f:
            out[d] = f[sensor]
    return out


class LectorManoReal:
    """Postura REAL de la OrcaHand (rad, convenio del sim, orden de ctrl 0..16) y la corriente de
    cada motor (fracción de max_current), leídas en un hilo propio: estado_real() tarda 2-4 ms y
    comparte el bus con el envío a 50 Hz, así que no se llama desde la física. Solo lee mientras
    alguien la pide (grabando o en [I]): en la teleoperación normal no toca el bus."""

    def __init__(self, wrapper, periodo_s=0.066, pedida_s=1.0):
        self.w = wrapper
        self.periodo_s = float(periodo_s)
        self.pedida_s = float(pedida_s)
        self._lock = threading.Lock()
        self._q = self._frac = None
        self._t = 0.0
        self._t_pedida = -1e9
        self._activo = False
        self._hilo = None
        self._aviso = None

    def iniciar(self):
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle, name="IA-ManoReal", daemon=True)
        self._hilo.start()
        return self

    def leer_una_vez(self):
        """Una lectura (lo que hace el hilo en cada vuelta). True si se pudo."""
        if not getattr(self.w, "conectado", False):
            return False
        e = self.w.estado_real()
        joints = list(getattr(self.w, "_joints", []) or [])
        q = self.w._desde_sdk({j: v[1] for j, v in e.items()})
        if q is None or len(joints) != len(q) or not np.all(np.isfinite(q)):
            return False
        mapa = self.w.JOINT_A_CTRL
        n = max(mapa.values()) + 1
        qq, ff = np.full(n, np.nan), np.full(n, np.nan)
        for k, j in enumerate(joints):
            i = mapa.get(j)
            if i is not None:
                qq[i] = float(q[k])
                v = e.get(j)
                ff[i] = float(v[2]) if v is not None and v[2] is not None else np.nan
        with self._lock:
            self._q, self._frac, self._t = qq, ff, time.monotonic()
        return True

    def _bucle(self):
        while self._activo:
            t0 = time.monotonic()
            if t0 - self._t_pedida <= self.pedida_s:     # alguien la está usando
                try:
                    self.leer_una_vez()
                except Exception as ex:                # nunca debe tirar nada: se graba sin ella
                    if self._aviso is None:
                        self._aviso = str(ex)
                        print(f"⚠️  IA: no puedo leer la postura real de la mano ({ex}); sigo sin ella.")
            time.sleep(max(0.0, self.periodo_s - (time.monotonic() - t0)))

    def ultima(self, max_edad_s=0.3):
        """(q (17,), corriente (17,)) de la última lectura, o None si es vieja o no hay."""
        self._t_pedida = time.monotonic()
        with self._lock:
            if self._q is None or time.monotonic() - self._t > max_edad_s:
                return None
            return self._q.copy(), self._frac.copy()

    def cerrar(self):
        self._activo = False


class FuenteObjetosCamara:
    """Lista de objetos que publica el proceso de la cámara de cabeza (skill_shield.autonomo.camara
    con --canal-herramientas): pelotas y herramientas, con su pose en el mundo (marco del MJCF) si
    hay calibración (camara_robot.json)."""

    def __init__(self, canal, proceso=None, max_edad_s=0.6):
        self.canal, self.proceso, self.max_edad_s = canal, proceso, float(max_edad_s)
        self._lista, self._t_leido = [], 0.0
        self.calibrada = None

    @classmethod
    def lanzar(cls, cfg, raiz, nombre=CANAL_OBJETOS_IA):
        """Crea el canal y lanza la cámara de cabeza (la misma que el modo autónomo)."""
        from skill_shield.autonomo.canal import Canal
        canal = Canal(nombre, capacidad=64 << 10, crear=True)
        py = cfg.get("AUTONOMO_PYTHON_PERCEPCION") or sys.executable
        cmd = [py, "-m", "skill_shield.autonomo.camara", "--canal-herramientas", nombre,
               "--padre", str(os.getpid())]
        if cfg.get("AUTONOMO_CAM_MXID"):
            cmd += ["--mxid", str(cfg["AUTONOMO_CAM_MXID"])]
        if cfg.get("IA_OBJETOS_VENTANA", False):
            cmd.append("--ventana")
        try:
            proc = subprocess.Popen(cmd, cwd=raiz)
        except Exception:
            canal.cerrar()
            raise
        print(f"📷 IA: cámara de cabeza lanzada para ver los objetos al grabar (PID {proc.pid}).")
        return cls(canal, proc)

    def __call__(self):
        ahora = time.time()
        if ahora - self._t_leido < 0.03:              # publica a 10 Hz: no hace falta más
            return self._lista
        self._t_leido = ahora
        if self.proceso is not None and self.proceso.poll() is not None:
            self._lista = []
            return self._lista
        try:
            b, t = self.canal.leer()
        except Exception:
            b, t = None, None
        if b is None or t is None or ahora - t > self.max_edad_s:
            self._lista = []
            return self._lista
        try:
            d = json.loads(b)
        except ValueError:
            return self._lista
        self.calibrada = bool(d.get("calibrada", False))
        self._lista = list(d.get("lista") or [])
        return self._lista

    def estado(self):
        if self.proceso is not None and self.proceso.poll() is not None:
            return "la cámara de cabeza se cerró (¿la usa otro programa?)"
        if self.calibrada is None:
            return "esperando a la cámara de cabeza"
        if not self.calibrada:
            return "cámara de cabeza sin calibrar (python calibrar_camara_robot.py --auto)"
        return "ok"

    def cerrar(self):
        if self.proceso is not None and self.proceso.poll() is None:
            self.proceso.terminate()
            try:
                self.proceso.wait(timeout=3.0)
            except Exception:
                self.proceso.kill()
        try:
            self.canal.cerrar()
        except Exception:
            pass


class FuenteObjetosSim:
    """Objetos de MuJoCo (p. ej. la mesa del taller [3]) con su pose EXACTA, en el mismo formato
    que la cámara, para grabar demos en simulación. Se lee desde el hilo de la física."""

    def __init__(self, datos, cuerpos, visible=None, tipo="herramienta", tam=None):
        self.datos = datos
        self.cuerpos = dict(cuerpos)                  # nombre → id del body
        self.visible = visible
        self.tipo = tipo
        self.tam = dict(tam or {})

    def __call__(self):
        if self.visible is not None and not self.visible():
            return []
        out = []
        for nombre, bid in self.cuerpos.items():
            T = np.eye(4)
            T[:3, :3] = np.asarray(self.datos.xmat[bid], float).reshape(3, 3)
            T[:3, 3] = np.asarray(self.datos.xpos[bid], float)
            limpio = str(nombre).replace("taller_", "")
            out.append(dict(id=int(bid), nombre=limpio, tipo="pelota" if "pelota" in limpio else self.tipo,
                            T_mundo=T, estable=True, tam=self.tam.get(nombre)))
        return out

    def estado(self):
        return "ok" if (self.visible is None or self.visible()) else "la mesa del taller [3] está escondida"


class ObjetosCombinados:
    """La primera fuente que tenga objetos (la cámara real antes que el simulador)."""

    def __init__(self, fuentes):
        self.fuentes = [f for f in fuentes if f is not None]

    def __call__(self):
        for f in self.fuentes:
            lista = f()
            if lista:
                return lista
        return []

    def estado(self):
        estados = [getattr(f, "estado", lambda: "ok")() for f in self.fuentes]
        return "ok" if "ok" in estados else (estados[0] if estados else "sin fuentes de objetos")

    def cerrar(self):
        for f in self.fuentes:
            c = getattr(f, "cerrar", None)
            if c is not None:
                c()


class FuentesGrabacion:
    """Lo que el grabador añade a cada fila (formato 3) y lo que la reproducción necesita en vivo.
    brazo_real: () → q (6,) o None · mano_real: LectorManoReal · tactil: () → {dedo: [fx, fy, fz]}
    · objetos: () → lista de detecciones (formato de pelotas.py / herramientas.py)."""

    def __init__(self, nombres_brazo, nombres_mano, cadena=None, brazo_real=None, mano_real=None,
                 tactil=None, objetos=None, radio_objeto=0.35, mapa_tactil=None):
        self.nombres_brazo, self.nombres_mano = list(nombres_brazo), list(nombres_mano)
        self.cadena = cadena
        self.brazo_real, self.mano_real, self.tactil, self.objetos = brazo_real, mano_real, tactil, objetos
        self.radio = float(radio_objeto)
        self.mapa_tactil = dict(mapa_tactil) if mapa_tactil else None   # sensor → dedo físico
        self.iniciar_episodio()

    def columnas(self):
        return ([f"br:{n}" for n in self.nombres_brazo] + [f"hr:{n}" for n in self.nombres_mano]
                + [f"hc:{n}" for n in self.nombres_mano]
                + [f"tac:{d}:{e}" for d in DEDOS for e in EJES_TACTIL] + [f"obj:{c}" for c in COLS_OBJ])

    def iniciar_episodio(self):
        self._objeto = None                     # el objetivo: se fija con la 1ª detección cercana
        self._n = dict(filas=0, br=0, hr=0, tac=0, obj=0)

    # ── lecturas sueltas: nunca lanzan ──
    def leer_brazo_real(self):
        if self.brazo_real is None:
            return None
        try:
            q = self.brazo_real()
        except Exception:
            return None
        if q is None:
            return None
        q = np.asarray(q, float).ravel()
        return q if len(q) == len(self.nombres_brazo) and np.all(np.isfinite(q)) else None

    def leer_mano_real(self):
        if self.mano_real is None:
            return None
        try:
            r = self.mano_real.ultima()
        except Exception:
            return None
        return r if r is not None and len(r[0]) == len(self.nombres_mano) else None

    def leer_tactil(self):
        """{dedo FÍSICO: [fx, fy, fz]} (N) o None."""
        if self.tactil is None:
            return None
        try:
            return tactil_por_dedo_fisico(self.tactil(), self.mapa_tactil)
        except Exception:                       # p. ej. el guante aún no existe
            return None

    def lista_objetos(self):
        if self.objetos is None:
            return []
        try:
            return self.objetos() or []
        except Exception:
            return []

    def palma(self, q_brazo):
        if self.cadena is None or q_brazo is None or len(q_brazo) != 6:
            return None
        try:
            return self.cadena.fk(np.asarray(q_brazo, float))["palma"]
        except Exception:
            return None

    def objeto_cercano(self, q_brazo):
        """El objeto que se puede coger más cercano a la palma (dict de episodios.objeto_desde_deteccion)."""
        return elegir_objeto(self.lista_objetos(), self.palma(q_brazo), self.radio)

    def _objeto_actual(self, q_brazo):
        lista = self.lista_objetos()
        if self._objeto is None:
            o = elegir_objeto(lista, self.palma(q_brazo), self.radio)
            if o is not None:
                self._objeto = {k: o[k] for k in ("nombre", "clase", "tipo", "id", "clave")}
            return o
        for d in lista:
            if f"{d.get('tipo')}:{d.get('id')}" == self._objeto["clave"]:
                return objeto_desde_deteccion(d)
        return None

    def muestra(self, q_brazo_sim):
        """Una fila (columnas()) con lo que hay ahora, NaN en lo que falte. Desde el hilo de la física."""
        nb, nm = len(self.nombres_brazo), len(self.nombres_mano)
        q_br = self.leer_brazo_real()
        mr = self.leer_mano_real()
        tac = tactil_array(self.leer_tactil())
        o = self._objeto_actual(q_br if q_br is not None else q_brazo_sim)
        fila = q_br.tolist() if q_br is not None else [NAN] * nb
        fila += (mr[0].tolist() + mr[1].tolist()) if mr is not None else [NAN] * (2 * nm)
        fila += tac.ravel().tolist()
        if o is not None:
            fila += [float(o["id"]) if isinstance(o["id"], (int, float)) else NAN,
                     *o["T"][:3, 3].tolist(), *mat_a_quat(o["T"][:3, :3]).tolist(), o["tam"]]
        else:
            fila += [NAN] * len(COLS_OBJ)
        self._n["filas"] += 1
        self._n["br"] += q_br is not None
        self._n["hr"] += mr is not None
        self._n["tac"] += bool(np.all(np.isfinite(tac[:, 2])))
        self._n["obj"] += o is not None
        return fila

    def meta(self):
        """Para el .json del episodio: el objeto objetivo y la fracción de filas con cada dato."""
        n = max(self._n["filas"], 1)
        return dict(objeto=dict(self._objeto) if self._objeto else None,
                    fuentes={k: round(self._n[k] / n, 3) for k in ("br", "hr", "tac", "obj")})

    def resumen(self, q_brazo_sim=None):
        """Una línea con lo que hay disponible AHORA (se imprime al empezar a grabar)."""
        mr = self.leer_mano_real()
        if mr is None and hasattr(self.mano_real, "leer_una_vez"):   # el hilo aún no ha leído
            try:
                self.mano_real.leer_una_vez()
                mr = self.leer_mano_real()
            except Exception:
                pass
        partes = ["brazo real ✓" if self.leer_brazo_real() is not None else "brazo real ✗",
                  "mano real ✓" if mr is not None else "mano real ✗",
                  "táctiles ✓" if np.all(np.isfinite(tactil_array(self.leer_tactil())[:, 2])) else "táctiles ✗"]
        if self.objetos is None:
            partes.append("objeto ✗ (ni cámara de cabeza ni objetos del simulador)")
        else:
            o = self.objeto_cercano(q_brazo_sim)
            if o is not None:
                partes.append(f"objeto ✓ ({o['nombre']})")
            else:
                est = getattr(self.objetos, "estado", lambda: "ok")()
                partes.append("objeto ✗ (" + (est if est != "ok" else
                                              f"ninguno a menos de {self.radio:.2f} m de la palma") + ")")
        return " · ".join(partes)

    def observacion(self, q_brazo_sim, q_mano_sim):
        """Lo que necesita politica.Politica en cada paso (el objeto se fija aparte, al arrancar)."""
        mr = self.leer_mano_real()
        return dict(q_brazo=np.asarray(q_brazo_sim, float), q_mano=np.asarray(q_mano_sim, float),
                    q_brazo_real=self.leer_brazo_real(), q_mano_real=None if mr is None else mr[0],
                    tactil=self.leer_tactil())

    def cerrar(self):
        for f in (self.mano_real, self.objetos):
            c = getattr(f, "cerrar", None)
            if c is not None:
                try:
                    c()
                except Exception:
                    pass


def crear_fuentes_teleop(cfg, raiz, nombres_brazo, nombres_mano, fr5=None, mano_real=None, tactil=None,
                         objetos_sim=None, camara_libre=False, hardware=False):
    """Las fuentes del formato 3 para Skill_Shield_Nivel_6.py ([G] y [I]). cfg: globals() de la app."""
    from skill_shield.teleop_fr5.cinematica import CadenaFR5
    cadena = None
    try:
        cadena = CadenaFR5.desde_mjcf(cfg.get("RUTA_XML"))
    except Exception as e:
        print(f"⚠️  IA: sin cinemática del FR5 ({e}): no se podrá elegir el objeto más cercano a la palma.")
    brazo_real = (lambda: fr5.q_real_rad()) if fr5 is not None else None
    lector = None
    if mano_real is not None:
        lector = LectorManoReal(mano_real, periodo_s=float(cfg.get("IA_MANO_REAL_PERIODO_S", 0.066))).iniciar()
    cam = None
    if hardware and cfg.get("IA_OBJETOS_CAMARA", True):
        if camara_libre:
            try:
                cam = FuenteObjetosCamara.lanzar(cfg, raiz)
            except Exception as e:
                print(f"⚠️  IA: no se pudo lanzar la cámara de cabeza para ver los objetos ({e}).")
        else:
            print("ℹ️  IA: la cámara de cabeza la usa la teleoperación (MULTICAMARA_ACTIVA = True): se "
                  "grabará SIN objetos. Para grabar agarres con objetos: MULTICAMARA_ACTIVA = False.")
    objetos = ObjetosCombinados([cam, objetos_sim]) if (cam is not None or objetos_sim is not None) else None
    return FuentesGrabacion(nombres_brazo, nombres_mano, cadena=cadena, brazo_real=brazo_real,
                            mano_real=lector, tactil=tactil, objetos=objetos,
                            radio_objeto=float(cfg.get("IA_OBJETO_RADIO_M", 0.35)),
                            mapa_tactil=cfg.get("ORCA_TACTIL_SENSOR_DE_DEDO"))
