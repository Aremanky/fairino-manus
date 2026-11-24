"""
AGARRE ASISTIDO del modo GRABACION (16-nov): el planificador lleva la mano a la postura de agarre y
desde ahí el operador hace el agarre en modo RELATIVO (con embrague). Eso es lo que se graba: la
red aprende desde el MISMO punto en el que empezará al ejecutarse (tras el planificador).

En la ventana de la cámara:
  [O]  cambia el objeto a coger (por defecto, el más cercano a la palma; naranja en MuJoCo)
  [L]  planifica cómo LLEVAR la mano a la postura de agarre de ese objeto —la misma a la que llega la
       Tarea del modo autónomo antes de bajar (Planificador.planificar_previa)—. Con el PEDAL
       pisado el brazo va; al soltarlo se para. Al llegar, AFINAR: GRABACION_AFINAR_S quieto para que
       la cámara mida bien el objeto. [L] mientras planifica o va = cancelar.
  LISTO → pisa el PEDAL: embrague puesto, modo relativo y EMPIEZA A GRABARSE el episodio. Haces el
       agarre con el guante y subes unos cm. SUELTA el pedal: el robot se para y el episodio se
       guarda. [M] = salió mal. Volver a pisar: relativo SIN grabar (dejar el objeto, apartarse).
       [L]: otro agarre (planifica desde donde esté). [K]: vuelta a la teleoperación normal.
  [B]  bloquea el giro: en relativo solo cuenta la traslación y el giro es el del planificador
       (para pelotas suele bastar, y los datos salen con menos ruido).

MODO RELATIVO (embrague, como los robots quirúrgicos): al pisar se fijan dos referencias —dónde
está tu mano (lo que mide la teleoperación, ya llevado al mundo del robot: obj["p_muneca"],
obj["R_mano"]) y dónde está la del robot— y desde ahí el robot copia tus DESPLAZAMIENTOS y GIROS,
escalados (GRABACION_ESCALA_POS, GRABACION_ESCALA_GIRO: 0,5 = la mitad, tiembla menos y se apunta
mejor), no tu postura. Codo y hombro los pone la IK (la solución más cercana a la postura actual:
el planificador ya dejó el brazo bien colocado). Los dedos siguen yendo directos del guante.
Seguridad: límites, velocidad máxima (GRABACION_V_MAX), saltos de la IK y choques (con el mapa de
obstáculos de la cámara de cabeza: el brazo frena lo más cerca posible sin chocar).
"""
import math
import threading
import time
import traceback
import unicodedata

import numpy as np

from skill_shield.teleop_fr5.cinematica import OFFSET_MUNECA_ORCA, T_de, rot_eje


def escalar_giro(R, s):
    """El mismo giro que R, con el ángulo multiplicado por s (alrededor del mismo eje)."""
    R = np.asarray(R, float)
    c = min(1.0, max(-1.0, (float(np.trace(R)) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.eye(3)
    eje = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(eje)
    if n < 1e-9:                                      # ~180°: eje de la columna dominante de R + I
        k = int(np.argmax(np.diag(R)))
        eje = (R + np.eye(3))[:, k]
        n = np.linalg.norm(eje)
    return rot_eje(eje / n, s * ang)


class TeleopRelativa:
    """Modo relativo con embrague. paso() → postura del brazo a mandar (o la de antes, si no puede)."""

    def __init__(self, cadena, escala_pos=0.5, escala_giro=0.5, v_max=0.8, salto_max=0.5,
                 margen_lim=math.radians(3.0)):
        self.c = cadena
        self.escala_pos, self.escala_giro = float(escala_pos), float(escala_giro)
        self.v_max, self.salto_max = float(v_max), float(salto_max)
        self.lo = cadena.limites[:, 0] + margen_lim
        self.hi = cadena.limites[:, 1] - margen_lim
        self.ref = None
        self.q = None
        self.info = dict(estado="suelto", motivo="")

    @property
    def enganchado(self):
        return self.ref is not None

    def enganchar(self, obj, q_robot):
        """Pedal pisado: referencias de tu mano (obj) y de la del robot (q_robot)."""
        T_r = self.c.fk_marcos(np.asarray(q_robot, float))[6]
        self.ref = dict(p_h=np.asarray(obj["p_muneca"], float).copy(), R_h=np.asarray(obj["R_mano"], float).copy(),
                        R_r=T_r[:3, :3].copy(), p_r=T_r[:3, :3] @ OFFSET_MUNECA_ORCA + T_r[:3, 3])
        self.q = np.asarray(q_robot, float).copy()
        self.info = dict(estado="enganchado", motivo="")

    def soltar(self):
        self.ref = None
        self.info = dict(estado="suelto", motivo="")

    def objetivo(self, obj, bloquear_giro=False):
        """mano_mount (4x4, mundo) que corresponde a la mano del operador ahora."""
        r = self.ref
        p = r["p_r"] + self.escala_pos * (np.asarray(obj["p_muneca"], float) - r["p_h"])
        if bloquear_giro:
            R = r["R_r"]
        else:
            R = escalar_giro(np.asarray(obj["R_mano"], float) @ r["R_h"].T, self.escala_giro) @ r["R_r"]
        return T_de(R, p - R @ OFFSET_MUNECA_ORCA)

    def paso(self, obj, dt=None, bloquear_giro=False, col=None):
        """→ (q, info). Sin embrague → (None, …). Sin medida de tu mano → se queda donde está."""
        if self.ref is None:
            return None, self.info
        if obj is None:
            self.info = dict(estado="sin_medida", motivo="no se ve tu mano")
            return self.q.copy(), self.info
        sols = self.c.ik_analitica(self.objetivo(obj, bloquear_giro), q_ref=self.q)
        if not sols:
            self.info = dict(estado="fuera_de_alcance", motivo="el brazo no llega ahí")
            return self.q.copy(), self.info
        q = np.asarray(min(sols, key=lambda s: float(np.max(np.abs(np.asarray(s) - self.q)))), float)
        if float(np.max(np.abs(q - self.q))) > self.salto_max:
            self.info = dict(estado="salto", motivo="la IK tendría que cambiar de postura: no se mueve")
            return self.q.copy(), self.info
        q = np.clip(q, self.lo, self.hi)
        if dt:
            paso = self.v_max * float(dt)
            q = self.q + np.clip(q - self.q, -paso, paso)
        estado, motivo = "ok", ""
        if col is not None:
            cols = col.colisiones(q)
            if cols:
                motivo = "colision " + ",".join(f"{a}-{b}" for a, b, _ in cols[:2])
                if col.libre(self.q):                 # lo más lejos posible sin chocar (como la IK)
                    a_ok, a_mal = 0.0, 1.0
                    for _ in range(6):
                        a = 0.5 * (a_ok + a_mal)
                        if col.libre(self.q + a * (q - self.q)):
                            a_ok = a
                        else:
                            a_mal = a
                    q = self.q + a_ok * (q - self.q)
                else:
                    q = self.q.copy()
                estado = "bloqueado"
        self.q = q
        self.info = dict(estado=estado, motivo=motivo)
        return q.copy(), self.info


class AgarreAsistido:
    """La secuencia del agarre asistido (ver cabecera). La lleva Skill_Shield_Nivel_6.py en el modo
    GRABACION: pedir_plan() con [L], pedal() con cada pisada/suelta, paso() en cada vuelta."""

    def __init__(self, cfg, raiz, cadena, fabrica_planificador=None):
        g = {k: v for k, v in dict(cfg).items() if isinstance(k, str) and k.isupper()}
        self.cfg, self.raiz, self.cadena = g, raiz, cadena
        self.rel = TeleopRelativa(cadena, escala_pos=float(g.get("GRABACION_ESCALA_POS", 0.5)),
                                  escala_giro=float(g.get("GRABACION_ESCALA_GIRO", 0.5)),
                                  v_max=float(g.get("GRABACION_V_MAX", 0.8)),
                                  salto_max=float(g.get("GRABACION_SALTO_MAX_RAD", 0.5)))
        self.bloquear_giro = bool(g.get("GRABACION_BLOQUEAR_GIRO", False))
        self.afinar_s = float(g.get("GRABACION_AFINAR_S", 1.0))
        self.vel = math.radians(float(g.get("AUTONOMO_TAREA_VEL_DEG_S", 15.0)))
        self.acel = math.radians(float(g.get("AUTONOMO_TAREA_ACEL_DEG_S2", 30.0)))
        self.desvio_max = math.radians(float(g.get("AUTONOMO_TAREA_DESVIO_MAX_DEG", 8.0)))
        self.excluir_r = float(g.get("AUTONOMO_TAREA_EXCLUIR_RADIO", 0.15))
        self._fabrica = fabrica_planificador
        self._pl = None
        self.estado, self.aviso = "libre", ""
        self.plan = self.ejec = None
        self.objetivo = None                       # la detección a la que se lleva la mano
        self.pisado = False
        self.q_cmd = None
        self.n_grabados = 0
        self._hilo, self._res, self._gen = None, None, 0
        self._t_afinar, self._t_desvio = None, None

    # ── estado ──
    @property
    def controla_brazo(self):
        """True: el brazo lo mueve el agarre asistido (no la teleoperación normal)."""
        return self.estado != "libre"

    @property
    def necesita_humano(self):
        """True: hace falta medir tu mano (la teleoperación tiene que estar calculando)."""
        return self.estado in ("listo", "agarrando", "relativo")

    @property
    def quiere_mover(self):
        """True: el FR5 debe estar armado (mandando) ahora."""
        if not self.pisado:
            return False
        return self.estado == "llevando" or (self.estado in ("agarrando", "relativo") and self.rel.enganchado)

    def _crear_planificador(self):
        import os
        from skill_shield.autonomo.tarea import Planificador
        T = None
        ruta = self.cfg.get("CALIB_ROBOT_ARCHIVO", "calibraciones/datos/camara_robot.json")
        ruta = ruta if os.path.isabs(ruta) else os.path.join(self.raiz, ruta)
        if os.path.exists(ruta):
            from skill_shield.autonomo.calibracion import cargar
            T = cargar(ruta)["T_mundo_cam"]
        cadena = self.cadena
        if self.cfg.get("RUTA_XML"):                  # su propia cinemática (va en otro hilo), como la Tarea
            from skill_shield.teleop_fr5.cinematica import cargar_cadena
            cadena = cargar_cadena(self.cfg["RUTA_XML"], verbose=False)
        return Planificador(cadena, self.cfg, T)

    # ── órdenes ──
    def pedir_plan(self, q_actual, det, mapa=None):
        """[L]: planificar (en otro hilo) cómo llevar la mano a la postura de agarre de det."""
        if det is None or det.get("T_mundo") is None:
            self.aviso = "no hay objeto que coger: elige uno con [O]"
            return False
        self.rel.soltar()
        self.pisado = False
        self.objetivo = det
        self.estado, self.aviso = "planificando", ""
        self.plan, self.ejec, self._res = None, None, None
        self._gen += 1
        if det.get("tipo") == "pelota":
            caja = det.get("caja")
            r = float(det.get("diametro_m") or (2 * float(caja[0]) if caja else 0.07)) / 2.0
            excluir = [(np.asarray(det["T_mundo"], float)[:3, 3], r + 0.03)]
        else:
            excluir = [(np.asarray(det["T_mundo"], float)[:3, 3], self.excluir_r)]
        self._hilo = threading.Thread(target=self._planificar, name="Grabacion-plan", daemon=True,
                                      args=(self._gen, np.asarray(q_actual, float), det, mapa, excluir))
        self._hilo.start()
        return True

    def _planificar(self, gen, q, det, mapa, excluir):
        from skill_shield.autonomo.tarea import ErrorPlan
        t0 = time.monotonic()
        try:
            if self._pl is None:
                self._pl = self._fabrica() if self._fabrica is not None else self._crear_planificador()
            res_m, centros = mapa if mapa is not None else (0.02, None)
            self._pl.poner_mapa(centros, res_m, excluir)
            res = ("ok", self._pl.planificar_previa(q, det), time.monotonic() - t0)
        except ErrorPlan as e:
            res = ("error", str(e), time.monotonic() - t0)
        except Exception as e:                          # fallo de programa: que se vea
            traceback.print_exc()
            res = ("error", f"fallo al planificar: {e}", time.monotonic() - t0)
        if gen == self._gen:                            # si entretanto se canceló, no vale
            self._res = res

    def cancelar(self, aviso=""):
        self._gen += 1
        self.rel.soltar()
        self.estado, self.aviso = "libre", aviso
        self.plan = self.ejec = None
        self.pisado = False
        self._t_desvio = None

    def pedal(self, pisado, q_robot=None, obj_humano=None):
        """Cada pisada / suelta. → "grabar" (empieza un episodio), "parar_grabacion" o None."""
        self.pisado = bool(pisado)
        if pisado:
            if self.estado in ("listo", "relativo"):
                if obj_humano is None:
                    self.aviso = "no se ve tu mano (¿teleoperación calibrada con [A]?): no se engancha"
                    return None
                self.rel.enganchar(obj_humano, q_robot)
                self.aviso = ""
                if self.estado == "listo":
                    self.estado = "agarrando"
                    return "grabar"
            return None
        if self.rel.enganchado:
            self.rel.soltar()
        if self.estado == "agarrando":
            self.estado = "relativo"
            self.n_grabados += 1
            return "parar_grabacion"
        return None

    # ── cada vuelta ──
    def paso(self, dt, q_actual=None, q_real=None, obj_humano=None, col=None, mover=True):
        """→ postura (6) que hay que mandar al brazo, o None (no mandar nada).
        mover: el FR5 está mandando (o no hay FR5): solo entonces avanza el plan."""
        if self.estado == "planificando" and self._res is not None:
            tipo, valor, seg = self._res
            self._res = None
            if tipo == "ok" and q_real is not None \
                    and float(np.max(np.abs(np.asarray(q_real, float) - valor.q0))) > math.radians(3.0):
                tipo, valor = "error", "el robot se ha movido mientras se planificaba: pulsa [L] otra vez"
            if tipo == "ok":
                from skill_shield.autonomo.tarea import (Ejecutor, MANO_ABIERTA, MANO_PELOTA_ABIERTA,
                                                         MANO_PINZA_ABIERTA, _postura_cfg)
                self.plan = valor
                # (la mano real la sigue llevando el guante; el Ejecutor solo da el brazo)
                posturas = dict(
                    abierta=_postura_cfg(self.cfg, "AUTONOMO_TAREA_MANO_ABIERTA", MANO_ABIERTA),
                    pelota_abierta=_postura_cfg(self.cfg, "AUTONOMO_TAREA_MANO_PELOTA_ABIERTA", MANO_PELOTA_ABIERTA),
                    pinza_abierta=dict(MANO_PINZA_ABIERTA))
                self.ejec = Ejecutor(valor, posturas, self.vel, self.acel, t_mano=(0.0, 0.0))
                self.q_cmd = self.ejec.q()
                self.estado = "llevando"
                self.aviso = f"plan en {seg:.1f} s: {valor.nombre}"
            else:
                self.estado, self.aviso = "libre", f"no se puede llevar la mano: {valor}"
        if self.estado == "llevando":
            if self.pisado and mover and dt:
                self.ejec.avanzar(float(dt))
            q = self.ejec.q()
            if q_real is not None and self.pisado and mover:
                if float(np.max(np.abs(np.asarray(q_real, float) - q))) > self.desvio_max:
                    self._t_desvio = self._t_desvio or time.monotonic()
                    if time.monotonic() - self._t_desvio > 0.6:
                        self.cancelar("el robot no sigue al plan (desvío): parado")
                        return None
                else:
                    self._t_desvio = None
            self.q_cmd = q
            if self.ejec.terminado:
                self.estado, self._t_afinar = "afinando", time.monotonic()
            return q
        if self.estado == "afinando":
            if time.monotonic() - self._t_afinar >= self.afinar_s:
                self.estado = "listo"
            return self.q_cmd
        if self.estado == "listo":
            return self.q_cmd
        if self.estado in ("agarrando", "relativo"):
            if self.rel.enganchado:
                q, info = self.rel.paso(obj_humano, dt, self.bloquear_giro, col)
                self.q_cmd = q
                return q
            return self.q_cmd
        return None

    def progreso(self):
        return self.ejec.progreso() if self.ejec is not None else 0.0

    def texto(self):
        """Una línea para el HUD (sin tildes: cv2.putText no las pinta)."""
        nombre = (self.objetivo or {}).get("nombre", "?")
        e = self.estado
        if e == "libre":
            t = "[L] llevar la mano al objetivo | [O] otro objeto"
        elif e == "planificando":
            t = f"PLANIFICANDO la llegada a {nombre}... ([L] cancela)"
        elif e == "llevando":
            t = f"LLEVANDO LA MANO a {nombre}: pisa el pedal ({100 * self.progreso():.0f} %) - mano abierta"
        elif e == "afinando":
            t = "AFINANDO con la camara..."
        elif e == "listo":
            t = ("LISTO: suelta el pedal y vuelve a pisarlo para coger (SE GRABA)" if self.pisado else
                 "LISTO: pisa el pedal para coger (SE GRABA) - empieza con la mano abierta")
        else:
            grabando = e == "agarrando"
            info = self.rel.info if self.rel.enganchado else {}
            t = (("GRABANDO" if grabando else "RELATIVO (sin grabar)")
                 + f" x{self.rel.escala_pos:g}" + (" giro bloqueado" if self.bloquear_giro else "")
                 + (f" - {info.get('motivo')}" if info.get("motivo") else "")
                 + ("" if grabando else " | [L] otro | [K] salir | [M] fallido"))
        if self.aviso and e in ("libre", "relativo", "listo", "llevando"):
            t += f" | {self.aviso}"
        return "".join(c for c in unicodedata.normalize("NFKD", t) if ord(c) < 128)
