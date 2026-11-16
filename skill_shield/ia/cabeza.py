"""
Cámara de CABEZA dentro de la teleoperación — modo GRABACION del menú (16-nov).

La teleoperación de siempre (cámaras A y C cruzadas + guante) MÁS lo que ve el modo autónomo con
la cámara de cabeza (AUTONOMO_CAM_MXID, la antigua B), para grabar demos con el mundo:
  · se lanza el MISMO proceso de cámara que en el autónomo (skill_shield.autonomo.camara) con las
    herramientas (ArUco), las pelotas y el mapa de obstáculos, y se habla con él por memoria
    compartida (canal.py);
  · el objeto OBJETIVO es automático: el que se está grabando (fijado al empezar, fuentes.py) o,
    sin grabar, el que se puede coger más cercano a la palma. No cuenta como obstáculo: se le
    manda al proceso de la cámara (lo quita del mapa) y tiene su zona de agarre en las colisiones;
  · el mapa entra en las colisiones de la IK (teleop_fr5/colision_mapa.py) → el brazo FRENA;
  · objetos, objetivo (naranja) y obstáculos cerca de la mano se dibujan en la ventana de MuJoCo;
  · cada fila del episodio lleva si el brazo iba frenado (ev:frenado).

    cab = CamaraCabeza.lanzar(cfg, raiz)
    mundo = MundoGrabacion(cab, fuentes, modelo=modelo, cfg=cfg)
    mundo.col = ColisionConMapa(teleop_v2.ik.col); teleop_v2.ik.col = mundo.col
    en cada vuelta: mundo.paso(datos, q_sim, q_real, info_ik); mundo.dibujar(visor, base)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

from skill_shield.ia.episodios import TIPOS_COGIBLES, elegir_objeto
from skill_shield.ia.fuentes import FuenteObjetosCamara
from skill_shield.teleop_fr5.colision_mapa import ColisionConMapa, zona_de

CANAL_HERR = "skill_grab_herramientas"
CANAL_MAPA = "skill_grab_mapa"
CANAL_QBRAZO = "skill_grab_qbrazo"
CANAL_MANDO = "skill_grab_mando"


def _clave(d):
    return f"{d.get('tipo')}:{d.get('id')}"


class CamaraCabeza:
    """El proceso de la cámara de cabeza (herramientas, pelotas y mapa) y sus canales."""

    def __init__(self, c_herr, c_mando, c_q=None, c_mapa=None, proceso=None):
        self.c_herr, self.c_mando, self.c_q, self.c_mapa = c_herr, c_mando, c_q, c_mapa
        self.proceso = proceso
        self.fuente_objetos = FuenteObjetosCamara(c_herr, proceso)   # (la usan fuentes.py y el dibujo)
        self._objetivo = "sin mandar"

    @classmethod
    def lanzar(cls, cfg, raiz, con_mapa=True):
        from skill_shield.autonomo.canal import Canal
        n_max = int(cfg.get("AUTONOMO_MAPA_MAX_DIBUJO", 6000))
        c_herr = Canal(CANAL_HERR, capacidad=64 << 10, crear=True)
        c_mando = Canal(CANAL_MANDO, capacidad=4 << 10, crear=True)
        c_q = Canal(CANAL_QBRAZO, capacidad=16 << 10, crear=True) if con_mapa else None
        c_mapa = Canal(CANAL_MAPA, capacidad=16 + 12 * n_max, crear=True) if con_mapa else None
        py = cfg.get("AUTONOMO_PYTHON_PERCEPCION") or sys.executable
        cmd = [py, "-m", "skill_shield.autonomo.camara", "--canal-herramientas", CANAL_HERR,
               "--canal-mando", CANAL_MANDO, "--padre", str(os.getpid())]
        if con_mapa:
            cmd += ["--canal-qbrazo", CANAL_QBRAZO, "--canal-mapa", CANAL_MAPA]
        if cfg.get("AUTONOMO_CAM_MXID"):
            cmd += ["--mxid", str(cfg["AUTONOMO_CAM_MXID"])]
        if cfg.get("GRABACION_CAM_VENTANA", False):
            cmd.append("--ventana")
        try:
            proc = subprocess.Popen(cmd, cwd=raiz)
        except Exception:
            for c in (c_herr, c_mando, c_q, c_mapa):
                if c is not None:
                    c.cerrar()
            raise
        print(f"📷 GRABACIÓN: cámara de cabeza lanzada (PID {proc.pid}): herramientas, pelotas"
              + (" y mapa de obstáculos" if con_mapa else "") + ".")
        return cls(c_herr, c_mando, c_q, c_mapa, proc)

    def objetos(self):
        return self.fuente_objetos()

    def estado(self):
        return self.fuente_objetos.estado()

    def publicar_pose(self, q, real, caps_mano=None):
        """Pose del brazo (y piezas de la mano) para que el robot no salga como obstáculo."""
        if self.c_q is not None:
            from skill_shield.autonomo.mapa import empaquetar_pose
            self.c_q.escribir(empaquetar_pose(q, real, caps_mano))

    def poner_objetivo(self, id_):
        """El objeto que se va a coger: el proceso de la cámara lo quita del mapa."""
        if id_ != self._objetivo:
            self._objetivo = id_
            self.c_mando.escribir(json.dumps({"objetivo_id": id_}).encode())

    def mapa(self, max_edad_s=3.0):
        """(resolución, centros de los vóxeles ocupados (N x 3)) o None si no hay mapa reciente."""
        if self.c_mapa is None:
            return None
        b, t = self.c_mapa.leer()
        if not b or t is None or time.time() - t > max_edad_s:
            return None
        from skill_shield.autonomo.mapa import desempaquetar
        return desempaquetar(b)

    def cerrar(self):
        if self.proceso is not None and self.proceso.poll() is None:
            self.proceso.terminate()
            try:
                self.proceso.wait(timeout=3.0)
            except Exception:
                self.proceso.kill()
        for c in (self.c_herr, self.c_mando, self.c_q, self.c_mapa):
            if c is not None:
                try:
                    c.cerrar()
                except Exception:
                    pass


class MundoGrabacion:
    """Lo que hace el modo GRABACION en cada vuelta de la teleoperación (ver cabecera)."""

    def __init__(self, cabeza, fuentes, modelo=None, cfg=None, periodo_mapa_s=0.25):
        cfg = cfg or {}
        self.cab, self.fuentes = cabeza, fuentes
        self.col = None                                    # ColisionConMapa de la IK (si frena)
        self.zona_extra = float(cfg.get("GRABACION_ZONA_OBJETO_M", 0.25))
        self.ver = bool(cfg.get("GRABACION_MAPA_VER", True))
        self.max_dibujo = int(cfg.get("GRABACION_MAPA_MAX_DIBUJO", 400))
        self.radio_dibujo = float(cfg.get("GRABACION_MAPA_RADIO_DIBUJO_M", 0.5))
        self.periodo_mapa_s = float(periodo_mapa_s)
        self.piezas = None
        if modelo is not None:
            try:
                from skill_shield.autonomo.mapa import PiezasMano
                self.piezas = PiezasMano(modelo)
            except Exception as e:
                print(f"⚠️  GRABACIÓN: sin las piezas de la mano para el mapa ({e}): se quita con una cápsula.")
        self.lista, self.objetivo, self.mapa_ult = [], None, None
        self.elegido = None                                # clave elegida con [O] (None = el más cercano)
        self.frenando, self.motivo_freno = False, ""
        self._t_mapa, self._sin_mapa_avisado, self._base = 0.0, False, None
        self.palma = None

    def nueva_colision(self, base, cfg=None):
        """Envuelve el ModeloColision de la IK (teleop_v2.ik.col) con el mapa. Devuelve el nuevo."""
        cfg = cfg or {}
        self.col = ColisionConMapa(base, margen=float(cfg.get("GRABACION_MAPA_MARGEN_M", 0.02)))
        if self.mapa_ult is not None:
            self._poner_mapa()
        return self.col

    def _objetivo(self, lista, q_brazo):
        """La detección del objetivo: la que se está grabando, la elegida con [O] o la más cercana
        a la palma."""
        clave = self.fuentes.objetivo_clave() if self.fuentes is not None else None
        if clave is None and self.elegido is not None and any(_clave(d) == self.elegido for d in lista):
            clave = self.elegido
        if clave is None:
            palma = self.fuentes.palma(q_brazo) if self.fuentes is not None else None
            radio = self.fuentes.radio if self.fuentes is not None else 0.35
            o = elegir_objeto(lista, palma, radio)
            clave = o["clave"] if o is not None else None
        return next((d for d in lista if _clave(d) == clave), None) if clave else None

    def siguiente_objetivo(self):
        """[O]: el siguiente objeto que se puede coger (de más cerca a más lejos de la palma) y, tras
        el último, otra vez «el más cercano». → nombre del elegido (o None = automático)."""
        cands = []
        for d in self.lista:
            if d.get("T_mundo") is None or (d.get("tipo") or "herramienta") not in TIPOS_COGIBLES:
                continue
            p = np.asarray(d["T_mundo"], float)[:3, 3]
            dist = float(np.linalg.norm(p - self.palma)) if self.palma is not None else 0.0
            cands.append((dist, _clave(d), d))
        cands.sort(key=lambda c: c[0])
        claves = [c[1] for c in cands]
        if not claves:
            self.elegido = None
            return None
        i = claves.index(self.elegido) + 1 if self.elegido in claves else 0
        self.elegido = claves[i] if i < len(claves) else None
        return next((c[2].get("nombre") for c in cands if c[1] == self.elegido), None)

    def objetivo_para_plan(self):
        """La detección a la que llevar la mano ([L]): el objetivo de ahora, con su T_mundo."""
        o = self.objetivo
        return o if o is not None and o.get("T_mundo") is not None else None

    def _poner_mapa(self):
        if self.col is None:
            return
        if self.mapa_ult is None:
            self.col.quitar_mapa()
            return
        res, C = self.mapa_ult
        self.col.poner_mapa(C, res, [zona_de(self.objetivo, self.zona_extra)] if self.objetivo else [])

    def paso(self, datos, q_brazo_sim, q_real=None, info_ik=None):
        q_real = None if q_real is None else np.asarray(q_real, float)
        q = q_real if q_real is not None else np.asarray(q_brazo_sim, float)
        # 1. pose del brazo para quitarlo del mapa (y las piezas de la mano, si el brazo es el real)
        caps = None
        if self.piezas is not None and q_real is not None:
            try:
                caps = self.piezas.capsulas(datos)
            except Exception:
                caps = None
        self.cab.publicar_pose(q, q_real is not None, caps)
        # 2. objetos y objetivo (el proceso de la cámara lo quita del mapa)
        self.lista = self.cab.objetos()
        self.objetivo = self._objetivo(self.lista, q)
        self.cab.poner_objetivo(self.objetivo.get("id") if self.objetivo else None)
        self.palma = self.fuentes.palma(q) if self.fuentes is not None else None
        # 3. mapa → colisiones de la IK
        ahora = time.monotonic()
        if ahora - self._t_mapa >= self.periodo_mapa_s:
            self._t_mapa = ahora
            self.mapa_ult = self.cab.mapa()
            if self.mapa_ult is None and self.col is not None and not self._sin_mapa_avisado:
                self._sin_mapa_avisado = True
                print("⚠️  GRABACIÓN: no llega el mapa de obstáculos (" + self.cab.estado()
                      + "): el brazo solo frena con los obstáculos de config.")
            elif self.mapa_ult is not None:
                self._sin_mapa_avisado = False
            self._poner_mapa()
        # 4. ¿la IK está frenando el brazo por el mapa o la base del objeto?
        info = info_ik or {}
        motivo = str(info.get("motivo", ""))
        self.frenando = info.get("estado") == "bloqueado" and ("mapa" in motivo or "mesa" in motivo)
        self.motivo_freno = motivo if self.frenando else ""
        return self.objetivo

    def texto(self):
        """Una línea para el HUD de la teleoperación (sin tildes: cv2.putText no las pinta)."""
        if self.frenando:
            return "FRENADO: obstaculo (" + ("base del objeto" if "mesa" in self.motivo_freno else "mapa") + ")"
        n = self.col.n_voxeles if self.col is not None else 0
        obj = self.objetivo.get("nombre") if self.objetivo else "ninguno cerca"
        if self.elegido is not None and self.objetivo is not None and _clave(self.objetivo) == self.elegido:
            obj += " (elegido [O])"
        return f"Objetivo: {obj} | mapa: {n} cubos" if self.mapa_ult is not None else f"Objetivo: {obj} | sin mapa"

    def dibujar(self, visor, base=None):
        """Objetos (objetivo en naranja) y los obstáculos cerca de la palma en la ventana de MuJoCo,
        a partir del geom 'base' de user_scn (detrás del brazo fantasma, que se redibuja cada vuelta)."""
        if not self.ver:
            return
        from skill_shield.autonomo.herramientas import dibujar_herramientas
        from skill_shield.autonomo.mapa import dibujar_voxeles
        with visor.lock():
            scn = visor.user_scn
            if base is None:
                if self._base is None:
                    self._base = scn.ngeom
                base = self._base
            scn.ngeom = base
            dibujar_herramientas(scn, self.lista, self.objetivo.get("id") if self.objetivo else None)
            if self.mapa_ult is not None and self.palma is not None:
                res, C = self.mapa_ult
                if len(C):
                    d = np.linalg.norm(np.asarray(C, float) - self.palma, axis=1)
                    cerca = np.nonzero(d < self.radio_dibujo)[0]
                    cerca = cerca[np.argsort(d[cerca])[:self.max_dibujo]]
                    dibujar_voxeles(scn, np.asarray(C)[cerca], res)
