"""
Brazo FANTASMA en el visor de MuJoCo (9-nov).

Con el gemelo digital enseñando la pose REAL del FR5, hace falta ver también a dónde se le
está mandando ir. Esto dibuja, en visor.user_scn (solo visual: no toca la física):
  · el OBJETIVO del brazo como un esqueleto translúcido (cápsulas entre los orígenes de los
    eslabones joint1..joint6 y la punta, esferas en las articulaciones);
  · dos puntos: la PUNTA OBJETIVO (verde) y la PUNTA REAL (roja), con una línea entre ellos
    si están separados.

El objetivo es la configuración articular (rad, MJCF) que el bucle le da al GeneradorSuave:
la que sale de la IK de teleoperación o de los sliders. La cinemática se calcula en un MjData
aparte (mj_kinematics), así que no interfiere con la simulación.
"""
import numpy as np


class FantasmaBrazo:
    def __init__(self, modelo, arm_joints, cuerpo_punta="mano_mount",
                 rgba_brazo=(0.1, 0.85, 1.0, 0.35), rgba_objetivo=(0.1, 1.0, 0.2, 0.9),
                 rgba_real=(1.0, 0.2, 0.2, 0.9), radio=0.025):
        import mujoco
        self._mj = mujoco
        self.modelo = modelo
        self.datos = mujoco.MjData(modelo)
        jids = [mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm_joints]
        self.qadr = [int(modelo.jnt_qposadr[j]) for j in jids]
        cuerpos = [int(modelo.jnt_bodyid[j]) for j in jids]
        base = int(modelo.body_parentid[cuerpos[0]])
        punta = mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_BODY, cuerpo_punta)
        self.punta = int(punta) if punta >= 0 else cuerpos[-1]
        self.cadena = [base] + cuerpos + ([self.punta] if self.punta != cuerpos[-1] else [])
        self.rgba_brazo = np.asarray(rgba_brazo, dtype=np.float32)
        self.rgba_obj = np.asarray(rgba_objetivo, dtype=np.float32)
        self.rgba_real = np.asarray(rgba_real, dtype=np.float32)
        self.radio = float(radio)
        self._base = None          # primer geom de user_scn que es nuestro
        self._conectar = getattr(mujoco, "mjv_connector", None) or getattr(mujoco, "mjv_makeConnector")

    # ── cinemática ──
    def puntos(self, q_rad, qpos_resto=None):
        """Orígenes de la cadena (base, joint1..6, punta) con el brazo en q_rad."""
        if qpos_resto is not None:
            self.datos.qpos[:] = qpos_resto
        for a, q in zip(self.qadr, q_rad):
            self.datos.qpos[a] = float(q)
        self._mj.mj_kinematics(self.modelo, self.datos)
        return [self.datos.xpos[b].copy() for b in self.cadena]

    # ── dibujo ──
    def _geom(self, scn, i, tipo, tam, pos, rgba):
        self._mj.mjv_initGeom(scn.geoms[i], tipo, np.asarray(tam, dtype=float),
                              np.asarray(pos, dtype=float), np.eye(3).flatten(), rgba)

    def _cilindro(self, scn, i, tipo, ancho, a, b, rgba):
        g = scn.geoms[i]
        self._mj.mjv_initGeom(g, tipo, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba)
        try:
            self._conectar(g, tipo, float(ancho), np.asarray(a, dtype=float), np.asarray(b, dtype=float))
        except TypeError:                                 # firma antigua (coordenadas sueltas)
            self._conectar(g, tipo, float(ancho), *[float(v) for v in a], *[float(v) for v in b])

    def dibujar(self, visor, q_objetivo, punta_real=None, qpos_resto=None):
        """Redibuja el fantasma con el brazo en q_objetivo (rad). punta_real: xyz o None."""
        mj = self._mj
        pts = self.puntos(q_objetivo, qpos_resto)
        with visor.lock():
            scn = visor.user_scn
            if self._base is None:
                self._base = scn.ngeom
            i = self._base
            n_max = scn.maxgeom
            # esqueleto del objetivo
            for a, b in zip(pts[:-1], pts[1:]):
                if i >= n_max:
                    break
                if np.linalg.norm(b - a) < 1e-4:          # orígenes coincidentes: nada que unir
                    continue
                self._cilindro(scn, i, mj.mjtGeom.mjGEOM_CAPSULE, self.radio, a, b, self.rgba_brazo)
                i += 1
            for p in pts[1:-1]:                           # articulaciones
                if i >= n_max:
                    break
                self._geom(scn, i, mj.mjtGeom.mjGEOM_SPHERE, (self.radio * 1.4,) * 3, p, self.rgba_brazo)
                i += 1
            # punta objetivo (verde) y punta real (roja)
            if i < n_max:
                self._geom(scn, i, mj.mjtGeom.mjGEOM_SPHERE, (self.radio * 1.6,) * 3, pts[-1], self.rgba_obj)
                i += 1
            if punta_real is not None and i < n_max:
                self._geom(scn, i, mj.mjtGeom.mjGEOM_SPHERE, (self.radio * 1.6,) * 3, punta_real, self.rgba_real)
                i += 1
                if np.linalg.norm(np.asarray(punta_real) - pts[-1]) > 0.01 and i < n_max:
                    self._cilindro(scn, i, mj.mjtGeom.mjGEOM_LINE, 2.0, pts[-1], punta_real, self.rgba_real)
                    i += 1
            scn.ngeom = i
        return float(np.linalg.norm(np.asarray(punta_real) - pts[-1])) if punta_real is not None else None

    def ocultar(self, visor):
        if self._base is None:
            return
        with visor.lock():
            visor.user_scn.ngeom = self._base
