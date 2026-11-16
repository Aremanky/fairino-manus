"""
Reproducción de una política entrenada (16-nov). La comparten [I] (Skill_Shield_Nivel_6.py), la
evaluación (evaluar.py) y la fase «Agarre IA» de la Tarea (siguiente paso). No importa torch: el
modelo llega ya cargado en `red` (act.cargar_ia), con su función `predecir`.

    red = cargar_ia(ruta)                      # skill_shield/ia/act.py
    pol = Politica(red, tarea="agarre_pelota")
    pol.reiniciar(obs)                         # fija el objeto y llena el historial
    paso = pol.paso(obs)                       # → brazo (6) o None, mano (17), progreso 0…1, error
    cmd = filtro.aplicar(paso, brazo_ctrl_anterior, mano_ctrl_anterior)

obs (un instante): q_brazo (6, sim), q_mano (17, sim) y, si los hay, q_brazo_real, q_mano_real,
tactil ({dedo: [fx, fy, fz]}). En reiniciar() además «objeto» (episodios.objeto_desde_deteccion):
se CONGELA ahí, como al grabar, porque la mano lo tapa al acercarse. Las entradas de la red se
construyen con las mismas funciones que al entrenar (episodios.caracteristicas).
"""
from collections import deque, namedtuple

import numpy as np

from skill_shield.ia.episodios import (R_de_rot6d, SinObjeto, accion_a_joints, caracteristicas,
                                       crudo_de_obs, rot6d)
from skill_shield.teleop_fr5.cinematica import OFFSET_PALMA_ORCA

Paso = namedtuple("Paso", "brazo mano progreso accion error dist_objeto")
Comando = namedtuple("Comando", "brazo mano motivo parar")


class Politica:
    def __init__(self, red, tarea=None, cadena=None, peso_exp=0.15):
        self.red = red
        self.receta = red["receta"]
        self.H, self.C = int(self.receta["ventana"]), int(self.receta["chunk"])
        tareas = list(self.receta.get("tareas") or [])
        if tarea is None:
            self.tarea_idx = 0
        elif isinstance(tarea, (int, np.integer)):
            self.tarea_idx = int(tarea)
        elif tarea in tareas:
            self.tarea_idx = tareas.index(tarea)
        else:
            raise ValueError(f"el modelo no conoce la tarea '{tarea}' (conoce: {', '.join(tareas)})")
        self.necesita_objeto = bool(self.receta.get("entorno"))
        self.usa_ik = self.receta["accion"] == "mano_en_objeto"
        if cadena is None and (self.necesita_objeto or self.usa_ik):
            import config
            from skill_shield.teleop_fr5.cinematica import CadenaFR5
            cadena = CadenaFR5.desde_mjcf(config.RUTA_XML)
        self.cadena = cadena
        self.peso_exp = float(peso_exp)
        self.objeto = None
        self.hist = None
        self.bloques = deque(maxlen=self.C)
        self.error = None
        self.ultimo = None

    @property
    def nombres_brazo(self):
        return list(self.receta.get("nombres_brazo") or [])

    @property
    def nombres_mano(self):
        return self.receta.get("nombres_mano")

    @property
    def mueve_brazo(self):
        return bool(self.nombres_brazo)

    @property
    def dt(self):
        return 1.0 / float(self.receta.get("fps") or 30.0)

    @property
    def tarea(self):
        tareas = list(self.receta.get("tareas") or [])
        return tareas[self.tarea_idx] if self.tarea_idx < len(tareas) else None

    def _entrada(self, obs):
        crudo = crudo_de_obs(self.receta, obs, self.objeto)
        estado, entorno, _ = caracteristicas(self.receta, crudo, self.cadena)
        return estado[0], (None if entorno is None else entorno[0]), crudo["q_brazo"][0]

    def reiniciar(self, obs):
        """Empieza (o vuelve a empezar, p. ej. al pisar el pedal). False (y self.error) si falta algo."""
        self.error = None
        self.bloques.clear()
        self.hist = None
        self.objeto = obs.get("objeto") if self.necesita_objeto else None
        if self.necesita_objeto and self.objeto is None:
            self.error = "no veo ningún objeto que coger cerca de la mano"
            return False
        try:
            s, _, _ = self._entrada(obs)
        except (SinObjeto, ValueError) as e:
            self.error = str(e)
            return False
        self.hist = deque([s] * self.H, maxlen=self.H)     # igual que el relleno al entrenar
        return True

    def paso(self, obs):
        """Una predicción (un fotograma). Devuelve Paso; si algo falla, error ≠ None."""
        if self.hist is None and not self.reiniciar(obs):
            return Paso(None, None, None, None, self.error, None)
        try:
            s, e, q_b = self._entrada(obs)
        except (SinObjeto, ValueError) as err:
            return Paso(None, None, None, None, str(err), None)
        self.hist.append(s)
        try:                        # cualquier fallo de la red = parada controlada, nunca un cuelgue
            acc, prog = self.red["predecir"](np.asarray(self.hist, float), e, self.tarea_idx)
        except Exception as err:
            return Paso(None, None, None, None, f"fallo de la red: {err}", None)
        self.bloques.append((np.asarray(acc, float), None if prog is None else np.asarray(prog, float)))
        a, p = self._ensamblar()
        brazo, mano, err = accion_a_joints(self.receta, a, self.objeto, q_b if len(q_b) else None, self.cadena)
        # distancia de la PALMA (no de mano_mount, que está ~14 cm detrás) al centro del objeto
        dist = float(np.linalg.norm(R_de_rot6d(a[3:9]) @ OFFSET_PALMA_ORCA + a[:3])) if self.usa_ik else None
        self.ultimo = Paso(brazo, mano, p, a, err, dist)
        return self.ultimo

    def _ensamblar(self):
        """Temporal ensembling: la acción de AHORA según cada bloque reciente (el k-ésimo más reciente
        aporta su elemento k), con pesos exp(−peso_exp·k): más peso a lo más reciente. En modo
        'mano_en_objeto' todas las acciones están en el marco del objeto (congelado), así que se
        pueden promediar directamente."""
        items = [(k, b, p) for k, (b, p) in enumerate(reversed(self.bloques)) if k < len(b)]
        w = np.exp(-self.peso_exp * np.array([k for k, _, _ in items], float))
        w /= w.sum()
        a = sum(wi * b[k] for wi, (k, b, _) in zip(w, items))
        if self.usa_ik:
            a = np.array(a, float)
            a[3:9] = rot6d(R_de_rot6d(a[3:9]))
        ps = [(wi, float(p[k])) for wi, (k, _, p) in zip(w, items) if p is not None]
        prog = sum(wi * v for wi, v in ps) / sum(wi for wi, _ in ps) if ps else None
        return np.asarray(a, float), prog


class FiltroSeguridad:
    """La red propone y el filtro decide, antes de mover nada:
      · error de la red (IK imposible, sin objeto…) o valores no válidos → no se mueve y PARA
      · la palma lejos del objeto (más de max_dist_objeto m) → PARA
      · salto de un joint del brazo mayor que salto_max (rad) respecto al comando anterior (p. ej.
        la IK cambia de rama) → PARA
      · límites de cada joint (brazo y mano)
      · velocidad máxima de cada joint del brazo (v_max_brazo · dt por fotograma)
      · choque (libre(q) → False) → ese fotograma el brazo no avanza (se cuenta en n_choques)"""

    def __init__(self, lim_brazo=None, lim_mano=None, v_max_brazo=1.0, dt=1.0 / 30, salto_max=None,
                 max_dist_objeto=None, libre=None):
        self.lim_brazo = None if lim_brazo is None else np.asarray(lim_brazo, float).reshape(-1, 2)
        self.lim_mano = None if lim_mano is None else np.asarray(lim_mano, float).reshape(-1, 2)
        self.v_max, self.dt = float(v_max_brazo), float(dt)
        self.salto_max = None if salto_max is None else float(salto_max)
        self.max_dist = None if max_dist_objeto is None else float(max_dist_objeto)
        self.libre = libre
        self.n_choques = 0

    def aplicar(self, paso, brazo_ant=None, mano_ant=None):
        ant_b = None if brazo_ant is None else np.asarray(brazo_ant, float)
        ant_m = None if mano_ant is None else np.asarray(mano_ant, float)
        if paso.error:
            return Comando(ant_b, ant_m, paso.error, True)
        if self.max_dist is not None and paso.dist_objeto is not None and paso.dist_objeto > self.max_dist:
            return Comando(ant_b, ant_m, f"la red lleva la mano a {paso.dist_objeto:.2f} m del objeto "
                                         f"(máx. {self.max_dist:.2f})", True)
        mano, brazo, motivo = paso.mano, paso.brazo, None
        if mano is not None:
            mano = np.asarray(mano, float)
            if not np.all(np.isfinite(mano)):
                return Comando(ant_b, ant_m, "la red ha dado valores no válidos (mano)", True)
            if self.lim_mano is not None and len(self.lim_mano) == len(mano):
                mano = np.clip(mano, self.lim_mano[:, 0], self.lim_mano[:, 1])
        if brazo is not None:
            brazo = np.asarray(brazo, float)
            if not np.all(np.isfinite(brazo)):
                return Comando(ant_b, ant_m, "la red ha dado valores no válidos (brazo)", True)
            if ant_b is not None and self.salto_max is not None:
                salto = float(np.max(np.abs(brazo - ant_b)))
                if salto > self.salto_max:
                    return Comando(ant_b, mano, f"la red pide un salto de {np.degrees(salto):.0f}° en el brazo "
                                                f"(máx. {np.degrees(self.salto_max):.0f}°)", True)
            if self.lim_brazo is not None and len(self.lim_brazo) == len(brazo):
                brazo = np.clip(brazo, self.lim_brazo[:, 0], self.lim_brazo[:, 1])
            if ant_b is not None:
                paso_max = self.v_max * self.dt
                brazo = ant_b + np.clip(brazo - ant_b, -paso_max, paso_max)
            if self.libre is not None and not self.libre(brazo):
                self.n_choques += 1
                motivo = "choque: el brazo no avanza"
                brazo = ant_b
        return Comando(brazo, mano, motivo, False)
