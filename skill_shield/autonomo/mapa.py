"""
Mapa de obstáculos (paso 7 de la hoja de ruta v2, 10-nov) — funciones puras (numpy + scipy).

La profundidad de la cámara de cabeza se convierte en una rejilla de cubos (vóxeles) en el
marco del MUNDO (el del MJCF), usando la calibración cámara→robot (camara_robot.json):

  imagen de profundidad → puntos 3D (cámara) → mundo → se quitan el ROBOT (sus cápsulas
  con la pose real, más un margen), el SUELO y lo que se pida excluir (la herramienta que
  se va a coger) → cada vóxel acumula evidencia:
     · OCUPADO  si cae un punto dentro (varias imágenes seguidas, no un píxel suelto);
     · LIBRE    si la cámara ve a través de él (la profundidad medida en su píxel está más
                lejos que el vóxel);
     · DESCONOCIDO si está tapado por algo (detrás de un objeto) o lleva mucho sin verse.
  Lo que no se ve durante AUTONOMO_MAPA_OLVIDO_S vuelve a «desconocido» (el robot es móvil:
  si se mueve la peana, lo antiguo deja de valer; también hay un botón «Borrar mapa»).

Para las colisiones (planificador, calibración automática…) se precalcula la DISTANCIA de
cada vóxel al obstáculo más cercano (transformada de distancia): comprobar una cápsula del
robot es mirar esa distancia en unos pocos puntos de su eje. Ver ColisionMapa.
"""
import math
import time

import numpy as np

DESCONOCIDO, LIBRE, OCUPADO = 0, 1, 2


def nube_desde_profundidad(prof_mm, K, paso=2, z_min=0.2, z_max=2.5):
    """Puntos (N x 3, metros, marco de la cámara OpenCV) de una imagen de profundidad en mm.
    paso = 2 → se usa 1 de cada 2 píxeles en cada eje (sobra para vóxeles de 2 cm)."""
    p = prof_mm[::paso, ::paso].astype(np.float32) * 0.001
    h, w = p.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * paso + (paso - 1) * 0.5
    v = v * paso + (paso - 1) * 0.5
    ok = (p > z_min) & (p < z_max)
    z = p[ok]
    x = (u[ok] - K[0, 2]) / K[0, 0] * z
    y = (v[ok] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1)


def distancia_a_segmentos(P, A, B):
    """Distancia (N,) de cada punto de P al más cercano de los segmentos A[k]-B[k]."""
    d = np.full(len(P), np.inf, dtype=np.float32)
    for a, b in zip(A, B):
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0) if L2 > 1e-12 else np.zeros(len(P))
        dist = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        d = np.minimum(d, dist)
    return d


class MapaOcupacion:
    """Rejilla de vóxeles con evidencia (log-odds) de ocupado / libre."""

    L_OCUPADO, L_LIBRE = 0.85, 0.4          # cuánto suma un acierto / resta «verlo a través»
    L_MIN, L_MAX = -2.0, 3.5
    UMBRAL_OCUPADO, UMBRAL_LIBRE = 1.2, -0.6

    def __init__(self, limites, resolucion=0.02, olvido_s=30.0, z_suelo=None, margen_robot=0.03,
                 margen_suelo=0.015, min_puntos=2):
        lim = np.asarray(limites, float).reshape(3, 2)
        self.res = float(resolucion)
        self.origen = lim[:, 0]
        self.n = np.maximum(np.ceil((lim[:, 1] - lim[:, 0]) / self.res).astype(int), 1)
        self.olvido_s = float(olvido_s)
        self.z_suelo = z_suelo
        self.margen_robot = float(margen_robot)
        self.margen_suelo = float(margen_suelo)
        self.min_puntos = int(min_puntos)
        self.L = np.zeros(tuple(self.n), np.float32)
        self.t_visto = np.zeros(tuple(self.n), np.float64)      # time.time() de la última vez que se vio
        ix, iy, iz = np.meshgrid(*[np.arange(k) for k in self.n], indexing="ij")
        self._centros = (np.stack([ix, iy, iz], -1).reshape(-1, 3) + 0.5) * self.res + self.origen
        self._dist = None                                         # transformada de distancia (caché)
        self._dist_desc = None
        self.excluir = []                                         # cajas (centro, semi) a ignorar
        self.ultima = dict(puntos=0, robot=0, suelo=0, ms=0.0)

    # ── utilidades ──
    def borrar(self):
        self.L[:] = 0.0
        self.t_visto[:] = 0.0
        self._dist = self._dist_desc = None

    def indices(self, P):
        """Índices enteros (N x 3) de los puntos y máscara de los que caen dentro."""
        idx = np.floor((P - self.origen) / self.res).astype(int)
        dentro = np.all((idx >= 0) & (idx < self.n), axis=1)
        return idx, dentro

    def estado(self, ahora=None):
        """Rejilla con DESCONOCIDO / LIBRE / OCUPADO (lo que lleva > olvido_s sin verse: desconocido)."""
        ahora = time.time() if ahora is None else ahora
        e = np.full(self.L.shape, DESCONOCIDO, np.int8)
        reciente = (ahora - self.t_visto) <= self.olvido_s
        e[(self.L >= self.UMBRAL_OCUPADO) & reciente] = OCUPADO
        e[(self.L <= self.UMBRAL_LIBRE) & reciente] = LIBRE
        return e

    def centros_ocupados(self, ahora=None):
        e = self.estado(ahora)
        return self._centros[(e == OCUPADO).reshape(-1)]

    # ── actualización con una imagen de profundidad ──
    def actualizar(self, prof_mm, K, T_mundo_cam, capsulas_robot=(), ahora=None, paso=2):
        """capsulas_robot: lista de (a, b, radio) en el mundo (la pose REAL del robot)."""
        t0 = time.perf_counter()
        ahora = time.time() if ahora is None else ahora
        R, t = T_mundo_cam[:3, :3], T_mundo_cam[:3, 3]
        P = nube_desde_profundidad(prof_mm, K, paso) @ R.T + t
        n_total = len(P)
        idx, dentro = self.indices(P)
        P, idx = P[dentro], idx[dentro]
        quitar = np.zeros(len(P), bool)
        n_suelo = 0
        if self.z_suelo is not None:
            s = P[:, 2] < self.z_suelo + self.margen_suelo
            n_suelo = int(s.sum())
            quitar |= s
        n_robot = 0
        if len(capsulas_robot):
            A = np.array([c[0] for c in capsulas_robot], float)
            B = np.array([c[1] for c in capsulas_robot], float)
            r = np.array([c[2] for c in capsulas_robot], float)
            # cada cápsula con su radio: se compara contra la más restrictiva (radio máx + margen)
            for a, b, rr in zip(A, B, r):
                d = distancia_a_segmentos(P, [a], [b])
                q = d < rr + self.margen_robot
                n_robot += int((q & ~quitar).sum())
                quitar |= q
        for c, semi in self.excluir:
            quitar |= np.all(np.abs(P - np.asarray(c)) <= np.asarray(semi), axis=1)
        idx = idx[~quitar]

        # aciertos: vóxeles con al menos min_puntos puntos en esta imagen
        aciertos = np.zeros(self.L.shape, np.int32)
        np.add.at(aciertos, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
        golpe = aciertos >= self.min_puntos

        # libres: la cámara ve MÁS LEJOS que el vóxel en su píxel (pasa a través de él)
        visto, libre = self._visibilidad(prof_mm, K, T_mundo_cam)
        libre &= ~golpe
        self.L[golpe] += self.L_OCUPADO
        self.L[libre] -= self.L_LIBRE
        np.clip(self.L, self.L_MIN, self.L_MAX, out=self.L)
        self.t_visto[golpe | visto] = ahora
        self._dist = self._dist_desc = None
        self.ultima = dict(puntos=n_total, robot=n_robot, suelo=n_suelo,
                           ms=round((time.perf_counter() - t0) * 1000, 1))

    def _visibilidad(self, prof_mm, K, T_mundo_cam):
        """(visto, libre) por vóxel. visto = en el campo de visión con profundidad medida en su
        píxel y NO tapado; libre = además la medida está claramente más lejos que el vóxel."""
        h, w = prof_mm.shape
        Tcm = np.linalg.inv(T_mundo_cam)
        C = self._centros @ Tcm[:3, :3].T + Tcm[:3, 3]
        z = C[:, 2]
        delante = z > 0.15
        u = np.full(len(C), -1, int)
        v = np.full(len(C), -1, int)
        u[delante] = np.round(C[delante, 0] / z[delante] * K[0, 0] + K[0, 2]).astype(int)
        v[delante] = np.round(C[delante, 1] / z[delante] * K[1, 1] + K[1, 2]).astype(int)
        en_imagen = delante & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        medida = np.zeros(len(C), np.float32)
        medida[en_imagen] = prof_mm[v[en_imagen], u[en_imagen]].astype(np.float32) * 0.001
        valida = en_imagen & (medida > 0)
        tol = 1.5 * self.res
        libre = valida & (medida > z + tol)
        visto = valida & (medida > z - tol)                      # delante o en la superficie medida
        return visto.reshape(self.L.shape), libre.reshape(self.L.shape)

    # ── distancias para las colisiones ──
    def distancias(self, desconocido_ocupado=False, ahora=None):
        """Distancia (m) de cada vóxel al obstáculo más cercano (0 dentro de uno)."""
        cache = "_dist_desc" if desconocido_ocupado else "_dist"
        if getattr(self, cache) is None:
            from scipy.ndimage import distance_transform_edt
            e = self.estado(ahora)
            obst = e == OCUPADO
            if desconocido_ocupado:
                obst |= self._tapados(e)
            d = distance_transform_edt(~obst) * self.res if obst.any() else np.full(self.L.shape, np.inf, np.float32)
            setattr(self, cache, d.astype(np.float32))
        return getattr(self, cache)

    def _tapados(self, e):
        """Vóxeles desconocidos que están DEBAJO/DETRÁS de algo ocupado en la misma columna
        vertical (lo más probable: el interior o la sombra de un objeto). No se marcan como
        obstáculo las zonas que simplemente quedan fuera de la vista."""
        ocup = e == OCUPADO
        encima = np.flip(np.maximum.accumulate(np.flip(ocup, axis=2), axis=2), axis=2)   # hay algo ocupado más arriba
        return (e == DESCONOCIDO) & encima

    def distancia_en(self, P, desconocido_ocupado=False):
        """Distancia al obstáculo más cercano en los puntos P (fuera de la rejilla: infinito)."""
        d = self.distancias(desconocido_ocupado)
        idx, dentro = self.indices(np.asarray(P, float))
        out = np.full(len(P), np.inf, np.float32)
        out[dentro] = d[idx[dentro, 0], idx[dentro, 1], idx[dentro, 2]]
        return out


class ColisionMapa:
    """Choques de las cápsulas del robot (ModeloColision) contra el mapa de obstáculos.
    Misma interfaz que ModeloColision.colisiones: lista de (cápsula, "mapa", penetración)."""

    def __init__(self, modelo, mapa, margen=0.02, desconocido_ocupado=True, paso=0.01):
        self.m = modelo
        self.mapa = mapa
        self.margen = float(margen)
        self.desc = bool(desconocido_ocupado)
        self.paso = float(paso)

    def colisiones(self, q, marcos=None):
        out = list(self.m.colisiones(q, marcos))
        for c in self.m.capsulas(q, marcos):
            if c.nombre == "base":
                continue
            n = max(2, int(math.ceil(np.linalg.norm(c.b - c.a) / self.paso)) + 1)
            pts = c.a + np.linspace(0.0, 1.0, n)[:, None] * (c.b - c.a)
            d = float(np.min(self.mapa.distancia_en(pts, self.desc)))
            # la distancia es al CENTRO del vóxel ocupado más cercano: se resta medio vóxel
            pen = c.r + self.margen + 0.5 * self.mapa.res - d
            if pen > 0:
                out.append((c.nombre, "mapa", pen))
        return out

    def libre(self, q, marcos=None):
        return not self.colisiones(q, marcos)

    def capsulas(self, q, marcos=None):
        return self.m.capsulas(q, marcos)


def capsulas_robot(modelo, q):
    """(a, b, radio) de las cápsulas del robot en la pose q, para quitar el robot del mapa."""
    return [(c.a, c.b, c.r) for c in modelo.capsulas(np.asarray(q, float))]


# ── para mandar el mapa a otros procesos (canal de memoria compartida) ──
def empaquetar(centros, res, max_voxeles=6000):
    """Bytes: float32 [res, n, x0, y0, z0, x1, …]. Si hay demasiados, se queda con una muestra."""
    c = np.asarray(centros, np.float32).reshape(-1, 3)
    if len(c) > max_voxeles:
        c = c[np.linspace(0, len(c) - 1, max_voxeles).astype(int)]
    return np.concatenate([np.array([res, len(c)], np.float32), c.ravel()]).tobytes()


def desempaquetar(b):
    a = np.frombuffer(b, np.float32)
    if len(a) < 2:
        return 0.0, np.zeros((0, 3), np.float32)
    n = int(a[1])
    return float(a[0]), a[2:2 + 3 * n].reshape(-1, 3)


def dibujar_voxeles(scn, centros, res, rgba=(1.0, 0.45, 0.0, 0.55), inicio=None):
    """Añade los vóxeles como cajas a una escena de MuJoCo (visor.user_scn o Renderer.scene).
    Devuelve cuántos ha dibujado."""
    import mujoco
    i = scn.ngeom if inicio is None else inicio
    tam = np.full(3, 0.5 * res * 0.92)
    mat = np.eye(3).flatten()
    rgba = np.asarray(rgba, np.float32)
    n = 0
    for c in centros:
        if i >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_BOX, tam, np.asarray(c, float), mat, rgba)
        i += 1
        n += 1
    scn.ngeom = i
    return n


def modelo_colision(cadena, cfg, T_mundo_cam=None, radio_mano=None):
    """Modelo de colisión de la teleoperación (cápsulas del FR5 + mano; soporte, suelo y
    TELEOP_FR5_OBSTACULOS de config) y, si se sabe, la cámara de cabeza como obstáculo.
    cfg: el módulo config o un dict."""
    from skill_shield.teleop_fr5.ik_teleop import Caja, Entorno, ModeloColision, entorno_soporte_hombros
    from skill_shield.teleop_fr5.obstaculos import cajas_desde_config
    g = cfg if isinstance(cfg, dict) else {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    ent = entorno_soporte_hombros(g["SOPORTE_SUELO_Z"]) if g.get("SOPORTE_HOMBROS", False) \
        else Entorno(z_suelo=0.0)
    ent.cajas = cajas_desde_config(g, ent.z_suelo, verbose=False)
    if T_mundo_cam is not None:
        ent.cajas.append(Caja("camara_cabeza", np.asarray(T_mundo_cam)[:3, 3], (0.07, 0.07, 0.07)))
    m = ModeloColision(cadena, ent)
    if radio_mano is not None:
        m.R_MANO = float(radio_mano)
    return m


class ConstructorMapa:
    """El mapa dentro del proceso de la cámara: carga la calibración (y la recarga si cambia),
    lee la pose real del brazo y los mandos de la web, actualiza el mapa a AUTONOMO_MAPA_HZ y lo
    publica. paso(prof_mm, K) se llama con cada imagen de profundidad nueva."""

    def __init__(self, cfg, raiz, canal_q=None, canal_mapa=None, canal_mando=None):
        import os
        g = cfg if isinstance(cfg, dict) else {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
        self.g = g
        ruta = g.get("CALIB_ROBOT_ARCHIVO", "camara_robot.json")
        self.ruta_calib = ruta if os.path.isabs(ruta) else os.path.join(raiz, ruta)
        z_suelo = g.get("SOPORTE_SUELO_Z") if g.get("SOPORTE_HOMBROS", False) else 0.0
        lim = g.get("AUTONOMO_MAPA_LIMITES") or ((-1.0, 0.6), (-1.1, 0.4), (z_suelo, 0.6))
        self.mapa = MapaOcupacion(lim, float(g.get("AUTONOMO_MAPA_RES", 0.02)),
                                  olvido_s=float(g.get("AUTONOMO_MAPA_OLVIDO_S", 30.0)), z_suelo=z_suelo,
                                  margen_robot=float(g.get("AUTONOMO_MAPA_MARGEN_ROBOT", 0.03)))
        from skill_shield.teleop_fr5.cinematica import cargar_cadena
        self.modelo = modelo_colision(cargar_cadena(g.get("RUTA_XML"), verbose=False), g)
        self.periodo = 1.0 / max(float(g.get("AUTONOMO_MAPA_HZ", 4.0)), 0.5)
        self.max_dibujo = int(g.get("AUTONOMO_MAPA_MAX_DIBUJO", 6000))
        self.canal_q, self.canal_mapa, self.canal_mando = canal_q, canal_mapa, canal_mando
        self.T, self._mtime = None, None
        self._t_ult, self._t_calib = 0.0, 0.0
        self._borrado_visto = None
        self.hz, self._n, self._t_hz = 0.0, 0, time.monotonic()
        self.aviso = ""
        self.robot_quitado = False

    def _recargar_calibracion(self):
        import os
        try:
            m = os.path.getmtime(self.ruta_calib)
        except OSError:
            m = None
        if m != self._mtime:
            self._mtime = m
            if m is None:
                self.T = None
            else:
                from skill_shield.autonomo.calibracion import cargar
                self.T = cargar(self.ruta_calib)["T_mundo_cam"]
                self.mapa.borrar()                       # calibración nueva: lo de antes no vale

    def _mandos(self):
        if self.canal_mando is None:
            return
        import json
        b, _ = self.canal_mando.leer()
        if not b:
            return
        try:
            m = json.loads(b)
        except ValueError:
            return
        bid = m.get("mapa_borrar_id")
        if bid is not None and bid != self._borrado_visto:
            if self._borrado_visto is not None or m.get("mapa_borrar_nuevo"):
                self.mapa.borrar()
            self._borrado_visto = bid
        self.mapa.excluir = [(np.asarray(c, float), np.asarray(s, float)) for c, s in (m.get("mapa_excluir") or [])]

    def _q_real(self):
        if self.canal_q is None:
            return None
        b, t = self.canal_q.leer()
        if not b or t is None or time.time() - t > 0.5:
            return None
        a = np.frombuffer(b, np.float64)
        return a[:6].copy() if len(a) >= 7 and a[6] > 0.5 else None    # a[6] = 1 si es la pose REAL

    def paso(self, prof_mm, K):
        ahora = time.monotonic()
        if ahora - self._t_ult < self.periodo:
            return False
        self._t_ult = ahora
        if ahora - self._t_calib > 1.0:
            self._t_calib = ahora
            self._recargar_calibracion()
        self._mandos()
        if self.T is None:
            self.aviso = "sin calibrar (camara_robot.json)"
            return False
        q = self._q_real()
        self.robot_quitado = q is not None
        self.aviso = "" if q is not None else "sin pose real del robot: puede salir como obstáculo"
        self.mapa.actualizar(prof_mm, K, self.T, capsulas_robot(self.modelo, q) if q is not None else ())
        if self.canal_mapa is not None:
            b = empaquetar(self.mapa.centros_ocupados(), self.mapa.res, self.max_dibujo)
            if len(b) <= self.canal_mapa.capacidad:
                self.canal_mapa.escribir(b)
        self._n += 1
        if ahora - self._t_hz >= 2.0:
            self.hz, self._n, self._t_hz = self._n / (ahora - self._t_hz), 0, ahora
        return True

    def estado(self):
        if self.T is None:
            return dict(estado="sin", texto="sin calibrar")
        e = self.mapa.estado()
        n_ocu = int((e == OCUPADO).sum())
        return dict(estado="ok" if self.robot_quitado else "aviso",
                    texto=f"{n_ocu} vóxeles" + ("" if self.robot_quitado else " (robot sin quitar)"),
                    ocupados=n_ocu, libres=int((e == LIBRE).sum()), res=self.mapa.res,
                    hz=round(self.hz, 1), ms=self.mapa.ultima.get("ms"), aviso=self.aviso)
