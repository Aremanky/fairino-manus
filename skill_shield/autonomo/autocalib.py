"""
Autocorrección de la calibración cámara→robot con el PROPIO BRAZO (12-nov).

El robot es móvil y la calibración (camara_robot.json) se hace de vez en cuando; con el
tiempo se desajusta unos centímetros (la cámara se mueve en su soporte, el tablero no era
perfecto…). Se nota en dos cosas:
  · la mano va a coger la pelota a un sitio desplazado (la pelota y el robot no cuadran);
  · el brazo, al salir en la imagen, deja cubos en el mapa al lado de donde el gemelo cree
    que está (no se quitan con sus cápsulas) → colisiones falsas.

Pero la cámara VE el brazo, y sabemos dónde está (cinemática + joints reales). Cuando el brazo
está quieto y se ve, se compara la nube de profundidad con las mallas del gemelo (ICP solo de
traslación: para cada punto medido, el punto de la malla más cercano; se mueve la nube la
mediana de esas diferencias; se repite estrechando el radio). Lo que sale es cuánto hay que
mover lo que ve la cámara para que el brazo medido caiga sobre el del gemelo: se aplica a
TODO lo que ve la cámara (mapa, pelotas, marcadores). Como se mide justo donde trabaja el
brazo, corrige sobre todo ahí, que es donde importa.

Solo traslación (los giros pequeños de la cámara se notan como traslación en la zona de
trabajo); se acumula poco a poco (AUTONOMO_AUTOCALIB_GANANCIA) y con un máximo
(AUTONOMO_AUTOCALIB_MAX_M). Se guarda en <calibración>.correccion.json para la próxima vez y
se olvida si se vuelve a calibrar.
"""
import json
import os
import time

import numpy as np


class ConCorreccion:
    """Para los detectores (marcadores, pelotas): T_mundo_cam = corrección @ la del archivo."""
    _T_archivo = None
    correccion = None                       # 4x4 o None (la pone camara.py con la del mapa)

    @property
    def T_mundo_cam(self):
        T = self._T_archivo
        return T if (T is None or self.correccion is None) else self.correccion @ T

    @T_mundo_cam.setter
    def T_mundo_cam(self, T):
        self._T_archivo = T


def T_traslacion(t):
    T = np.eye(4)
    T[:3, 3] = np.asarray(t, float)
    return T


class AutoCalib:
    """estimar(P_mundo, q) → dict(t, n, residuo) o None. P_mundo: puntos de la profundidad ya en
    el mundo (con la corrección actual); t: lo que hay que sumarles para que el brazo cuadre."""

    def __init__(self, cadena, ruta_xml, paso=0.01, dedos=((-0.03, 0.17, 0.0), 0.11), n_min=250):
        from skill_shield.autonomo.choque_mano import CUERPOS_BRAZO, MARCO_CADENA, nubes_mjcf, reducir
        self.c = cadena
        self.n_min = int(n_min)
        nubes = nubes_mjcf(ruta_xml, 0.006)
        self.partes = [(MARCO_CADENA[k], reducir(nubes[k], paso)) for k in CUERPOS_BRAZO
                       if k in nubes and len(nubes[k])]
        # la mano: sin los dedos (no sabemos cómo están; se quitan también los puntos medidos ahí)
        m = cadena.fk_marcos(np.zeros(6))
        T_w_mm = np.linalg.inv(m[5]) @ m[6]                       # wrist3 → mano_mount
        c_mm, self.r_dedos = np.asarray(dedos[0], float), float(dedos[1])
        self.c_dedos = T_w_mm[:3, :3] @ c_mm + T_w_mm[:3, 3]      # en wrist3
        mano = nubes.get("mano")
        if mano is not None and len(mano):
            mano = reducir(mano, paso)
            self.partes.append((5, mano[np.linalg.norm(mano - self.c_dedos, axis=1) > self.r_dedos]))

    def modelo(self, q):
        marcos = self.c.fk_marcos(np.asarray(q, float))
        M = np.vstack([P @ marcos[k][:3, :3].T + marcos[k][:3, 3] for k, P in self.partes])
        dedos = marcos[5][:3, :3] @ self.c_dedos + marcos[5][:3, 3]
        return M, dedos

    def coincide(self, q, t, prof_mm, K, T_mundo_cam, paso=2, tol=0.03):
        """Comprobación independiente del ICP (12-nov): el brazo del gemelo, movido con la
        corrección, proyectado en la imagen de profundidad. En cada píxel donde el gemelo dice que
        se ve el brazo: «sí» si la cámara mide esa distancia (±tol), «no» si mide MÁS LEJOS (ve a
        través de donde debería estar el brazo); si mide más cerca, algo lo tapa y no cuenta.
        Devuelve (píxeles del brazo en la imagen, fracción de «sí»)."""
        M, _ = self.modelo(q)
        Tcm = np.linalg.inv(T_mundo_cam)
        Pc = (M - np.asarray(t, float)) @ Tcm[:3, :3].T + Tcm[:3, 3]      # donde lo vería la cámara
        Pc = Pc[Pc[:, 2] > 0.2]
        if not len(Pc):
            return 0, 0.0
        h, w = prof_mm.shape[:2]
        u = np.round((Pc[:, 0] / Pc[:, 2] * K[0, 0] + K[0, 2]) / paso).astype(int)
        v = np.round((Pc[:, 1] / Pc[:, 2] * K[1, 1] + K[1, 2]) / paso).astype(int)
        hp, wp = (h + paso - 1) // paso, (w + paso - 1) // paso
        k = (u >= 0) & (u < wp) & (v >= 0) & (v < hp)
        if not k.any():
            return 0, 0.0
        zbuf = np.full((hp, wp), np.inf)
        np.minimum.at(zbuf, (v[k], u[k]), Pc[k, 2])                  # la cara del brazo que se ve
        vv, uu = np.nonzero(np.isfinite(zbuf))
        z_mod = zbuf[vv, uu]
        z_med = prof_mm[::paso, ::paso][vv, uu].astype(float) * 0.001
        hay = z_med > 0
        tol_z = tol + 0.01 * z_mod * z_mod
        si = hay & (np.abs(z_med - z_mod) <= tol_z)
        no = hay & (z_med > z_mod + tol_z)
        n = int(si.sum() + no.sum())
        return n, (float(si.sum()) / n if n else 0.0)

    def estimar(self, P, q, radio_ini=0.08, prof_mm=None, K=None, T_mundo_cam=None,
                min_pixeles=300, min_coincide=0.5):
        """Con prof_mm, K y T_mundo_cam (la calibración con la corrección actual) se exige además
        que el brazo se vea de verdad donde dice el ICP (coincide): si no, None."""
        r = self._icp(P, q, radio_ini)
        if r is None or prof_mm is None:
            return r
        n, f = self.coincide(q, r["t"], prof_mm, K, T_mundo_cam)
        r.update(pixeles=n, coincide=f)
        if n < min_pixeles or f < min_coincide:
            return None
        n0, f0 = self.coincide(q, np.zeros(3), prof_mm, K, T_mundo_cam)
        if f < f0:
            return None                                   # sin mover cuadraba mejor: no tocar
        return r

    def _icp(self, P, q, radio_ini=0.08):
        from scipy.spatial import cKDTree
        if P is None or len(P) < self.n_min:
            return None
        M, dedos = self.modelo(q)
        arbol = cKDTree(M)
        lo, hi = M.min(0) - radio_ini, M.max(0) + radio_ini
        P = P[np.all((P > lo) & (P < hi), axis=1)]
        if len(P) < self.n_min:
            return None
        d, _ = arbol.query(P, distance_upper_bound=radio_ini)
        P = P[np.isfinite(d)]
        P = P[np.linalg.norm(P - dedos, axis=1) > self.r_dedos]
        if len(P) < self.n_min:
            return None
        if len(P) > 6000:
            P = P[np.random.default_rng(0).choice(len(P), 6000, replace=False)]
        t = np.zeros(3)
        for radio in (0.08,) * 6 + (0.06,) * 4 + (0.045,) * 3 + (0.035, 0.03, 0.025) + (0.02,) * 4:
            d, i = arbol.query(P + t, distance_upper_bound=radio)
            ok = np.isfinite(d)
            if ok.sum() < self.n_min:
                return None
            t = t + np.median(M[i[ok]] - (P[ok] + t), axis=0)
        d, _ = arbol.query(P + t, distance_upper_bound=0.03)
        ok = np.isfinite(d)
        if ok.sum() < self.n_min:
            return None
        return dict(t=t, n=int(ok.sum()), residuo=float(np.median(d[ok])), frac=float(ok.mean()))


class Correccion:
    """La corrección acumulada (traslación en el mundo), con su archivo."""

    def __init__(self, ruta_calib, ganancia=0.5, max_m=0.15, residuo_max=0.012, paso_max=0.08):
        self.ruta = os.path.splitext(ruta_calib)[0] + ".correccion.json"
        self.paso_max = float(paso_max)
        self.ganancia, self.max_m, self.residuo_max = float(ganancia), float(max_m), float(residuo_max)
        self.t = np.zeros(3)
        self.calib_mtime = None
        self.n_medidas, self.t_ultima, self.ultima = 0, 0.0, None
        self._t_guardado, self._t_guardado_v = 0.0, np.zeros(3)

    def T(self):
        return T_traslacion(self.t)

    def nueva_calibracion(self, mtime):
        """Calibración (re)cargada: si el archivo de corrección es de ella, se sigue; si no, a 0."""
        self.calib_mtime = mtime
        self.t = np.zeros(3)
        try:
            with open(self.ruta, encoding="utf-8") as f:
                d = json.load(f)
            if mtime is not None and abs(float(d.get("calib_mtime", -1)) - mtime) < 1e-3:
                t = np.asarray(d.get("t", (0, 0, 0)), float)
                if t.shape == (3,) and np.linalg.norm(t) <= self.max_m:
                    self.t = t
        except (OSError, ValueError, TypeError):
            pass
        self._t_guardado_v = self.t.copy()

    def aplicar(self, medida, ahora=None):
        """Suma (con ganancia) una medida de AutoCalib.estimar. True si se ha usado."""
        ahora = time.time() if ahora is None else ahora
        self.ultima = medida
        if medida is None or medida["residuo"] > self.residuo_max:
            return False
        if np.linalg.norm(medida["t"]) > self.paso_max:
            return False                                   # un salto así de una vez: no es fiable
        nueva = self.t + self.ganancia * np.asarray(medida["t"], float)
        if np.linalg.norm(nueva) > self.max_m:
            return False                                   # demasiado: algo raro (no es el brazo)
        self.t = nueva
        self.n_medidas += 1
        self.t_ultima = ahora
        if np.linalg.norm(self.t - self._t_guardado_v) > 0.002 and ahora - self._t_guardado > 5.0:
            self.guardar(ahora)
        return True

    def borrar(self):
        """Vuelta a la calibración del archivo, sin corrección (botón «Borrar mapa»)."""
        self.t = np.zeros(3)
        self.n_medidas = 0
        try:
            os.remove(self.ruta)
        except OSError:
            pass
        self._t_guardado_v = self.t.copy()

    def guardar(self, ahora=None):
        ahora = time.time() if ahora is None else ahora
        try:
            with open(self.ruta, "w", encoding="utf-8") as f:
                json.dump(dict(t=[round(float(v), 4) for v in self.t], calib_mtime=self.calib_mtime,
                               fecha=time.strftime("%Y-%m-%d %H:%M:%S")), f)
            self._t_guardado, self._t_guardado_v = ahora, self.t.copy()
        except OSError:
            pass

    def texto(self):
        n = np.linalg.norm(self.t) * 100
        return f"corrección {n:.1f} cm ({', '.join(f'{v * 100:+.1f}' for v in self.t)})"
