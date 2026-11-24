"""
Extrínseca A↔B AUTOMÁTICA con el cuerpo del operador (28-oct). Sin tablero ni pasos manuales.

Por qué
───────
El registro quieto del 28-oct dio 36–46 cm de separación (sesgo casi constante) entre lo que
miden A y B del mismo hombro/codo/muñeca: la calibración con el tablero ChArUco del 27-oct ya
no valía (una cámara se movió, o el tablero a 2 m no daba precisión). Con la extrínseca mal
la fusión descarta siempre una cámara ("!") y todo sale del estéreo de una sola OAK.

Cómo
────
Las dos cámaras ven los MISMOS puntos del cuerpo (nariz, ojos, orejas, hombros, codos,
muñecas, caderas, rodillas: índices de MediaPipe Pose). Cada uno, con la profundidad estéreo
de su cámara, da un par (p_A, p_B) del mismo punto 3D. Con unos cientos de pares de unos
segundos (mejor si mueves los brazos):
  1. RANSAC + Kabsch: transformación rígida robusta (los pares malos —oclusión, landmark
     equivocado, profundidad sobre ropa oscura— quedan fuera);
  2. ajuste fino ANISÓTROPO: cada punto pesa con su incertidumbre real (lateral = píxel·z/f,
     profundidad = k·z²), así el estéreo ruidoso en profundidad no tira del resultado.
Se acepta si el residuo mediano es pequeño y hay pares suficientes y repartidos.

Vigilancia
──────────
Durante la teleoperación se sigue midiendo la discrepancia A↔B de los puntos del cuerpo con
la extrínseca en uso. Si pasa de umbral_mal durante t_mal_s (alguien movió una cámara), se
recalibra sola con los fotogramas siguientes y se sustituye en caliente si mejora.

Convención: p_A = R_ab · p_B + t_ab (la de camaras_extrinsecas.json).
"""
import json
import math
import os
import shutil
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

# nariz, ojos (int.), orejas, hombros, codos, muñecas, caderas, rodillas
LANDMARKS_CUERPO = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26)


@dataclass
class ParametrosAutoExt:
    vis_min: float = 0.6            # visibilidad de MediaPipe en las DOS cámaras
    max_pares: int = 4000
    min_pares: int = 250
    min_fotogramas: int = 25
    min_extension_m: float = 0.25   # 2º valor singular de la nube (no colineal)
    ransac_iter: int = 300
    ransac_umbral_m: float = 0.08
    sigma_pix: float = 4.0
    sigma_z_k: float = 0.006
    acepta_residuo_m: float = 0.045 # residuo mediano máx. para aceptar
    acepta_frac_inl: float = 0.45
    # comprobación de la extrínseca en uso y vigilancia
    t_comprobar_s: float = 1.0      # al calibrar: tras esto, ¿la actual ya vale?
    umbral_ok_m: float = 0.05       # mediana A↔B por debajo de esto = la actual vale
    umbral_mal_m: float = 0.08      # durante la teleop: por encima, sospecha
    t_mal_s: float = 3.0            # ... durante este tiempo seguido → recalibrar
    ventana_s: float = 2.0
    t_max_calib_s: float = 15.0     # al calibrar: si no se consigue, se sigue con la cámara A
    refinar_cada_s: float = 10.0    # durante la teleop: re-ajuste con todo lo acumulado (se aplica
                                    # si usa ≥1,5× más puntos que el que está en uso y no empeora)
    max_ajuste: int = 900           # pares como mucho en cada ajuste (submuestra): coste acotado
    # vigilancia ADAPTADA al residuo real: con cámaras de verdad, A y B ven caras distintas del
    # cuerpo y la mediana A↔B "buena" ronda 8-10 cm (en la sala, 28-oct). Se recalibra solo si
    # supera  max(umbral_mal_m, factor_mal · residuo_base + margen_mal_m)
    cada_n_teleop: int = 3          # durante la teleop solo se toman pares 1 de cada N fotogramas
    factor_mal: float = 1.5
    margen_mal_m: float = 0.03


def kabsch(P, Q, w=None):
    """R, t que minimizan Σ w‖P − (R Q + t)‖²."""
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    cp, cq = w @ P, w @ Q
    H = ((Q - cq) * w[:, None]).T @ (P - cp)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cp - R @ cq


def _rotvec(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _covs(P, s_lat, s_z):
    """Covarianzas (N,3,3) de puntos con σ lateral s_lat y σ a lo largo del rayo s_z:
    C = s_lat²·I + (s_z² − s_lat²)·r rᵀ (r = dirección del rayo = P/‖P‖)."""
    r = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    C = (s_lat ** 2)[:, None, None] * np.eye(3)[None]
    return C + ((s_z ** 2 - s_lat ** 2)[:, None, None]) * (r[:, :, None] * r[:, None, :])


def _skew(V):
    """Matrices [v]× de (N,3) → (N,3,3)."""
    Z = np.zeros((len(V), 3, 3))
    Z[:, 0, 1], Z[:, 0, 2] = -V[:, 2], V[:, 1]
    Z[:, 1, 0], Z[:, 1, 2] = V[:, 2], -V[:, 0]
    Z[:, 2, 0], Z[:, 2, 1] = -V[:, 1], V[:, 0]
    return Z


def resolver_pares(PA, PB, SA, SB, nf, par, rng=None):
    """Extrínseca (R, t, info) a partir de pares de puntos (marco A, marco B) y sus σ
    (SA, SB = (N,2) con σ lateral y σ de profundidad). Todo vectorizado: unos ms."""
    N = N0 = len(PA)
    if N < 10:
        return None
    rng = rng or np.random.default_rng(0)
    # submuestra para que el coste no crezca con el búfer
    if N > par.max_ajuste:
        idx = rng.choice(N, par.max_ajuste, replace=False)
        PA, PB, SA, SB = PA[idx], PB[idx], SA[idx], SB[idx]
        N = len(PA)
    mejor = None
    for _ in range(par.ransac_iter):
        idx = rng.choice(N, 3, replace=False)
        q = PB[idx]
        if np.linalg.norm(np.cross(q[1] - q[0], q[2] - q[0])) < 0.02:
            continue                                    # casi colineales
        R, t = kabsch(PA[idx], PB[idx])
        res = np.linalg.norm(PA - (PB @ R.T + t), axis=1)
        inl = res < par.ransac_umbral_m
        n_in = int(inl.sum())
        if mejor is None or n_in > mejor[0] or (n_in == mejor[0] and res[inl].mean() < mejor[3]):
            mejor = (n_in, R, t, float(res[inl].mean()) if n_in else 1e9)
    if mejor is None or mejor[0] < 6:
        return None
    R, t = mejor[1], mejor[2]
    inl = np.linalg.norm(PA - (PB @ R.T + t), axis=1) < par.ransac_umbral_m
    R, t = kabsch(PA[inl], PB[inl])
    CA = _covs(PA, SA[:, 0], SA[:, 1])
    CB = _covs(PB, SB[:, 0], SB[:, 1])
    # ajuste anisótropo (Gauss-Newton con Huber), vectorizado
    for _ in range(10):
        res = np.linalg.norm(PA - (PB @ R.T + t), axis=1)
        m = res < par.ransac_umbral_m
        if m.sum() < 6:
            break
        rpb = PB[m] @ R.T
        r = PA[m] - (rpb + t)
        C = CA[m] + np.einsum("ij,njk,lk->nil", R, CB[m], R)
        Ci = np.linalg.inv(C)
        m2 = np.einsum("ni,nij,nj->n", r, Ci, r)
        w = np.where(m2 <= 9.0, 1.0, 3.0 / np.sqrt(np.maximum(m2, 1e-12)))      # Huber (δ = 3σ)
        # ∂r/∂ω = [R p_B]×, ∂r/∂t = −I
        J = np.concatenate([_skew(rpb), np.broadcast_to(-np.eye(3), (len(r), 3, 3))], axis=2)
        CiJ = Ci @ J
        H = np.einsum("n,nki,nkj->ij", w, J, CiJ)
        g = np.einsum("n,nki,nk->i", w, CiJ, r)
        try:
            dx = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
        except np.linalg.LinAlgError:
            break
        R = _rotvec(dx[:3]) @ R
        t = t + dx[3:]
        if np.linalg.norm(dx) < 1e-6:
            break
    res = np.linalg.norm(PA - (PB @ R.T + t), axis=1)
    inl = res < par.ransac_umbral_m
    med = float(np.median(res[inl])) if inl.any() else float("inf")
    frac = float(inl.mean())
    n_in_total = int(inl.sum())
    ok = bool(inl.sum() >= min(par.min_pares, N) * 0.5 and frac >= par.acepta_frac_inl
              and med <= par.acepta_residuo_m)
    info = dict(n=N, n_total=N0, fotogramas=nf, inliers=n_in_total, frac_inliers=frac, residuo_m=med, aceptada=ok,
                residuo_todos_m=float(np.median(res)))
    return R, t, info


class AutoExtrinseca:
    """Búfer circular de pares (punto del cuerpo en A, el mismo en B) + ajuste. El ajuste pesado
    va en un hilo aparte (resolver_async) para no frenar el bucle de la cámara."""

    def __init__(self, par_fusion=None, par: ParametrosAutoExt = None):
        self.par = par or ParametrosAutoExt()
        self.pf = par_fusion                     # Parametros de la fusión (parche de profundidad)
        M = self.par.max_pares
        self._T = np.zeros(M)
        self._F = np.zeros(M, dtype=np.int64)
        self._PA = np.zeros((M, 3))
        self._PB = np.zeros((M, 3))
        self._SA = np.zeros((M, 2))
        self._SB = np.zeros((M, 2))
        self._n = 0                              # pares válidos en el búfer
        self._i = 0                              # siguiente posición a escribir
        self.n_fot = 0
        self.t_mal = None
        self.meds = deque(maxlen=400)            # (t, mediana A↔B del fotograma) para vigilar
        self.ultimo = None                       # info del último ajuste
        self._pool = None
        self._fut = None

    def reiniciar(self, t=None):
        self._n = self._i = 0
        self.n_fot = 0
        self.t_mal = None
        self.meds.clear()
        self._fut = None                         # un ajuste en curso se ignora

    @property
    def n_pares(self):
        return self._n

    # ── datos ────────────────────────────────────────────────────────────────
    def _puntos(self, obs):
        """{índice: (p, s_lat, s_z)} de los landmarks del cuerpo visibles y con profundidad."""
        out = {}
        for i in LANDMARKS_CUERPO:
            pix, vis = obs.pixel(i)
            if pix is None or vis < self.par.vis_min:
                continue
            z = obs.z_en(pix[0], pix[1], self.pf) if self.pf is not None else None
            if z is None:
                continue
            v = max(vis, 0.3)
            out[i] = (obs.intr.rayo(*pix) * z, self.par.sigma_pix * z / obs.intr.f / v,
                      self.par.sigma_z_k * z * z / v)
        return out

    def alimentar(self, obs_a, obs_b, t, R=None, tt=None):
        """Añade los pares de un fotograma. Con (R, tt) = extrínseca en uso, guarda además la
        mediana A↔B de ESTE fotograma para la vigilancia (coste fijo por fotograma)."""
        if obs_a is None or obs_b is None:
            return 0
        a = self._puntos(obs_a)
        if not a:
            return 0
        b = self._puntos(obs_b)
        comunes = [i for i in a if i in b]
        if not comunes:
            return 0
        M = self.par.max_pares
        for i in comunes:
            k = self._i
            self._T[k], self._F[k] = t, self.n_fot
            self._PA[k], self._PB[k] = a[i][0], b[i][0]
            self._SA[k] = a[i][1:]
            self._SB[k] = b[i][1:]
            self._i = (k + 1) % M
            self._n = min(self._n + 1, M)
        self.n_fot += 1
        if R is not None and len(comunes) >= 3:
            PA = np.array([a[i][0] for i in comunes])
            PB = np.array([b[i][0] for i in comunes])
            self.meds.append((float(t), float(np.median(np.linalg.norm(PA - (PB @ R.T + tt), axis=1)))))
        return len(comunes)

    def _arrays(self, desde_t=None):
        n = self._n
        if n == 0:
            return None
        sel = slice(0, n)
        T, F = self._T[sel], self._F[sel]
        m = np.ones(n, dtype=bool) if desde_t is None else (T >= desde_t)
        if not m.any():
            return None
        return (self._PA[sel][m].copy(), self._PB[sel][m].copy(), self._SA[sel][m].copy(),
                self._SB[sel][m].copy(), int(len(np.unique(F[m]))))

    # ── comprobación / vigilancia ────────────────────────────────────────────
    def discrepancia(self, R, t, desde_t=None):
        """Mediana de ‖p_A − (R p_B + t)‖ de los pares (desde desde_t) o None."""
        a = self._arrays(desde_t)
        if a is None or len(a[0]) < 20:
            return None
        return float(np.median(np.linalg.norm(a[0] - (a[1] @ R.T + t), axis=1)))

    def discrepancia_reciente(self, ahora):
        v = [m for tm, m in self.meds if tm >= ahora - self.par.ventana_s]
        return float(np.median(v)) if len(v) >= 5 else None

    def vigilar(self, ahora, umbral):
        """True cuando la mediana A↔B reciente lleva t_mal_s seguidos por encima de 'umbral'."""
        d = self.discrepancia_reciente(ahora)
        if d is None:
            return False
        if d > umbral:
            if self.t_mal is None:
                self.t_mal = ahora
            return (ahora - self.t_mal) >= self.par.t_mal_s
        self.t_mal = None
        return False

    def listo(self):
        if self._n < self.par.min_pares or self.n_fot < self.par.min_fotogramas:
            return False
        PA = self._PA[:self._n]
        sv = np.linalg.svd(PA - PA.mean(axis=0), compute_uv=False) / math.sqrt(len(PA))
        return bool(sv[1] >= self.par.min_extension_m / 2)

    def progreso(self):
        return min(1.0, self._n / self.par.min_pares)

    # ── ajuste ───────────────────────────────────────────────────────────────
    def resolver(self, rng=None):
        """(R, t, info) o None (en este hilo)."""
        a = self._arrays()
        if a is None:
            return None
        sol = resolver_pares(*a, self.par, rng)
        if sol is not None:
            self.ultimo = sol[2]
        return sol

    def resolver_async(self):
        """Lanza el ajuste en un hilo aparte con una COPIA del búfer. Devuelve False si ya hay
        uno en curso. El resultado se recoge con resultado()."""
        if self._fut is not None:
            return False
        a = self._arrays()
        if a is None:
            return False
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AutoExt")
        self._fut = self._pool.submit(resolver_pares, *a, self.par, None)
        return True

    @property
    def en_curso(self):
        return self._fut is not None

    def resultado(self):
        """None si no hay ajuste terminado; si lo hay, (R, t, info) o (None, None, None)."""
        if self._fut is None or not self._fut.done():
            return None
        fut, self._fut = self._fut, None
        try:
            sol = fut.result()
        except Exception as e:                   # pragma: no cover
            print(f"⚠️  Ajuste de la extrínseca falló: {e}")
            sol = None
        if sol is None:
            return (None, None, None)
        self.ultimo = sol[2]
        return sol


def guardar_extrinsecas(ruta, R, t, info=None, mxid_a=None, mxid_b=None):
    """Escribe camaras_extrinsecas.json (mismo formato que calibraciones/calibrar_extrinseca.py) y deja una
    copia de la anterior como .bak_<fecha>."""
    if os.path.exists(ruta):
        try:
            shutil.copy2(ruta, ruta + time.strftime(".bak_%Y%m%d_%H%M%S"))
        except Exception:
            pass
        if mxid_a is None or mxid_b is None:
            try:
                with open(ruta, encoding="utf-8") as f:
                    d0 = json.load(f)
                mxid_a = mxid_a or d0.get("mxid_a")
                mxid_b = mxid_b or d0.get("mxid_b")
            except Exception:
                pass
    info = info or {}
    d = {
        "mxid_a": mxid_a, "mxid_b": mxid_b,
        "R_ab": np.asarray(R).tolist(), "t_ab": np.asarray(t).tolist(),
        "convencion": "p_A = R_ab @ p_B + t_ab",
        "metodo": "cuerpo (autocalib.py)",
        "n_muestras": int(info.get("inliers", 0)),
        "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "residuo_mediano_mm": round(1000 * float(info.get("residuo_m", float("nan"))), 1),
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    return ruta
