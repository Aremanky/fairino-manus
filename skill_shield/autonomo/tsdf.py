"""
Fusión TSDF del mapa de obstáculos y mesas como planos (17-nov) — funciones puras.

FUSIÓN TSDF (en vez de contar impactos)
  Cada cubo guarda la DISTANCIA CON SIGNO a la superficie más cercana a lo largo del rayo de la
  cámara (truncada: ±1 = «lejos de cualquier superficie», 0 = justo en la superficie, negativo =
  detrás) y un PESO (cuántas veces se ha medido). Con cada imagen de profundidad, los cubos que
  caen en el rayo de un píxel se promedian con lo que ya tenían. El ruido del estéreo de una imagen
  a otra se PROMEDIA (en vez de acumularse en capas gruesas como al contar impactos) y las
  superficies salen finas. Es lo que hacen KinectFusion, voxblox o nvblox.
  · truncado: trunc_0 + k · z² (el ruido del estéreo crece con la distancia al cuadrado);
  · lo que está a más de dist_max (1,7 m por defecto) no pone superficies (ahí el estéreo tiene
    ruido de varios cm), pero sí dice que hasta dist_max no hay nada (espacio libre);
  · píxeles del propio robot, lo excluido y el suelo: solo espacio libre hasta ellos.

  Dos motores con la MISMA cuenta: numpy (siempre) y PyTorch en la GPU NVIDIA (si hay CUDA). Lo
  pesado (proyectar los ~750 000 cubos en la imagen y decidir qué se actualiza) va en la GPU; la
  media ponderada, en numpy (es la fuente de verdad: así borrar, vaciar o excluir son operaciones
  normales). Si la GPU falla, se sigue en numpy y se avisa.

MESAS COMO PLANOS
  La tapa de una mesa o una cajonera (metal liso, vista de canto) es lo peor para el estéreo: sale
  como una capa gruesa o una rampa de cubos. Se buscan planos HORIZONTALES en la nube (RANSAC: los
  puntos voladores no cuentan), se separan en trozos conexos y de cada uno se guarda su altura y su
  contorno en planta. Si se ven varias imágenes seguidas, en el mapa la tapa queda como una sola
  capa a la altura medida y lo que haya justo encima (el ruido) se borra; el planificador la usa
  además como suelo para la mano y el brazo (y la rellena hacia abajo: macizo).
"""
import math
import time

import numpy as np


# ══════════════════════════ fusión TSDF ══════════════════════════

def _proyectar_np(Cw, T_cam_mundo, Ks, h, w):
    C = Cw @ T_cam_mundo[:3, :3].T + T_cam_mundo[:3, 3]
    z = C[:, 2]
    delante = z > 0.10
    zz = np.where(delante, z, 1.0)
    u = np.rint(C[:, 0] / zz * Ks[0, 0] + Ks[0, 2]).astype(np.int64)
    v = np.rint(C[:, 1] / zz * Ks[1, 1] + Ks[1, 2]).astype(np.int64)
    en = delante & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return np.nonzero(en)[0], u, v, z


def observar_numpy(Cw, T_cam_mundo, d_sup, d_lib, Ks, trunc_0, k_trunc):
    """Qué cubos mide esta imagen y con qué valor. Cw: centros de los cubos (N x 3, mundo).
    d_sup: profundidad (m) de los píxeles que pueden poner SUPERFICIE (0 = no); d_lib: hasta dónde
    se sabe que está LIBRE en cada píxel (0 = nada). Ks: intrínsecos de esas imágenes.
    Devuelve (índices de los cubos actualizados, su valor observado en [-1, 1], índices de los
    cubos tapados: con una superficie claramente delante)."""
    h, w = d_sup.shape
    ii, u, v, z = _proyectar_np(Cw, T_cam_mundo, Ks, h, w)
    ds, dl, zz = d_sup[v[ii], u[ii]], d_lib[v[ii], u[ii]], z[ii]
    tr = trunc_0 + k_trunc * zz * zz
    sdf = ds - zz
    sup = (ds > 0) & (sdf > -tr)
    lib = ~sup & (dl > 0) & (dl - zz > tr)
    upd = sup | lib
    obs = np.where(sup, np.clip(sdf / tr, -1.0, 1.0), 1.0)
    tapado = (ds > 0) & (sdf <= -tr)
    return ii[upd], obs[upd].astype(np.float32), ii[tapado]


class ObservadorTorch:
    """Lo mismo que observar_numpy, en la GPU (PyTorch + CUDA). Los centros de los cubos se suben
    una vez. Lanza una excepción si no hay CUDA (el que lo usa se queda con numpy)."""

    def __init__(self, Cw, dispositivo="cuda"):
        import torch
        if dispositivo.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("PyTorch no ve ninguna GPU con CUDA")
        self.torch = torch
        self.dev = torch.device(dispositivo)
        self.Cw = torch.as_tensor(np.asarray(Cw, np.float32), device=self.dev)
        self.nombre = torch.cuda.get_device_name(self.dev) if self.dev.type == "cuda" else "cpu"

    def __call__(self, Cw, T_cam_mundo, d_sup, d_lib, Ks, trunc_0, k_trunc):
        t = self.torch
        T = t.as_tensor(np.asarray(T_cam_mundo, np.float32), device=self.dev)
        ds_img = t.as_tensor(np.ascontiguousarray(d_sup, np.float32), device=self.dev)
        dl_img = t.as_tensor(np.ascontiguousarray(d_lib, np.float32), device=self.dev)
        h, w = ds_img.shape
        C = self.Cw @ T[:3, :3].T + T[:3, 3]
        z = C[:, 2]
        delante = z > 0.10
        zz = t.where(delante, z, t.ones_like(z))
        u = t.round(C[:, 0] / zz * float(Ks[0, 0]) + float(Ks[0, 2])).long()
        v = t.round(C[:, 1] / zz * float(Ks[1, 1]) + float(Ks[1, 2])).long()
        en = delante & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ii = t.nonzero(en).squeeze(1)
        ui, vi, zi = u[ii], v[ii], z[ii]
        ds, dl = ds_img[vi, ui], dl_img[vi, ui]
        tr = float(trunc_0) + float(k_trunc) * zi * zi
        sdf = ds - zi
        sup = (ds > 0) & (sdf > -tr)
        lib = (~sup) & (dl > 0) & (dl - zi > tr)
        upd = sup | lib
        obs = t.where(sup, t.clamp(sdf / tr, -1.0, 1.0), t.ones_like(sdf))
        tapado = (ds > 0) & (sdf <= -tr)
        return (ii[upd].cpu().numpy(), obs[upd].float().cpu().numpy(), ii[tapado].cpu().numpy())


def crear_observador(Cw, motor="auto"):
    """motor: "auto" (GPU si hay, si no numpy), "gpu"/"cuda" o "numpy". Devuelve (función, nombre)."""
    if motor in ("auto", "gpu", "cuda"):
        try:
            o = ObservadorTorch(Cw)
            return o, f"GPU ({o.nombre})"
        except Exception as e:
            if motor != "auto":
                print(f"⚠️  [mapa] TSDF en la GPU no disponible ({e}): sigo con numpy (CPU).")
    return observar_numpy, "numpy (CPU)"


def fundir(D, W, idx, obs, w_max):
    """Media ponderada de lo observado en los cubos idx (D y W planos, se modifican)."""
    if not len(idx):
        return
    wj = W[idx]
    D[idx] = (D[idx] * wj + obs) / (wj + 1.0)
    W[idx] = np.minimum(wj + 1.0, w_max)


# ══════════════════════════ mesas como planos ══════════════════════════

def planos_horizontales(P, res=0.02, n_max=3, tol=0.012, min_puntos=300, inclinacion_max_deg=8.0,
                        area_min=0.02, iteraciones=80, rng=None):
    """Planos casi horizontales de la nube P (mundo): [(z, vértices xy de su contorno, área m²)],
    un elemento por trozo conexo de cada plano (dos mesas a la misma altura salen separadas)."""
    from scipy.ndimage import label
    from scipy.spatial import ConvexHull
    P = np.asarray(P, float).reshape(-1, 3)
    if len(P) < min_puntos:
        return []
    rng = np.random.default_rng(0) if rng is None else rng
    if len(P) > 8000:
        P = P[rng.choice(len(P), 8000, replace=False)]
    cos_max = math.cos(math.radians(inclinacion_max_deg))
    out, resto = [], P
    for _ in range(n_max):
        if len(resto) < min_puntos:
            break
        mejor, mejor_n = None, None
        for _ in range(iteraciones):
            A, B, C = resto[rng.choice(len(resto), 3, replace=False)]
            n = np.cross(B - A, C - A)
            nn = np.linalg.norm(n)
            if nn < 1e-9 or abs(n[2]) / nn < cos_max:
                continue
            n /= nn
            inl = np.abs((resto - A) @ n) < tol
            if mejor is None or inl.sum() > mejor.sum():
                mejor, mejor_n = inl, n
        if mejor is None or mejor.sum() < min_puntos:
            break
        Q = resto[mejor]
        resto = resto[~mejor]
        # trozos conexos en planta (celdas de res)
        ij = np.floor(Q[:, :2] / res).astype(int)
        lo = ij.min(0)
        g = np.zeros(tuple(ij.max(0) - lo + 1), bool)
        g[ij[:, 0] - lo[0], ij[:, 1] - lo[1]] = True
        etq, n_etq = label(g, structure=np.ones((3, 3), bool))
        pe = etq[ij[:, 0] - lo[0], ij[:, 1] - lo[1]]
        for k in range(1, n_etq + 1):
            celdas = int((etq == k).sum())
            if celdas * res * res < area_min:
                continue
            Qk = Q[pe == k]
            try:
                hull = ConvexHull(Qk[:, :2])
            except Exception:
                continue
            out.append((float(np.median(Qk[:, 2])), Qk[hull.vertices, :2].copy(), celdas * res * res))
    return out


class SeguidorSuperficies:
    """Mesas vistas en varias imágenes seguidas. Cada una: altura (mediana de las últimas medidas),
    contorno (el último) y cuántas veces se ha visto. Estable a partir de n_estable."""

    def __init__(self, n_estable=3, olvido_s=30.0, tol_z=0.025):
        self.n_estable, self.olvido_s, self.tol_z = int(n_estable), float(olvido_s), float(tol_z)
        self.pistas = []

    def actualizar(self, planos, ahora=None):
        ahora = time.time() if ahora is None else ahora
        for z, poli, area in planos:
            c = poli.mean(0)
            pista = None
            for p in self.pistas:
                if abs(p["z"] - z) < self.tol_z and (_dentro(p["poli"], c) or _dentro(poli, p["poli"].mean(0))):
                    pista = p
                    break
            if pista is None:
                self.pistas.append(dict(zs=[z], z=z, poli=poli, area=area, n=1, t=ahora))
            else:
                pista["zs"] = (pista["zs"] + [z])[-10:]
                pista["z"] = float(np.median(pista["zs"]))
                if area >= 0.7 * pista["area"]:             # un trozo tapado no encoge la mesa
                    pista["poli"], pista["area"] = poli, area
                pista["n"] += 1
                pista["t"] = ahora
        self.pistas = [p for p in self.pistas if ahora - p["t"] <= self.olvido_s]

    def estables(self):
        return [(p["z"], p["poli"]) for p in self.pistas if p["n"] >= self.n_estable]

    def borrar(self):
        self.pistas = []


def _dentro(poli, p):
    """¿El punto p (xy) está dentro del polígono convexo poli?"""
    poli = np.asarray(poli, float)
    if len(poli) < 3:
        return False
    d = np.roll(poli, -1, axis=0) - poli
    r = np.asarray(p, float) - poli
    cruz = d[:, 0] * r[:, 1] - d[:, 1] * r[:, 0]
    return bool(np.all(cruz >= 0) or np.all(cruz <= 0))


def dentro_poligono(poli, P2):
    """Máscara de los puntos P2 (N x 2) dentro del polígono convexo poli."""
    poli = np.asarray(poli, float)
    P2 = np.asarray(P2, float)
    if len(poli) < 3:
        return np.zeros(len(P2), bool)
    d = np.roll(poli, -1, axis=0) - poli
    signos = []
    for a, e in zip(poli, d):
        r = P2 - a
        signos.append(e[0] * r[:, 1] - e[1] * r[:, 0])
    S = np.stack(signos, 1)
    return np.all(S >= 0, axis=1) | np.all(S <= 0, axis=1)
