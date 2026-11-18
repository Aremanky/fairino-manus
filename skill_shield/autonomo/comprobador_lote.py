"""
Comprobación del mapa EN LOTE (17-nov): mapa de distancias + GPU.

El planificador comprobaba cada postura contra los cubos del mapa con un árbol de búsqueda: ~1800
puntos (mano, lo que lleva, mallas del brazo, cápsulas) por postura y miles de posturas por plan. Era
casi la mitad del tiempo de planificar.

Ahora:
  · una vez por mapa: la DISTANCIA de cada cubo de una rejilla al cubo ocupado más cercano
    (transformada de distancia, en la misma rejilla que los cubos: alineada con ellos);
  · para un lote de posturas (todas las de un tramo a la vez): se colocan sus puntos con la FK y se
    mira la distancia de su cubo en la rejilla (una lectura por punto, en la GPU si hay PyTorch con CUDA,
    si no en numpy; el resultado es el mismo).
  · Es CONSERVADOR: la distancia real de un punto es ≥ la de su cubo − media diagonal del cubo. Si con
    eso ya está claramente libre, la postura no necesita la búsqueda exacta; si no, motivo() la hace
    como siempre. Así no cambia ninguna decisión, solo cuánto cuesta tomarla.
"""
import math

import numpy as np


class MapaDistancias:
    """Distancia (m) de cada cubo de una rejilla alineada con los cubos ocupados C al ocupado más cercano."""

    def __init__(self, C, res, margen=0.40):
        from scipy.ndimage import distance_transform_edt
        C = np.asarray(C, float).reshape(-1, 3)
        self.res = float(res)
        k = int(math.ceil(margen / self.res))
        self.origen = C.min(0) - (k + 0.5) * self.res          # centros de la rejilla = posiciones de los cubos
        self.n = np.ceil((C.max(0) - C.min(0)) / self.res).astype(int) + 2 * k + 2
        occ = np.zeros(tuple(self.n), bool)
        idx = np.floor((C - self.origen) / self.res).astype(int)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        self.margen = k * self.res                             # fuera de la rejilla: al menos esto de lejos
        # con un borde de una celda que vale «margen»: lo de fuera se lleva (clip) a ese borde y la lectura
        # es una sola indexación plana, sin máscaras (lo de fuera está a más de «margen» de todo)
        D = np.full(tuple(self.n + 2), self.margen, np.float32)
        D[1:-1, 1:-1, 1:-1] = distance_transform_edt(~occ) * self.res
        self.D = D
        self._plano = D.reshape(-1)
        self._pasos = np.array([D.shape[1] * D.shape[2], D.shape[2], 1], np.int64)
        # lo que puede fallar una lectura: media diagonal del cubo del punto + lo que un cubo del mapa se
        # separe del centro de su celda (0 si vienen de la rejilla del mapa, que es lo normal)
        desvio = float(np.max(np.linalg.norm(C - (self.origen + (idx + 0.5) * self.res), axis=1))) if len(C) else 0.0
        self.h = 0.5 * math.sqrt(3.0) * self.res + desvio + 1e-6
        self._D_gpu = None

    def leer(self, P):
        """Distancia de la rejilla en los puntos P (N x 3, numpy); fuera de ella, self.margen."""
        idx = np.floor((np.asarray(P, float) - self.origen) / self.res).astype(np.int64) + 1
        np.clip(idx, 0, self.n + 1, out=idx)
        return self._plano[idx @ self._pasos]

    def leer_gpu(self, P, torch, dev):
        """Lo mismo con PyTorch (P tensor N x 3 en dev). Devuelve un tensor."""
        if self._D_gpu is None:
            self._D_gpu = torch.as_tensor(self._plano, device=dev)
            self._o_gpu = torch.as_tensor(self.origen, dtype=torch.float32, device=dev)
            self._max_gpu = torch.as_tensor(self.n + 1, dtype=torch.int64, device=dev)
            self._pasos_gpu = torch.as_tensor(self._pasos, dtype=torch.int64, device=dev)
        idx = torch.floor((P - self._o_gpu) / self.res).long() + 1
        idx = torch.minimum(torch.clamp(idx, min=0), self._max_gpu)
        return self._D_gpu[(idx * self._pasos_gpu).sum(dim=1)]


def crear_motor(motor="auto"):
    """(torch, dispositivo) si se pide/puede la GPU; (None, None) para numpy."""
    if motor in ("auto", "gpu", "cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return torch, torch.device("cuda")
        except Exception:
            pass
    return None, None


def libres_en_lote(md, puntos_por_postura, umbrales, torch=None, dev=None):
    """puntos_por_postura: array (N, M, 3) (todas las posturas con los mismos M puntos); umbrales: (M,).
    Devuelve (N,) bool: True = claramente libre (ningún punto más cerca de un cubo que su umbral)."""
    N, Mp, _ = puntos_por_postura.shape
    if torch is not None:
        P = torch.as_tensor(np.ascontiguousarray(puntos_por_postura.reshape(-1, 3), np.float32), device=dev)
        U = torch.as_tensor(np.asarray(umbrales, np.float32), device=dev)
        d = md.leer_gpu(P, torch, dev).reshape(N, Mp)
        return torch.all(d - md.h >= U[None, :], dim=1).cpu().numpy()
    d = md.leer(puntos_por_postura.reshape(-1, 3)).reshape(N, Mp)
    return np.all(d - md.h >= np.asarray(umbrales, np.float32)[None, :], axis=1)


def minimo_transformado(md, R, t, pts, torch=None, dev=None):
    """Para cada postura n: mín sobre los puntos pts (M x 3, en su marco) llevados con (R[n], t[n]) al
    marco del mapa de distancias, de (distancia leída − md.h) — una cota INFERIOR de la distancia real
    del objeto a la nube del mapa. R: (N, 3, 3), t: (N, 3). La transformación también va en la GPU."""
    N = len(R)
    if torch is not None:
        Rg = torch.as_tensor(np.asarray(R, np.float32), device=dev)
        tg = torch.as_tensor(np.asarray(t, np.float32), device=dev)
        pg = torch.as_tensor(np.asarray(pts, np.float32), device=dev)
        P = torch.matmul(Rg, pg.T).transpose(1, 2) + tg[:, None, :]
        d = md.leer_gpu(P.reshape(-1, 3), torch, dev).reshape(N, -1)
        return (d.min(dim=1).values - md.h).cpu().numpy()
    P = np.matmul(R, np.asarray(pts, float).T).transpose(0, 2, 1) + t[:, None, :]   # (matmul: BLAS, no einsum)
    return md.leer(P.reshape(-1, 3)).reshape(N, -1).min(axis=1) - md.h


def dist_segmento_lote(P, a, b):
    """(N, M) distancia de los puntos P (N, M, 3) al segmento a-b."""
    ab = np.asarray(b, float) - np.asarray(a, float)
    L2 = float(ab @ ab)
    t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0) if L2 > 1e-12 else np.zeros(P.shape[:2])
    return np.linalg.norm(P - (a + t[..., None] * ab), axis=-1)


def dist_caja_lote(P, caja):
    """(N, M) distancia con signo de los puntos P (N, M, 3) a una ik_teleop.Caja."""
    l = (P - caja.centro) @ caja.R
    q = np.abs(l) - caja.semi
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


def exacto_en_lote(md, arbol, P, U, torch=None, dev=None):
    """(N,) bool EXACTO: True si en la postura n ningún punto P[n, m] está a menos de U[m] de la nube del
    árbol (lo mismo que una búsqueda en el árbol de todos los puntos, postura a postura).
    Cómo: el mapa de distancias (GPU si hay) descarta de golpe los puntos que seguro están lejos; solo los
    que quedan cerca —pocos— se buscan en el árbol, todos en una llamada y con todos los núcleos."""
    N, Mp = P.shape[:2]
    U = np.broadcast_to(np.asarray(U, float), (Mp,))
    Pf = np.ascontiguousarray(P.reshape(-1, 3))
    if torch is not None:
        Pg = torch.as_tensor(Pf.astype(np.float32), device=dev)
        Ug = torch.as_tensor(U.astype(np.float32), device=dev).repeat(N)
        idx = torch.nonzero(md.leer_gpu(Pg, torch, dev) - md.h < Ug).reshape(-1).cpu().numpy()
    else:
        idx = np.flatnonzero(md.leer(Pf) - md.h < np.tile(U, N))
    ok = np.ones(N, bool)
    if len(idx):
        Ui = U[idx % Mp]
        try:
            d, _ = arbol.query(Pf[idx], distance_upper_bound=float(Ui.max()), workers=-1)
        except TypeError:                                    # scipy viejo: sin workers
            d, _ = arbol.query(Pf[idx], distance_upper_bound=float(Ui.max()))
        ok[idx[d < Ui] // Mp] = False
    return ok
