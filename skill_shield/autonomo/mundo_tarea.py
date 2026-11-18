"""
Mundo SIMPLE del planificador de la tarea (18-nov, revisión del modo autónomo).

Antes la mesa se representaba de ~8 formas a la vez (cubos del mapa, superficie de la escena fija,
tapas fijas, planos del mapa, la mesa medida junto a la pelota, discos de apoyo, suelo de columnas
quitadas y el disco bajo la pelota), cada una con su margen. Aquí:

  · MESAS = LOSAS: una losa es la tapa de una mesa (altura z y polígono en planta) extruida hacia
    abajo hasta su fondo (por defecto, el suelo: las mesas y muebles se tratan como macizos).
    Salen de la escena fija (escena_fija.json), de los planos que encuentra el mapa (TSDF) y de la
    mesa medida alrededor del objeto (pelotas.py).
  · EL RESTO = los cubos del mapa en vivo, SIN la capa de la mesa (lo que la cámara ve de la propia
    mesa, con su ruido de unos cm por encima de la tapa: la losa ya la representa).
  · EL OBJETIVO se quita del mapa con su forma (lo hace Tarea con «excluir»).
  · TRES márgenes: mano↔cubos (AUTONOMO_TAREA_MARGEN_MAPA), brazo↔cubos (AUTONOMO_TAREA_MARGEN_BRAZO)
    y mano/brazo↔mesa (AUTONOMO_TAREA_HOLGURA_MESA_MANO / _BRAZO).

Sin numpy más que lo básico (nada de GPU): se puede probar en cualquier PC.
"""
import numpy as np


class Losa:
    """Tapa de una mesa (z_tapa, polígono convexo en planta) maciza hasta z_fondo."""

    __slots__ = ("z", "z_fondo", "poli", "lo", "hi", "_a", "_e")

    def __init__(self, z, poli, z_fondo):
        P = np.asarray(poli, float).reshape(-1, 2)
        c = P.mean(0)
        ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
        P = P[np.argsort(ang)]                               # en orden (antihorario)
        self.z, self.z_fondo, self.poli = float(z), float(min(z_fondo, z)), P
        self.lo, self.hi = P.min(0), P.max(0)
        self._a = P
        self._e = np.roll(P, -1, axis=0) - P

    def __repr__(self):
        return f"Losa(z={self.z:+.3f}, fondo={self.z_fondo:+.3f}, {len(self.poli)} vértices)"

    def dist_planta(self, P2):
        """Distancia (m) en planta de cada punto al polígono: 0 dentro, > 0 fuera."""
        P2 = np.asarray(P2, float).reshape(-1, 2)
        if len(self.poli) < 3:
            return np.full(len(P2), np.inf)
        r = P2[:, None, :] - self._a[None]                   # (N, E, 2)
        cruz = self._e[None, :, 0] * r[..., 1] - self._e[None, :, 1] * r[..., 0]
        dentro = np.all(cruz >= -1e-12, axis=1)              # antihorario: todo a la izquierda
        ee = np.einsum("ek,ek->e", self._e, self._e)
        t = np.clip(np.einsum("nek,ek->ne", r, self._e) / np.maximum(ee, 1e-18), 0.0, 1.0)
        d = np.linalg.norm(r - t[..., None] * self._e[None], axis=-1).min(1)
        d[dentro] = 0.0
        return d

    def mascara(self, P, holgura=0.0):
        """Puntos (mundo) a menos de 'holgura' de la losa (dentro, debajo de la tapa + holgura, por
        encima del fondo − holgura y a menos de holgura de su borde en planta)."""
        P = np.asarray(P, float).reshape(-1, 3)
        h = float(holgura)
        out = np.zeros(len(P), bool)
        cand = ((P[:, 2] < self.z + h) & (P[:, 2] > self.z_fondo - abs(h))
                & np.all(P[:, :2] > self.lo - abs(h), axis=1) & np.all(P[:, :2] < self.hi + abs(h), axis=1))
        if cand.any():
            out[np.nonzero(cand)[0]] = self.dist_planta(P[cand, :2]) <= max(h, 0.0)
        return out


def losas_de(superficies, fondo):
    """[(z, polígono)] → [Losa]. fondo(z) da la z del fondo de cada una."""
    out = []
    for s in superficies or ():
        z, poli = s[0], s[1]
        poli = np.asarray(poli, float).reshape(-1, 2)
        if len(poli) >= 3:
            out.append(Losa(float(z), poli, float(fondo(float(z)))))
    return out


def mascara_losas(losas, P, holgura=0.0):
    P = np.asarray(P, float).reshape(-1, 3)
    out = np.zeros(len(P), bool)
    for L in losas or ():
        out |= L.mascara(P, holgura)
    return out


def quitar_capa_mesa(C, losas, capa, res):
    """Quita de los cubos C (N x 3) los que son la PROPIA mesa: dentro de una losa en planta (+ medio
    cubo) y por debajo de su tapa + capa (capa: número o array por cubo, el ruido de la profundidad
    a esa distancia). Devuelve (C sin ellos, cuántos se han quitado)."""
    C = np.asarray(C, float).reshape(-1, 3)
    if not len(C) or not losas:
        return C, 0
    capa = np.broadcast_to(np.asarray(capa, float), (len(C),))
    quita = np.zeros(len(C), bool)
    for L in losas:
        cand = (C[:, 2] < L.z + capa) & (C[:, 2] > L.z_fondo - res)
        if cand.any():
            idx = np.nonzero(cand)[0]
            quita[idx] |= L.dist_planta(C[idx, :2]) <= 0.5 * res
    return C[~quita], int(quita.sum())


def losa_bajo(losas, p, tol):
    """La losa sobre la que está apoyado el punto p (su tapa a menos de tol de p[2]) o None."""
    p = np.asarray(p, float)
    mejor = None
    for L in losas or ():
        if abs(L.z - p[2]) <= tol and L.dist_planta(p[None, :2])[0] <= 0.0:
            if mejor is None or abs(L.z - p[2]) < abs(mejor.z - p[2]):
                mejor = L
    return mejor
