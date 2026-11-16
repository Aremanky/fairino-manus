"""
Choques de la teleoperación contra el MAPA de obstáculos de la cámara de cabeza (16-nov, modo GRABACION).

Envuelve el ModeloColision de la IK (cápsulas del FR5 + mano contra soporte, suelo y
TELEOP_FR5_OBSTACULOS) y le añade los vóxeles ocupados del mapa (los publica el proceso de la
cámara, skill_shield/autonomo/mapa.py). Tiene la MISMA interfaz, así que la IK hace con ellos lo
que ya hacía con cualquier choque: lleva el brazo lo más lejos posible sin chocar y se queda ahí
(estado «bloqueado», motivo «colision …-mapa») → el brazo FRENA ante la mesa, una caja, una persona…

ZONA DE AGARRE alrededor del objeto que se va a coger. Ahí los vóxeles no cuentan: la mesa justo
debajo del objeto, el borde del propio objeto y el ruido del estéreo (una capa de varios cm en la
mesa vista de lado) impedirían cogerlo con unas cápsulas que son gruesas a propósito. En su lugar,
nada del brazo ni de la mano puede bajar de la BASE del objeto (la superficie en la que está:
choque «…-mesa»). Lo que sobresale por encima del objeto dentro de la zona sigue contando.
La misma idea que las «zonas de apoyo» del planificador del modo autónomo (tarea.py).
"""
import math

import numpy as np


def zona_de(det, radio_extra=0.25, alto_extra=0.05, fondo=0.06):
    """Zona de agarre de una detección (formato de pelotas.py / herramientas.py: T_mundo, caja,
    tipo) → (centro xy, radio horizontal, z de la base, z de arriba, z de abajo de la zona) o None."""
    T = det.get("T_mundo") if isinstance(det, dict) else None
    if T is None:
        return None
    T = np.asarray(T, float).reshape(4, 4)
    c = T[:3, 3]
    caja = det.get("caja")
    if det.get("tipo") == "pelota" or caja is None:
        r = float(caja[0]) if caja else 0.5 * float(det.get("diametro_m") or det.get("tam") or 0.07)
        semi_alto, r_h = r, r
    else:
        s = np.abs(np.asarray(caja, float).reshape(3))
        A = np.abs(T[:3, :3]) @ s                 # semi-extensión de la caja girada en x, y, z del mundo
        semi_alto, r_h = float(A[2]), float(math.hypot(A[0], A[1]))
    z_base = float(c[2] - semi_alto)
    return (c[:2].copy(), r_h + float(radio_extra), z_base, float(c[2] + semi_alto + alto_extra),
            z_base - float(fondo))


class ColisionConMapa:
    """ModeloColision + mapa. base: el ModeloColision de la IK (teleop_v2.ik.col)."""

    def __init__(self, base, margen=0.02, paso=0.015, radio_dedos=0.012):
        self.base = base
        self.margen, self.paso, self.radio_dedos = float(margen), float(paso), float(radio_dedos)
        self._arbol, self._res, self.zonas = None, 0.02, []
        self.n_voxeles = 0
        self.activo = True

    def __getattr__(self, k):            # entorno, PARES_PROPIOS, c, R_MANO… los del modelo de siempre
        if k == "base":
            raise AttributeError(k)
        return getattr(self.base, k)

    def poner_mapa(self, centros, res, zonas=()):
        """centros: vóxeles ocupados (N x 3, mundo). zonas: [zona_de(...)] (sin None). → nº de vóxeles."""
        C = np.asarray(centros if centros is not None else [], float).reshape(-1, 3)
        zonas = [z for z in zonas if z is not None]
        quedan = np.ones(len(C), bool)
        for cxy, rh, _zb, z_arriba, z_abajo in zonas:
            quedan &= ~((np.linalg.norm(C[:, :2] - cxy, axis=1) < rh) & (C[:, 2] < z_arriba) & (C[:, 2] > z_abajo))
        C = C[quedan]
        arbol = None
        if len(C):
            from scipy.spatial import cKDTree
            arbol = cKDTree(C)
        # (cada referencia se cambia de golpe: la IK puede estar consultando en otro hilo)
        self._res = float(res or 0.02)
        self.zonas = zonas
        self._arbol = arbol
        self.n_voxeles = len(C)
        return len(C)

    def quitar_mapa(self):
        self._arbol, self.zonas, self.n_voxeles = None, [], 0

    def colisiones(self, q, marcos=None):
        """Las del modelo de siempre + (cápsula, "mapa", penetración) + (cápsula, "mesa", cuánto baja)."""
        out = list(self.base.colisiones(q, marcos))
        arbol, zonas, res = self._arbol, self.zonas, self._res
        if not self.activo or (arbol is None and not zonas):
            return out
        for c in self.base.capsulas(q, marcos):
            if c.nombre == "base":
                continue
            a, b = np.asarray(c.a, float), np.asarray(c.b, float)
            n = max(2, int(math.ceil(np.linalg.norm(b - a) / self.paso)) + 1)
            pts = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
            if arbol is not None:
                # la distancia es al CENTRO del vóxel más cercano: se resta medio vóxel
                pen = c.r + self.margen + 0.5 * res - float(np.min(arbol.query(pts)[0]))
                if pen > 0:
                    out.append((c.nombre, "mapa", pen))
            # la base del objeto hace de mesa dentro de la zona: los dedos pueden rozarla (radio
            # fino); el resto del brazo, con su radio de siempre
            rr = self.radio_dedos if c.nombre == "mano" else float(c.r)
            for cxy, rh, z_base, _za, _zab in zonas:
                dentro = np.linalg.norm(pts[:, :2] - cxy, axis=1) < rh
                if dentro.any():
                    baja = z_base - (pts[dentro, 2] - rr)
                    if float(baja.max()) > 0:
                        out.append((c.nombre, "mesa", float(baja.max())))
        return out

    def libre(self, q, marcos=None):
        return not self.colisiones(q, marcos)

    def camino_libre(self, q0, q1, n=24):
        for s in np.linspace(0.0, 1.0, n):
            if not self.libre(q0 + s * (q1 - q0)):
                return False
        return True
