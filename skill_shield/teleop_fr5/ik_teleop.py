"""
IK de teleoperación del FR5: configuración, límites, colisiones y "buscar otro camino".

Cada fotograma recibe el objetivo (posición de la muñeca de la OrcaHand + orientación de la
mano + codo sugerido) y devuelve los 6 joints:

  1. IK ANALÍTICA: las (hasta) 8 configuraciones exactas que ponen la mano donde se pide.
  2. FILTRO: límites articulares (los reales del FR5 estrechados con margen), colisiones
     (brazo consigo mismo, suelo, columna del soporte) con cápsulas, altura del codo.
  3. ELECCIÓN: coste = continuidad (distancia articular a la solución anterior, la base pesa
     más) + lado del codo (el más parecido al codo humano) + penalización de "configuración
     girada". Se elige la de MENOR coste, pero no se salta a otra rama de la IK si la actual
     sigue valiendo: cambiar de rama es un movimiento grande.
  4. SI NO HAY SOLUCIÓN EXACTA (fuera de alcance, límite, colisión): IK numérica desde la
     solución anterior con la posición como prioridad y la orientación como tarea secundaria
     (deja libre primero la pronosupinación, luego toda la orientación). El brazo llega lo más
     cerca posible SIN violar límites ni chocar.
  5. "OTRO CAMINO": si la mejor configuración está en OTRA rama (salto articular grande) o la
     rama actual lleva un tiempo atascada (error > umbral), se planifica la transición: línea
     recta en el espacio articular comprobada contra colisiones; si choca, se prueban puntos
     intermedios (codo levantado, muñeca primero...). La transición se ejecuta a velocidad
     limitada y durante ella no se persigue el objetivo. Si no hay camino libre, el brazo se
     queda donde está y se informa.

No usa MuJoCo: la cinemática es la de `cinematica.CadenaFR5` y las colisiones, cápsulas.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from skill_shield.teleop_fr5.cinematica import (
    CadenaFR5, OFFSET_MUNECA_ORCA, T_de, wrap,
)


# ══════════════════════════════════════════════════════════════════════════════
# colisiones con cápsulas
# ══════════════════════════════════════════════════════════════════════════════

def _dist_segmentos(p1, q1, p2, q2):
    """Distancia mínima entre los segmentos p1-q1 y p2-q2 (Ericson, Real-Time Collision Detection).
    Aritmética escalar (28-oct): se llama decenas de veces por fotograma y numpy sobre vectores
    de 3 elementos cuesta más en sobrecarga que en cálculo."""
    p1x, p1y, p1z = float(p1[0]), float(p1[1]), float(p1[2])
    p2x, p2y, p2z = float(p2[0]), float(p2[1]), float(p2[2])
    d1x, d1y, d1z = float(q1[0]) - p1x, float(q1[1]) - p1y, float(q1[2]) - p1z
    d2x, d2y, d2z = float(q2[0]) - p2x, float(q2[1]) - p2y, float(q2[2]) - p2z
    rx, ry, rz = p1x - p2x, p1y - p2y, p1z - p2z
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    f = d2x * rx + d2y * ry + d2z * rz
    if a <= 1e-12 and e <= 1e-12:
        return math.sqrt(rx * rx + ry * ry + rz * rz)
    if a <= 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = d1x * rx + d1y * ry + d1z * rz
        if e <= 1e-12:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = d1x * d2x + d1y * d2y + d1z * d2z
            den = a * e - b * b
            s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    dx = rx + d1x * s - d2x * t
    dy = ry + d1y * s - d2y * t
    dz = rz + d1z * s - d2z * t
    return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass
class Capsula:
    nombre: str
    a: np.ndarray
    b: np.ndarray
    r: float


@dataclass
class Caja:
    """Obstáculo o zona prohibida con forma de caja (5-nov): centro y semi-lados en el MUNDO,
    R = orientación (columnas = ejes de la caja; None = alineada con el mundo)."""
    nombre: str
    centro: np.ndarray
    semi: np.ndarray
    R: np.ndarray = None
    zona: bool = False            # True = zona prohibida (no es un objeto: espacio reservado)

    def __post_init__(self):
        self.centro = np.asarray(self.centro, dtype=float)
        self.semi = np.asarray(self.semi, dtype=float)
        self.R = np.eye(3) if self.R is None else np.asarray(self.R, dtype=float)
        self.radio_env = float(np.linalg.norm(self.semi))      # esfera envolvente

    def sdf(self, p):
        """Distancia con signo de p a la caja (negativa dentro) y su gradiente (normal hacia
        fuera, unitaria)."""
        l = self.R.T @ (np.asarray(p, dtype=float) - self.centro)
        q = np.abs(l) - self.semi
        fuera = np.maximum(q, 0.0)
        nf = float(np.linalg.norm(fuera))
        if nf > 1e-12:                                   # fuera: hacia el punto más cercano
            g = np.sign(l) * fuera / nf
            return nf, self.R @ g
        i = int(np.argmax(q))                            # dentro: hacia la cara más cercana
        g = np.zeros(3)
        g[i] = 1.0 if l[i] >= 0.0 else -1.0
        return float(q[i]), self.R @ g

    def segmento(self, a, b, n_iter=18):
        """Punto del segmento a-b más cercano a la caja: (punto, normal, distancia con signo).
        La distancia con signo a un convexo es convexa a lo largo del segmento → búsqueda
        ternaria."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        d = b - a
        lo, hi = 0.0, 1.0
        for _ in range(n_iter):
            m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
            if self.sdf(a + m1 * d)[0] <= self.sdf(a + m2 * d)[0]:
                hi = m2
            else:
                lo = m1
        p = a + 0.5 * (lo + hi) * d
        sd, n = self.sdf(p)
        return p, n, sd


def _lejos_caja(a, b, r, caja, umbral):
    """Descarte rápido por esferas envolventes (cápsula a-b de radio r contra la caja)."""
    c = 0.5 * (np.asarray(a) + np.asarray(b))
    rc = 0.5 * float(np.linalg.norm(np.asarray(b) - np.asarray(a))) + r
    return float(np.linalg.norm(c - caja.centro)) - rc - caja.radio_env > umbral


@dataclass
class Entorno:
    """Obstáculos fijos: suelo (z), cápsulas (columna del soporte...) y cajas (mesa, peana,
    cámaras, zonas prohibidas — 5-nov, ver skill_shield/teleop_fr5/obstaculos.py)."""
    z_suelo: float = None
    capsulas: list = field(default_factory=list)
    margen: float = 0.03
    cajas: list = field(default_factory=list)


def entorno_soporte_hombros(z_suelo=-0.975):
    """Columna + cabezal del soporte de hombros (fairino_description/mjcf/decoracion/
    soporte_hombros.xml): columna en x = 0.26, de z = −0.78 a 0.43, sección 0.12 × 0.15."""
    return Entorno(z_suelo=z_suelo, capsulas=[
        Capsula("columna", np.array([0.26, 0.0, -0.80]), np.array([0.26, 0.0, 0.43]), 0.10),
        Capsula("cabezal", np.array([0.26, 0.0, 0.43]), np.array([0.26, 0.0, 0.55]), 0.16),
        Capsula("bastidor", np.array([-0.21, -0.375, -0.82]), np.array([-0.21, 0.375, -0.82]), 0.05),
    ])


class ModeloColision:
    """Cápsulas del FR5 + OrcaHand a partir de la FK, y pares que se comprueban."""
    # radios (m) con algo de margen
    R_BASE, R_BRAZO, R_ANTEBRAZO, R_MUNECA, R_MANO = 0.09, 0.07, 0.06, 0.055, 0.07

    def __init__(self, cadena: CadenaFR5, entorno: Entorno = None):
        self.c = cadena
        self.entorno = entorno or Entorno()

    def capsulas(self, q, marcos=None):
        m = self.c.fk_marcos(q) if marcos is None else marcos
        base = self.c.T_base[:3, 3]
        hombro, codo, wrist1 = m[1][:3, 3], m[2][:3, 3], m[3][:3, 3]
        brida = m[5][:3, 3]
        Tt = m[6]
        punta = Tt[:3, :3] @ np.array([0.0, 0.26, 0.0]) + Tt[:3, 3]      # punta de los dedos
        return [
            Capsula("base", base, hombro, self.R_BASE),
            Capsula("brazo", hombro, codo, self.R_BRAZO),
            Capsula("antebrazo", codo, wrist1, self.R_ANTEBRAZO),
            Capsula("muneca", wrist1, brida, self.R_MUNECA),
            Capsula("mano", brida, punta, self.R_MANO),
        ]

    PARES_PROPIOS = (("mano", "brazo"), ("mano", "base"), ("muneca", "brazo"), ("muneca", "base"),
                     ("antebrazo", "base"))

    def colisiones(self, q, marcos=None):
        """Lista de (nombre_a, nombre_b, penetración) de todas las colisiones de q."""
        caps = {c.nombre: c for c in self.capsulas(q, marcos)}
        out = []
        for a, b in self.PARES_PROPIOS:
            ca, cb = caps[a], caps[b]
            d = _dist_segmentos(ca.a, ca.b, cb.a, cb.b) - ca.r - cb.r
            if d < 0.0:
                out.append((a, b, -d))
        ent = self.entorno
        for nombre in ("brazo", "antebrazo", "muneca", "mano"):
            c = caps[nombre]
            if ent.z_suelo is not None:
                z = min(float(c.a[2]), float(c.b[2])) - c.r - ent.margen
                if z < ent.z_suelo:
                    out.append((nombre, "suelo", ent.z_suelo - z))
            for o in ent.capsulas:
                d = _dist_segmentos(c.a, c.b, o.a, o.b) - c.r - o.r - ent.margen
                if d < 0.0:
                    out.append((nombre, o.nombre, -d))
            for o in ent.cajas:
                if _lejos_caja(c.a, c.b, c.r, o, ent.margen):
                    continue
                d = o.segmento(c.a, c.b)[2] - c.r - ent.margen
                if d < 0.0:
                    out.append((nombre, o.nombre, -d))
        return out

    def libre(self, q, marcos=None):
        return not self.colisiones(q, marcos)

    def camino_libre(self, q0, q1, n=24):
        """¿La interpolación lineal articular q0→q1 está libre de colisiones?"""
        for s in np.linspace(0.0, 1.0, n):
            if not self.libre(q0 + s * (q1 - q0)):
                return False
        return True


# ══════════════════════════════════════════════════════════════════════════════
# controlador
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParametrosIK:
    margen_limites: float = math.radians(5.0)     # rad de margen sobre los límites del FR5
    limites_extra: dict = field(default_factory=dict)   # {índice joint: (lo, hi)} estrechados ("humanos")
    w_continuidad: tuple = (2.0, 1.0, 1.0, 0.6, 0.6, 0.4)   # peso por joint de la distancia a q_prev
    w_codo: float = 2.5           # m⁻¹: coste por metro de distancia del codo al codo sugerido
    w_codo_lado: float = 0.8      # coste por estar el codo al lado contrario del humano
    salto_rama: float = 1.0       # rad: distancia articular (ponderada) que ya es "otra rama"
    histeresis_rama: float = 0.35 # la otra rama debe ser mejor por este coste para cambiar
    t_atasco_s: float = 0.6       # s con error alto antes de buscar otro camino
    error_atasco_m: float = 0.04  # m de error de posición que cuenta como atasco
    v_transicion: float = 1.2     # rad/s máx. de cada joint durante una transición planificada
    v_seguimiento: float = 3.0    # rad/s máx. de cada joint siguiendo el objetivo (el FR5 llega a 3.15)
    hold_objetivo_s: float = 0.5  # s que se mantiene el último objetivo si deja de llegar
    z_codo_min: float = None      # m: altura mínima del codo del robot (None = sin límite)
    # ── postura humana (ver PosturaHumana) ──
    postura_humana: bool = True   # el robot solo adopta posturas que un brazo humano puede adoptar
    codo_signo: int = None        # +1 / −1: lado "bíceps" de joint3; None = se decide solo
    codo_flex_max_deg: float = 145.0
    codo_hiperext_deg: float = 5.0        # ° de hiperextensión tolerada (ruido)
    muneca_q4_deg: float = 80.0           # |joint4| ≤ esto (inclinación de la mano en el plano del brazo)
    muneca_q5_deg: float = 60.0           # |joint5 − 90°| ≤ esto (inclinación perpendicular)
    hombro_cono: bool = True              # el brazo del robot dentro del cono del hombro humano
    hombro_az_min: float = -45.0          # ° (igual que retargeting.LimitesAnatomicos)
    hombro_az_max: float = 130.0
    hombro_ext_elev: float = 55.0
    ejes: tuple = ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))   # (fuera, arriba, delante) del robot


class PosturaHumana:
    """
    Límites HUMANOS reales aplicados a la postura del ROBOT (no solo a la medida). Un brazo tipo
    UR tiene 6 ejes y el humano 7 (le falta la rotación del húmero), así que de las 8
    configuraciones de la IK solo una se parece a un brazo: las demás llegan al mismo punto con
    la muñeca volteada (joint5 < 0 y joint4 girado ~180°), el codo "al revés" o el hombro
    dado la vuelta. Aquí se excluyen:
      · CODO: flexión solo hacia el lado del bíceps (signo fijo de joint3), 0 … codo_flex_max.
        El lado se decide solo: con el brazo hacia delante-abajo, el antebrazo se dobla hacia
        ARRIBA (como un humano con el húmero en neutro). Configurable con codo_signo.
      · MUÑECA: la mano no se tuerce más que una humana respecto al antebrazo:
        |joint4| ≤ muneca_q4_deg y |joint5 − 90°| ≤ muneca_q5_deg (joint4 = 0, joint5 = +90° es
        la mano alineada con el antebrazo). Esto elimina la muñeca volteada.
      · HOMBRO: el brazo (hombro→codo) del robot dentro del cono del hombro humano (azimut
        permitido según la elevación, en los ejes fuera/arriba/delante del robot): nada de
        brazo por detrás de la espalda ni cruzado más allá del pecho.
    Las dos primeras son cajas articulares (las respeta también la IK numérica); la tercera se
    comprueba sobre la FK y además se convierte en cajas conservadoras de joint1/joint2.
    """

    def __init__(self, cadena: CadenaFR5, par: ParametrosIK, lo, hi):
        self.c = cadena
        self.par = par
        self.ejes = np.column_stack([np.asarray(v, float) for v in par.ejes])   # (fuera, arriba, delante) → mundo
        self.codo_signo = par.codo_signo if par.codo_signo in (1, -1) else self._elegir_lado_codo(lo, hi)
        self.lo, self.hi = self.cajas(lo, hi)

    # — direcciones en ejes del torso del robot —
    def a_torso(self, v):
        return self.ejes.T @ np.asarray(v, float)

    @staticmethod
    def az_elev(d):
        elev = math.degrees(math.acos(float(np.clip(-d[1], -1.0, 1.0))))
        az = math.degrees(math.atan2(float(d[0]), float(d[2])))
        return az, elev

    def rango_azimut(self, elev):
        p = self.par
        if elev <= p.hombro_ext_elev:
            az_max = 180.0
        elif elev >= 90.0:
            az_max = p.hombro_az_max
        else:
            t = (elev - p.hombro_ext_elev) / (90.0 - p.hombro_ext_elev)
            az_max = 180.0 + t * (p.hombro_az_max - 180.0)
        return p.hombro_az_min, az_max

    def q12_para_direccion(self, d_mundo, q_ref=None):
        """(q1, q2) que ponen el brazo (hombro→codo) del robot en la dirección d_mundo, en forma
        cerrada (cadena UR: en el marco DH 0 el brazo apunta a −(c1·c2, s1·c2, s2)). Rama "normal"
        (cos q2 ≥ 0); si q_ref se da, de las dos ramas se devuelve la más cercana."""
        dh = self.c.dh_equivalente()
        R0 = (self.c.T_base @ dh["T_0"])[:3, :3]
        d = R0.T @ (np.asarray(d_mundo, float) / np.linalg.norm(d_mundo))
        th2 = math.asin(float(np.clip(-d[2], -1.0, 1.0)))
        c2 = math.cos(th2)
        th1 = 0.0 if abs(c2) < 1e-9 else math.atan2(-d[1] / c2, -d[0] / c2)
        s1, s2 = dh["signos"][0], dh["signos"][1]
        cands = [(th1 * s1, th2 * s2), (wrap(th1 + math.pi) * s1, (math.pi - th2) * s2)]
        if q_ref is None:
            return cands[0]
        return min(cands, key=lambda q: abs(wrap(q[0] - q_ref[0])) + abs(wrap(q[1] - q_ref[1])))

    def _elegir_lado_codo(self, lo, hi):
        """Signo de joint3 con el que el antebrazo se dobla hacia ARRIBA (lado del bíceps) con el
        brazo apuntando hacia delante-abajo."""
        d_ref = self.ejes @ np.array([0.3, -0.5, 0.8])
        q1, q2 = self.q12_para_direccion(d_ref)
        arriba = self.ejes[:, 1]
        mejor = {}
        for sg in (1, -1):
            m = self.c.fk_marcos(np.array([q1, q2, sg * 1.2, 0.0, math.pi / 2, 0.0]))
            fore = m[3][:3, 3] - m[2][:3, 3]
            mejor[sg] = float(fore @ arriba) / np.linalg.norm(fore)
        return 1 if mejor[1] >= mejor[-1] else -1

    def cajas(self, lo, hi):
        """Cajas articulares humanas: codo de un lado, muñeca alrededor de la recta, hombro
        (joint1/joint2) por el cono (envolvente conservadora)."""
        p = self.par
        lo, hi = lo.copy(), hi.copy()
        flex = math.radians(p.codo_flex_max_deg)
        hip = math.radians(p.codo_hiperext_deg)
        if self.codo_signo > 0:
            lo[2], hi[2] = max(lo[2], -hip), min(hi[2], flex)
        else:
            lo[2], hi[2] = max(lo[2], -flex), min(hi[2], hip)
        q4 = math.radians(p.muneca_q4_deg)
        lo[3], hi[3] = max(lo[3], -q4), min(hi[3], q4)
        q5 = math.radians(p.muneca_q5_deg)
        lo[4], hi[4] = max(lo[4], math.pi / 2 - q5), min(hi[4], math.pi / 2 + q5)
        if p.hombro_cono:
            # recorrer el cono humano (elevación 15°..165°, azimut permitido) y ver qué q1, q2 usa
            q1s, q2s = [], []
            for elev in np.linspace(15.0, 165.0, 16):
                az_lo, az_hi = self.rango_azimut(elev)
                for az in np.linspace(az_lo, az_hi, 16):
                    e, a = math.radians(elev), math.radians(az)
                    d_t = np.array([math.sin(e) * math.sin(a), -math.cos(e), math.sin(e) * math.cos(a)])
                    q1, q2 = self.q12_para_direccion(self.ejes @ d_t)
                    q1s.append(q1); q2s.append(q2)
            if q1s:
                mg = math.radians(10.0)
                lo[0], hi[0] = max(lo[0], min(q1s) - mg), min(hi[0], max(q1s) + mg)
                lo[1], hi[1] = max(lo[1], min(q2s) - mg), min(hi[1], max(q2s) + mg)
        return lo, hi

    def valida(self, q, fk):
        """(válida, motivo) de la postura del robot según los criterios humanos."""
        p = self.par
        u = fk["codo"] - fk["hombro"]
        f = fk["marcos"][3][:3, 3] - fk["codo"]           # antebrazo del FR5 (codo → joint4)
        # codo: flexión y lado
        flex = math.degrees(math.acos(float(np.clip(u @ f / (np.linalg.norm(u) * np.linalg.norm(f)), -1, 1))))
        if flex > p.codo_flex_max_deg + 1e-6:
            return False, "codo>max"
        if self.codo_signo * q[2] < -math.radians(p.codo_hiperext_deg):
            return False, "codo al reves"
        # muñeca
        if abs(q[3]) > math.radians(p.muneca_q4_deg) + 1e-6:
            return False, "muneca q4"
        if abs(q[4] - math.pi / 2) > math.radians(p.muneca_q5_deg) + 1e-6:
            return False, "muneca q5"
        # hombro: cono
        if p.hombro_cono:
            az, elev = self.az_elev(self.a_torso(u))
            if 15.0 < elev < 165.0:
                lo_az, hi_az = self.rango_azimut(elev)
                if not (lo_az - 5.0 <= az <= hi_az + 5.0):
                    return False, "hombro fuera del cono"
        return True, ""


class ControlIKFR5:
    def __init__(self, cadena: CadenaFR5, colision: ModeloColision = None, par: ParametrosIK = None):
        self.c = cadena
        self.col = colision or ModeloColision(cadena)
        self.par = par or ParametrosIK()
        lo = cadena.limites[:, 0] + self.par.margen_limites
        hi = cadena.limites[:, 1] - self.par.margen_limites
        for i, (a, b) in self.par.limites_extra.items():
            lo[i], hi[i] = max(lo[i], a), min(hi[i], b)
        self.postura = PosturaHumana(cadena, self.par, lo, hi) if self.par.postura_humana else None
        if self.postura is not None:
            lo, hi = self.postura.lo, self.postura.hi
        self.lo, self.hi = lo, hi
        self.q = None                 # última solución
        self.rama = None              # firma de la rama actual
        self.t_err = None             # desde cuándo hay error alto
        self.transicion = None        # dict(camino=[...], i, t0) durante un cambio de rama
        self.info = {}
        self.n_cambios_rama = 0

    # ── utilidades ───────────────────────────────────────────────────────────
    def en_limites(self, q):
        return bool(np.all(q >= self.lo) and np.all(q <= self.hi))

    @staticmethod
    def firma_rama(q):
        """(hombro, codo, muñeca) ∈ {±1}³: identifica la rama de la IK de un brazo UR.
        codo: signo de joint3; muñeca: signo de joint5; hombro: sentido de joint1 respecto a 0."""
        return (1 if q[2] >= 0 else -1, 1 if q[4] >= 0 else -1)

    def _dist_q(self, qa, qb):
        d = np.array([wrap(a - b) for a, b in zip(qa, qb)])
        return float(np.sqrt(np.sum(np.asarray(self.par.w_continuidad) * d * d)))

    def _coste(self, q, fk, obj, q_prev):
        par = self.par
        c = 0.0
        if q_prev is not None:
            c += self._dist_q(q, q_prev)
        pe = obj.get("p_codo_sugerido")
        if pe is not None:
            c += par.w_codo * float(np.linalg.norm(fk["codo"] - pe))
            # lado del codo respecto al plano hombro–muñeca–vertical
            v = fk["muneca"] - fk["hombro"]
            n = np.cross(v, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(n) > 1e-6:
                s_r = float((fk["codo"] - fk["hombro"]) @ n)
                s_h = float((pe - fk["hombro"]) @ n)
                if s_r * s_h < 0:
                    c += par.w_codo_lado
        return c

    def _valida(self, q):
        """(válida, motivo) respecto a límites, codo y colisiones."""
        if not self.en_limites(q):
            return False, "limites"
        fk = self.c.fk(q)
        if self.par.z_codo_min is not None and fk["codo"][2] < self.par.z_codo_min:
            return False, "codo bajo"
        if self.postura is not None:
            ok, motivo = self.postura.valida(q, fk)
            if not ok:
                return False, "postura " + motivo
        cols = self.col.colisiones(q, fk["marcos"])
        if cols:
            return False, "colision " + ",".join(f"{a}-{b}" for a, b, _ in cols[:2])
        return True, ""

    # ── resolución por fotograma ─────────────────────────────────────────────
    def resolver(self, obj, t, dt=None):
        """
        obj: dict con p_muneca (mundo), R_mano (3x3), p_codo_sugerido (opcional).
        Devuelve (q, info). info["estado"] ∈ {"exacta", "aprox", "transicion", "bloqueado"}.
        """
        par = self.par
        q_prev = self.q
        R, p_w = obj["R_mano"], np.asarray(obj["p_muneca"], dtype=float)
        T_tool = T_de(R, p_w - R @ OFFSET_MUNECA_ORCA)
        sols = self.c.ik_analitica(T_tool, q_ref=q_prev, con_rama=True)

        # — transición en curso: seguir el camino planificado, actualizando el destino con la
        #   solución actual de la rama de destino (el objetivo sigue moviéndose) si el tramo
        #   que queda sigue libre de colisiones —
        if self.transicion is not None:
            tr = self.transicion
            for s_, rama_ in sols:
                if rama_ == self.rama and self._valida(s_)[0]:
                    desde = q_prev if len(tr["camino"]) == 1 else tr["camino"][-2]
                    if self.col.camino_libre(desde, s_):
                        tr["camino"][-1] = s_
                        tr["destino"] = s_
                    break
            q, fin = self._avanzar_transicion(t, dt)
            self.q = q
            self.info = dict(estado="transicion", error=None, rama=self.rama,
                             paso=tr["i"] if self.transicion else -1)
            if fin:
                self.transicion = None
            return q, self.info

        cands = []
        descartes = {}
        for s, rama in sols:
            ok, motivo = self._valida(s)
            if not ok:
                descartes[motivo.split()[0]] = descartes.get(motivo.split()[0], 0) + 1
                continue
            fk = self.c.fk(s)
            cands.append((self._coste(s, fk, obj, q_prev), s, fk, rama))
        cands.sort(key=lambda c: c[0])
        if not cands:
            return self._aproximar(obj, q_prev, t, "sin solucion exacta " + str(descartes), dt)

        coste, q, fk, rama = cands[0]
        if q_prev is not None and self.rama is not None:
            # La rama actual, si sigue siendo válida, se mantiene salvo que otra sea claramente
            # mejor (histéresis): cambiar de rama es un movimiento grande que hay que planificar.
            misma = [c for c in cands if c[3] == self.rama]
            if misma and misma[0][0] <= coste + par.histeresis_rama:
                coste, q, fk, rama = misma[0]
            salto = self._dist_q(q, q_prev) >= par.salto_rama
            if salto and rama == self.rama:
                # Salto grande DENTRO de la misma rama: joint6 que da la vuelta al llegar a su
                # tope o paso por una singularidad del hombro. Nunca se ejecuta el salto (sería
                # "una vuelta"): se sigue con la IK numérica desde donde estamos, relajando la
                # orientación (primero el giro de la mano) hasta que la solución exacta vuelva
                # a estar cerca.
                return self._aproximar(obj, q_prev, t, "salto en la misma rama (tope de q6 / singularidad)", dt)
            # Cambio de rama CON salto articular grande → transición planificada. (Cerca de una
            # singularidad las ramas se tocan: el cambio es continuo y no hace falta planificar.)
            if rama != self.rama and salto:
                camino = self._planificar(q_prev, q)
                if camino is None:
                    return self._aproximar(obj, q_prev, t, "sin camino libre a otra rama", dt)
                self.n_cambios_rama += 1
                self.transicion = dict(camino=camino, i=0, t_prev=t, destino=q)
                self.rama = rama
                q, _ = self._avanzar_transicion(t, dt)
                self.q = q
                self.info = dict(estado="transicion", error=None, rama=rama, paso=0,
                                 motivo="cambio de rama", n_cambios=self.n_cambios_rama)
                return q, self.info
        elif q_prev is not None and self._dist_q(q, q_prev) >= par.salto_rama:
            # primera solución, lejos de la postura actual del robot: ir despacio y sin chocar
            camino = self._planificar(q_prev, q)
            if camino is None:
                return self._aproximar(obj, q_prev, t, "sin camino libre a la primera solucion", dt)
            self.transicion = dict(camino=camino, i=0, t_prev=t, destino=q)
            self.rama = rama
            q, _ = self._avanzar_transicion(t, dt)
            self.q = q
            self.info = dict(estado="transicion", error=None, rama=rama, paso=0, motivo="arranque")
            return q, self.info

        q, limitado = self._limitar_velocidad(q, q_prev, dt, p_w)
        self.q = q
        self.rama = rama
        self.t_err = None
        self.info = dict(estado="exacta", error=0.0 if not limitado else
                         float(np.linalg.norm(self.c.punto(q, OFFSET_MUNECA_ORCA) - p_w)),
                         rama=rama, n_sol=len(sols), n_validas=len(cands), descartes=descartes,
                         coste=coste, vel_limitada=limitado)
        return q, self.info

    def _limitar_velocidad(self, q_obj, q_prev, dt, p_w):
        """Hacia q_obj a lo sumo v_seguimiento·dt por joint. Si el paso limitado choca o sale de
        límites, aproximación numérica desde q_prev (orientación relajada). Devuelve (q, limitado)."""
        if q_prev is None or dt is None or dt <= 0:
            return q_obj, False
        paso = self.par.v_seguimiento * float(dt)
        d = q_obj - q_prev
        if float(np.abs(d).max()) <= paso:
            return q_obj, False
        q = q_prev + np.clip(d, -paso, paso)
        ok, _ = self._valida(q)
        if ok:
            return q, True
        q2, ep, eo = self._ik_num(q_prev, p_w, None, 0.0, None)
        ok, _ = self._valida(q2)
        if ok and float(np.abs(q2 - q_prev).max()) <= paso:
            return q2, True
        return q_prev.copy(), True

    def _aproximar(self, obj, q_prev, t, motivo, dt=None):
        """Lo más cerca posible: posición primero, orientación después (pronosupinación libre y,
        si hace falta, toda la orientación). Siempre con velocidad acotada."""
        par = self.par
        R, p_w = obj["R_mano"], np.asarray(obj["p_muneca"], dtype=float)
        q0 = q_prev if q_prev is not None else self.c.limites.mean(axis=1)
        mejor = None
        for w_ori, eje_libre in ((1.0, [0.0, 1.0, 0.0]), (0.25, [0.0, 1.0, 0.0]), (0.05, None), (0.0, None)):
            q, ep, eo = self._ik_num(q0, p_w, R, w_ori, eje_libre)
            ok, _ = self._valida(q)
            if ok and (mejor is None or ep < mejor[1] - 1e-3):
                mejor = (q, ep, eo)
            if ok and ep < 0.01:
                break
        if mejor is None:
            # ni siquiera la aproximación es válida (colisión al acercarse): retroceder un poco
            # hacia una postura segura = quedarse donde estamos
            q = q0
            ep = float(np.linalg.norm(self.c.punto(q, OFFSET_MUNECA_ORCA) - p_w))
            estado = "bloqueado"
        else:
            q, ep, _ = mejor
            estado = "aprox"
        if q_prev is not None:
            q, _ = self._limitar_velocidad(q, q_prev, dt, p_w)
            ep = float(np.linalg.norm(self.c.punto(q, OFFSET_MUNECA_ORCA) - p_w))
        # ¿atascado? (error alto un rato) → intentar otra rama relajando la continuidad
        if ep > par.error_atasco_m:
            if self.t_err is None:
                self.t_err = t
            elif t - self.t_err >= par.t_atasco_s and self.transicion is None:
                alt, alt_rama = self._buscar_otra_rama(obj, q, p_w)
                if alt is not None:
                    camino = self._planificar(q, alt)
                    if camino is not None:
                        self.n_cambios_rama += 1
                        self.transicion = dict(camino=camino, i=0, t_prev=t, destino=alt)
                        self.t_err = None
                        self.rama = alt_rama
                        q2, _ = self._avanzar_transicion(t, None)
                        self.q = q2
                        self.info = dict(estado="transicion", error=ep, rama=self.rama,
                                         paso=0, motivo="atasco → otra rama", n_cambios=self.n_cambios_rama)
                        return q2, self.info
                self.t_err = t          # reintentar más tarde
        else:
            self.t_err = None
        self.q = q
        self.info = dict(estado=estado, error=ep, rama=self.rama, motivo=motivo)
        return q, self.info

    def _ik_num(self, q0, p_w, R, w_ori, eje_libre):
        lo_guard, hi_guard = self.c.limites.copy(), None
        # IK numérica con los límites estrechados del controlador
        c = self.c
        lim_orig = c.limites
        c.limites = np.column_stack([self.lo, self.hi])
        try:
            q, ep, eo = c.ik_numerica(q0, p_w, R, OFFSET_MUNECA_ORCA, w_pos=1.0, w_ori=w_ori,
                                      n_iter=40, eje_libre=eje_libre, paso_max=0.2)
        finally:
            c.limites = lim_orig
        return q, ep, eo

    def _buscar_otra_rama(self, obj, q_act, p_w):
        """Otra configuración (válida) que acerque más la muñeca, aunque esté lejos de q_act:
        soluciones analíticas de una versión relajada del objetivo (muñeca en su sitio, mano con
        la orientación que más cerca quede) → si no hay, IK numérica desde varias semillas."""
        R = obj["R_mano"]
        T_tool = T_de(R, p_w - R @ OFFSET_MUNECA_ORCA)
        mejor = None
        for s, rama in self.c.ik_analitica(T_tool, q_ref=q_act, con_rama=True):
            ok, _ = self._valida(s)
            if ok:
                fk = self.c.fk(s)
                coste = self._coste(s, fk, obj, None)
                if mejor is None or coste < mejor[0]:
                    mejor = (coste, s, rama)
        if mejor is not None:
            return mejor[1], mejor[2]
        semillas = [q_act.copy()]
        for dq3 in (-2.0, 2.0):                       # codo al otro lado
            s = q_act.copy(); s[2] = -s[2]; s[1] += dq3 * 0.3
            semillas.append(np.clip(s, self.lo, self.hi))
        s = q_act.copy(); s[0] = wrap(s[0] + math.pi); s[1] = -math.pi - s[1]     # hombro girado
        semillas.append(np.clip(s, self.lo, self.hi))
        for s0 in semillas:
            q, ep, eo = self._ik_num(s0, p_w, R, 0.1, [0.0, 1.0, 0.0])
            ok, _ = self._valida(q)
            if ok and (mejor is None or ep < mejor[0]):
                mejor = (ep, q)
        if mejor is not None and mejor[0] < self.par.error_atasco_m * 0.7:
            return mejor[1], self.rama
        return None, None

    # ── planificación de la transición ───────────────────────────────────────
    def _planificar(self, q0, q1):
        """Camino articular q0 → q1 libre de colisiones: recta; si choca, dos tramos por un punto
        intermedio (varios candidatos). Devuelve lista de puntos (incluye q1) o None."""
        if self.col.camino_libre(q0, q1):
            return [q1]
        qm = 0.5 * (q0 + q1)
        candidatos = []
        # codo levantado / brazo recogido
        for dq1 in (-0.6, 0.6, 0.0):
            for dq2 in (-0.5, 0.5):
                v = qm.copy(); v[1] += dq1; v[2] += dq2
                candidatos.append(v)
        # mover la muñeca primero y el brazo después
        v = q0.copy(); v[3:] = q1[3:]
        candidatos.append(v)
        v = q1.copy(); v[3:] = q0[3:]
        candidatos.append(v)
        for v in candidatos:
            v = np.clip(v, self.lo, self.hi)
            if self.col.libre(v) and self.col.camino_libre(q0, v) and self.col.camino_libre(v, q1):
                return [v, q1]
        return None

    def _avanzar_transicion(self, t, dt):
        """Siguiente punto de la transición a velocidad limitada. Devuelve (q, terminada)."""
        tr = self.transicion
        dt = float(dt) if dt is not None else max(1e-3, float(t - tr["t_prev"]))
        tr["t_prev"] = t
        q = self.q.copy()
        destino = tr["camino"][tr["i"]]
        paso = self.par.v_transicion * dt
        d = destino - q
        if float(np.abs(d).max()) <= paso:
            q = destino.copy()
            tr["i"] += 1
            if tr["i"] >= len(tr["camino"]):
                return q, True
            return q, False
        q = q + np.clip(d, -paso, paso)
        return q, False
