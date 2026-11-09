"""
IK DIFERENCIAL PONDERADA para la teleoperación del FR5 (28-oct, tras el análisis del vídeo).

Por qué sustituye a la IK analítica de ik_teleop.ControlIKFR5
──────────────────────────────────────────────────────────────
La analítica pide un objetivo 6D exacto (muñeca + orientación completa de la mano). Con 6
ejes eso fija la postura entera: el codo humano solo sirve para elegir una de 8 soluciones,
y los límites de postura humana (cajas rígidas) dejan muchas veces CERO soluciones válidas con
una medida ruidosa → aproximación numérica que suelta la orientación, saltos en la misma rama
en el tope de q6 / singularidades y transiciones planificadas durante las que el robot no te
sigue (diagnostico.py lo mide: con la trayectoria LIMPIA ya fallaba con el brazo colgando o
lateral-abajo, y con 2 cm de ruido aparecían decenas de transiciones).

Aquí, cada fotograma, se parte del q anterior y se da un paso de mínimos cuadrados
amortiguados con varias tareas PONDERADAS (no exactas):

    posición de la muñeca de la OrcaHand        peso alto      (lo que más se nota)
    dirección del codo (⟂ a hombro→muñeca)      peso medio     (el codo del robot sigue al tuyo)
    dirección "adelante" de la mano             peso bajo
    giro de la mano (pronosupinación)           peso más bajo
    límites HUMANOS (codo, muñeca, hombro)      coste que crece al salirse, NO prohibición
    continuidad (q anterior)                    regularización muy débil

Consecuencias:
  · UNA SOLA RAMA: la elegida al arrancar (la de la postura de reposo). No hay cambios de rama
    ni transiciones automáticas: el robot nunca "se va a su aire".
  · Con un objetivo ruidoso o inalcanzable no hay "sin solución": el robot va al mejor
    compromiso (posición primero) y el error queda acotado.
  · Singularidades (brazo estirado, muñeca recta): el amortiguamiento crece al bajar la
    manipulabilidad, sin saltos.
  · Límites del FR5 y colisiones (cápsulas de ik_teleop.ModeloColision) sí son duros.

Misma interfaz que ControlIKFR5 (resolver, q, rama, lo, hi, en_limites, col, postura...),
así que TeleopBrazoFR5 / el simulador / el diagnóstico usan una u otra sin cambios.
Estados: "exacta" (muñeca < 1,5 cm y mano < 15°), "aprox", "transicion" (solo al arrancar,
lejos del objetivo, a velocidad reducida) y "bloqueado" (una colisión frena el movimiento).
"""
import math
from dataclasses import dataclass

import numpy as np

from skill_shield.teleop_fr5.cinematica import CadenaFR5, OFFSET_MUNECA_ORCA
from skill_shield.teleop_fr5.ik_teleop import (
    ControlIKFR5, ModeloColision, ParametrosIK, PosturaHumana, _lejos_caja,
)


def _clip01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _puntos_cercanos(p1, q1, p2, q2):
    """Puntos más cercanos entre los segmentos p1-q1 y p2-q2 (Ericson). Aritmética escalar:
    se llama decenas de veces por fotograma y las operaciones de numpy sobre 3 elementos
    cuestan más en sobrecarga que en cálculo."""
    p1x, p1y, p1z = p1[0], p1[1], p1[2]
    p2x, p2y, p2z = p2[0], p2[1], p2[2]
    d1x, d1y, d1z = q1[0] - p1x, q1[1] - p1y, q1[2] - p1z
    d2x, d2y, d2z = q2[0] - p2x, q2[1] - p2y, q2[2] - p2z
    rx, ry, rz = p1x - p2x, p1y - p2y, p1z - p2z
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    f = d2x * rx + d2y * ry + d2z * rz
    if a <= 1e-12 and e <= 1e-12:
        s, t = 0.0, 0.0
    elif a <= 1e-12:
        s, t = 0.0, _clip01(f / e)
    else:
        c = d1x * rx + d1y * ry + d1z * rz
        if e <= 1e-12:
            t, s = 0.0, _clip01(-c / a)
        else:
            b = d1x * d2x + d1y * d2y + d1z * d2z
            den = a * e - b * b
            s = _clip01((b * f - c * e) / den) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, _clip01(-c / a)
            elif t > 1.0:
                t, s = 1.0, _clip01((b - c) / a)
    return (np.array([p1x + d1x * s, p1y + d1y * s, p1z + d1z * s]),
            np.array([p2x + d2x * t, p2y + d2y * t, p2z + d2z * t]))


def _esfera(c):
    """(centro x, y, z, radio envolvente) de una cápsula, en floats."""
    ax, ay, az = float(c.a[0]), float(c.a[1]), float(c.a[2])
    bx, by, bz = float(c.b[0]), float(c.b[1]), float(c.b[2])
    h = 0.5 * math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
    return (0.5 * (ax + bx), 0.5 * (ay + by), 0.5 * (az + bz), h + c.r)


def _lejos(ea, eb, umbral):
    """Descarte rápido: dos cápsulas cuyas esferas envolventes están a más de 'umbral'."""
    d = math.sqrt((ea[0] - eb[0]) ** 2 + (ea[1] - eb[1]) ** 2 + (ea[2] - eb[2]) ** 2)
    return d - ea[3] - eb[3] > umbral


# último joint (índice 0..5) que mueve cada cápsula del robot (ModeloColision.capsulas)
_ULTIMO_JOINT = {"base": -1, "brazo": 1, "antebrazo": 2, "muneca": 4, "mano": 5}


def _suave01(x, lo, hi):
    t = min(max((x - lo) / max(hi - lo, 1e-9), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class ParametrosIKDif:
    w_pos: float = 1.0            # m⁻¹ (la unidad de referencia)
    w_codo: float = 0.35          # dirección del codo (componente ⟂ a hombro→muñeca), m
    codo_flex_ini_deg: float = 10.0   # flexión humana por debajo de la cual el codo no "dice" lado
    codo_flex_pleno_deg: float = 35.0
    w_post: float = 0.30          # m/rad hacia la solución analítica de nuestra rama (si la hay)

    w_hold: float = 0.03           # inalcanzable: atracción (débil) a la última postura buena
    paso_compromiso_m: float = 0.03   # objetivo inalcanzable: cuánto "tira" la posición por paso
    w_dir: float = 0.20           # m/rad: "adelante" de la mano (dirección del antebrazo)
    w_roll: float = 0.08          # m/rad: giro alrededor de "adelante"
    w_lim: float = 1.2            # m/rad fuera de la caja humana (coste, no prohibición)
    margen_suave_deg: float = 4.0 # el coste empieza este margen ANTES del borde de la caja
    margen_duro_deg: float = 20.0 # límite duro = caja humana ensanchada esto (∩ límites del FR5)
    w_reg: float = 0.02           # m/rad hacia el q anterior (continuidad)
    lam: float = 0.02             # amortiguamiento base
    lam_sing: float = 0.12        # amortiguamiento extra cerca de una singularidad
    manip_ref: float = 0.03       # manipulabilidad de posición "normal" (por debajo crece lam)
    n_iter: int = 3               # iteraciones por fotograma (el objetivo se mueve: no hace falta converger)
    paso_iter: float = 0.15       # rad máx. por iteración
    err_exacta_m: float = 0.015
    err_exacta_deg: float = 15.0
    err_arranque_m: float = 0.05  # al arrancar, velocidad reducida hasta acercarse a esto
    w_col: float = 3.0            # repulsión de colisiones (misma prioridad que la posición)
    d_col_activa: float = 0.07    # m: a menos de esto de un obstáculo empieza a apartarse
    # Banda (coste) de q1/q2: la que usa la rama de la postura humana para TODO el espacio de
    # trabajo de un brazo humano con el soporte de hombros inclinado (muestreo de 576 posturas
    # humanas → IK analítica: q1 p1..p99 = −97..126°, q2 = −133..17°). Fuera de ella el robot
    # estaría "dando la vuelta" alrededor del eje de joint1.
    banda_q1_deg: tuple = (-105.0, 135.0)
    banda_q2_deg: tuple = (-150.0, 30.0)
    margen_eje_j1: float = 0.07   # m: el centro de muñeca del FR5 no se acerca al eje de joint1 a
                                  # menos de d4 + esto (allí se cambia de "lado de hombro")


class ControlIKDiferencial:
    def __init__(self, cadena: CadenaFR5, colision: ModeloColision = None, par: ParametrosIK = None,
                 pdif: ParametrosIKDif = None):
        self.c = cadena
        self.col = colision or ModeloColision(cadena)
        self.par = par or ParametrosIK()
        self.pd = pdif or ParametrosIKDif()
        lo = cadena.limites[:, 0] + self.par.margen_limites
        hi = cadena.limites[:, 1] - self.par.margen_limites
        for i, (a, b) in self.par.limites_extra.items():
            lo[i], hi[i] = max(lo[i], a), min(hi[i], b)
        self.lo_fr5, self.hi_fr5 = lo.copy(), hi.copy()
        # cajas HUMANAS (las mismas que la IK analítica) → aquí son coste
        self.postura = PosturaHumana(cadena, self.par, lo, hi) if self.par.postura_humana else None
        if self.postura is not None:
            self.lo_h, self.hi_h = self.postura.lo.copy(), self.postura.hi.copy()
            # El cono del hombro como caja de q1/q2 es una envolvente pensada para la IK analítica
            # y con el soporte inclinado deja fuera posturas humanas (brazo colgando: q2 ≈ −110°
            # o −174°). Aquí el hombro solo tiene los límites del FR5; codo y muñeca, la caja
            # humana como coste + margen_duro_deg como tope.
            # 9-nov: la banda de q1 se midió con la base montada como en el diseño del soporte
            # (brazo hacia DELANTE con q1 = 0). Si la base real está girada sobre la placa
            # (giro_base_grados, diagnosticos/ajustar_gemelo.py), la misma postura humana
            # necesita otro q1: la banda se desplaza lo que haga falta para que "delante" siga
            # cayendo donde caía.
            q1_delante = self.postura.q12_para_direccion(np.asarray(self.par.ejes[2], float))[0]
            desplaz = math.degrees(math.atan2(math.sin(q1_delante), math.cos(q1_delante)))
            banda_q1 = (self.pd.banda_q1_deg[0] + desplaz, self.pd.banda_q1_deg[1] + desplaz)
            for i_, (a_, b_) in ((0, banda_q1), (1, self.pd.banda_q2_deg)):
                self.lo_h[i_] = max(lo[i_], math.radians(a_))
                self.hi_h[i_] = min(hi[i_], math.radians(b_))
            if abs(desplaz) > 1.0:
                fuera_lim = (max(0.0, math.radians(banda_q1[1]) - hi[0])
                             + max(0.0, lo[0] - math.radians(banda_q1[0])))
                print(f"🔄 IK: base girada {desplaz:+.0f}° respecto al diseño del soporte → banda de q1 "
                      f"[{banda_q1[0]:.0f}, {banda_q1[1]:.0f}]°")
                if fuera_lim > math.radians(1.0):
                    print(f"⚠ IK: {math.degrees(fuera_lim):.0f}° de la banda humana de q1 quedan FUERA del "
                          f"recorrido del joint1 (±{math.degrees(hi[0]):.0f}°): el robot NO podrá seguir esas "
                          "posturas. Gira el FR5 sobre la placa hasta que ajustar_gemelo.py dé 0°.")
            md = math.radians(self.pd.margen_duro_deg)
            self.lo = np.maximum(lo, self.lo_h - md)
            self.hi = np.minimum(hi, self.hi_h + md)
            self.lo[:2], self.hi[:2] = lo[:2], hi[:2]       # hombro: tope duro = el del FR5
            # el codo NUNCA cruza al otro lado (sería cambiar de rama) y la muñeca nunca pasa
            # por q5 = 0 (muñeca volteada): esos dos bordes sí son duros
            hip = math.radians(max(self.par.codo_hiperext_deg, 1.0))
            if self.postura.codo_signo > 0:
                self.lo[2] = max(self.lo[2], -hip)
            else:
                self.hi[2] = min(self.hi[2], hip)
            self.lo[4] = max(self.lo[4], math.radians(8.0))
            self.hi[4] = min(self.hi[4], math.pi - math.radians(8.0))
        else:
            self.lo_h, self.hi_h = lo.copy(), hi.copy()
            self.lo, self.hi = lo, hi
        self.hombro_robot = cadena.fk(np.zeros(6))["hombro"]
        # eje de joint1 (fijo en el mundo) y radio del "hueco" del hombro (d4 de la cadena UR):
        # el centro de muñeca (joint5) a menos de d4 del eje es inalcanzable y al cruzar ese
        # cilindro la IK cambia de lado de hombro. Mantenerlo fuera = una sola rama de hombro.
        self.eje_j1 = cadena.fk(np.zeros(6))["ejes"][0]
        self.o_j1 = cadena.fk_marcos(np.zeros(6))[0][:3, 3]
        try:
            self.r_hueco = abs(float(cadena.dh_equivalente()["params"]["d4"]))
        except Exception:
            self.r_hueco = 0.1021          # d4 oficial del FR5
        self.q = None
        self.q_buena = None               # última postura de referencia alcanzable
        self._manip = 0.0
        self.rama = None
        self.arrancando = True
        self.transicion = None            # compatibilidad (nunca hay transiciones planificadas)
        self.n_cambios_rama = 0
        self.info = {}

    # ── compatibilidad con ControlIKFR5 ──────────────────────────────────────
    def en_limites(self, q):
        return bool(np.all(q >= self.lo - 1e-9) and np.all(q <= self.hi + 1e-9))

    firma_rama = staticmethod(ControlIKFR5.firma_rama)

    def _valida(self, q):
        if not self.en_limites(q):
            return False, "limites"
        cols = self.col.colisiones(q)
        if cols:
            return False, "colision " + ",".join(f"{a}-{b}" for a, b, _ in cols[:2])
        return True, ""

    def reiniciar(self):
        self.arrancando = True

    # ── tareas ───────────────────────────────────────────────────────────────
    @staticmethod
    def _err_rot(R_obj, R):
        """Vector de rotación (mundo) que lleva R a R_obj."""
        dR = R_obj @ R.T
        c = (np.trace(dR) - 1.0) / 2.0
        ang = math.acos(min(1.0, max(-1.0, c)))
        v = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        if ang < 1e-9:
            return 0.5 * v
        if ang > math.pi - 1e-4:
            # 180°: eje = autovector de dR con autovalor 1
            w, V = np.linalg.eigh(0.5 * (dR + dR.T))
            return V[:, int(np.argmax(w))] * ang
        return ang / (2.0 * math.sin(ang)) * v

    def _referencia(self, obj, q_prev):
        """Postura de referencia: la solución ANALÍTICA exacta más cercana a q_prev que esté en
        NUESTRA rama (mismo lado de codo y de muñeca) y dentro de la banda del hombro (mismo lado
        de hombro: no da la vuelta al eje de joint1), dentro de límites y sin colisión. Puede estar
        lejos: entonces el robot va hacia ella a velocidad limitada SIGUIÉNDOTE con la muñeca (no
        es una transición: la posición sigue mandando). None si el objetivo no es alcanzable."""
        R = obj["R_mano"]
        p_w = np.asarray(obj["p_muneca"], dtype=float)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p_w - R @ OFFSET_MUNECA_ORCA
        firma = self.firma_rama(q_prev)
        mg = math.radians(self.pd.margen_suave_deg)
        mejor = None
        for s_, _ in self.c.ik_analitica(T, q_ref=q_prev, con_rama=True):
            if self.firma_rama(s_) != firma or not self.en_limites(s_):
                continue
            if not all(self.lo_h[i] - mg <= s_[i] <= self.hi_h[i] + mg for i in (0, 1)):
                continue
            d = self._dist(s_, q_prev)
            if mejor is None or d < mejor[0]:
                mejor = (d, s_)
        if mejor is None or not self.col.libre(mejor[1]):
            return None
        return mejor[1]

    def _dist(self, qa, qb):
        d = np.array([((a - b + math.pi) % (2 * math.pi)) - math.pi for a, b in zip(qa, qb)])
        return float(np.sqrt(np.sum(np.asarray(self.par.w_continuidad) * d * d)))

    def _jac_punto(self, m, p, k, WO=None):
        """Jacobiana lineal de un punto p rígidamente unido al eslabón que mueve el joint k.
        WO = (ejes de los joints en el mundo (6,3), orígenes (6,3)) ya calculados."""
        J = np.zeros((3, 6))
        if k < 0:
            return J
        if WO is None:
            WO = self._ejes_mundo(m)
        W, O = WO
        J[:, :k + 1] = np.cross(W[:k + 1], p - O[:k + 1]).T
        return J

    def _ejes_mundo(self, m):
        W = np.einsum("nij,nj->ni", np.array([mi[:3, :3] for mi in m[:6]]), self.c._ejes)
        O = np.array([mi[:3, 3] for mi in m[:6]])
        return W, O

    def _repulsion(self, q, m, WO=None):
        """Filas (J, e) que apartan las cápsulas del robot de los obstáculos (suelo, soporte) y
        de sí mismo cuando están a menos de d_col_activa. None si no hay nada cerca."""
        pd = self.pd
        WO = WO if WO is not None else self._ejes_mundo(m)
        caps = {c.nombre: c for c in self.col.capsulas(q, m)}
        esf = {k: _esfera(c) for k, c in caps.items()}
        ent = self.col.entorno
        if getattr(self, "_esf_ent", None) is None:
            self._esf_ent = [_esfera(o) for o in ent.capsulas]
        filas, rhs = [], []

        def fila(n, J, d):
            filas.append(pd.w_col * (n @ J)[None, :])
            rhs.append(np.array([pd.w_col * (pd.d_col_activa - d)]))

        for nombre in ("brazo", "antebrazo", "muneca", "mano"):
            c = caps[nombre]
            k = _ULTIMO_JOINT[nombre]
            if ent.z_suelo is not None:
                p = c.a if c.a[2] <= c.b[2] else c.b
                d = float(p[2]) - c.r - ent.margen - ent.z_suelo
                if d < pd.d_col_activa:
                    fila(np.array([0.0, 0.0, 1.0]), self._jac_punto(m, p, k, WO), d)
            for o, eo in zip(ent.capsulas, self._esf_ent):
                if _lejos(esf[nombre], eo, pd.d_col_activa + ent.margen):
                    continue
                pa, pb = _puntos_cercanos(c.a, c.b, o.a, o.b)
                v = pa - pb
                dist = float(np.linalg.norm(v))
                d = dist - c.r - o.r - ent.margen
                if d < pd.d_col_activa and dist > 1e-6:
                    fila(v / dist, self._jac_punto(m, pa, k, WO), d)
            for o in ent.cajas:                       # 5-nov: mesa, peana, cámaras, zonas
                if _lejos_caja(c.a, c.b, c.r, o, pd.d_col_activa + ent.margen):
                    continue
                pa, n, sd = o.segmento(c.a, c.b)
                d = sd - c.r - ent.margen
                if d < pd.d_col_activa:
                    fila(n, self._jac_punto(m, pa, k, WO), d)
        for a_, b_ in self.col.PARES_PROPIOS:
            ca, cb = caps[a_], caps[b_]
            if _lejos(esf[a_], esf[b_], pd.d_col_activa):
                continue
            pa, pb = _puntos_cercanos(ca.a, ca.b, cb.a, cb.b)
            v = pa - pb
            dist = float(np.linalg.norm(v))
            d = dist - ca.r - cb.r
            if d < pd.d_col_activa and dist > 1e-6:
                Ja = self._jac_punto(m, pa, _ULTIMO_JOINT[a_], WO)
                Jb = self._jac_punto(m, pb, _ULTIMO_JOINT[b_], WO)
                fila(v / dist, Ja - Jb, d)
        # banda del hombro (q1, q2): barrera de la misma prioridad que la posición, para que el
        # robot no "dé la vuelta" alrededor del eje de joint1 persiguiendo un objetivo
        ms = math.radians(pd.margen_suave_deg)
        for i in (0, 1):
            lo_s, hi_s = self.lo_h[i] + ms, self.hi_h[i] - ms
            if q[i] > hi_s or q[i] < lo_s:
                r_ = (hi_s - q[i]) if q[i] > hi_s else (lo_s - q[i])
                f_ = np.zeros(6)
                f_[i] = pd.w_col
                filas.append(f_[None, :])
                rhs.append(np.array([pd.w_col * r_]))
        # hueco del hombro: centro de muñeca (origen del marco de joint5) lejos del eje de joint1
        p5 = m[4][:3, 3]
        v = p5 - self.o_j1
        rad = v - float(v @ self.eje_j1) * self.eje_j1
        r = float(np.linalg.norm(rad))
        r_act = self.r_hueco + pd.margen_eje_j1
        if r < r_act and r > 1e-6:
            filas.append(pd.w_col * ((rad / r) @ self._jac_punto(m, p5, 3, WO))[None, :])
            rhs.append(np.array([pd.w_col * (r_act - r)]))
        if not filas:
            return None
        return np.vstack(filas), np.concatenate(rhs)

    def _sistema(self, q, obj, q_post=None, w_post=None):
        """Tarea 1 (J1, e1) = posición de la muñeca de la OrcaHand.
        Tarea 2 (J2, e2) = orientación, dirección del codo, límites humanos y postura de
        referencia, apiladas con sus pesos. + diagnóstico."""
        pd = self.pd
        m = self.c.fk_marcos(q)
        Tt = m[6]
        Rt = Tt[:3, :3]
        p_w = np.asarray(obj["p_muneca"], dtype=float)
        p = Rt @ OFFSET_MUNECA_ORCA + Tt[:3, 3]
        WO = self._ejes_mundo(m)
        Jl, Ja = np.cross(WO[0], p - WO[1]).T, WO[0].T
        J1, e1 = Jl, p_w - p
        cerca = self._repulsion(q, m, WO)
        if cerca is not None:
            J1 = np.vstack([J1, cerca[0]])
            e1 = np.concatenate([e1, cerca[1]])
        filas, rhs = [], []
        # orientación: "adelante" (peso w_dir) y giro alrededor de él (peso w_roll)
        R_obj = obj["R_mano"]
        e_o = self._err_rot(R_obj, Rt)
        f = R_obj[:, 1] / np.linalg.norm(R_obj[:, 1])
        W = pd.w_dir * (np.eye(3) - np.outer(f, f)) + pd.w_roll * np.outer(f, f)
        filas.append(W @ Ja)
        rhs.append(W @ e_o)
        # codo: hacia DÓNDE se dobla (componente ⟂ a hombro→muñeca), como el codo humano. Con el
        # brazo humano casi recto esa dirección no existe: el peso baja a 0 y manda la continuidad.
        d_b, d_a = obj.get("d_brazo"), obj.get("d_antebrazo")
        w_c = 0.0
        if d_b is not None and d_a is not None and pd.w_codo > 0:
            u = p_w - self.hombro_robot
            nu = float(np.linalg.norm(u))
            flex = math.degrees(math.acos(float(np.clip(np.dot(d_b, d_a), -1.0, 1.0))))
            w_c = pd.w_codo * _suave01(flex, pd.codo_flex_ini_deg, pd.codo_flex_pleno_deg)
            if nu > 1e-6 and w_c > 1e-3:
                u /= nu
                P = np.eye(3) - np.outer(u, u)
                n = P @ np.asarray(d_b, float)
                nn = float(np.linalg.norm(n))
                if nn > 1e-3:
                    n /= nn
                    o = P @ (m[2][:3, 3] - self.hombro_robot)        # desplazamiento actual del codo
                    h = max(float(np.linalg.norm(o)), 0.05)
                    Je = self._jac_punto(m, m[2][:3, 3], 1, WO)
                    filas.append(w_c * (P @ Je))
                    rhs.append(w_c * (h * n - o))
        # límites humanos como coste (empieza margen_suave antes del borde)
        ms = math.radians(pd.margen_suave_deg)
        lim_activos = []
        for i in range(2, 6):                       # q1/q2: banda como barrera (tarea 1)
            lo_s, hi_s = self.lo_h[i] + ms, self.hi_h[i] - ms
            if q[i] > hi_s:
                r = hi_s - q[i]
            elif q[i] < lo_s:
                r = lo_s - q[i]
            else:
                continue
            fila = np.zeros(6)
            fila[i] = pd.w_lim
            filas.append(fila[None, :])
            rhs.append(np.array([pd.w_lim * r]))
            if abs(r) > ms:
                lim_activos.append(i)
        # postura de referencia (solución analítica cercana de la misma rama)
        if q_post is not None:
            wp = pd.w_post if w_post is None else w_post
            filas.append(wp * np.eye(6))
            rhs.append(wp * np.array([((a - b + math.pi) % (2 * math.pi)) - math.pi
                                             for a, b in zip(q_post, q)]))
        mu = math.sqrt(max(np.linalg.det(Jl @ Jl.T), 0.0))
        diag = dict(err_pos=float(np.linalg.norm(p_w - p)),
                    err_dir=math.degrees(math.acos(float(np.clip(Rt[:, 1] @ f, -1.0, 1.0)))),
                    lim=lim_activos, manip=mu, w_codo=w_c)
        return (J1, e1), (np.vstack(filas), np.concatenate(rhs)), diag

    def _diag(self, q, obj):
        """Lo que hace falta para el estado/motivo (sin jacobianas ni colisiones: barato)."""
        Tt = self.c.fk_marcos(q)[6]
        Rt = Tt[:3, :3]
        p = Rt @ OFFSET_MUNECA_ORCA + Tt[:3, 3]
        f = obj["R_mano"][:, 1]
        f = f / np.linalg.norm(f)
        ms = math.radians(self.pd.margen_suave_deg)
        lim = [i for i in range(2, 6)
               if q[i] > self.hi_h[i] or q[i] < self.lo_h[i]]
        return dict(err_pos=float(np.linalg.norm(np.asarray(obj["p_muneca"], float) - p)),
                    err_dir=math.degrees(math.acos(float(np.clip(Rt[:, 1] @ f, -1.0, 1.0)))),
                    lim=lim, manip=self._manip)

    def _paso(self, q, obj, q_prev, q_post, compromiso=False):
        """Un paso de la IK.
        Normal (hay referencia alcanzable): PRIORIDAD — posición de la muñeca (+ barreras de
        colisión / hombro) primero, y orientación, codo, límites humanos y referencia en el
        espacio nulo.
        Compromiso (objetivo inalcanzable: hueco del hombro, colisión, fuera de alcance): todas
        las tareas PONDERADAS juntas y con la referencia = última postura buena, así el robot
        se acerca lo que puede sin irse a una postura rara persiguiendo lo imposible."""
        (J1, e1), (J2, e2), d = self._sistema(q, obj, q_post, self.pd.w_hold if compromiso else None)
        self._manip = d["manip"]
        pd = self.pd
        lam = pd.lam + pd.lam_sing * max(0.0, 1.0 - d["manip"] / pd.manip_ref) ** 2
        J2 = np.vstack([J2, pd.w_reg * np.eye(6)])
        e2 = np.concatenate([e2, pd.w_reg * (q_prev - q)])
        if compromiso:
            e1 = e1.copy()
            n = float(np.linalg.norm(e1[:3]))
            if n > pd.paso_compromiso_m:
                e1[:3] *= pd.paso_compromiso_m / n
            J = np.vstack([J1, J2])
            e = np.concatenate([e1, e2])
            dq = np.linalg.solve(J.T @ J + lam ** 2 * np.eye(6), J.T @ e)
        else:
            J1p = J1.T @ np.linalg.inv(J1 @ J1.T + lam ** 2 * np.eye(J1.shape[0]))
            dq1 = J1p @ e1
            N1 = np.eye(6) - J1p @ J1
            J2n = J2 @ N1
            dq2 = np.linalg.solve(J2n.T @ J2n + (pd.lam ** 2) * np.eye(6), J2n.T @ (e2 - J2 @ dq1))
            dq = dq1 + N1 @ dq2
        mx = float(np.abs(dq).max())
        if mx > pd.paso_iter:
            dq *= pd.paso_iter / mx
        return np.clip(q + dq, self.lo, self.hi)

    # ── resolución por fotograma ─────────────────────────────────────────────
    def resolver(self, obj, t, dt=None):
        q_prev = self.q.copy() if self.q is not None else self.c.limites.mean(axis=1)
        q_prev = np.clip(q_prev, self.lo, self.hi)
        q_ref = None if self.arrancando else self._referencia(obj, q_prev)
        compromiso = q_ref is None and not self.arrancando
        if q_ref is not None:
            self.q_buena = q_ref.copy()
        q_post = q_ref if q_ref is not None else (self.q_buena if compromiso else None)
        q = q_prev.copy()
        for _ in range(self.pd.n_iter):
            q = self._paso(q, obj, q_prev, q_post, compromiso)
        d0 = self._diag(q, obj)
        # velocidad: al arrancar (lejos del objetivo) más despacio
        v = self.par.v_transicion if self.arrancando else self.par.v_seguimiento
        limitado = False
        if dt is not None and dt > 0:
            paso = v * float(dt)
            dq = q - q_prev
            if float(np.abs(dq).max()) > paso:
                q = q_prev + np.clip(dq, -paso, paso)
                limitado = True
        # colisiones: lo más lejos posible por el segmento q_prev → q sin chocar
        bloqueado, choque = False, ""
        cols = self.col.colisiones(q)
        if cols:
            choque = ",".join(f"{a}-{b}" for a, b, _ in cols[:2])
            if self.col.libre(q_prev):
                a_ok, a_mal = 0.0, 1.0
                for _ in range(6):
                    a = 0.5 * (a_ok + a_mal)
                    if self.col.libre(q_prev + a * (q - q_prev)):
                        a_ok = a
                    else:
                        a_mal = a
                q = q_prev + a_ok * (q - q_prev)
            else:
                # ya dentro (no debería pasar): solo se acepta si reduce la penetración
                pen = lambda qq: sum(p_ for _, _, p_ in self.col.colisiones(qq))
                if pen(q) >= pen(q_prev):
                    q = q_prev.copy()
            bloqueado = True
        d = self._diag(q, obj)
        if self.arrancando and d["err_pos"] < self.pd.err_arranque_m:
            self.arrancando = False
        if self.arrancando and self.q is not None:
            estado, motivo = "transicion", "arranque (velocidad reducida)"
        elif bloqueado and d["err_pos"] > 0.02:
            estado, motivo = "bloqueado", "colision " + choque
        elif d["err_pos"] < self.pd.err_exacta_m and d["err_dir"] < self.pd.err_exacta_deg:
            estado, motivo = "exacta", ""
        else:
            estado = "aprox"
            if limitado and d0["err_pos"] < d["err_pos"] - 1e-3:
                motivo = "siguiendo (velocidad limitada)"
            elif float(np.linalg.norm(np.asarray(obj["p_muneca"]) - self.hombro_robot)) > 0.97 * self._alcance():
                motivo = "fuera de alcance"
            elif compromiso:
                motivo = "inalcanzable: se acerca sin cambiar de postura"
            elif d["lim"]:
                motivo = "limite humano " + ",".join(f"q{i + 1}" for i in d["lim"])
            else:
                motivo = "compromiso posicion/orientacion"
        self.q = q
        self.rama = self.firma_rama(q)
        self.info = dict(estado=estado, error=d["err_pos"], error_dir=d["err_dir"], rama=self.rama,
                         motivo=motivo, vel_limitada=limitado, manip=d["manip"], lim_humanos=d["lim"],
                         referencia=q_ref is not None, compromiso=compromiso)
        return q, self.info

    def _alcance(self):
        if not hasattr(self, "_alc"):
            p = self.c.fk(np.array([0.0, 0.0, 0.0, 0.0, math.pi / 2, 0.0]))["muneca"]
            self._alc = float(np.linalg.norm(p - self.hombro_robot))
        return self._alc
