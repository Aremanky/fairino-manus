"""
Límites HUMANOS de hombro y codo + retargeting fiel al ARCTOS (6-oct).

Tres problemas vistos en la grabación del 6-oct y cómo se atacan aquí:

1) MEDIDAS IMPOSIBLES DEL HUMANO. El estéreo/MediaPipe a veces dan posturas que
   ningún brazo puede hacer (antebrazo doblado hacia atrás, brazo metido en el
   torso). limitar_postura_humana() proyecta tus direcciones (brazo, antebrazo) al
   rango anatómico ANTES de convertirlas en objetivos del robot:
     · hombro: azimut permitido según la elevación (cruzar por delante, abrir hacia
       atrás, extensión con el brazo bajo)
     · codo: flexión 0..CODO_FLEX_MAX y rotación del húmero acotada (el antebrazo
       no puede apuntar hacia atrás con el brazo colgando, etc.)

2) "BRAZO RECTO → ROBOT DOBLADO". objetivos_codo_palma() colocaba la palma del
   robot a la FRACCIÓN lineal de tu alcance y la recortaba al 95 %. Como el ARCTOS
   tiene el antebrazo+mano mucho más largo que el brazo (0.28 / 0.53 m frente a tu
   ~0.30 / 0.36 m), ese 95 % obliga a doblar su codo ~45° con tu brazo RECTO, y el
   lado hacia el que se dobla lo decide el ruido (en el vídeo: q3=2.23 con tu codo a
   6°, es decir, 40° hacia el lado contrario). Aquí la distancia hombro→palma del
   robot sale de TU flexión de codo con la ley del coseno y SUS longitudes:
   tu codo a 0° → su codo a 0°, tu codo a 90° → su codo a 90°.

3) LÍMITES DEL ROBOT CON SENTIDO FÍSICO. joint2/joint3 tenían ±2.5 rad: el codo
   podía plegarse más que un humano y el brazo bajaba hasta el suelo (q2 = 2.5 →
   codo a 6 cm del suelo). limites_robot() calcula, con la geometría cargada:
     · joint3 recto (máximo alcance) y flexión máx. CODO_FLEX_MAX a cada lado
     · joint2 máximo para que el codo quede a CODO_Z_MIN del suelo
     · joint1 dentro del rango de azimut humano (sin vueltas por ±180°)

Direcciones en ejes (fuera, arriba, delante) del torso, como en angulos_humano.py.
"""
import math

import mujoco
import numpy as np

from config import *

from skill_shield.brazo.ik_codo_muneca import (   # 7-oct: espacio de trabajo (puntos 1 y 3)
    CODO_Z_MIN as _CODO_Z_MIN_OBJ, REUBICAR_OBJETIVOS, reubicar_direccion, reubicar_palma,
)
from skill_shield.brazo import base_brazo         # 27-oct: base inclinada (soporte de hombros)

# ── Valores por defecto (se pueden sobrescribir en config.py) ────────────────
_DEF = {
    "LIMITES_HUMANOS":     True,
    "HOMBRO_AZ_MIN":       -45.0,   # ° cruzando por delante del pecho (aducción horizontal)
    "HOMBRO_AZ_MAX":       130.0,   # ° abriendo hacia atrás con el brazo alto (abd. horizontal)
    "HOMBRO_EXT_ELEV":      55.0,   # ° de elevación por debajo de la cual se permite ir hacia atrás (extensión)
    "HOMBRO_AZ_LIBRE_ELEV": 20.0,   # ° con el brazo casi colgando el azimut es ruido: no se limita
    "CODO_FLEX_MAX":       145.0,   # ° flexión máxima del codo humano
    "HUMERO_ROT_EXT":       95.0,   # ° rotación externa máx. (antebrazo hacia fuera)
    "HUMERO_ROT_INT":      110.0,   # ° rotación interna máx. (antebrazo cruzando / a la espalda)
    "CODO_Z_MIN":           0.12,   # m: altura mínima del codo del robot (suelo + grosor del eslabón)
    "Q1_MARGEN":            0.35,   # rad de margen en joint1 sobre el rango de azimut humano
    "Q2_MIN":              -0.5,    # rad: hombro un poco hacia atrás (cerca de la vertical)
    "RETARGET_FIEL":        True,   # distancia de la palma por ley del coseno (punto 2)
    "CODO_ESPEJO_ERROR":    0.06,   # m: si el codo del robot está más lejos que esto, probar el otro lado
    "CODO_ESPEJO_MEJORA":   0.7,    # el otro lado se acepta si deja el coste por debajo del 70 %
}
for _k, _v in _DEF.items():
    globals().setdefault(_k, _v)


def _u(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def _rot(eje, ang):
    """Matriz de rotación (Rodrigues) de 'ang' rad alrededor de 'eje' unitario."""
    x, y, z = eje
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def az_elev(d):
    """(azimut, elevación) en GRADOS de una dirección (fuera, arriba, delante)."""
    elev = math.degrees(math.acos(float(np.clip(-d[1], -1.0, 1.0))))
    az = math.degrees(math.atan2(float(d[0]), float(d[2])))
    return az, elev


def dir_desde_az_elev(az, elev):
    a, e = math.radians(az), math.radians(elev)
    s = math.sin(e)
    return np.array([s * math.sin(a), -math.cos(e), s * math.cos(a)])


# ══════════════════════════════════════════════════════════════════════════════
# 1) LÍMITES ANATÓMICOS SOBRE LA MEDIDA HUMANA
# ══════════════════════════════════════════════════════════════════════════════

def rango_azimut(elev):
    """[az_min, az_max] (°) permitido para el brazo con esa elevación.
    Con el brazo bajo se puede llevar hacia atrás (extensión, hasta ±180°);
    con el brazo alto, abrirlo hacia atrás está limitado a HOMBRO_AZ_MAX."""
    if elev <= HOMBRO_EXT_ELEV:
        az_max = 180.0
    elif elev >= 90.0:
        az_max = HOMBRO_AZ_MAX
    else:
        t = (elev - HOMBRO_EXT_ELEV) / (90.0 - HOMBRO_EXT_ELEV)
        az_max = 180.0 + t * (HOMBRO_AZ_MAX - 180.0)
    return HOMBRO_AZ_MIN, az_max


def limitar_hombro(d_b):
    """Dirección del brazo dentro del cono del hombro (conserva la elevación)."""
    az, el = az_elev(d_b)
    if el < HOMBRO_AZ_LIBRE_ELEV or el > 180.0 - HOMBRO_AZ_LIBRE_ELEV:
        return np.asarray(d_b, dtype=float), False
    lo, hi = rango_azimut(el)
    if lo <= az <= hi:
        return np.asarray(d_b, dtype=float), False
    # Fuera: al límite más cercano (teniendo en cuenta la vuelta de ±180°)
    d_lo = (lo - az) % 360.0
    d_hi = (az - hi) % 360.0
    az_n = lo if d_lo <= d_hi else hi
    return dir_desde_az_elev(az_n, el), True


def _referencia_neutra(d_b):
    """Hacia dónde iría el antebrazo al flexionar el codo SIN rotar el húmero:
    la dirección 'delante' llevada por el giro mínimo colgando → d_b.
    (Colgando: hacia delante. Brazo al frente: hacia arriba. T-pose: hacia delante.)
    None con el brazo casi vertical hacia arriba (giro indefinido)."""
    abajo = np.array([0.0, -1.0, 0.0])
    c = float(np.clip(abajo @ d_b, -1.0, 1.0))
    if c < -0.85:                       # brazo arriba: no se comprueba la rotación
        return None
    eje = np.cross(abajo, d_b)
    n = float(np.linalg.norm(eje))
    if n < 1e-6:                        # colgando
        return np.array([0.0, 0.0, 1.0])
    return _rot(eje / n, math.acos(c)) @ np.array([0.0, 0.0, 1.0])


def limitar_codo(d_b, d_a):
    """Antebrazo dentro de: flexión 0..CODO_FLEX_MAX y rotación del húmero acotada."""
    d_b = np.asarray(d_b, dtype=float)
    d_a = np.asarray(d_a, dtype=float)
    flex = math.degrees(math.acos(float(np.clip(d_b @ d_a, -1.0, 1.0))))
    perp = _u(d_a - (d_a @ d_b) * d_b)
    if perp is None or flex < 15.0:     # casi recto: el plano de flexión es ruido
        return d_a, False
    tocado = False
    if flex > CODO_FLEX_MAX:
        flex, tocado = CODO_FLEX_MAX, True
    ref = _referencia_neutra(d_b)
    if ref is not None:
        ref = _u(ref - (ref @ d_b) * d_b)
    if ref is not None:
        # rotación externa = antebrazo hacia FUERA (signo + aquí)
        rot = -math.degrees(math.atan2(float(np.cross(ref, perp) @ d_b), float(ref @ perp)))
        rot_n = float(np.clip(rot, -HUMERO_ROT_INT, HUMERO_ROT_EXT))
        if rot_n != rot:
            perp = _rot(d_b, -math.radians(rot_n)) @ ref
            tocado = True
    if not tocado:
        return d_a, False
    f = math.radians(flex)
    return math.cos(f) * d_b + math.sin(f) * perp, True


class LimitesHumanos:
    """Aplica los límites y cuenta cuántas veces actúan (diagnóstico en la terminal)."""

    def __init__(self):
        self.n_hombro = 0
        self.n_codo = 0

    def __call__(self, d_b, d_a):
        if not LIMITES_HUMANOS:
            return d_b, d_a
        d_b0 = np.asarray(d_b, dtype=float)
        d_b, th = limitar_hombro(d_b0)
        if th:      # el antebrazo gira CON el brazo (se conserva la flexión del codo)
            eje = np.cross(d_b0, d_b)
            n = float(np.linalg.norm(eje))
            if n > 1e-9:
                ang = math.acos(float(np.clip(d_b0 @ d_b, -1.0, 1.0)))
                d_a = _rot(eje / n, ang) @ np.asarray(d_a, dtype=float)
        d_a, tc = limitar_codo(d_b, d_a)
        self.n_hombro += th
        self.n_codo += tc
        return d_b, d_a

    def texto(self):
        return f"[límites humanos: hombro {self.n_hombro}, codo {self.n_codo}]"


# ══════════════════════════════════════════════════════════════════════════════
# 2) LÍMITES DEL ROBOT Y GEOMETRÍA EXTRA (se llama después de preparar_hibrido)
# ══════════════════════════════════════════════════════════════════════════════

def _fk_palma(modelo, datos, hib, q3):
    """(palma, hombro, codo, eje de joint2) para joint1..3 = q3, sin tocar la simulación."""
    idx = hib["qpos3"]
    guarda = datos.qpos[idx].copy()
    datos.qpos[idx] = q3
    mujoco.mj_kinematics(modelo, datos)
    bm = hib["b_mano"]
    palma = np.array(datos.xpos[bm] + datos.xmat[bm].reshape(3, 3) @ hib["off_punta"])
    hombro = np.array(datos.xanchor[hib["j_hombro"]])
    codo = np.array(datos.xanchor[hib["j_codo"]])
    eje2 = np.array(datos.xaxis[hib["j_hombro"]])
    datos.qpos[idx] = guarda
    mujoco.mj_kinematics(modelo, datos)
    return palma, hombro, codo, eje2


def _q_imit(joint, ang):
    """Ángulo humano (rad) → valor del joint con ARM_IMITACION (sin recortar)."""
    _f, s, off, k = ARM_IMITACION[joint]
    return off + s * k * ang


def _signo_espejo(modelo, datos, hib):
    """Sentido en que hay que girar el hombro para pasar el codo al otro lado sin mover
    la palma: +1 si joint2 y joint3 giran en sentidos opuestos (ARCTOS), -1 si en el
    mismo (FR5). Se mide: el signo que deja la palma más cerca de donde estaba."""
    q = np.clip(np.array([0.0, 0.5 * (hib["lo"][1] + hib["hi"][1]), hib["q3_recto"] - 0.8]),
                hib["lo"], hib["hi"])
    p0 = _fk_palma(modelo, datos, hib, q)[0]
    mejor, signo = None, 1.0
    for s in (1.0, -1.0):
        hib["espejo_signo"] = s
        e = float(np.linalg.norm(_fk_palma(modelo, datos, hib, semilla_codo_espejo(q, hib))[0] - p0))
        if mejor is None or e < mejor:
            mejor, signo = e, s
    return signo


def completar_hib(modelo, datos, hib, verbose=True):
    """Añade a 'hib' el codo recto real, la longitud efectiva del antebrazo y
    recorta hib['lo'] / hib['hi'] a límites con sentido físico (ver cabecera)."""
    # — joint3 recto: el que da MÁS alcance hombro→palma —
    # (FR5: se busca en todo el rango de joint3; antes 1.2..1.9, que solo valía para el ARCTOS)
    _lo3, _hi3 = LIMITES_CTRL_BRAZO[2]
    qs = np.linspace(max(_lo3, -math.pi), min(_hi3, math.pi), 721)
    dist = []
    for q in qs:
        palma, hombro, _, _ = _fk_palma(modelo, datos, hib, np.array([0.0, 0.0, q]))
        dist.append(float(np.linalg.norm(palma - hombro)))
    q3_recto = float(qs[int(np.argmax(dist))])
    alcance = float(max(dist))
    # — longitud del brazo EN SU PLANO (la que decide la altura del codo) —
    _, hombro, codo, eje2 = _fk_palma(modelo, datos, hib, np.array([0.0, 0.0, q3_recto]))
    v = codo - hombro
    L1p = float(np.linalg.norm(v - (v @ eje2) * eje2))
    hib["q3_recto"] = q3_recto
    hib["alcance_fiel"] = alcance
    hib["L1_ef"] = float(hib["L_brazo"])
    hib["L2_ef"] = alcance - float(hib["L_brazo"])     # recto → circunferencia de radio 0

    # — límites —
    flex = math.radians(CODO_FLEX_MAX)
    z_h = float(hombro[2])
    # FR5: los límites salen de ARM_IMITACION (el mismo mapeo que la imitación), no de
    # suponer "q2 = 0 vertical" y "q1 = -azimut" (solo cierto en el ARCTOS, donde da
    # exactamente los valores de antes).
    e_min = math.acos(float(np.clip((z_h - CODO_Z_MIN) / max(L1p, 1e-6), -1.0, 1.0)))
    if base_brazo.BASE_INCLINADA:
        # 27-oct: base inclinada → joint1/joint2 no son "azimut/elevación". Sus límites salen
        # de recorrer con el mapeo exacto todo el rango humano (elevación desde la del codo
        # a CODO_Z_MIN hasta el brazo arriba y, en cada una, el azimut de rango_azimut) +
        # márgenes.
        (q1a, q1b), (q2a, q2b) = base_brazo.rangos_q12(
            lambda e: tuple(math.radians(v) for v in rango_azimut(math.degrees(e))),
            e_min, math.pi)
        q1_lo, q1_hi = q1a - Q1_MARGEN, q1b + Q1_MARGEN
        q2_lo, q2_hi = q2a - abs(Q2_MIN), q2b + abs(Q2_MIN)
    else:
        q2_lo, q2_hi = sorted((_q_imit("joint2", e_min),               # codo a CODO_Z_MIN
                               _q_imit("joint2", math.pi - Q2_MIN)))  # pasado de la vertical
        _q1 = (_q_imit("joint1", math.radians(HOMBRO_AZ_MIN)), _q_imit("joint1", math.pi))
        q1_lo, q1_hi = min(_q1) - Q1_MARGEN, max(_q1) + Q1_MARGEN
    lo = np.array([max(hib["lo"][0], q1_lo), max(hib["lo"][1], q2_lo),
                   max(hib["lo"][2], q3_recto - flex)])
    hi = np.array([min(hib["hi"][0], q1_hi), min(hib["hi"][1], q2_hi),
                   min(hib["hi"][2], q3_recto + flex)])
    hib["lo"], hib["hi"] = lo, hi
    hib["espejo_signo"] = _signo_espejo(modelo, datos, hib)   # ARCTOS +1 · FR5 -1
    if verbose:
        print(f"🦴 Límites del brazo (humanos+suelo): q1 [{lo[0]:.2f}, {hi[0]:.2f}]  "
              f"q2 [{lo[1]:.2f}, {hi[1]:.2f}]  q3 [{lo[2]:.2f}, {hi[2]:.2f}]  "
              f"(codo recto q3={q3_recto:.3f}, hombro a {z_h:.2f} m)")
        f_abajo = math.degrees(hi[2] - q3_recto)
        if f_abajo < CODO_FLEX_MAX - 1:
            print(f"⚠️  Codo 'hacia abajo' (q3 > {q3_recto:.2f}) limitado a {f_abajo:.0f}° por el "
                  f"rango del actuador de joint3: posturas tipo 'mano a la cara' no caben.")
    return hib


# ══════════════════════════════════════════════════════════════════════════════
# 3) OBJETIVOS FIELES: TU FLEXIÓN DE CODO → LA DEL ROBOT
# ══════════════════════════════════════════════════════════════════════════════

def _codo_en_circunferencia(hombro, palma, codo_ref, L1, L2):
    """Codo del robot a L1 del hombro y L2 de la palma, en el lado de codo_ref."""
    v = palma - hombro
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        return np.asarray(codo_ref, dtype=float)
    u = v / d
    if d >= L1 + L2 - 1e-6:
        return hombro + u * L1
    a = (L1 * L1 - L2 * L2 + d * d) / (2.0 * d)
    h = math.sqrt(max(L1 * L1 - a * a, 0.0))
    w = np.asarray(codo_ref, dtype=float) - hombro
    n = w - float(w @ u) * u
    nn = float(np.linalg.norm(n))
    if nn < 1e-6:
        return hombro + a * u
    return hombro + a * u + (n / nn) * h


def objetivos_codo_palma_fiel(datos, d_b, d_a, hib, longitudes_humano):
    """
    · DIRECCIÓN hombro→palma: la tuya (brazo + antebrazo + mano con TUS huesos).
    · DISTANCIA hombro→palma: la que tendría el ARCTOS con TU flexión de codo
      (ley del coseno con SUS longitudes). Tu brazo recto = su brazo recto.
    · CODO: en la circunferencia compatible, del lado de tu codo.
    """
    L1h, L2h = longitudes_humano
    L3h = HUMANO_MUNECA_PALMA
    d_b = np.asarray(d_b, dtype=float)
    d_a = np.asarray(d_a, dtype=float)
    dir_palma = _u(L1h * d_b + (L2h + L3h) * d_a)
    if dir_palma is None:
        dir_palma = d_b
    cos_f = float(np.clip(d_b @ d_a, -1.0, 1.0))
    L1, L2 = hib["L1_ef"], hib["L2_ef"]
    dist = math.sqrt(max(L1 * L1 + L2 * L2 + 2.0 * L1 * L2 * cos_f, 1e-6))
    hombro = np.array(datos.xanchor[hib["j_hombro"]])
    v = dist * (hib["ejes"] @ dir_palma)
    d_codo = hib["ejes"] @ d_b
    if REUBICAR_OBJETIVOS:          # 7-oct: comprimir abajo + horizontal estable
        v, h_ref, w = reubicar_palma(datos, hib, hombro, v)
        d_codo, _ = reubicar_direccion(d_codo, L1, float(hombro[2]), _CODO_Z_MIN_OBJ, h_ref, w)
    palma = hombro + v
    palma[2] = max(palma[2], ARM_Z_MINIMA)
    codo_ref = hombro + L1 * d_codo
    codo = _codo_en_circunferencia(hombro, palma, codo_ref, L1, L2)
    return codo, palma


def semilla_codo_espejo(q, hib):
    """Misma palma con el codo doblado al OTRO lado del plano del brazo, sin girar
    la base: q2' = q2 + 2β, q3' = 2·q3_recto − q3."""
    q = np.asarray(q, dtype=float).copy()
    phi = hib["q3_recto"] - q[2]
    L1, L2 = hib["L1_ef"], hib["L2_ef"]
    beta = math.atan2(L2 * math.sin(phi), L1 + L2 * math.cos(phi))
    q[1] = q[1] + hib.get("espejo_signo", 1.0) * 2.0 * beta
    q[2] = 2.0 * hib["q3_recto"] - q[2]
    return np.clip(q, hib["lo"], hib["hi"])
