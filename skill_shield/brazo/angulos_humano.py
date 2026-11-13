"""
Ángulos del brazo humano (MediaPipe Pose + estéreo → marco del torso) y pronosupinación.

(Extraído de Skill_Shield_Nivel_6.py — código sin cambios, solo reubicado.)
"""
import numpy as np
import math

from config import *

from skill_shield.vision.profundidad_3d import pixel_a_3d
from skill_shield.brazo import base_brazo   # 27-oct: base inclinada (soporte de hombros)


# ══════════════════════════════════════════════════════════════════════════════
# ÁNGULOS DEL BRAZO HUMANO (MediaPipe Pose → marco del torso)
# ══════════════════════════════════════════════════════════════════════════════

def _unitario(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def _angulos_desde_puntos(P, vis, brazo: str = "derecho",
                          vis_min: float = POSE_VISIBILIDAD_MIN):
    """
    Devuelve los ángulos del brazo humano, medidos respecto al TORSO (no a la
    cámara), o None si los puntos necesarios no se ven con confianza.

    Marco del torso (se construye con hombros y caderas):
        derecha : de tu hombro izquierdo al derecho
        arriba  : de la cadera al hombro
        delante : hacia donde miras
    Así los ángulos no cambian si te giras o te mueves delante de la cámara.

    Devuelve un dict con (todo en GRADOS):
        "azimut"       giro del brazo (hombro→codo) alrededor de la vertical.
                       0 = apunta al frente | +90 = hacia fuera (lado del brazo)
                       | -90 = cruzado hacia el otro lado | ±180 = hacia atrás
        "elevacion"    0 = brazo colgando | 90 = horizontal | 180 = hacia arriba
        "codo"         flexión del codo. 0 = brazo estirado | ~150 = muy doblado
        "azimut_muneca" igual que "azimut" pero para la línea hombro→MUÑECA
        "azimut_fiable" False si el brazo está casi vertical (el azimut es ruido)
        "ref"          "caderas" o "vertical" (cómo se obtuvo el eje 'arriba')
    """
    # P(i)   → punto 3D (np.array, metros) del landmark i, en ejes de cámara
    #          (x = derecha de la imagen, y = ABAJO, z = alejándose de la cámara)
    # vis(i) → confianza 0..1 de ese punto (0 = no disponible)

    if brazo == "derecho":
        i_h, i_c, i_m, lado = PS_HOMBRO_DER, PS_CODO_DER, PS_MUNECA_DER, 1.0
    else:
        i_h, i_c, i_m, lado = PS_HOMBRO_IZQ, PS_CODO_IZQ, PS_MUNECA_IZQ, -1.0

    # Puntos imprescindibles: hombro, codo y muñeca del brazo + ambos hombros
    for i in (i_h, i_c, i_m, PS_HOMBRO_IZQ, PS_HOMBRO_DER):
        if vis(i) < vis_min:
            return None

    # Eje "arriba": con las caderas si se ven; si no (torso cortado por el
    # encuadre), con la vertical de la cámara (y de imagen crece hacia ABAJO).
    if ARRIBA_REF == "caderas" and vis(PS_CADERA_IZQ) >= vis_min and vis(PS_CADERA_DER) >= vis_min:
        arriba = _unitario((P(PS_HOMBRO_IZQ) + P(PS_HOMBRO_DER)) / 2.0
                           - (P(PS_CADERA_IZQ) + P(PS_CADERA_DER)) / 2.0)
        ref = "caderas"
    else:
        arriba, ref = np.array([0.0, -1.0, 0.0]), "vertical"
    derecha = _unitario(P(PS_HOMBRO_DER) - P(PS_HOMBRO_IZQ))
    if arriba is None or derecha is None:
        return None
    derecha = _unitario(derecha - np.dot(derecha, arriba) * arriba)   # ortogonalizar
    if derecha is None:
        return None
    delante = np.cross(arriba, derecha)       # ya es unitario (ortogonales)

    def a_torso(v):
        return np.array([np.dot(v, derecha), np.dot(v, arriba), np.dot(v, delante)])

    u = _unitario(P(i_c) - P(i_h))     # brazo (hombro→codo)
    w = _unitario(P(i_m) - P(i_c))     # antebrazo (codo→muñeca)
    s = _unitario(P(i_m) - P(i_h))     # hombro→muñeca
    if u is None or w is None or s is None:
        return None

    r, a, f = a_torso(u)
    elevacion = math.degrees(math.acos(float(np.clip(-a, -1.0, 1.0))))
    azimut    = math.degrees(math.atan2(lado * r, f))
    codo      = math.degrees(math.acos(float(np.clip(np.dot(u, w), -1.0, 1.0))))
    rs, _, fs = a_torso(s)
    azimut_m  = math.degrees(math.atan2(lado * rs, fs))
    sv = a_torso(P(i_m) - P(i_h))      # hombro→muñeca en METROS, ejes del torso
    wr, wa, wf = a_torso(w)             # antebrazo en ejes del torso

    return {
        "azimut":        azimut,
        "elevacion":     elevacion,
        "codo":          codo,
        "azimut_muneca": azimut_m,
        "azimut_fiable": AZIMUT_ELEV_MIN < elevacion < 180.0 - AZIMUT_ELEV_MIN,
        "ref":           ref,
        "antebrazo": w,        # unitario codo→muñeca, ejes de cámara
        "arriba":    arriba,   # vertical del torso, ejes de cámara
        "derecha":   derecha,  # ejes del torso (para el eje del codo del robot)
        "delante":   delante,
        "lado":      lado,
        "hombro_muneca": np.array([lado * sv[0], sv[1], sv[2]]),   # (fuera, arriba, delante) [m]
        # PASO 3 — DIRECCIONES unitarias en (fuera, arriba, delante), como el TRON 2:
        "dir_brazo":     np.array([lado * r,  a,  f]),    # hombro→codo
        "dir_antebrazo": np.array([lado * wr, wa, wf]),   # codo→muñeca
    }

def _lm_hand(hand_world_lm, i):
    l = hand_world_lm.landmark[i]
    return np.array([l.x, l.y, l.z], float)

def _wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


# (Aquí había una versión ANTIGUA de calcular_pronosupinacion, con "arriba", que
#  ANULABA a la nueva —la de "eje_ref", al principio del archivo— porque Python se
#  queda con la última definición. Eliminada.)

def calcular_angulos_brazo_humano(pose_world_lm, brazo: str = "derecho",
                                  vis_min: float = POSE_VISIBILIDAD_MIN):
    """Ángulos del brazo usando los puntos 3D 'world' de MediaPipe Pose
    (profundidad ESTIMADA por la red neuronal, poco fiable en el eje z)."""
    lm = pose_world_lm.landmark
    return _angulos_desde_puntos(
        lambda i: np.array([lm[i].x, lm[i].y, lm[i].z]),
        lambda i: lm[i].visibility, brazo, vis_min)


def profundidad_robusta(depth_frame: np.ndarray, px: int, py: int,
                        radio: int = PROF_RADIO_PX, percentil: float = PROF_PERCENTIL):
    """Profundidad (m) en un parche alrededor de (px, py), en coordenadas del mapa
    de profundidad. Usa un percentil BAJO (no la mediana): en los bordes del brazo
    el parche mezcla brazo (cerca) y fondo (lejos), y así nos quedamos con el brazo."""
    h, w = depth_frame.shape
    if not (0 <= px < w and 0 <= py < h):
        return None
    region  = depth_frame[max(0, py - radio):min(h, py + radio + 1),
                          max(0, px - radio):min(w, px + radio + 1)]
    validos = region[region > 0]
    if len(validos) < PROF_MIN_PIXELES:
        return None
    return float(np.percentile(validos, percentil)) / 1000.0


class EstimadorLongitudesBrazo:
    """
    Mide cuánto miden el brazo (hombro→codo) y el antebrazo (codo→muñeca) del
    usuario, y sirve para dos cosas:
      · comprobar que un codo detectado es CREÍBLE (huesos con su longitud normal)
      · reconstruir el codo cuando MediaPipe lo coloca mal o no lo ve

    Las longitudes se APRENDEN SOLO tras llamar a iniciar_calibracion() (al pulsar
    [A]): se promedian 'n_objetivo' fotogramas válidos y luego se CONGELAN.
    Si se pasa 'manual' = (brazo, antebrazo) en metros, no hace falta calibrar.
    """
    BANDA_BRAZO     = (0.22, 0.38)   # límites absolutos razonables (m)
    BANDA_ANTEBRAZO = (0.18, 0.34)

    def __init__(self, manual=None, tol: float = LONG_BRAZO_TOL,
                 n_objetivo: int = CALIB_HUESOS_FRAMES):
        self.manual     = manual
        self.tol        = tol
        self.n_objetivo = n_objetivo
        self.h1, self.h2 = [], []
        self._fijas     = None     # (brazo, antebrazo) ya congeladas
        self.calibrando = False
        self.ultimo_rel = None     # último vector hombro→codo CREÍBLE (para desambiguar)
        self.motivo_fallo = ""     # por qué falló la última medición estéreo (diagnóstico)

    # — Estado de la calibración —
    @property
    def calibrado(self) -> bool:
        return self.manual is not None or self._fijas is not None

    @property
    def progreso(self) -> int:
        return len(self.h1)

    @property
    def longitudes(self):
        if self.manual is not None:
            return tuple(self.manual)
        return self._fijas

    def iniciar_calibracion(self):
        """Empieza a aprender los huesos (se llama al pulsar [A])."""
        if self.manual is not None:
            return
        self.h1, self.h2 = [], []
        self._fijas      = None
        self.calibrando  = True
        self.ultimo_rel  = None

    def codo_creible(self, S, E, W, aprender: bool = True) -> bool:
        l1, l2 = float(np.linalg.norm(E - S)), float(np.linalg.norm(W - E))
        en_banda = (self.BANDA_BRAZO[0] <= l1 <= self.BANDA_BRAZO[1] and
                    self.BANDA_ANTEBRAZO[0] <= l2 <= self.BANDA_ANTEBRAZO[1])
        ref = self.longitudes
        if ref is None:
            ok = en_banda
        else:
            ok = (abs(l1 - ref[0]) <= self.tol * ref[0] and
                  abs(l2 - ref[1]) <= self.tol * ref[1])
        if ok and aprender and en_banda and self.calibrando:
            self.h1.append(l1)
            self.h2.append(l2)
            if len(self.h1) >= self.n_objetivo:
                self._fijas     = (float(np.median(self.h1)), float(np.median(self.h2)))
                self.calibrando = False
                print(f"🦴 Huesos calibrados: brazo={self._fijas[0]:.3f} m  "
                      f"antebrazo={self._fijas[1]:.3f} m")
        if ok:
            self.ultimo_rel = E - S
        return ok

    def reconstruir_codo(self, S, W, abajo=np.array([0.0, 1.0, 0.0])):
        """
        Posición del codo sabiendo hombro S, muñeca W y las longitudes de los dos
        huesos. Si el brazo está estirado, el codo cae en la línea S→W. Si no, hay
        una circunferencia de posiciones posibles; se elige el punto más cercano a
        donde estaba el codo en el último fotograma creíble (el codo se mueve de forma
        continua) y, si no hay ninguno, el más 'caído' hacia 'abajo' (gravedad, ejes cámara).
        """
        ref = self.longitudes
        if ref is None:
            return None
        L1, L2 = ref
        v = W - S
        d = float(np.linalg.norm(v))
        if d < 1e-6:
            return None
        u = v / d
        if d >= L1 + L2:                       # brazo estirado (o más): codo en la línea
            if d > (L1 + L2) * (1.0 + CODO_ESTIRADO_MARGEN):
                return None                    # muñeca mal medida → no inventar un codo recto
            return S + u * (d * L1 / (L1 + L2))
        if d <= abs(L1 - L2) + 1e-3:
            return None
        a = (L1 ** 2 - L2 ** 2 + d ** 2) / (2.0 * d)
        h = math.sqrt(max(L1 ** 2 - a ** 2, 0.0))
        pref = self.ultimo_rel if self.ultimo_rel is not None else abajo
        n = pref - np.dot(pref, u) * u
        nn = float(np.linalg.norm(n))
        if nn < 1e-6:
            n = np.cross(u, np.array([1.0, 0.0, 0.0]))
            nn = float(np.linalg.norm(n))
            if nn < 1e-6:
                return None
        return S + u * a + (n / nn) * h


def corregir_muneca_por_longitud(S, px, py, z_medida, L, ancho, alto, f_px, z_ref=None,
                                  radio=None):
    """
    La muñeca NO puede estar más lejos del hombro que brazo+antebrazo (L).
    Si lo está, lo que falla es su PROFUNDIDAD (p.ej. el estéreo midió la pared de
    detrás de la mano); su posición en la IMAGEN (px, py) sí es fiable. Se recoloca
    sobre el mismo rayo de la cámara, justo a distancia L del hombro S.
    De las dos soluciones (delante / detrás) se elige la más cercana a z_ref
    (profundidad relativa de MediaPipe) o, si no la hay, a la medida.
    Devuelve el punto corregido, o None si la medida ya era posible.
    """
    d = np.array([(px - ancho / 2.0) / f_px, (py - alto / 2.0) / f_px, 1.0])  # rayo (z = 1)
    if np.linalg.norm(d * z_medida - S) <= L:
        return None
    # Se recoloca a distancia `radio` del hombro (por defecto L = brazo estirado).
    # Usar L SIEMPRE forzaba el codo a 0° (brazo recto) cada vez que actuaba.
    L = L if radio is None else float(min(radio, L))
    a = float(d @ d)
    b = -2.0 * float(d @ S)
    c = float(S @ S) - L * L
    disc = b * b - 4.0 * a * c
    if disc < 0.0:                       # el rayo no llega a la esfera: su punto más cercano
        z = float(d @ S) / a
    else:
        r = math.sqrt(disc)
        z1, z2 = (-b - r) / (2.0 * a), (-b + r) / (2.0 * a)   # delante / detrás del hombro
        ref = z_ref if z_ref is not None else z_medida
        z = z1 if abs(z1 - ref) <= abs(z2 - ref) else z2
    return d * max(z, PROF_Z_MIN)


class PuntosCamara:
    """
    Puntos 3D del brazo medidos por UNA cámara, en el marco de ESA cámara
    (x = derecha de la imagen, y = ABAJO, z = alejándose de la cámara).
    Es el resultado de los pasos 1-3b de la medición estéreo; el paso 4 (comprobar o
    reconstruir el codo) y los ángulos se hacen después, sobre estos puntos o sobre
    los de varias cámaras fusionados (ver skill_shield/vision/fusion_brazo.py).
    """
    __slots__ = ("pts", "fuente", "vis", "codo_mp", "f_px")

    def __init__(self, pts, fuente, vis, codo_mp, f_px):
        self.pts     = pts       # {índice Pose: np.array 3D (m)}
        self.fuente  = fuente    # {índice Pose: "E" | "E+" | "M" | "L"}
        self.vis     = vis       # {índice Pose: visibilidad 0..1 de MediaPipe}
        self.codo_mp = codo_mp   # codo con profundidad de MediaPipe (respaldo) o None
        self.f_px    = f_px


def puntos_brazo_estereo(pose_lm, frame_depth, f_px: float,
                         brazo: str = "derecho",
                         vis_min: float = POSE_VISIBILIDAD_MIN,
                         ancho: int = ANCHO_CAM, alto: int = ALTO_CAM,
                         estimador: "EstimadorLongitudesBrazo" = None,
                         pose_world_lm=None, fallos: list = None,
                         exigir_hombro_opuesto: bool = True):
    """
    Pasos 1-3b de la medición estéreo de UNA cámara:
      · posición en la imagen (x, y)  ← MediaPipe Pose (pose_landmarks, normalizados)
      · profundidad (z)               ← mapa estéreo, en el píxel de cada articulación
    Si una articulación no tiene profundidad estéreo válida, en vez de perder el
    brazo entero se rellena (ver PROF_RELLENO_MEDIAPIPE): parche más grande y, si
    tampoco, profundidad relativa de MediaPipe (pose_world_lm) anclada al estéreo.

    Devuelve un PuntosCamara, o None si faltan puntos imprescindibles (los motivos
    se AÑADEN a la lista 'fallos').

    exigir_hombro_opuesto=False (cámara SECUNDARIA, vista de lado): el hombro del otro
    brazo suele quedar tapado por el torso y no hace falta, porque el marco del torso
    se construye con los puntos de la cámara principal. Entonces el hombro se mide en
    su propio píxel (con la cámara de lado la mano no lo tapa).
    """
    nombres = {PS_HOMBRO_IZQ: "hombro izq", PS_HOMBRO_DER: "hombro der",
               PS_CODO_IZQ: "codo izq",     PS_CODO_DER: "codo der",
               PS_MUNECA_IZQ: "muñeca izq", PS_MUNECA_DER: "muñeca der",
               PS_CADERA_IZQ: "cadera izq", PS_CADERA_DER: "cadera der"}
    if fallos is None:
        fallos = []

    lm  = pose_lm.landmark
    wl  = pose_world_lm.landmark if pose_world_lm is not None else None
    dh, dw = frame_depth.shape

    if brazo == "derecho":
        i_h, i_c, i_m, i_h2 = PS_HOMBRO_DER, PS_CODO_DER, PS_MUNECA_DER, PS_HOMBRO_IZQ
    else:
        i_h, i_c, i_m, i_h2 = PS_HOMBRO_IZQ, PS_CODO_IZQ, PS_MUNECA_IZQ, PS_HOMBRO_DER
    caderas = (PS_CADERA_IZQ, PS_CADERA_DER)

    def usable(i):
        l = lm[i]
        return l.visibility >= vis_min and 0.0 <= l.x < 1.0 and 0.0 <= l.y < 1.0

    def motivo_no_usable(i):
        l = lm[i]
        if l.visibility < vis_min:
            return f"{nombres[i]}: visibilidad {l.visibility:.2f} < {vis_min:.2f}"
        return f"{nombres[i]}: fuera de imagen"

    # ── 1) Puntos imprescindibles VISIBLES en la imagen ───────────────────────
    codo_calibrable = estimador is not None and estimador.longitudes is not None
    req = (i_h, i_h2, i_m) if exigir_hombro_opuesto else (i_h, i_m)
    for i in req:
        if not usable(i):
            fallos.append(motivo_no_usable(i))
    if not usable(i_c) and not codo_calibrable:
        fallos.append(motivo_no_usable(i_c) + " (y sin huesos calibrados para reconstruirlo)")
    if fallos:
        return None

    # ── 2) Profundidad estéreo de cada punto (parche normal → parche grande) ──
    def z_estereo(xn, yn, percentil):
        for radio, etiqueta in ((PROF_RADIO_PX, "E"), (2 * PROF_RADIO_PX, "E+")):
            z = profundidad_robusta(frame_depth, int(xn * dw), int(yn * dh),
                                    radio=radio, percentil=percentil)
            if z is not None and PROF_Z_MIN < z < PROF_Z_MAX:
                return z, etiqueta
            if not PROF_RELLENO_MEDIAPIPE:
                break          # comportamiento antiguo: un solo intento
        return None, None

    z, fuente = {}, {}
    # Hombros: medidos en el PECHO (ver HOMBRO_PROF_HACIA_TORSO)
    if exigir_hombro_opuesto or usable(i_h2):
        x_med = (lm[i_h].x + lm[i_h2].x) / 2.0
        y_med = (lm[i_h].y + lm[i_h2].y) / 2.0
        for i in (i_h, i_h2):
            xs = lm[i].x + HOMBRO_PROF_HACIA_TORSO * (x_med - lm[i].x)
            ys = lm[i].y + HOMBRO_PROF_HACIA_TORSO * (y_med - lm[i].y)
            zi, f = z_estereo(xs, ys, 50)
            if zi is not None:
                z[i], fuente[i] = zi, f
    else:                                  # cámara secundaria: solo el hombro del brazo
        zi, f = z_estereo(lm[i_h].x, lm[i_h].y, 50)
        if zi is not None:
            z[i_h], fuente[i_h] = zi, f
    for i in (i_c, i_m) + caderas:
        if usable(i):
            zi, f = z_estereo(lm[i].x, lm[i].y, PROF_PERCENTIL)
            if zi is not None:
                z[i], fuente[i] = zi, f

    # Descarta puntos cuya profundidad se va muy lejos del resto (= se midió el fondo)
    if z:
        z_med = float(np.median(list(z.values())))
        for i in list(z):
            if abs(z[i] - z_med) > PROF_DISPERSION_Z_MAX:
                fallos.append(f"{nombres[i]}: estéreo incoherente ({z[i]:.2f} m vs {z_med:.2f} m)")
                del z[i], fuente[i]

    # ── 3) Relleno de los que faltan con MediaPipe anclado al estéreo ─────────
    anclas = [a for a in (i_h, i_h2) + caderas if a in z and fuente[a] != "M"]

    def z_mediapipe(i):
        """Profundidad de MediaPipe anclada al estéreo del primer punto ancla."""
        a = anclas[0]
        return float(np.clip(z[a] + (wl[i].z - wl[a].z), PROF_Z_MIN + 0.01, PROF_Z_MAX - 0.01))

    puede_mp = PROF_RELLENO_MEDIAPIPE and wl is not None and bool(anclas)

    # ── 3a) Estéreo de codo y muñeca VALIDADO con MediaPipe (idea del TRON 2) ──
    # (ver comentarios completos en la versión anterior de esta función: solo se valida
    #  el CODO, y solo con huesos calibrados; la muñeca la protege el paso 3b.)
    if puede_mp and PROF_TOL_MEDIAPIPE > 0 and codo_calibrable:
        if i_c in z and fuente[i_c] != "M":
            if abs(z[i_c] - z_mediapipe(i_c)) > PROF_TOL_MEDIAPIPE:
                del z[i_c], fuente[i_c]
    # El codo sin estéreo se prefiere RECONSTRUIR con los huesos (paso 4, más exacto);
    # MediaPipe solo se usa para él si no hay huesos calibrados.
    necesarios = req + ((i_c,) if usable(i_c) and not codo_calibrable else ())
    for i in necesarios:
        if i in z:
            continue
        if puede_mp:
            z[i], fuente[i] = z_mediapipe(i), "M"
        else:
            fallos.append(f"{nombres[i]}: sin profundidad estéreo"
                          + ("" if anclas else " (ni en hombros ni en caderas)"))
    if any(i not in z for i in req):
        return None

    pts = {i: np.array(pixel_a_3d(lm[i].x * ancho, lm[i].y * alto, zi, ancho, alto, f_px))
           for i, zi in z.items()}

    # ── 3b) Muñeca imposible (más lejos del hombro que brazo+antebrazo) ───────
    # Suele ser el estéreo leyendo el FONDO tras la mano (brazo en alto, mano
    # pequeña en la imagen). Se corrige solo su profundidad, sobre su rayo.
    if estimador is not None and estimador.longitudes is not None:
        _L = sum(estimador.longitudes) * (1.0 + CODO_ESTIRADO_MARGEN)
        # Si la muñeca es imposible, se recoloca a la ÚLTIMA distancia hombro-muñeca
        # buena (así se conserva la flexión del codo en vez de enderezarlo de golpe).
        _W = corregir_muneca_por_longitud(
            pts[i_h], lm[i_m].x * ancho, lm[i_m].y * alto, float(pts[i_m][2]),
            _L, ancho, alto, f_px,
            z_ref=z_mediapipe(i_m) if puede_mp else None,
            radio=getattr(estimador, "ultima_d_sw", None))
        if _W is None:
            estimador.ultima_d_sw = float(np.linalg.norm(pts[i_m] - pts[i_h]))
        else:
            pts[i_m], fuente[i_m] = _W, "L"

    # Respaldo del codo (paso 4): profundidad de MediaPipe anclada al estéreo
    codo_mp = None
    if puede_mp and usable(i_c):
        codo_mp = np.array(pixel_a_3d(lm[i_c].x * ancho, lm[i_c].y * alto,
                                      z_mediapipe(i_c), ancho, alto, f_px))

    vis = {i: float(lm[i].visibility) for i in pts}
    vis[i_c] = float(lm[i_c].visibility)
    return PuntosCamara(pts, fuente, vis, codo_mp, f_px)


def angulos_desde_puntos_3d(pts: dict, fuente: dict, vis_hcm: tuple, codo_mp,
                            estimador, brazo: str, vis_min: float, fallos: list):
    """
    Pasos 4 y final de la medición estéreo, sobre unos puntos 3D YA medidos (de una
    cámara o fusionados de varias; en ejes x=derecha, y=ABAJO, z=profundidad):
      · comprueba el codo con las longitudes de los huesos y lo RECONSTRUYE si es falso
      · calcula los ángulos y direcciones del brazo (marco del torso)
    'vis_hcm' = (visibilidad hombro, codo, muñeca). 'pts' y 'fuente' se modifican.
    Devuelve el dict de ángulos o None (motivos añadidos a 'fallos').
    """
    if brazo == "derecho":
        i_h, i_c, i_m = PS_HOMBRO_DER, PS_CODO_DER, PS_MUNECA_DER
    else:
        i_h, i_c, i_m = PS_HOMBRO_IZQ, PS_CODO_IZQ, PS_MUNECA_IZQ

    # ── 4) Comprobación / reconstrucción del codo ─────────────────────────────
    codo_estimado = False
    if estimador is not None:
        codo_medido = pts.get(i_c)
        if codo_medido is not None and not estimador.codo_creible(
                pts[i_h], codo_medido, pts[i_m], aprender=vis_hcm[1] >= 0.7):
            del pts[i_c]                        # codo falso → se intenta reconstruir
        if i_c not in pts:
            E = estimador.reconstruir_codo(pts[i_h], pts[i_m])
            if E is not None:
                pts[i_c], fuente[i_c] = E, "R"
                codo_estimado = True
            elif codo_medido is not None:
                # No se puede reconstruir (p.ej. codo MUY doblado, mano en el hombro):
                # mejor el codo medido, aunque sea dudoso, que perder el brazo.
                pts[i_c] = codo_medido
            elif codo_mp is not None:
                pts[i_c] = codo_mp
                fuente[i_c] = "M"
    if i_c not in pts:
        fallos.append("codo: sin posición (ni medida ni reconstruible)")
        return None

    res = _angulos_desde_puntos(
        lambda i: pts[i],
        lambda i: 1.0 if i in pts else 0.0,
        brazo, vis_min)
    if res is None:
        fallos.append("geometría degenerada (hombros superpuestos)")
        return None
    res["codo_estimado"] = codo_estimado
    res["vis"]     = tuple(vis_hcm)
    res["fuentes"] = f"h:{fuente.get(i_h, '?')} c:{fuente.get(i_c, '?')} m:{fuente.get(i_m, '?')}"
    if estimador is not None:
        estimador.motivo_fallo = ""
    return res


def calcular_angulos_brazo_estereo(pose_lm, frame_depth, f_px: float,
                                   brazo: str = "derecho",
                                   vis_min: float = POSE_VISIBILIDAD_MIN,
                                   ancho: int = ANCHO_CAM, alto: int = ALTO_CAM,
                                   estimador: "EstimadorLongitudesBrazo" = None,
                                   pose_world_lm=None):
    """
    Ángulos del brazo usando la profundidad REAL de la OAK-D (UNA cámara).
    Es la composición de puntos_brazo_estereo() + angulos_desde_puntos_3d(); con dos
    cámaras se fusionan los puntos entre ambas (fusion_brazo.py). Mismo resultado que
    antes de separarla.

    Con un 'estimador':
      · un codo con huesos de longitud imposible se RECONSTRUYE con hombro y muñeca
      · si devuelve None, 'estimador.motivo_fallo' explica por qué
    El resultado lleva "fuentes" (de dónde salió la profundidad de hombro/codo/muñeca:
    E = estéreo, E+ = estéreo parche grande, M = MediaPipe anclado, R = reconstruido).
    """
    fallos = []
    pc = puntos_brazo_estereo(pose_lm, frame_depth, f_px, brazo, vis_min, ancho, alto,
                              estimador, pose_world_lm, fallos)
    if pc is None:
        if estimador is not None:
            estimador.motivo_fallo = "; ".join(fallos) if fallos else "puntos insuficientes"
        return None
    if brazo == "derecho":
        i_h, i_c, i_m = PS_HOMBRO_DER, PS_CODO_DER, PS_MUNECA_DER
    else:
        i_h, i_c, i_m = PS_HOMBRO_IZQ, PS_CODO_IZQ, PS_MUNECA_IZQ
    res = angulos_desde_puntos_3d(pc.pts, pc.fuente,
                                  (pc.vis[i_h], pc.vis[i_c], pc.vis[i_m]),
                                  pc.codo_mp, estimador, brazo, vis_min, fallos)
    if res is None and estimador is not None:
        estimador.motivo_fallo = "; ".join(fallos) if fallos else "puntos insuficientes"
    return res


def angulos_humano_a_q_brazo(azimut_rad: float, elevacion_rad: float, codo_rad: float):
    """
    PASO 4 — ángulos del brazo humano (rad) → objetivos de joint1, joint2 y joint3
    del ARCTOS (rad), con la tabla ARM_IMITACION de config.py y recortados a
    IMITAR_LIMITES y a LIMITES_CTRL_BRAZO.
    """
    fuentes = {"azimut": azimut_rad, "elevacion": elevacion_rad, "codo": codo_rad}
    # 27-oct: con la base INCLINADA (soporte de hombros) la tabla lineal no vale: mapeo
    # exacto con la cinemática del FR5 (base_brazo.py). Con la base vertical, lo de siempre.
    q_inc = (base_brazo.q_desde_angulos(azimut_rad, elevacion_rad, codo_rad)
             if base_brazo.BASE_INCLINADA else None)
    q = []
    for k, joint in enumerate(("joint1", "joint2", "joint3")):
        fuente, signo, offset, escala = ARM_IMITACION[joint]
        v = offset + signo * escala * fuentes[fuente] if q_inc is None else float(q_inc[k])
        lo, hi = IMITAR_LIMITES[joint]
        lo = max(lo, LIMITES_CTRL_BRAZO[k][0])
        hi = min(hi, LIMITES_CTRL_BRAZO[k][1])
        q.append(float(np.clip(v, lo, hi)))
    return np.array(q)


def corregir_con_tpose(ang: dict, tpose_med):
    """
    Ajusta (azimut, elevación, codo) [GRADOS] para que la T-pose MEDIDA en la
    calibración coincida con la T-pose ideal T_POSE_HUMANA.
      · azimut : desplazamiento (con envoltura a ±180)
      · elevación: lineal a trozos con 3 puntos fijos 0→0, T_medida→T_ideal, 180→180
                   (así el brazo colgando sigue en 0 y el brazo arriba en 180)
      · codo   : desplazamiento, sin bajar de 0
    Si tpose_med es None, devuelve los ángulos tal cual.
    """
    az, el, co = ang["azimut"], ang["elevacion"], ang["codo"]
    if tpose_med is None:
        return az, el, co
    az_t, el_t, co_t = tpose_med
    az_i, el_i, co_i = T_POSE_HUMANA
    az = az - (az_t - az_i)
    az = (az + 180.0) % 360.0 - 180.0
    if el <= el_t:
        el = el * el_i / el_t
    else:
        el = el_i + (el - el_t) * (180.0 - el_i) / (180.0 - el_t)
    el = float(np.clip(el, 0.0, 180.0))
    co = max(0.0, co - (co_t - co_i))
    return az, el, co


def angulos_humano_a_q_brazo_rel(az_deg, el_deg, co_deg, ref_deg, q_ref):
    """
    Modo RELATIVO. La pose de reposo del sim (q_ref = ctrl[0:3] al empezar) equivale a
    la T-pose medida (ref_deg = az, elev, codo en GRADOS). Cada joint se mueve el
    incremento del ángulo humano respecto a esa T-pose, con el signo/escala de
    ARM_IMITACION, y se recorta a IMITAR_LIMITES y LIMITES_CTRL_BRAZO.
    """
    d_az = (az_deg - ref_deg[0] + 180.0) % 360.0 - 180.0
    fuentes = {"azimut":    math.radians(d_az),
               "elevacion": math.radians(el_deg - ref_deg[1]),
               "codo":      math.radians(co_deg - ref_deg[2])}
    dq_inc = None
    if base_brazo.BASE_INCLINADA:
        # 27-oct: base inclinada → incremento con el mapeo exacto (tu postura − tu T-pose)
        q_a = base_brazo.q_desde_angulos(*(math.radians(v) for v in (az_deg, el_deg, co_deg)))
        q_t = base_brazo.q_desde_angulos(*(math.radians(v) for v in ref_deg))
        dq_inc = q_a - q_t
        dq_inc[0] = (dq_inc[0] + math.pi) % (2.0 * math.pi) - math.pi
    q = []
    for k, joint in enumerate(("joint1", "joint2", "joint3")):
        fuente, signo, _offset, escala = ARM_IMITACION[joint]
        v = q_ref[k] + (signo * escala * fuentes[fuente] if dq_inc is None else float(dq_inc[k]))
        lo, hi = IMITAR_LIMITES[joint]
        lo = max(lo, LIMITES_CTRL_BRAZO[k][0])
        hi = min(hi, LIMITES_CTRL_BRAZO[k][1])
        q.append(float(np.clip(v, lo, hi)))
    return np.array(q)


def limitar_velocidad_articular(q_obj, q_prev, dt: float, vel_max: float):
    """Mueve q_prev hacia q_obj, como mucho vel_max·dt rad por joint en este fotograma."""
    paso = vel_max * max(dt, 1e-3)
    return q_prev + np.clip(np.asarray(q_obj) - np.asarray(q_prev), -paso, paso)


def _fmt_ang(a) -> str:
    """Formato corto de ángulos para la terminal."""
    if a is None:
        return "sin datos"
    return f"az={a['azimut']:6.1f}°  elev={a['elevacion']:5.1f}°  codo={a['codo']:5.1f}°"




def calcular_pronosupinacion(hand_world_lm, antebrazo, eje_ref):
    """Giro del antebrazo (rad) alrededor de su eje, medido desde el EJE DEL CODO
    DEL ROBOT (horizontal, perpendicular al plano del brazo según el azimut).
    A diferencia de la versión anterior, funciona con el antebrazo vertical.
    None si no es fiable."""
    a = _unitario(antebrazo)
    if a is None or eje_ref is None:
        return None
    h = eje_ref - (eje_ref @ a) * a                    # eje del codo, ⟂ antebrazo
    if np.linalg.norm(h) < 0.4 * np.linalg.norm(eje_ref):
        return None                                    # antebrazo casi paralelo al eje del codo
    h = _unitario(h)
    lat = _lm_hand(hand_world_lm, LM_MENIQUE_MCP) - _lm_hand(hand_world_lm, LM_INDICE_MCP)
    lat_p = lat - (lat @ a) * a                        # proyección ⟂ antebrazo
    if np.linalg.norm(lat_p) < 0.5 * np.linalg.norm(lat):
        return None                                    # nudillos casi alineados con el antebrazo
    lat_p = _unitario(lat_p)
    return math.atan2(float(np.cross(h, lat_p) @ a), float(h @ lat_p))
