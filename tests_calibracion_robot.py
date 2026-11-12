"""
Pruebas de la calibración cámara de cabeza → robot (calibrar_camara_robot.py), sin robot ni cámara:

    python -m pytest tests_calibracion_robot.py -q

Las distancias «de verdad» de estas pruebas se calculan aquí mismo, por fuerza bruta sobre las
piezas del gemelo (mallas del MJCF) y con medidas escritas a mano, sin pasar por el comprobador
de choques que se está probando.
"""
import math
import os
import sys
import types

import numpy as np
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg_real                                           # noqa: E402
import calibrar_camara_robot as crc                                 # noqa: E402
from skill_shield.autonomo import calibracion as C                  # noqa: E402
from skill_shield.autonomo import choque_mano as CM                 # noqa: E402
from skill_shield.teleop_fr5.cinematica import CadenaFR5            # noqa: E402

RUTA_MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fairino_description", "mjcf", "fairino_mjcf.xml")

# ── escenario de las pruebas (todo escrito a mano) ────────────────────────────────────────────
# Cámara de cabeza: encima de la columna del soporte (x = 0,26), mirando hacia −Y (delante).
T_MUNDO_CAM = np.array([[-1.0, 0.0, 0.0, 0.26],
                        [0.0, 0.0, -1.0, -0.085],
                        [0.0, -1.0, 0.0, 0.525],
                        [0.0, 0.0, 0.0, 1.0]])
K_CAM = np.array([[985.0, 0.0, 676.0], [0.0, 985.0, 507.0], [0.0, 0.0, 1.0]])
ANCHO, ALTO = 1352, 1014
# Tablero pequeño (7x5 de 20 mm) pegado a la palma (la palma mira hacia −x de la brida).
TABLERO_20 = dict(cuadros=(7, 5), lado=0.020, marcador=0.015, diccionario="DICT_4X4_50")
T_BRIDA_TABLERO_20 = np.array([[0.0, 0.0, 1.0, -0.03],
                               [1.0, 0.0, 0.0, -0.10],
                               [0.0, 1.0, 0.0, 0.31],
                               [0.0, 0.0, 0.0, 1.0]])
# Postura de inicio con el tablero delante de la cámara y la muñeca casi recta (grados)
Q0_DEG = [93.0, -17.0, 53.0, 43.0, 105.0, 25.0]

MARCO = {"upperarm_Link": 1, "forearm_Link": 2, "wrist1_Link": 3}   # marco de fk_marcos de cada eslabón


def _conf(**cambios):
    """Config de las pruebas: la del repositorio con los valores de la calibración fijados aquí,
    para que las pruebas no dependan de lo que se ajuste en config.py."""
    g = {k: getattr(cfg_real, k) for k in dir(cfg_real) if k.isupper()}
    g.update(CALIB_ROBOT_TABLERO=dict(TABLERO_20), CALIB_MANO_HOLGURA=0.03, CALIB_MANO_SALIENTES=0.04,
             CALIB_MANO_ACERCAR_MAX=0.01, CALIB_TABLERO_HOLGURA=0.04, CALIB_TABLERO_MARGEN=0.02,
             CALIB_ENTORNO_HOLGURA=0.05, CALIB_AUTO_RADIO_MANO=0.13,
             FR5_MARGEN_LIMITES_DEG=3.0,
             FR5_LIMITES_BLANDOS_DEG=((-175.0, 175.0), (-265.0, 85.0), (-10.0, 150.0),
                                      (-105.0, 85.0), (5.0, 175.0), (-175.0, 175.0)),
             TELEOP_FR5_OBSTACULOS=[], TELEOP_FR5_CAMARAS=[], TELEOP_FR5_ZONAS_PROHIBIDAS=[])
    g.update(cambios)
    return g


@pytest.fixture
def calib(monkeypatch):
    """calibrar_camara_robot con la config de las pruebas."""
    def poner(**cambios):
        g = _conf(**cambios)
        monkeypatch.setattr(crc, "cfg", types.SimpleNamespace(**g))
        monkeypatch.setattr(crc, "TABLERO", dict(g["CALIB_ROBOT_TABLERO"]))
        monkeypatch.setattr(crc, "RADIO_MANO", float(g["CALIB_AUTO_RADIO_MANO"]))
        # lo que calibrar_camara_robot lee de config al importarse, fijado aquí. La amplitud es la
        # más amplia (la de antes del 10-nov): las pruebas comprueban las reglas, no el recorte.
        for nombre, valor in dict(N_AUTO=20, VEL_MAX=10.0, AMPLITUD=(30, 20, 25, 45, 45, 70), ESPERA_S=1.0,
                                  ERROR_MAX=5.0, N_VERIFICAR=4, AUTO_RESET=True, SIN_AVANCE_S=5.0).items():
            monkeypatch.setattr(crc, nombre, valor)
        return crc
    return poner


@pytest.fixture(scope="module")
def cadena():
    return CadenaFR5.desde_mjcf(RUTA_MJCF)


@pytest.fixture(scope="module")
def piezas():
    """Mallas del gemelo como nubes de puntos, con un muestreo más fino que el de producción."""
    nubes = CM.nubes_mjcf(RUTA_MJCF, 0.004)
    return dict(mano=nubes["mano"], arboles={k: cKDTree(nubes[k]) for k in MARCO})


def _hueco_mano_brazo(cadena, piezas, q, solo=None):
    """Distancia (m) de la OrcaHand al antebrazo, al giro del J4 y al brazo, por fuerza bruta
    (inf si está a más de 10 cm de todo). solo: nombre de un único eslabón con el que medir."""
    m = cadena.fk_marcos(np.asarray(q, float))
    mano_mundo = piezas["mano"] @ m[5][:3, :3].T + m[5][:3, 3]
    d = math.inf
    for nombre, arbol in piezas["arboles"].items():
        if solo is not None and nombre != solo:
            continue
        T = np.linalg.inv(m[MARCO[nombre]])
        d = min(d, float(arbol.query(mano_mundo @ T[:3, :3].T + T[:3, 3], distance_upper_bound=0.10)[0].min()))
    return d


def _recorrido(q0, plan, v_max_deg=10.0, cada_deg=1.0):
    """Todas las posturas por las que pasa el robot yendo de q0 a cada postura del plan, tal
    como lo mueve fr5_servo (cada joint con su tope de velocidad), muestreadas cada ~1°."""
    salto = max(1, int(round(cada_deg / (v_max_deg * 0.04))))
    actual = np.asarray(q0, float)
    for p in plan:
        camino = C.simular_camino(actual, np.asarray(p["q"], float), v_max_deg)
        yield from camino[::salto]
        yield camino[-1]
        actual = np.asarray(p["q"], float)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 1. La muñeca no se dobla hasta aplastar lo que sale de la mano
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_plan_no_cierra_el_hueco_entre_la_mano_y_el_brazo(calib, cadena, piezas):
    """Rotura que caza: el planificador dobla la muñeca (J5 hacia 140°) hasta dejar la parte de
    atrás de la mano —por donde salen los cables— a ~3 cm del antebrazo y del giro del J5, cuando
    al empezar había más de 6 cm: lo que sobresale de la mano queda aplastado."""
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    assert _hueco_mano_brazo(cadena, piezas, q0) > 0.060          # así empieza: más de 6 cm libres

    plan, info, _ = crc_.planificar_medida(cadena, q0, T_MUNDO_CAM, T_BRIDA_TABLERO_20, K_CAM, ANCHO, ALTO)

    assert info["elegidas"] >= 15
    hueco_min = min(_hueco_mano_brazo(cadena, piezas, q) for q in _recorrido(q0, plan))
    assert hueco_min >= 0.045, f"el plan deja la mano a {hueco_min * 100:.1f} cm del brazo (empezó a más de 6)"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 2. Las posturas del plan son alcanzables: dentro de lo que fr5_servo deja pasar
# ══════════════════════════════════════════════════════════════════════════════════════════════
# Con la config de las pruebas, fr5_servo recorta lo que envía a (grados de la controladora):
#   topes del FR5 (±175, −265…85, ±160, −265…85, ±175, ±175) con 3° de margen, y los límites
#   blandos (J3 −10…150, J4 −105…85, J5 5…175) con 0,5° de margen.
SERVO_MIN_DEG = [-172.0, -262.0, -9.5, -104.5, 5.5, -172.0]
SERVO_MAX_DEG = [172.0, 82.0, 149.5, 82.0, 172.0, 172.0]


def test_plan_dentro_de_los_limites_que_aplica_el_servo(calib, cadena):
    """Rotura que caza: el planificador sortea posturas en los rangos del MJCF (J4 hasta 85°),
    pero fr5_servo recorta el objetivo a los límites blandos menos su margen (J4 ≤ 82°): el robot
    no llega nunca, la calibración se queda esperando y el trayecto real ya no es el comprobado."""
    crc_ = calib()
    q0 = np.radians([82.4, -0.4, 21.5, 66.3, 77.6, -26.6])        # J4 a 16° de su tope

    plan, info, _ = crc_.planificar_medida(cadena, q0, T_MUNDO_CAM, T_BRIDA_TABLERO_20, K_CAM, ANCHO, ALTO)

    assert info["elegidas"] >= 15
    Q = np.degrees(np.array([p["q"] for p in plan]))
    assert np.all(Q >= np.array(SERVO_MIN_DEG)), f"por debajo del mínimo: {Q.min(0).round(1).tolist()}"
    assert np.all(Q <= np.array(SERVO_MAX_DEG)), f"por encima del máximo: {Q.max(0).round(1).tolist()}"


def test_exploracion_dentro_de_los_limites_que_aplica_el_servo(calib, cadena):
    """Rotura que caza: lo mismo en los giros pequeños de la exploración. Con J4 a 74°, el giro
    de +10° pide 84°: cabe en el MJCF (85°) pero fr5_servo no pasa de 82°."""
    crc_ = calib()
    q0 = np.radians([82.4, -0.4, 21.5, 74.0, 77.6, -26.6])
    col = crc_.crear_seguridad(cadena, q0)

    posturas = crc_.posturas_exploracion(cadena, q0, col)

    assert len(posturas) >= 8
    Q = np.degrees(np.array(posturas))
    assert np.all(Q >= np.array(SERVO_MIN_DEG)), f"por debajo del mínimo: {Q.min(0).round(1).tolist()}"
    assert np.all(Q <= np.array(SERVO_MAX_DEG)), f"por encima del máximo: {Q.max(0).round(1).tolist()}"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 3. El tablero de verdad (tamaño A4) y la mano, contra el soporte y contra el propio robot
# ══════════════════════════════════════════════════════════════════════════════════════════════
# Tablero grande como el de las fotos: 7x5 cuadros de 37,5 mm, cogido con los dedos por un lado
# largo y saliendo ~20 cm por delante de la mano, inclinado 30° hacia el lado de la palma.
TABLERO_A4 = dict(cuadros=(7, 5), lado=0.0375, marcador=0.028, diccionario="DICT_4X4_50")
T_BRIDA_TABLERO_A4 = np.array([[0.0, -0.5, 0.8660254, -0.03],
                               [1.0, 0.0, 0.0, -0.18375],
                               [0.0, 0.8660254, 0.5, 0.28],
                               [0.0, 0.0, 0.0, 1.0]])
Q0_A4_DEG = [63.15, -17.83, 42.95, 34.98, 56.14, -18.34]      # tablero delante de la cámara, todo lejos


def _placa(T_bt, tablero, margen=0.012, paso=0.004):
    """Puntos de la placa real (dibujo + 12 mm de papel alrededor) en el marco de la brida."""
    nx, ny = tablero["cuadros"]
    xs = np.arange(-margen, nx * tablero["lado"] + margen + 1e-9, paso)
    ys = np.arange(-margen, ny * tablero["lado"] + margen + 1e-9, paso)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]) @ T_bt[:3, :3].T + T_bt[:3, 3]


def _caja(centro, semi, R=None, paso=0.004):
    """Puntos de la superficie de una caja (centro, semi-lados, giro) en el mundo."""
    ejes = [np.arange(-s, s + 1e-9, paso) if s > paso else np.array([-s, s]) for s in semi]
    X, Y, Z = np.meshgrid(*ejes, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    P = P[(np.abs(np.abs(P) - np.asarray(semi)) < 1e-9).any(axis=1)]
    return P @ (np.eye(3) if R is None else R).T + np.asarray(centro, float)


def _giro_y(grados):
    c, s = math.cos(math.radians(grados)), math.sin(math.radians(grados))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _en_el_mundo(cadena, puntos_brida, q_deg):
    T = cadena.fk_marcos(np.radians(q_deg))[5]
    return puntos_brida @ T[:3, :3].T + T[:3, 3]


def _dist(A, B):
    return float(cKDTree(B).query(A)[0].min())


# Piezas del soporte real (medidas del plano: decoracion/soporte_hombros.xml), en el mundo
COLUMNA = _caja((0.26011, 0.0, -0.18189), (0.06, 0.075, 0.595))
PLACA_HOMBRO_IZQ = _caja((0.40214, 0.0, 0.27309), (0.11, 0.11, 0.006), _giro_y(135.0))
PLACA_DE_ARRIBA = _caja((0.26011, 0.0, 0.41911), (0.11, 0.11, 0.006))       # debajo de la cabeza
PLETINA_DEL_PIE = _caja((0.26011, 0.0, -0.79589), (0.10, 0.40, 0.004))
MUNECA_INICIO_A4 = Q0_A4_DEG[3:]              # J4, J5, J6 de la postura de inicio


@pytest.fixture
def seguridad_a4(calib, cadena):
    crc_ = calib(CALIB_ROBOT_TABLERO=dict(TABLERO_A4))
    return crc_.crear_seguridad(cadena, np.radians(Q0_A4_DEG), T_BRIDA_TABLERO_A4, T_MUNDO_CAM)


def _pares_que_incumplen(seguridad, q_deg):
    return {(objeto, cosa) for objeto, cosa, _dist_m, _exigido_m in seguridad.mano.incumple(np.radians(q_deg))}


def test_inicio_del_escenario_a4_es_libre(seguridad_a4, cadena):
    """La postura de inicio del escenario (todo a más de medio metro del soporte) vale."""
    placa = _en_el_mundo(cadena, _placa(T_BRIDA_TABLERO_A4, TABLERO_A4), Q0_A4_DEG)
    assert _dist(placa, PLACA_HOMBRO_IZQ) > 0.5 and _dist(placa, COLUMNA) > 0.5
    assert seguridad_a4.libre(np.radians(Q0_A4_DEG))
    assert _pares_que_incumplen(seguridad_a4, Q0_A4_DEG) == set()


def test_tablero_contra_la_columna_no_es_libre(seguridad_a4, cadena):
    """Rotura que caza: el tablero solo se comprueba contra el antebrazo y el brazo; el soporte
    se mira con una cápsula de 13 cm alrededor de la mano, y un tablero A4 sobresale el doble.
    (La muñeca está igual que al empezar: solo se han movido J1-J3.)"""
    q = [-58.8, -120.9, 99.7] + MUNECA_INICIO_A4
    placa = _en_el_mundo(cadena, _placa(T_BRIDA_TABLERO_A4, TABLERO_A4), q)
    assert _dist(placa, COLUMNA) < 0.005                          # el tablero toca la columna
    assert not seguridad_a4.libre(np.radians(q))
    assert ("tablero", "soporte") in _pares_que_incumplen(seguridad_a4, q)


def test_mano_contra_el_pie_del_soporte_no_es_libre(seguridad_a4, cadena, piezas):
    """Rotura que caza: del bastidor del soporte el modelo de cápsulas solo conoce un tubo.
    (La muñeca está igual que al empezar.)"""
    q = [-38.2, -64.5, 32.4] + MUNECA_INICIO_A4
    assert _dist(_en_el_mundo(cadena, piezas["mano"], q), PLETINA_DEL_PIE) < 0.005
    assert not seguridad_a4.libre(np.radians(q))
    assert ("mano", "soporte") in _pares_que_incumplen(seguridad_a4, q)


def test_tablero_contra_la_placa_del_hombro_izquierdo(seguridad_a4, cadena):
    """Rotura que caza: la esquina del tablero toca la placa del otro hombro del soporte."""
    q = [47.9, -14.9, 65.0, 64.7, 122.6, -49.3]
    placa = _en_el_mundo(cadena, _placa(T_BRIDA_TABLERO_A4, TABLERO_A4), q)
    assert _dist(placa, PLACA_HOMBRO_IZQ) < 0.005
    assert ("tablero", "soporte") in _pares_que_incumplen(seguridad_a4, q)


def test_tablero_contra_la_cabeza(seguridad_a4, cadena):
    """Rotura que caza: la esquina del tablero llega a 2 cm de la placa sobre la que va la cabeza
    con la cámara y nadie lo mira (la cápsula de la mano queda lejos)."""
    q = [87.1, 5.5, 51.4, 77.3, 107.1, -56.3]
    placa = _en_el_mundo(cadena, _placa(T_BRIDA_TABLERO_A4, TABLERO_A4), q)
    assert _dist(placa, PLACA_DE_ARRIBA) < 0.025
    assert _pares_que_incumplen(seguridad_a4, q) & {("tablero", "soporte"), ("tablero", "cabezal")}


def test_tablero_contra_el_hombro_del_robot(seguridad_a4, cadena):
    """Rotura que caza: el tablero no se comprueba contra el hombro del propio FR5."""
    q = [-52.3, -195.9, 116.4, 68.9, 124.1, -42.6]
    meshes = os.path.join(os.path.dirname(RUTA_MJCF), "..", "meshes")
    hombro = np.vstack([CM.leer_stl(os.path.join(meshes, f"vis_shoulder_{i}.stl")) for i in range(5)]).reshape(-1, 3)
    m = cadena.fk_marcos(np.radians(q))
    T = np.linalg.inv(m[0])                                        # al marco de shoulder_Link
    placa = _en_el_mundo(cadena, _placa(T_BRIDA_TABLERO_A4, TABLERO_A4), q)
    assert _dist(placa @ T[:3, :3].T + T[:3, 3], hombro) < 0.005
    assert ("tablero", "shoulder_Link") in _pares_que_incumplen(seguridad_a4, q)


def test_un_tablero_que_toca_el_brazo_al_empezar_no_apaga_su_proteccion(cadena, piezas):
    """Rotura que caza: si al empezar el tablero ya estaba pegado al brazo (o la calibración
    aproximada lo sitúa mal), fijar_inicio rebajaba el hueco mínimo del tablero hasta 0 y el
    tablero dejaba de comprobarse: cualquier postura valía."""
    q0 = np.radians(Q0_DEG)
    m = cadena.fk_marcos(q0)
    # placa cruzada en mitad del antebrazo (que va de x = 0 a x = −0,395 en su marco)
    T_antebrazo_placa = np.array([[1.0, 0.0, 0.0, -0.27], [0.0, 1.0, 0.0, -0.05], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    T_brida_tablero = np.linalg.inv(m[5]) @ m[2] @ T_antebrazo_placa
    placa_en_antebrazo = _placa(T_antebrazo_placa, TABLERO_20)
    assert float(piezas["arboles"]["forearm_Link"].query(placa_en_antebrazo)[0].min()) < 0.005

    ch = CM.desde_config(cadena, _conf(), T_brida_tablero).fijar_inicio(q0)

    assert not ch.ok(q0)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 4. La exploración comprueba que el tablero mide lo que dice config antes de planificar nada
# ══════════════════════════════════════════════════════════════════════════════════════════════
# Giros pequeños desde la postura de inicio, como los de la exploración: (joint, grados)
GIROS_PEQUENOS = [(5, 15), (5, -15), (4, 12), (4, -12), (3, 10), (3, -10), (0, 6), (0, -6), (1, 5), (2, -6)]


def _muestras_exploracion(cadena, T_brida_tablero, escala_medida=1.0):
    """(poses de la brida, poses del tablero que da la cámara) en q0 y tras cada giro pequeño.
    escala_medida = lado de config / lado real: si el tablero real es mayor que el de config, el
    detector lo ve igual de orientado pero 'más cerca' (toda la traslación sale multiplicada)."""
    q0 = np.radians(Q0_DEG)
    Tb, Tc = [], []
    for j, grados in [(0, 0.0)] + GIROS_PEQUENOS:
        q = q0.copy()
        q[j] += math.radians(grados)
        T_mundo_brida = cadena.fk_marcos(q)[5]
        T_cam_tablero = np.linalg.inv(T_MUNDO_CAM) @ T_mundo_brida @ T_brida_tablero
        T_cam_tablero[:3, 3] *= escala_medida
        Tb.append(T_mundo_brida)
        Tc.append(T_cam_tablero)
    return Tb, Tc


def test_escala_de_un_tablero_bien_medido_es_uno(cadena):
    Tb, Tc = _muestras_exploracion(cadena, T_BRIDA_TABLERO_20)
    assert C.estimar_escala(Tb, Tc) == pytest.approx(1.0, abs=0.002)


def test_escala_delata_un_tablero_mas_grande_que_el_de_config(cadena):
    """Rotura que caza: config dice cuadros de 20 mm y el tablero impreso los tiene de 37,5 mm.
    La cámara lo coloca a casi la mitad de distancia y nadie lo nota hasta que algo choca."""
    Tb, Tc = _muestras_exploracion(cadena, T_BRIDA_TABLERO_A4, escala_medida=20.0 / 37.5)
    assert C.estimar_escala(Tb, Tc) == pytest.approx(37.5 / 20.0, abs=0.01)


def test_revision_de_la_exploracion_rechaza_un_tablero_mal_medido(cadena):
    Tb, Tc = _muestras_exploracion(cadena, T_BRIDA_TABLERO_A4, escala_medida=20.0 / 37.5)
    P = C.puntos_tablero((7, 5), 0.020)

    rev = C.revisar_exploracion(Tb, Tc, P, lado=0.020)

    assert not rev["ok"]
    assert "escala" in [codigo for codigo, _texto in rev["problemas"]]
    assert rev["lado_estimado"] == pytest.approx(0.0375, abs=0.0003)


def test_revision_de_la_exploracion_acepta_un_tablero_bien_medido(cadena):
    Tb, Tc = _muestras_exploracion(cadena, T_BRIDA_TABLERO_20)
    P = C.puntos_tablero((7, 5), 0.020)

    rev = C.revisar_exploracion(Tb, Tc, P, lado=0.020)

    assert rev["ok"] and rev["problemas"] == []
    assert np.linalg.norm(rev["aprox"]["T_mundo_cam"][:3, 3] - T_MUNDO_CAM[:3, 3]) < 0.002
    assert np.linalg.norm(rev["aprox"]["T_brida_tablero"][:3, 3] - T_BRIDA_TABLERO_20[:3, 3]) < 0.002


def _detectar(T_cam_tablero, tablero, rng, ruido_px=0.3, lado_real=None):
    """Lo que devolvería la cámara: esquinas del tablero REAL proyectadas con ruido de detección y
    solvePnP con las medidas de 'tablero' (las de config). None si alguna esquina sale de la imagen."""
    import cv2
    P_cfg = C.puntos_tablero(tablero["cuadros"], tablero["lado"])
    P_real = P_cfg if lado_real is None else C.puntos_tablero(tablero["cuadros"], lado_real)
    Pc = P_real @ T_cam_tablero[:3, :3].T + T_cam_tablero[:3, 3]
    if np.any(Pc[:, 2] < 0.15):
        return None
    uv = Pc[:, :2] / Pc[:, 2:3] * [K_CAM[0, 0], K_CAM[1, 1]] + [K_CAM[0, 2], K_CAM[1, 2]]
    if np.any(uv < 5) or np.any(uv[:, 0] > ANCHO - 5) or np.any(uv[:, 1] > ALTO - 5):
        return None
    uv = uv + rng.normal(0.0, ruido_px, uv.shape)
    ok, rvec, tvec = cv2.solvePnP(P_cfg, uv, K_CAM, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def test_revision_de_la_exploracion_no_da_falsas_alarmas_con_el_ruido_de_la_camara(cadena):
    """Rotura que caza: una tolerancia de escala tan justa (o un estimador tan nervioso) que un
    tablero bien medido se rechaza por el ruido normal de detección (0,3 px)."""
    q0 = np.radians(Q0_DEG)
    P = C.puntos_tablero((7, 5), 0.020)
    for semilla in range(15):
        rng = np.random.default_rng(semilla)
        Tb, Tc = [], []
        for j, grados in [(0, 0.0)] + GIROS_PEQUENOS:
            q = q0.copy()
            q[j] += math.radians(grados)
            T = _detectar(np.linalg.inv(T_MUNDO_CAM) @ cadena.fk_marcos(q)[5] @ T_BRIDA_TABLERO_20, TABLERO_20, rng)
            if T is not None:
                Tb.append(cadena.fk_marcos(q)[5])
                Tc.append(T)
        assert len(Tb) >= 9
        rev = C.revisar_exploracion(Tb, Tc, P, lado=0.020)
        assert rev["ok"], f"semilla {semilla}: {rev['problemas']}"
        assert abs(rev["escala"] - 1.0) < 0.02


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 5. El robot en marcha: sesión con un FR5 y una cámara simulados (sin ventana ni hardware)
# ══════════════════════════════════════════════════════════════════════════════════════════════
from skill_shield.hardware.fr5_servo import paso_limitado          # noqa: E402

PASO_S = 0.04          # tiempo simulado por vuelta del bucle (25 Hz)


class RelojSimulado:
    """Sustituye al módulo time dentro de calibrar_camara_robot: el tiempo lo mueve la simulación."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def time(self):
        return 1.7e9 + self.t


class FR5Simulado:
    """Lo que la calibración usa de FR5Servo, sobre un robot simulado que se mueve como lo mueve
    fr5_servo: cada joint hacia el último objetivo publicado con velocidad y aceleración limitadas,
    recortando el objetivo a 'limites' (rad), y solo mientras está habilitado y sin error."""

    def __init__(self, q0, limites=None, v_max_deg=10.0, colision_en=None):
        self.q = np.degrees(np.asarray(q0, float))
        self.v = np.zeros(6)
        self.limites = None if limites is None else np.degrees(np.asarray(limites, float))
        self.v_max = float(v_max_deg)
        self.objetivo = None
        self.habilitado = False
        self.error = False
        self.colision_en = colision_en            # función q_rad → True si la controladora da colisión
        self.publicados, self.recorrido, self.n_resets = [], [np.radians(self.q)], 0
        self.en_linea = True

    @property
    def en_error(self):
        return self.error

    def publicar(self, q_rad):
        self.objetivo = np.degrees(np.asarray(q_rad, float)[:6])
        self.publicados.append(np.asarray(q_rad, float)[:6].copy())

    def habilitar(self, si=True):
        self.habilitado = bool(si)
        if not si:
            self.v[:] = 0.0

    def resetear(self):
        self.habilitado, self.error, self.n_resets = False, False, self.n_resets + 1
        self.v[:] = 0.0

    def q_real_rad(self, max_edad_s=0.3):
        return np.radians(self.q)

    def estado(self):
        return dict(texto="simulado")

    def parar(self):
        self.habilitado = False

    def avanzar(self, dt):
        if self.habilitado and not self.error and self.objetivo is not None:
            obj = self.objetivo if self.limites is None else np.clip(self.objetivo, self.limites[:, 0], self.limites[:, 1])
            self.q, self.v = paso_limitado(self.q, self.v, obj, self.v_max, 300.0, dt)
            if self.colision_en is not None and self.colision_en(np.radians(self.q)):
                self.error, self.habilitado = True, False
                self.v[:] = 0.0
        self.recorrido.append(np.radians(self.q))


class PedalSimulado:
    """Operario que mantiene el pedal pisado; si el programa lo desarma (colisión…), lo suelta y
    vuelve a pisarlo un segundo después."""
    hombre_muerto = True

    def __init__(self, reloj):
        self._reloj, self._t_desarme = reloj, -1e9

    @property
    def armado(self):
        return self._reloj.monotonic() - self._t_desarme > 1.0

    def tecla(self, key):
        return None

    def actualizar(self):
        return None

    def desarmar(self):
        self._t_desarme = self._reloj.monotonic()

    def cerrar(self):
        pass


class SesionSimulada(crc.Sesion):
    """La Sesion de verdad (ir_a, recorrer, capturar, pedal, colisiones…) con el robot, la cámara y
    la ventana sustituidos por una simulación: la cámara ve el tablero donde lo pone el robot."""

    def __init__(self, cadena, reloj, fr5, T_brida_tablero, tablero, lado_real=None, semilla=0, tope_s=900.0,
                 mover=True, teclas=()):
        self._estado_inicial()
        self.mover = mover
        self._teclas = list(teclas)               # teclas que «pulsa» el operario, una por vuelta
        self.al_mirar = None                      # función(sesión) que se llama en cada vuelta (sucesos de la prueba)
        self.cadena, self.fr5, self._reloj = cadena, fr5, reloj
        self.robot = crc.Robot(fr5)
        self.pedal = PedalSimulado(reloj)
        self.K, self.dist = K_CAM, np.zeros(5)
        self.info = dict(mxid="SIMULADA", K=K_CAM, dist=np.zeros(5), ancho=ANCHO, alto=ALTO)
        self.P = C.puntos_tablero(tablero["cuadros"], tablero["lado"])
        self._tablero, self._T_bt, self._lado_real = tablero, T_brida_tablero, lado_real
        self._rng = np.random.default_rng(semilla)
        self._t_tope = reloj.monotonic() + tope_s

    def _mirar(self, hud=()):
        self._reloj.t += PASO_S
        if self._reloj.monotonic() > self._t_tope:
            raise TimeoutError("la sesión simulada lleva demasiado tiempo: el programa se ha quedado esperando")
        if self.al_mirar is not None:
            self.al_mirar(self)
        self.fr5.avanzar(PASO_S)
        T_ct = np.linalg.inv(T_MUNDO_CAM) @ self.cadena.fk_marcos(np.radians(self.fr5.q))[5] @ self._T_bt
        # el detector lo encuentra si lo ve por delante y a menos de 60° de canto (respecto al rayo de visión)
        centro = (self.P @ T_ct[:3, :3].T + T_ct[:3, 3]).mean(0)
        de_frente = float(T_ct[:3, 2] @ centro) / float(np.linalg.norm(centro)) > math.cos(math.radians(60.0))
        self.T_ct = _detectar(T_ct, self._tablero, self._rng, lado_real=self._lado_real) if de_frente else None
        self.txt_tab = "simulado" if self.T_ct is not None else "no se ve"
        return True

    def _tecla(self, hay_imagen):
        return self._teclas.pop(0) if self._teclas else -1

    def cerrar(self):
        self.fr5.habilitar(False)


@pytest.fixture
def reloj(monkeypatch):
    r = RelojSimulado()
    monkeypatch.setattr(crc, "time", r)
    return r


def test_ir_a_no_arranca_si_el_trayecto_desde_donde_esta_el_robot_cruza_el_soporte(calib, cadena, reloj):
    """Rotura que caza: antes de moverse, ir_a solo comprobaba la mano contra el brazo. Si el robot
    no está donde el plan creía (postura saltada, colisión a medias…), el trayecto real podía
    llevar el brazo a través de la columna del soporte sin que nadie lo mirase."""
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    q_otro_lado = np.radians([-26.1, -27.7, 56.4, 43.0, 105.0, 25.0])       # girando J1 119°: cruza la columna
    fr5 = FR5Simulado(q0)
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20)
    s.seguridad = crc_.crear_seguridad(cadena, q0, T_BRIDA_TABLERO_20, T_MUNDO_CAM)
    assert s.seguridad.libre(q_otro_lado)                          # el destino en sí es bueno

    llegado = s.ir_a(q_otro_lado)

    assert llegado is False
    assert fr5.publicados == []                                    # ni siquiera se le pide moverse
    assert np.degrees(np.abs(np.array(fr5.recorrido) - q0).max()) < 0.01


def test_ir_a_abandona_si_el_robot_se_para_antes_de_llegar(calib, cadena, reloj):
    """Rotura que caza: si la controladora recorta el objetivo (un límite blando más estrecho de
    lo que cree el programa), el robot se para antes de llegar y ir_a se quedaba esperando para
    siempre con el pedal pisado."""
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    q_obj = q0 + np.radians([0.0, 0.0, 0.0, 0.0, 0.0, 25.0])       # J6 de 25° a 50°
    limites_reales = np.radians([[-175.0, 175.0], [-265.0, 85.0], [-160.0, 160.0], [-265.0, 85.0],
                                 [-175.0, 175.0], [-175.0, 35.0]])  # …pero este robot no pasa de 35° en J6
    fr5 = FR5Simulado(q0, limites=limites_reales)
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20, tope_s=120.0)
    s.seguridad = crc_.crear_seguridad(cadena, q0, T_BRIDA_TABLERO_20, T_MUNDO_CAM)
    t0 = reloj.monotonic()

    llegado = s.ir_a(q_obj)

    assert llegado is False
    assert reloj.monotonic() - t0 < 30.0                           # no espera más de medio minuto
    assert math.degrees(fr5.q_real_rad()[5]) == pytest.approx(35.0, abs=0.1)


class FR5QueSeDesvia(FR5Simulado):
    """Robot que no obedece en J5: mientras está habilitado lo va doblando hacia el antebrazo a
    8 °/s, pida lo que pida el programa."""

    def avanzar(self, dt):
        j5 = self.q[4]
        super().avanzar(dt)
        if self.habilitado and not self.error:
            self.q[4] = j5 + 8.0 * dt
            self.recorrido[-1] = np.radians(self.q)


def test_la_vigilancia_para_el_robot_si_se_sale_de_lo_comprobado(calib, cadena, piezas, reloj):
    """Rotura que caza: una vez comprobado el trayecto, nadie mira dónde está el robot DE VERDAD
    mientras se mueve. Si no sigue lo pedido (aquí J5 se dobla solo hacia el antebrazo), nada lo
    para hasta que la mano aplasta lo que tenga detrás."""
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    fr5 = FR5QueSeDesvia(q0)
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20, tope_s=60.0)
    s.seguridad = crc_.crear_seguridad(cadena, q0, T_BRIDA_TABLERO_20, T_MUNDO_CAM)
    assert _hueco_mano_brazo(cadena, piezas, q0) > 0.060           # empieza con más de 6 cm libres

    with pytest.raises(crc_.Detenido):
        s.ir_a(q0 + np.radians([0.0, 0.0, 0.0, 0.0, 0.0, 25.0]))  # se le pide girar solo J6

    assert fr5.habilitado is False                                 # el robot queda parado…
    hueco_min = min(_hueco_mano_brazo(cadena, piezas, q) for q in fr5.recorrido[::5] + [fr5.recorrido[-1]])
    assert hueco_min > 0.033, f"…pero la mano llegó a {hueco_min * 100:.1f} cm del brazo"


def test_ir_a_vuelve_a_comprobar_el_trayecto_si_el_robot_aparece_en_otro_sitio(calib, cadena, reloj):
    """Rotura que caza: el trayecto se comprobaba una vez, al salir. Si a medio camino se suelta el
    pedal, alguien lleva el robot a mano a otro sitio (arrastre) y se vuelve a pisar, el robot
    seguía hacia el objetivo por un trayecto que nadie había mirado: aquí, a través de la columna."""
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    q_otro_lado = np.radians([-26.1, -27.7, 56.4, 43.0, 105.0, 25.0])       # desde aquí, ir a q0 cruza la columna
    fr5 = FR5Simulado(q0 + np.radians([0.0, 0.0, 0.0, 0.0, 0.0, 30.0]))     # sale de J6 +30°: trayecto bueno
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20, tope_s=120.0)
    s.seguridad = crc_.crear_seguridad(cadena, q0, T_BRIDA_TABLERO_20, T_MUNDO_CAM)
    assert s.seguridad.libre(q_otro_lado)
    t0 = reloj.monotonic()
    hecho = []

    def operario(sesion):
        t = reloj.monotonic() - t0
        if t > 1.0 and not hecho:                                  # suelta el pedal a medio camino…
            sesion.pedal.desarmar()
            hecho.append("suelta")
        elif t > 1.5 and hecho == ["suelta"]:                      # …y se lleva el robot a mano a otro sitio
            fr5.q = np.degrees(q_otro_lado)
            hecho.append("arrastra")                               # (el pedal vuelve a pisarse solo 1 s después)
    s.al_mirar = operario

    llegado = s.ir_a(q0)

    assert hecho == ["suelta", "arrastra"]
    assert llegado is False and "por el camino" in s.motivo_salto
    movido = float(np.degrees(np.abs(fr5.q_real_rad() - q_otro_lado).max()))
    assert movido < 0.5, f"desde donde lo dejaron, el robot se ha movido {movido:.1f}° sin comprobar el trayecto"


class FR5QuePierdeLaLectura(FR5Simulado):
    """Robot del que, durante un rato, no llega la postura real (corte en la comunicación)."""

    def __init__(self, q0, reloj, sin_lectura_entre):
        super().__init__(q0)
        self._reloj, self._ciego = reloj, sin_lectura_entre

    def q_real_rad(self, max_edad_s=0.3):
        t = self._reloj.monotonic()
        return None if self._ciego[0] <= t < self._ciego[1] else np.radians(self.q)


def test_sin_lectura_de_la_postura_el_robot_no_avanza(calib, cadena, reloj):
    """Rotura que caza: si deja de llegar la postura real, la vigilancia se queda ciega pero el
    robot seguía moviéndose hacia el objetivo. Sin lectura no se le mueve; cuando vuelve, sigue."""
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    t0 = reloj.monotonic()
    fr5 = FR5QuePierdeLaLectura(q0, reloj, (t0 + 1.0, t0 + 3.0))
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20, tope_s=120.0)
    s.seguridad = crc_.crear_seguridad(cadena, q0, T_BRIDA_TABLERO_20, T_MUNDO_CAM)
    donde = {}

    def apuntar(sesion):
        t = reloj.monotonic() - t0
        for nombre, cuando in (("al_poco_de_cortarse", 1.2), ("al_volver", 2.96)):
            if t >= cuando and nombre not in donde:
                donde[nombre] = fr5.q.copy()                       # (FR5Simulado guarda grados)
    s.al_mirar = apuntar

    llegado = s.ir_a(q0 + np.radians([0.0, 0.0, 0.0, 0.0, 0.0, 30.0]))     # 3 s de trayecto

    avance_a_ciegas = float(np.abs(donde["al_volver"] - donde["al_poco_de_cortarse"]).max())
    assert avance_a_ciegas < 0.1, f"sin lectura de la postura, el robot ha avanzado {avance_a_ciegas:.1f}°"
    assert llegado is True


def test_la_vigilancia_tambien_actua_mientras_se_captura(calib, cadena, piezas, reloj):
    """Rotura que caza: la vigilancia solo miraba durante los trayectos. Mientras se captura el
    pedal sigue pisado y el robot habilitado: si entonces se mueve solo, tampoco lo paraba nadie."""
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    fr5 = FR5QueSeDesvia(q0)
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20, tope_s=60.0)
    s.seguridad = crc_.crear_seguridad(cadena, q0, T_BRIDA_TABLERO_20, T_MUNDO_CAM)

    with pytest.raises(crc_.Detenido):
        for _ in range(6):                                         # varias capturas seguidas en la misma postura
            s.capturar(objetivo=q0)

    assert fr5.habilitado is False
    hueco_min = min(_hueco_mano_brazo(cadena, piezas, q) for q in fr5.recorrido[::5] + [fr5.recorrido[-1]])
    assert hueco_min > 0.033, f"la mano llegó a {hueco_min * 100:.1f} cm del brazo"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 6. La calibración automática entera (modo_auto) con el robot y la cámara simulados
# ══════════════════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def archivos(monkeypatch, tmp_path):
    """Los archivos que escribe la calibración, en una carpeta temporal."""
    rutas = dict(ARCHIVO=str(tmp_path / "camara_robot.json"), POSTURAS=str(tmp_path / "calib_posturas.json"),
                 MUESTRAS=str(tmp_path / "calib_muestras.json"))
    for nombre, ruta in rutas.items():
        monkeypatch.setattr(crc, nombre, ruta)
    return types.SimpleNamespace(**rutas)


def test_auto_se_detiene_del_todo_si_el_robot_no_obedece(calib, cadena, piezas, reloj, archivos):
    """Rotura que caza: si la vigilancia para el robot, el programa tiene que acabar ahí —parado,
    con un mensaje que diga por qué— y no seguir con la siguiente postura ni calcular nada."""
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    fr5 = FR5QueSeDesvia(q0)
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20, tope_s=120.0)

    with pytest.raises(SystemExit) as salida:
        crc_.modo_auto(s)

    assert "detenida" in str(salida.value).lower() and "no sigue el trayecto" in str(salida.value)
    assert fr5.habilitado is False
    hueco_min = min(_hueco_mano_brazo(cadena, piezas, q) for q in fr5.recorrido[::5] + [fr5.recorrido[-1]])
    assert hueco_min > 0.033
    assert not os.path.exists(archivos.ARCHIVO)                    # no se guarda ninguna calibración


def test_auto_se_para_tras_explorar_si_el_tablero_no_mide_lo_que_dice_config(calib, cadena, reloj, archivos):
    """Rotura que caza: config dice cuadros de 20 mm y el tablero montado los tiene de 37,5 mm. La
    calibración aproximada sale con un error que parece aceptable, pero sitúa la cámara y el
    tablero a decenas de cm de donde están; con eso se planifican y se recorren las posturas
    «seguras». Tiene que pararse al acabar de explorar y decir cuánto mide el cuadrado."""
    crc_ = calib()                                                 # CALIB_ROBOT_TABLERO: 20 mm
    q0 = np.radians(Q0_A4_DEG)
    fr5 = FR5Simulado(q0)
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_A4, TABLERO_20, lado_real=0.0375, tope_s=400.0)

    with pytest.raises(SystemExit) as salida:
        crc_.modo_auto(s)

    assert "37" in str(salida.value) and "CALIB_ROBOT_TABLERO" in str(salida.value)
    giro_max = float(np.degrees(np.abs(np.array(fr5.recorrido) - q0).max()))
    assert giro_max < 15.5, f"el robot ha girado {giro_max:.0f}°: ha pasado de la exploración"
    assert not os.path.exists(archivos.ARCHIVO)


def test_auto_no_arranca_con_la_mano_ya_pegada_al_antebrazo(calib, cadena, piezas, reloj, archivos):
    """Si el operario deja el robot con la muñeca tan doblada que la mano está a menos de 3 cm del
    antebrazo, la calibración no empieza: lo dice y no llega a mover el robot."""
    crc_ = calib()
    q_doblada = _doblar_j5_hasta_cerrar(cadena, piezas, np.radians(Q0_DEG), 0.045)
    assert _hueco_mano_brazo(cadena, piezas, q_doblada, "forearm_Link") < 0.027
    # un tablero colocado de modo que, en esa postura, la cámara lo vea de frente a medio metro
    T_cam_tablero = np.array([[1.0, 0, 0, -0.07], [0, 1.0, 0, -0.05], [0, 0, 1.0, 0.50], [0, 0, 0, 1.0]])
    T_brida_tablero = np.linalg.inv(cadena.fk_marcos(q_doblada)[5]) @ T_MUNDO_CAM @ T_cam_tablero
    fr5 = FR5Simulado(q_doblada)
    s = SesionSimulada(cadena, reloj, fr5, T_brida_tablero, TABLERO_20, tope_s=60.0)

    with pytest.raises(SystemExit) as salida:
        crc_.modo_auto(s)

    assert "postura de inicio" in str(salida.value) and "antebrazo" in str(salida.value)
    assert fr5.publicados == [] and len(fr5.recorrido) > 5
    assert np.degrees(np.abs(np.array(fr5.recorrido) - q_doblada).max()) < 1e-6


def _hueco_tablero_entorno(cadena, T_brida_tablero, tablero, recorrido):
    """Distancia mínima (m) de la placa del tablero a las piezas del soporte escritas arriba y a la
    cabeza con la cámara (un cilindro tumbado de 7,5 cm de radio, eje Y, de y = −0,09 a y = 0,06
    a la altura de la cámara), a lo largo de un recorrido."""
    placa = _placa(T_brida_tablero, tablero, paso=0.01)
    piezas_soporte = cKDTree(np.vstack([COLUMNA, PLACA_HOMBRO_IZQ, PLACA_DE_ARRIBA, PLETINA_DEL_PIE]))
    d = math.inf
    for q in recorrido:
        T = cadena.fk_marcos(q)[5]
        P = placa @ T[:3, :3].T + T[:3, 3]
        radial = np.maximum(np.hypot(P[:, 0] - 0.26, P[:, 2] - 0.525) - 0.075, 0.0)
        axial = np.maximum(np.maximum(-0.09 - P[:, 1], P[:, 1] - 0.06), 0.0)
        d = min(d, float(piezas_soporte.query(P)[0].min()), float(np.hypot(radial, axial).min()))
    return d


def test_auto_completa_la_calibracion_sin_cerrar_ningun_hueco(calib, cadena, piezas, reloj, archivos, capsys):
    """La calibración automática entera, con un tablero que mide lo que dice config: explora,
    planifica, mide las 20 posturas sin saltarse ninguna y acierta dónde está la cámara; y en todo
    el recorrido REAL del robot la mano no se acerca al brazo ni el tablero al soporte."""
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    fr5 = FR5Simulado(q0, limites=np.radians(np.column_stack([SERVO_MIN_DEG, SERVO_MAX_DEG])))
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20, tope_s=1500.0)

    ok = crc_.modo_auto(s)

    salida = capsys.readouterr().out
    assert ok is True, salida[-1500:]
    assert "la salto" not in salida, [l for l in salida.splitlines() if "la salto" in l]
    res = C.cargar(archivos.ARCHIVO)
    assert res["n_muestras"] >= 29                                 # inicio + 10 de explorar + ≥ 18 de medir
    assert np.linalg.norm(res["T_mundo_cam"][:3, 3] - T_MUNDO_CAM[:3, 3]) < 0.003
    assert C.angulo_R(res["T_mundo_cam"][:3, :3].T @ T_MUNDO_CAM[:3, :3]) < 0.3
    recorrido = fr5.recorrido[::10] + [fr5.recorrido[-1]]
    hueco_mano = min(_hueco_mano_brazo(cadena, piezas, q) for q in recorrido)
    assert hueco_mano >= 0.045, f"la mano llegó a {hueco_mano * 100:.1f} cm del brazo (empezó a más de 6)"
    hueco_tablero = _hueco_tablero_entorno(cadena, T_BRIDA_TABLERO_20, TABLERO_20, recorrido)
    assert hueco_tablero >= 0.04, f"el tablero llegó a {hueco_tablero * 100:.1f} cm del soporte o de la cámara"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 7. Lo planificado se comprueba al milímetro; lo que hace el robot de verdad, con unos mm de manga
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _doblar_j5_hasta_cerrar(cadena, piezas, q0, cierre_m, eslabon="forearm_Link"):
    """Postura con J5 doblado desde q0 (lo demás igual) hasta que el hueco entre la mano y el
    antebrazo, medido por fuerza bruta, es 'cierre_m' metros menor que en q0."""
    d0 = _hueco_mano_brazo(cadena, piezas, q0, eslabon)
    poco, mucho = 0.0, math.radians(40.0)
    for _ in range(30):
        medio = 0.5 * (poco + mucho)
        q = np.array(q0, float)
        q[4] += medio
        poco, mucho = (medio, mucho) if d0 - _hueco_mano_brazo(cadena, piezas, q, eslabon) < cierre_m else (poco, medio)
    q = np.array(q0, float)
    q[4] += 0.5 * (poco + mucho)
    return q


def test_ir_a_deja_volver_desde_una_postura_que_se_ha_pasado_tres_milimetros(calib, cadena, piezas, reloj):
    """Rotura que caza: el robot real nunca queda EXACTAMENTE donde se le pide. Si se para unos mm
    más allá de lo permitido (aquí 3 mm: el hueco se ha cerrado 13 mm y se permiten 10) y la
    comprobación del trayecto no tiene ninguna manga, ya no le deja moverse ni para volver: se
    saltarían todas las posturas que quedan."""
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    q_pasada = _doblar_j5_hasta_cerrar(cadena, piezas, q0, 0.013)
    fr5 = FR5Simulado(q_pasada)
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20, tope_s=120.0)
    s.seguridad = crc_.crear_seguridad(cadena, q0, T_BRIDA_TABLERO_20, T_MUNDO_CAM)
    assert not s.seguridad.libre(q_pasada)                         # está fuera de lo que se planifica

    llegado = s.ir_a(q0)

    assert llegado is True, s.motivo_salto
    assert np.degrees(np.abs(fr5.q_real_rad() - q0).max()) < 0.3


def test_ir_a_no_deja_seguir_cerrando_desde_una_postura_que_ya_se_ha_pasado(calib, cadena, piezas, reloj):
    """La manga es para volver, no para seguir: desde la misma postura pasada 3 mm, doblar la
    muñeca 5° más (otros ~8 mm de hueco) no se acepta y el robot no se mueve."""
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    q_pasada = _doblar_j5_hasta_cerrar(cadena, piezas, q0, 0.013)
    fr5 = FR5Simulado(q_pasada)
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20, tope_s=120.0)
    s.seguridad = crc_.crear_seguridad(cadena, q0, T_BRIDA_TABLERO_20, T_MUNDO_CAM)

    llegado = s.ir_a(q_pasada + np.radians([0.0, 0.0, 0.0, 0.0, 5.0, 0.0]))

    assert llegado is False and "antebrazo" in s.motivo_salto
    assert fr5.publicados == []


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 8. --repetir y --verificar mueven el robot con las mismas protecciones que --auto
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_repetir_tambien_vigila_al_robot(calib, cadena, piezas, reloj, archivos):
    """Rotura que caza: --repetir (y --verificar) recorrían las posturas guardadas sin el
    comprobador completo y sin vigilar la postura real: un robot que no obedece no lo paraba nadie."""
    import json
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    guardadas = [dict(q=q0.tolist(), capturar=True),
                 dict(q=(q0 + np.radians([0.0, 0.0, 0.0, 0.0, 0.0, 15.0])).tolist(), capturar=True),
                 dict(q=(q0 + np.radians([0.0, 0.0, 0.0, 8.0, 0.0, 0.0])).tolist(), capturar=True)]
    with open(archivos.POSTURAS, "w", encoding="utf-8") as f:
        json.dump(guardadas, f)
    fr5 = FR5QueSeDesvia(q0)
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20, tope_s=60.0)

    with pytest.raises(SystemExit) as salida:
        crc_.modo_repetir_o_verificar(False, s=s)

    assert "detenida" in str(salida.value).lower()
    assert fr5.habilitado is False
    hueco_min = min(_hueco_mano_brazo(cadena, piezas, q) for q in fr5.recorrido[::5] + [fr5.recorrido[-1]])
    assert hueco_min > 0.033, f"la mano llegó a {hueco_min * 100:.1f} cm del brazo"


def test_verificar_recorre_las_posturas_guardadas_y_mide_el_error(calib, cadena, reloj, archivos, capsys):
    """--verificar con una calibración guardada que es la verdadera: va a las posturas guardadas
    (saliendo de otra postura cercana), mide en ellas y da un error de décimas de mm."""
    import json
    crc_ = calib()
    q0 = np.radians(Q0_DEG)
    giros = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 15], [0, 0, 0, 8, 0, 0], [4, 0, 0, 0, 0, -10]]
    guardadas = [dict(q=(q0 + np.radians(g)).tolist(), capturar=True) for g in giros]
    with open(archivos.POSTURAS, "w", encoding="utf-8") as f:
        json.dump(guardadas, f)
    C.guardar(archivos.ARCHIVO, dict(T_mundo_cam=T_MUNDO_CAM, T_brida_tablero=T_BRIDA_TABLERO_20, metodo="a mano",
                                     error_medio_mm=0.0, error_max_mm=0.0),
              mxid="SIMULADA", K=K_CAM, dist=np.zeros(5), ancho=ANCHO, alto=ALTO, tablero=TABLERO_20,
              n_muestras=0, usadas=[])
    fr5 = FR5Simulado(q0 + np.radians([-3.0, 0.0, 2.0, 0.0, 0.0, 5.0]))
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20, tope_s=300.0)

    ok = crc_.modo_repetir_o_verificar(True, s=s)

    salida = capsys.readouterr().out
    assert ok is True, salida[-800:]
    assert "la salto" not in salida
    visitadas = [np.degrees(np.abs(np.array(fr5.recorrido) - np.array(p["q"])).max(axis=1).min()) for p in guardadas]
    assert max(visitadas) < 0.3, visitadas                         # ha pasado por las cuatro
    with open(os.path.splitext(archivos.ARCHIVO)[0] + "_verificacion.json", encoding="utf-8") as f:
        assert max(json.load(f)["errores_mm"]) < 1.5


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 9. --manual (el robot lo mueve el operario): los avisos dicen contra qué se acerca la mano
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_manual_avisa_de_que_la_mano_esta_contra_el_soporte(calib, cadena, piezas, reloj, archivos):
    """Rotura que caza: el aviso del modo manual solo sabía hablar del brazo; con la mano tocando
    el pie del soporte decía que estaba «a inf cm del brazo»."""
    crc_ = calib()
    q = np.radians([-38.2, -64.5, 32.4] + MUNECA_INICIO_A4)
    assert _dist(_en_el_mundo(cadena, piezas["mano"], np.degrees(q)), PLETINA_DEL_PIE) < 0.005
    fr5 = FR5Simulado(q)
    s = SesionSimulada(cadena, reloj, fr5, T_BRIDA_TABLERO_20, TABLERO_20, mover=False,
                       teclas=[-1, -1, -1, ord("q")])

    crc_.modo_manual(s)

    assert "CUIDADO" in s.mensaje and "soporte" in s.mensaje, s.mensaje
    assert "inf" not in s.mensaje


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 10. La Sesion de verdad (su arranque, su bucle, la detección del tablero y el dibujo), con el
#     robot simulado y una imagen sintética del tablero en lugar de la OAK-D
# ══════════════════════════════════════════════════════════════════════════════════════════════

class CamaraSintetica:
    """Hace de dispositivo, de cola de imágenes y de paquete de la OAK-D: en cada vuelta da una
    imagen (1352x1014) del tablero impreso tal como lo vería la cámara de cabeza con el robot
    donde esté, y hace avanzar el tiempo y el robot simulados."""

    def __init__(self, reloj, fr5, cadena, T_brida_tablero, lado=0.020, marcador=0.015, px=60):
        import cv2
        self._reloj, self._fr5, self._cadena, self._T_bt = reloj, fr5, cadena, T_brida_tablero
        impreso = cv2.aruco.CharucoBoard((7, 5), lado, marcador, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        dibujo = impreso.generateImage((7 * px, 5 * px), marginSize=0)
        self._papel = cv2.copyMakeBorder(dibujo, px // 2, px // 2, px // 2, px // 2, cv2.BORDER_CONSTANT, value=255)
        s = px / lado                               # píxel del papel → metros en el marco del tablero
        self._A = np.array([[1 / s, 0.0, -0.5 * lado], [0.0, 1 / s, -0.5 * lado], [0.0, 0.0, 1.0]])
        self._hay = False

    def tryGet(self):
        self._hay = not self._hay                   # una imagen nueva por vuelta
        if not self._hay:
            return None
        self._reloj.t += PASO_S
        self._fr5.avanzar(PASO_S)
        return self

    def getCvFrame(self):
        import cv2
        T_ct = np.linalg.inv(T_MUNDO_CAM) @ self._cadena.fk_marcos(np.radians(self._fr5.q))[5] @ self._T_bt
        H = K_CAM @ np.column_stack([T_ct[:3, 0], T_ct[:3, 1], T_ct[:3, 3]]) @ self._A
        gris = cv2.warpPerspective(self._papel, H, (ANCHO, ALTO), flags=cv2.INTER_AREA, borderValue=110)
        return cv2.cvtColor(cv2.GaussianBlur(gris, (3, 3), 0.6), cv2.COLOR_GRAY2BGR)

    def close(self):
        pass


def test_la_sesion_real_ve_el_tablero_captura_y_lleva_el_robot(calib, cadena, reloj, monkeypatch):
    """La Sesion sin sustituir nada suyo: arranca, encuentra el tablero en la imagen con el detector
    ChArUco del repositorio (config dice marcador de 12,5 mm; el impreso lo tiene de 15), captura
    una muestra que coincide con la pose verdadera y lleva el robot a otra postura."""
    crc_ = calib(CALIB_ROBOT_TABLERO=dict(TABLERO_20, marcador=0.0125))
    q0 = np.radians(Q0_DEG)
    fr5 = FR5Simulado(q0)
    camara = CamaraSintetica(reloj, fr5, cadena, T_BRIDA_TABLERO_20)
    monkeypatch.setattr(crc, "arrancar_fr5", lambda mover: fr5)
    monkeypatch.setattr(crc, "crear_pedal", lambda: PedalSimulado(reloj))
    monkeypatch.setattr(crc, "abrir_camara", lambda: (camara, camara, dict(mxid="SINTETICA", K=K_CAM, dist=np.zeros(5),
                                                                           ancho=ANCHO, alto=ALTO)))
    for nombre in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):      # sin pantalla
        monkeypatch.setattr(crc.cv2, nombre, lambda *a, **k: None)
    monkeypatch.setattr(crc.cv2, "waitKeyEx", lambda *a, **k: -1)

    s = crc_.Sesion(mover=True)
    try:
        assert s.esperar_tablero(["prueba"], segundos=5.0)
        muestra = s.capturar()
        s.seguridad = crc_.crear_seguridad(cadena, q0, T_BRIDA_TABLERO_20, T_MUNDO_CAM)
        q1 = q0 + np.radians([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
        llegado = s.ir_a(q1)
    finally:
        s.cerrar()

    T_real = np.linalg.inv(T_MUNDO_CAM) @ cadena.fk_marcos(q0)[5] @ T_BRIDA_TABLERO_20
    T_visto = np.array(muestra["T_cam_tablero"])
    assert np.linalg.norm(T_visto[:3, 3] - T_real[:3, 3]) < 0.001
    assert C.angulo_R(T_visto[:3, :3].T @ T_real[:3, :3]) < 0.2
    assert llegado is True and np.degrees(np.abs(fr5.q_real_rad() - q1).max()) < 0.3


def test_calibracion_gira_la_mano_180_alrededor_de_la_brida():
    """(12-nov) En la calibración la mano va dada la vuelta: su modelo se gira CALIB_MANO_GIRO_DEG
    alrededor del eje de la brida (z de wrist3) para las comprobaciones de choque."""
    import calibrar_camara_robot as CR
    from skill_shield.autonomo.choque_mano import desde_config
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    cad = cargar_cadena(CR.cfg.RUTA_XML, verbose=False)
    normal = desde_config(cad, CR.cfg)
    girada = CR.girar_mano(desde_config(cad, CR.cfg), 180.0)
    assert np.allclose(girada.mano[:, :2], -normal.mano[:, :2]) and np.allclose(girada.mano[:, 2], normal.mano[:, 2])
    assert girada._objetos["mano"] is girada.mano
    q = np.radians([0, -90, 90, -90, -90, 0])
    assert np.isfinite(girada.distancia(q)[0])


def test_girar_posturas_deja_el_tablero_donde_estaba():
    """(12-nov) Mano montada girada 180° en la brida: con el J6 girado al revés, el tablero (girado
    con la mano) queda en el mismo sitio del mundo; las que no caben en el J6 se quitan."""
    import calibrar_camara_robot as CR
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    cad = cargar_cadena(CR.cfg.RUTA_XML, verbose=False)
    lim = np.radians([[-175, 175]] * 6)
    T_bt = np.eye(4)
    T_bt[:3, 3] = (0.137, 0.003, 0.316)
    T_bt[:3, :3] = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0] * [1, 1, 1]
    if np.linalg.det(T_bt[:3, :3]) < 0:
        T_bt[:3, 2] *= -1
    T_bt2 = CR.T_brida_tablero_girado(T_bt, 180)
    posturas = [dict(q=list(np.radians([87, -35, 87, 21, 78, q6])), capturar=True) for q6 in (-104, 60, 2)]
    nuevas, quitadas = CR.girar_posturas(posturas, 180, lim)
    assert quitadas == 1 and len(nuevas) == 2                    # J6 = 2° → ±178°: no cabe
    for p_ant, p_nueva in zip(posturas[:2], nuevas):
        a = cad.fk_marcos(np.array(p_ant["q"]))[5] @ T_bt
        b = cad.fk_marcos(np.array(p_nueva["q"]))[5] @ T_bt2
        assert np.allclose(a, b, atol=1e-6)
