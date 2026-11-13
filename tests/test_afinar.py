"""
Pruebas del 12-nov (coger la pelota de verdad donde está), sin robot, sin cámara y sin MuJoCo:

    python -m pytest tests/test_afinar.py -q

  · AutoCalib: con la nube del brazo desplazada unos cm (calibración desajustada), el ICP da el
    desplazamiento al revés (±5 mm), aunque haya puntos de otras cosas.
  · Correccion: se acumula, no pasa del máximo, se guarda y se olvida al recalibrar.
  · Detectores: T_mundo_cam = corrección @ la del archivo.
  · Mapa: lo que hay dentro del robot (sus poses recientes) queda libre (sin «fantasmas»).
  · Planificador: los cubos pegados al robot en su postura actual no cuentan (son él).
  · Tarea: al llegar encima de la pelota se para a afinar; si la cámara la ve desplazada unos cm,
    rehace la bajada hacia donde está; si se ha ido lejos, para.
"""
import json
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del repo (tests/..)
sys.path.insert(0, RAIZ)

import config                                                        # noqa: E402
from skill_shield.autonomo import tarea as TA                        # noqa: E402
from skill_shield.autonomo.autocalib import AutoCalib, ConCorreccion, Correccion, T_traslacion   # noqa: E402
from skill_shield.autonomo.mapa import MapaOcupacion                 # noqa: E402
from skill_shield.hardware.fr5_servo import ACERCANDO, ENVIANDO      # noqa: E402
from skill_shield.teleop_fr5.cinematica import OFFSET_PALMA_ORCA, cargar_cadena   # noqa: E402

G = {k: getattr(config, k) for k in dir(config) if k.isupper()}
Q0 = np.radians([93.0, -17.0, 53.0, 43.0, 105.0, 25.0])
CAM = np.array([0.336, -0.049, 0.549])


@pytest.fixture(scope="module")
def cadena():
    return cargar_cadena(config.RUTA_XML, verbose=False)


@pytest.fixture(scope="module")
def ac(cadena):
    return AutoCalib(cadena, config.RUTA_XML)


def _visible(M, cam):
    """Aproximación a la cara que ve la cámara: de cada vecindad, la mitad más cercana."""
    from scipy.spatial import cKDTree
    a = cKDTree(M)
    d = np.linalg.norm(M - cam, axis=1)
    return M[np.array([d[i] <= np.median(d[a.query_ball_point(p, 0.05)]) for i, p in enumerate(M)])]


@pytest.mark.parametrize("err", [(0.03, -0.02, 0.04), (-0.05, 0.0, 0.02), (0.0, 0.06, -0.03)])
def test_icp_con_el_brazo_da_el_desajuste(ac, err):
    q = np.radians([100, -40, 90, -30, -90, 0])
    M, _ = ac.modelo(q)
    rng = np.random.default_rng(1)
    P = _visible(M, CAM) + np.array(err) + rng.normal(0, 0.003, (1, 3))
    P = np.vstack([P + rng.normal(0, 0.003, P.shape), rng.uniform(-1, 1, (3000, 3))])
    r = ac.estimar(P, q)
    assert r is not None and np.allclose(r["t"], -np.array(err), atol=0.005), r
    assert r["residuo"] < 0.008


def test_icp_sin_brazo_a_la_vista_no_inventa(ac):
    q = np.radians([100, -40, 90, -30, -90, 0])
    P = np.random.default_rng(2).uniform(-1, 1, (4000, 3)) + [3.0, 0, 0]      # nada cerca del brazo
    assert ac.estimar(P, q) is None


def test_correccion_acumula_guarda_y_se_olvida(tmp_path):
    ruta = str(tmp_path / "camara_robot.json")
    c = Correccion(ruta, ganancia=0.5, max_m=0.10)
    c.nueva_calibracion(123.0)
    assert c.aplicar(dict(t=np.array([0.04, 0, 0]), residuo=0.004, n=900))
    assert c.aplicar(dict(t=np.array([0.02, 0, 0]), residuo=0.004, n=900))
    assert np.allclose(c.t, [0.03, 0, 0])
    assert not c.aplicar(dict(t=np.array([0.0, 0.05, 0]), residuo=0.03, n=900))   # medida mala
    assert not c.aplicar(dict(t=np.array([0.3, 0, 0]), residuo=0.004, n=900))    # demasiado
    c.guardar()
    d = Correccion(ruta)
    d.nueva_calibracion(123.0)
    assert np.allclose(d.t, [0.03, 0, 0])                                       # la misma calibración
    d.nueva_calibracion(456.0)
    assert np.allclose(d.t, 0)                                                  # recalibrado: a cero
    assert json.load(open(c.ruta))["calib_mtime"] == 123.0


def test_detectores_usan_la_correccion():
    class D(ConCorreccion):
        pass
    d = D()
    T = np.eye(4)
    T[:3, 3] = (0.1, 0.2, 0.3)
    d.T_mundo_cam = T
    assert np.allclose(d.T_mundo_cam, T)
    d.correccion = T_traslacion((0.01, 0, -0.02))
    assert np.allclose(d.T_mundo_cam[:3, 3], (0.11, 0.2, 0.28))


def test_mapa_deja_libre_por_donde_pasa_el_robot():
    m = MapaOcupacion(((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.6)), 0.02, z_min=0.1)
    m.L[:] = 0.0
    caps = [(np.array([0.0, 0.0, 0.2]), np.array([0.3, 0.0, 0.2]), 0.05)]
    i, _ = m.indices(np.array([[0.15, 0.0, 0.2], [0.15, 0.3, 0.2]]))
    m.L[tuple(i.T)] = 3.0                                                       # «fantasma» y otra cosa
    assert m.vaciar(caps) > 0
    assert m.L[tuple(i[0])] <= m.L_MIN and m.L[tuple(i[1])] == 3.0


def test_cubos_pegados_al_robot_no_cuentan(cadena):
    pl = TA.Planificador(cadena, G)
    marcos = cadena.fk_marcos(Q0)
    codo = marcos[2][:3, 3]
    lejos = np.array([0.6, -0.9, 0.5])
    pl.poner_mapa(np.vstack([codo + [0.0, 0.0, 0.07], lejos]), 0.02)
    pl._empezar(Q0)
    assert pl.quitados_propios == 1 and len(pl._C) == 1 and np.allclose(pl._C[0], lejos)


# ── la tarea entera con el planificador de verdad ────────────────────────────────────────────
class FR5Falso:
    def __init__(self):
        self.conectado, self.en_error = True, False
        self.q = Q0.copy()
        self._estado = ACERCANDO

    def publicar(self, q):
        self.q = np.array(q)

    def habilitar(self, si=True):
        self._estado = ENVIANDO if si else 1

    def estado(self):
        return dict(estado=self._estado)


def _plano(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


BOLA, DEJAR = np.array([0.35, -0.45, -0.15]), np.array([0.15, -0.45, -0.15])


def _lista(bola):
    return [dict(id=100, nombre="Pelota roja", tipo="pelota", estable=True, diametro_m=0.067, edad_s=0.0,
                 T_mundo=_plano(bola).tolist()),
            dict(id=22, nombre="Punto de dejada", tipo="punto_dejada", estable=True, T_mundo=_plano(DEJAR).tolist())]


def _hasta_afinar(cadena, mover):
    fr5 = FR5Falso()
    t = TA.Tarea(dict(G, AUTONOMO_TAREA_MOVER_MANO=False), ".", fr5, simulacro=False,
                 fabrica_planificador=lambda: TA.Planificador(cadena, G))
    t.orden({"tarea": dict(accion="planificar", id="a1")},
            lambda: dict(q_real=fr5.q, herramientas=_lista(BOLA), objetivo_id=100, mapa=None))
    import time
    t0 = time.time()
    while t.estado == "planificando" and time.time() - t0 < 120:
        time.sleep(0.05)
        t.paso(0.0, fr5.q, False)
    assert t.estado == "lista", t.texto
    nombres = [f.nombre for f in t.plan.fases]
    assert "Afinar con la cámara" in nombres
    i_b = nombres.index("Bajar a la pelota")
    t.orden({"tarea": dict(accion="ejecutar", id="a2")}, lambda: dict(q_real=fr5.q))
    ctx = lambda: dict(q_real=fr5.q, herramientas=_lista(BOLA + mover), mapa=None)    # noqa: E731
    for _ in range(20000):
        t.paso(1 / 60, fr5.q, True, ctx)
        if t.estado == "error" or t.ejec.i > i_b:
            break
    return t, i_b


def _palma(cadena, q):
    T = cadena.fk_marcos(q)[6]
    return T[:3, :3] @ OFFSET_PALMA_ORCA + T[:3, 3]


def test_afinar_rehace_la_bajada_hacia_donde_esta(cadena):
    mover = np.array([0.02, -0.025, 0.01])
    t, i_b = _hasta_afinar(cadena, mover)
    assert t.estado != "error", t.texto
    fb = t.plan.fases[i_b]
    pz = t.plan.pelota
    assert np.allclose(pz["c"], BOLA + mover)
    T = cadena.fk_marcos(fb.qs[-1])[6]
    agarre = T[:3, :3] @ pz["punto"] + T[:3, 3]                            # palma o centro de la pinza
    d = np.linalg.norm(agarre - (BOLA + mover))
    assert abs(d - pz["d_palma"]) < 0.004                                   # donde está la pelota
    assert t.afinado_cm == pytest.approx(np.linalg.norm(mover) * 100, abs=0.1)
    fs = t.plan.fases[pz["i_subir"]]
    assert np.allclose(fs.qs, fb.qs[::-1])


def test_afinar_para_si_la_pelota_se_ha_ido(cadena):
    t, _ = _hasta_afinar(cadena, np.array([0.15, 0.0, 0.0]))
    assert t.estado == "error" and "Vuelve a planificar" in t.texto


# ── el constructor del mapa entero: brazo visto con la calibración desajustada ───────────────
def test_constructor_corrige_la_calibracion_y_quita_el_brazo(tmp_path):
    import shutil
    import time
    import cv2
    from skill_shield.autonomo.choque_mano import CUERPOS_BRAZO, MARCO_CADENA, nubes_mjcf
    from skill_shield.autonomo.mapa import ConstructorMapa, empaquetar_pose

    class Canal:
        def leer(self):
            return empaquetar_pose(Q0, True, None), time.time()
    shutil.copy(os.path.join(RAIZ, "camara_robot.json"), tmp_path)
    cm = ConstructorMapa(dict(G), str(tmp_path), Canal(), None, None)
    cm._recargar_calibracion()
    err = np.array([0.03, -0.02, 0.025])
    T_real = cm.T.copy()
    T_real[:3, 3] -= err                                   # la cámara no está donde dice el archivo
    cad = cm.modelo.c
    m = cad.fk_marcos(Q0)
    n = nubes_mjcf(config.RUTA_XML, 0.004)
    P = np.vstack([n[k] @ m[MARCO_CADENA[k]][:3, :3].T + m[MARCO_CADENA[k]][:3, 3] for k in CUERPOS_BRAZO])
    K = np.array([[450.0, 0, 320], [0, 450.0, 200], [0, 0, 1]])
    Tc = np.linalg.inv(T_real)
    Pc = P @ Tc[:3, :3].T + Tc[:3, 3]
    Pc = Pc[Pc[:, 2] > 0.2]
    u = np.round(Pc[:, 0] / Pc[:, 2] * 450 + 320).astype(int)
    v = np.round(Pc[:, 1] / Pc[:, 2] * 450 + 200).astype(int)
    k = (u >= 0) & (u < 640) & (v >= 0) & (v < 400)
    prof = np.full((400, 640), np.inf)
    np.minimum.at(prof, (v[k], u[k]), Pc[k, 2])
    prof = cv2.morphologyEx(np.where(np.isinf(prof), 0, prof * 1000).astype(np.uint16), cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8))
    assert (prof > 0).sum() > 5000                         # el brazo se ve
    for _ in range(16):
        cm._t_ult = 0.0
        cm.paso(prof, K)
        time.sleep(0.12)
    assert np.allclose(cm.correccion.t, -err, atol=0.008), cm.correccion.t
    assert len(cm.mapa.centros_ocupados()) <= 6            # sin el brazo como obstáculo
    assert os.path.exists(cm.correccion.ruta)              # guardada para la próxima vez


def test_icp_no_se_fia_si_el_brazo_no_esta_donde_dice(ac):
    """Puntos que el ICP podría «encajar» (cerca del brazo del gemelo) pero que en la imagen de
    profundidad no están donde iría el brazo: la comprobación con la imagen los rechaza."""
    q = np.radians([100, -40, 90, -30, -90, 0])
    M, _ = ac.modelo(q)
    T = np.eye(4)                                            # cámara en el origen mirando +z (mundo)
    T[:3, 3] = (0.0, 0.0, -1.2)
    K = np.array([[450.0, 0, 320], [0, 450.0, 240], [0, 0, 1]])
    P = _visible(M, np.array([0.0, 0.0, -1.2])) + [0.03, 0.0, 0.0]
    vacia = np.zeros((480, 640), np.uint16)                  # la cámara no ve nada ahí (todo lejos)
    vacia[:] = 4000
    assert ac.estimar(P, q, prof_mm=vacia, K=K, T_mundo_cam=T) is None


def test_las_hojas_de_los_marcadores_no_son_obstaculos():
    """Caja orientada (una hoja de pie, girada 30°): lo de dentro no cuenta (ni lo que ya estaba
    marcado), lo de debajo sí."""
    m = MapaOcupacion(((-0.5, 0.5), (-1.2, 0.0), (-0.2, 0.6)), 0.02, z_min=0.1, min_puntos=1, min_grupo=1)
    a = np.radians(30)
    R = np.array([[np.cos(a), 0, np.sin(a)], [np.sin(a), 0, -np.cos(a)], [0, 1, 0]])   # x, y=arriba, z=normal
    c = np.array([0.1, -0.8, 0.3])
    i, _ = m.indices(np.array([c, c + [0, 0, -0.25]]))
    m.L[tuple(i.T)] = 3.0                               # la hoja y la caja de debajo, ya marcadas
    m.excluir = [(c, np.array([0.13, 0.18, 0.035]), R)]
    K = np.array([[450.0, 0, 320], [0, 450.0, 240], [0, 0, 1]])
    m.actualizar(np.zeros((480, 640), np.uint16), K, np.eye(4), ahora=100.0)
    assert m.L[tuple(i[0])] <= 0.0 and m.L[tuple(i[1])] == 3.0


def test_hoja_del_marcador_llega_mas_abajo_que_el_marcador():
    """La caja de la hoja: centrada en el marcador de lado a lado, y más larga hacia abajo si la
    hoja cuelga (clave «hoja» del catálogo)."""
    from skill_shield.autonomo.herramientas import DetectorHerramientas
    import time
    d = DetectorHerramientas(dict(G, AUTONOMO_HERRAMIENTAS=[dict(id=20, nombre="X", lado=0.1, hoja=(1.0, 1.0, 3.0))]), ".")
    d.T_mundo_cam = np.eye(4)
    T = np.eye(4)
    T[:3, 3] = (0.0, 0.0, 1.0)
    d.seg[20] = type("S", (), dict(T=T @ d.cat[20]["T_marcador_herramienta"], t=time.time()))()
    (c, semi, R), = d.cajas_hojas()
    assert np.allclose(R, np.eye(3))
    assert np.isclose(semi[0], 0.1 + 0.02) and np.isclose(semi[1], 0.2 + 0.02)
    assert np.allclose(c, (0.0, -0.1, 1.0))                  # 1 lado hacia arriba (+y) y 3 hacia abajo


def test_halo_de_la_pelota_fuera_del_mapa_pero_la_mesa_se_queda():
    from skill_shield.autonomo.mapa import ConstructorMapa
    prof = np.full((480, 640), 1500, np.uint16)                 # fondo a 1,5 m
    prof[200:260, 300:360] = 1000                               # pelota a 1 m (cara)
    prof[230:300, 280:380] = np.where(prof[230:300, 280:380] == 1500, 1040, 1000)   # «halo» y mesa
    halo = [(330.0, 230.0, 45.0, 0.95, 1.40, 250.0)]           # (u, v, r, z0, z1, fila de la mesa)
    p2 = ConstructorMapa.quitar_pixeles(prof, halo)
    assert p2[230, 330] == 0 and p2[215, 310] == 0              # la pelota y su borde, fuera
    assert p2[290, 330] == prof[290, 330]                       # por debajo (la mesa), se queda
    assert p2[100, 100] == 1500                                 # lo demás, igual
    assert prof[230, 330] == 1000                               # sin tocar la original


def test_debajo_de_la_pelota_siempre_hay_suelo(cadena):
    """Aunque el mapa no tenga nada (halo quitado), los dedos no pasan por debajo de la pelota."""
    pl = TA.Planificador(cadena, G)
    pl.poner_mapa(None, 0.02)
    c, r = np.array([0.35, -0.45, -0.15]), 0.035
    pl.zonas_apoyo([(c, r + pl.zona_apoyo, r)])
    arbol, z = pl._suelo
    d, i = arbol.query(c[:2])
    assert d < 0.011 and np.isclose(z[i], c[2] - r + pl.margen_suave)
