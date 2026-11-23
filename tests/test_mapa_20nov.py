"""
(20-nov) Mejoras del mapa de obstáculos, sin cámaras ni robot:
  A) modos del estéreo (800P, extendida, subpíxel) y la distancia mínima que le toca a cada uno;
  B) la profundidad de la cámara A preparada, empaquetada y fundida en la MISMA TSDF (cámara secundaria);
  C) la transformada de distancia exacta por ejes (la de la GPU) da lo mismo que scipy;
  2) la pelota: sus píxeles dicen «libre hasta su cara» sin poner superficie, y es un obstáculo para cuRobo
     solo mientras se va hacia ella.

    python -m pytest tests/test_mapa_20nov.py -q
"""
import os
import sys
import types

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from skill_shield.autonomo import camara_a as CA                      # noqa: E402
from skill_shield.autonomo.curobo_fr5 import mundo as M                # noqa: E402
from skill_shield.autonomo.mapa import (ConstructorMapa, DESCONOCIDO, LIBRE, MapaOcupacion,   # noqa: E402
                                        OCUPADO)

K = np.array([[450.0, 0, 320], [0, 450.0, 240], [0, 0, 1]])
H, W_ = 480, 640
LIM = ((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.6))


# ── escena sintética: suelo (z = 0), una caja y una pelota, vistas desde cámaras que miran hacia abajo ──

def T_camara(x, y, z):
    """Cámara en (x, y, z) mirando hacia abajo (z de la cámara = −z del mundo)."""
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = (x, y, z)
    return T


def _rayos(T, K_=K, h=H, w=W_):
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u - K_[0, 2]) / K_[0, 0], (v - K_[1, 2]) / K_[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
    return d @ T[:3, :3].T, d


def _t_caja(o, dw, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (np.asarray(lo) - o) / dw, (np.asarray(hi) - o) / dw
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((tmax >= tmin) & (tmax > 0), np.maximum(tmin, 0), np.inf)


def _t_esfera(o, dw, c, r):
    oc = o - np.asarray(c, float)
    a = np.einsum("ij,ij->i", dw, dw)
    b = 2 * dw @ oc
    cc = oc @ oc - r * r
    disc = b * b - 4 * a * cc
    t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def render(T, cajas=(), esferas=(), suelo=True):
    """(profundidad en mm uint16, máscara de los píxeles de cada esfera)."""
    dw, dc = _rayos(T)
    o = T[:3, 3]
    t = np.full(len(dw), np.inf)
    if suelo:
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = (0.0 - o[2]) / dw[:, 2]
        t = np.where(ts > 0, ts, np.inf)
    for lo, hi in cajas:
        t = np.minimum(t, _t_caja(o, dw, lo, hi))
    mascaras = []
    for c, r in esferas:
        te = _t_esfera(o, dw, c, r)
        mascaras.append((te < t).reshape(H, W_))
        t = np.minimum(t, te)
    z = np.where(np.isfinite(t), t * dc[:, 2], 0.0).reshape(H, W_)
    return (z * 1000).astype(np.uint16), mascaras


def _mapa():
    return MapaOcupacion(LIM, 0.02, z_suelo=0.0, metodo="tsdf", motor="numpy", z_min=0.2, dist_max=1.7)


def _estado_en(m, P, ahora):
    e = m.estado(ahora)
    idx, dentro = m.indices(np.asarray(P, float))
    assert dentro.all()
    return e[idx[:, 0], idx[:, 1], idx[:, 2]]


# ═════════════ A) modos del estéreo ═════════════

def test_distancia_minima_de_cada_modo():
    assert CA.distancia_minima(False, False) == pytest.approx(0.35)       # lo de siempre (400P)
    assert CA.distancia_minima(True, False) == pytest.approx(0.70)        # 800P: el doble
    assert CA.distancia_minima(True, True) == pytest.approx(0.35)         # 800P + extendida: como antes
    assert CA.distancia_minima(False, True) == pytest.approx(0.175)


def test_modos_del_estereo_de_mejor_a_como_siempre():
    pytest.importorskip("cv2")
    from skill_shield.autonomo import camara
    cfg = types.SimpleNamespace(AUTONOMO_CAM_MONO_800P=True, AUTONOMO_CAM_EXTENDIDA=True, AUTONOMO_CAM_SUBPIXEL_BITS=5,
                                AUTONOMO_CAM_A_MONO_800P=True, AUTONOMO_CAM_A_EXTENDIDA=False,
                                AUTONOMO_CAM_A_SUBPIXEL_BITS=None)
    assert camara._modos_estereo(cfg, "cabeza") == [(True, True, 5), (True, False, 5), (None, None, None)]
    assert camara._modos_estereo(cfg, "a") == [(True, False, None), (None, None, None)]
    nada = types.SimpleNamespace()
    assert camara._modos_estereo(nada, "cabeza") == [(None, None, None)]


# ═════════════ C) transformada de distancia exacta por ejes ═════════════

@pytest.mark.parametrize("forma", [(80, 100, 45), (7, 13, 5), (30, 1, 9), (1, 1, 6)])
def test_edt_por_ejes_igual_que_scipy(forma):
    from scipy.ndimage import distance_transform_edt
    rng = np.random.default_rng(len(forma) + forma[0])
    o = rng.random(forma) < 0.01
    o.flat[0] = True
    esperado = (distance_transform_edt(~o) * 0.02).astype(np.float32)
    assert np.array_equal(M.edt_numpy_separable(o, 0.02), esperado)
    assert np.array_equal(M.edt(o, 0.02, "numpy"), esperado)
    assert np.allclose(M.edt(o, 0.02, "auto"), esperado, atol=1e-6)      # sin CUDA: scipy


def test_esdf_con_motor_igual_que_antes():
    from scipy.ndimage import distance_transform_edt
    rng = np.random.default_rng(3)
    occ = rng.random((20, 25, 12)) < 0.05
    res = 0.02
    antes = np.where(occ, -(distance_transform_edt(occ) * res - 0.5 * res),
                     distance_transform_edt(~occ) * res - 0.5 * res).astype(np.float32)
    for motor in ("numpy", "auto"):
        assert np.allclose(M.esdf(occ, res, motor), antes, atol=1e-6)
    assert np.all(M.esdf(np.zeros((3, 3, 3), bool), res) == 1.0)
    todo = M.esdf(np.ones((3, 3, 3), bool), res)                           # todo ocupado: no revienta
    assert todo.shape == (3, 3, 3) and np.all(todo < 0)


def test_mapa_distancias_con_el_edt_nuevo():
    m = _mapa()
    m.motor_esdf = "numpy"
    T = T_camara(0.0, 0.0, 1.0)
    prof, _ = render(T, cajas=[((-0.06, -0.06, 0.0), (0.06, 0.06, 0.10))])
    for i in range(6):
        m.actualizar(prof, K, T, ahora=100.0 + i)
    m.distancias(106.0)                                                    # (con el reloj de la prueba)
    d = m.distancia_en(np.array([[0.0, 0.0, 0.30], [0.4, 0.4, 0.5]]))
    assert 0.15 < d[0] < 0.25                                              # 20 cm por encima de la tapa
    assert d[1] > d[0]


# ═════════════ B) la cámara A en la misma TSDF ═════════════

def test_prof_a_empaquetar_y_desempaquetar():
    rng = np.random.default_rng(0)
    prof = rng.integers(0, 3000, (240, 320)).astype(np.uint16)
    T = T_camara(0.1, -0.2, 0.9)
    for mascara in (None, rng.random((240, 320)) < 0.1):
        b = CA.empaquetar_prof(prof, K / 2, T, mascara, t=123.5, dist_max=1.2)
        d = CA.desempaquetar_prof(b)
        assert np.array_equal(d["prof"], prof)
        assert np.allclose(d["K"], K / 2) and np.allclose(d["T_mundo_cam"], T)
        assert d["t"] == 123.5 and d["dist_max"] == 1.2
        assert (d["mascara"] is None) if mascara is None else np.array_equal(d["mascara"], mascara)
    assert CA.desempaquetar_prof(b"basura") is None


def test_preparar_prof_quita_lo_cercano_y_reduce():
    T = T_camara(0.0, 0.0, 0.6)
    prof, _ = render(T, cajas=[((-0.05, -0.05, 0.0), (0.05, 0.05, 0.30))])   # tapa a 0,30 m de la cámara
    sub, Ks, m = CA.preparar_prof(prof, K, None, None, T, paso=2, z_min=0.35)
    assert sub.shape == (H // 2, W_ // 2) and m is None
    assert Ks[0, 0] == pytest.approx(K[0, 0] / 2) and Ks[1, 2] == pytest.approx(K[1, 2] / 2)
    assert sub[H // 4, W_ // 4] == 0                                       # la tapa: demasiado cerca
    assert sub[10, 10] > 0                                                 # el suelo, a 0,6 m: se queda


def test_camara_a_sola_pone_obstaculos_como_secundaria():
    """Una caja que solo ve la A: sale ocupada; la autocorrección (ultimos_puntos) no se entera."""
    m = _mapa()
    T_a = T_camara(0.25, 0.0, 0.7)
    caja = ((0.20, -0.05, 0.0), (0.30, 0.05, 0.12))
    prof, _ = render(T_a, cajas=[caja])
    sub, Ks, msub = CA.preparar_prof(prof, K, None, None, T_a, paso=2, z_min=0.2)
    d = CA.desempaquetar_prof(CA.empaquetar_prof(sub, Ks, T_a, msub, dist_max=1.2))
    for i in range(6):
        m.actualizar(d["prof"], d["K"], d["T_mundo_cam"], paso=1, excluir_px=d["mascara"], z_min=0.0,
                     dist_max=d["dist_max"], filtrar=False, principal=False, ahora=200.0 + i)
    assert m.ultimos_puntos is None
    assert m.ultima_sec.get("puntos", 0) > 0
    assert _estado_en(m, [[0.25, 0.0, 0.11]], 206.0)[0] == OCUPADO        # la tapa de la caja
    assert _estado_en(m, [[0.25, 0.0, 0.40]], 206.0)[0] == LIBRE          # encima, visto y libre


def test_dos_camaras_ven_lo_que_una_sola_no():
    """La cabeza ve la mesa desde arriba; un bloque tapa el lado de una caja baja que solo ve la A (de lado)."""
    m1, m2 = _mapa(), _mapa()
    T_c = T_camara(0.0, 0.0, 1.0)
    T_a = np.eye(4)                                       # la A, de lado: mirando hacia +x, inclinada hacia abajo
    ang = np.radians(35)
    z_cam = np.array([np.cos(ang), 0.0, -np.sin(ang)])
    x_cam = np.array([0.0, -1.0, 0.0])
    T_a[:3, :3] = np.column_stack([x_cam, np.cross(z_cam, x_cam), z_cam])
    T_a[:3, 3] = (-0.45, 0.0, 0.45)
    tejado = ((-0.10, -0.15, 0.20), (0.15, 0.15, 0.22))   # una «balda» que tapa a la cabeza lo de debajo
    caja = ((-0.05, -0.04, 0.0), (0.03, 0.04, 0.08))      # bajo la balda
    escena = [tejado, caja]
    prof_c, _ = render(T_c, cajas=escena)
    prof_a, _ = render(T_a, cajas=escena)
    sub, Ks, _ = CA.preparar_prof(prof_a, K, None, None, T_a, paso=2, z_min=0.2)
    for i in range(6):
        for m in (m1, m2):
            m.actualizar(prof_c, K, T_c, ahora=300.0 + i)
        m2.actualizar(sub, Ks, T_a, paso=1, z_min=0.0, dist_max=1.2, filtrar=False, principal=False,
                      ahora=300.0 + i)
    lado = [[-0.05, 0.0, 0.05]]                           # la cara de la caja que mira a la A
    assert _estado_en(m1, lado, 306.0)[0] != OCUPADO      # la cabeza sola no la ve (está bajo la balda)
    assert _estado_en(m2, lado, 306.0)[0] == OCUPADO      # con la A, sí
    assert _estado_en(m2, [[0.0, 0.0, 0.21]], 306.0)[0] == OCUPADO        # y la balda sigue ahí


# ═════════════ 2) la pelota no es obstáculo, pero lo de delante se sabe libre ═════════════

def test_mascara_halos_igual_que_quitar_pixeles():
    rng = np.random.default_rng(1)
    prof = rng.integers(300, 1500, (H, W_)).astype(np.uint16)
    halos = [(320.0, 240.0, 30.0, 0.5, 1.2), (100.0, 100.0, 20.0, 0.3, 1.4, 110.0, None)]
    m = ConstructorMapa.mascara_halos(prof, halos)
    q = ConstructorMapa.quitar_pixeles(prof, halos)
    assert m.any()
    assert np.array_equal(q == 0, m | (prof == 0))
    assert ConstructorMapa.mascara_halos(prof, []).sum() == 0


def test_pelota_libre_delante_sin_superficie():
    T = T_camara(0.0, 0.0, 0.9)
    c, r = (0.0, 0.0, 0.035), 0.035
    prof, (m_pel,) = render(T, esferas=[(c, r)])
    encima = [[0.0, 0.0, 0.16], [0.0, 0.0, 0.24]]        # por donde baja la mano
    cara = [[0.0, 0.0, 0.07]]                               # la parte de arriba de la pelota

    con = _mapa()
    for i in range(6):
        con.actualizar(prof, K, T, ahora=400.0 + i, excluir_px=m_pel, margen_libre=0.01)
    assert np.all(_estado_en(con, encima, 406.0) == LIBRE)          # se sabe libre
    assert _estado_en(con, cara, 406.0)[0] != OCUPADO                # la pelota no es obstáculo
    assert con.ultima["pelotas"] > 0

    sin = _mapa()                                            # como antes: sus píxeles se tiraban
    quitada = np.where(m_pel, 0, prof).astype(np.uint16)
    for i in range(6):
        sin.actualizar(quitada, K, T, ahora=400.0 + i)
    assert _estado_en(sin, cara, 406.0)[0] != OCUPADO
    assert np.all(_estado_en(sin, encima, 406.0) == DESCONOCIDO)    # antes: sin saber qué había


def test_pelota_con_impactos_se_tira_como_antes():
    T = T_camara(0.0, 0.0, 0.9)
    prof, (m_pel,) = render(T, esferas=[((0.0, 0.0, 0.035), 0.035)])
    m = MapaOcupacion(LIM, 0.02, z_suelo=0.0, metodo="impactos", z_min=0.2)
    for i in range(6):
        m.actualizar(prof, K, T, ahora=500.0 + i, excluir_px=m_pel)
    assert _estado_en(m, [[0.0, 0.0, 0.07]], 506.0)[0] != OCUPADO


def test_capsula_objetivo_para_curobo():
    assert M.capsula_objetivo(None) is None
    c = M.capsula_objetivo(dict(c=np.array([0.1, -0.2, 0.05]), r=0.035), margen=0.01)
    assert c["name"] == "pelota_objetivo" and c["radius"] == pytest.approx(0.045)
    assert np.allclose(c["base"], [0.1, -0.2, 0.05]) and np.linalg.norm(np.subtract(c["tip"], c["base"])) < 1e-3


def test_pelota_obstaculo_solo_al_ir_hacia_ella():
    from skill_shield.autonomo.tarea import Planificador
    obj = dict(c=np.zeros(3), r=0.035)
    pl = types.SimpleNamespace(obj_curobo=True, obj_curobo_tramos=("ir a la pelota",), _objetivo_curobo=obj,
                               _carga=None)
    f = Planificador.objetivo_curobo
    assert f(pl, "ir a la pelota") is obj
    assert f(pl, "ir al punto de dejada") is None
    assert f(pl, "volver al inicio") is None
    pl._carga = np.zeros((5, 3))                             # ya la lleva: va en las esferas de la mano
    assert f(pl, "ir a la pelota") is None
    pl._carga, pl.obj_curobo = None, False
    assert f(pl, "ir a la pelota") is None
