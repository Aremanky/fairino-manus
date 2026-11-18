"""
(17-nov) Mapa con fusión TSDF, distancia máxima y mesas como planos, con una escena sintética como
la del laboratorio: la cámara de cabeza con su calibración real, una cajonera (tapa a z = 0,07 m),
una pared a ~2,7 m, ruido del estéreo que crece con la distancia y píxeles «voladores» en los bordes
(los que dibujaban la rampa). Sin cámara ni robot:

    python -m pytest tests/test_mapa_tsdf.py -q
"""
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from skill_shield.autonomo import calibracion as C                    # noqa: E402
from skill_shield.autonomo.mapa import (MapaOcupacion, OCUPADO, desempaquetar_visto,   # noqa: E402
                                        empaquetar_visto)
from skill_shield.autonomo.tsdf import (dentro_poligono, observar_numpy, planos_horizontales)  # noqa: E402

K = np.array([[450.0, 0, 320], [0, 450.0, 200], [0, 0, 1]])
H, W_ = 400, 640
Z_TAPA = 0.07
CAJONERA = ((0.10, 0.60), (-1.25, -0.83), (-0.975, Z_TAPA))
PARED = ((-0.95, -0.70), (-1.58, -1.40), (-0.975, 0.5))   # otra mesa / una persona, a ~1,9 m de la cámara
LIM = ((-1.0, 0.6), (-1.6, 0.4), (-0.975, 0.9))


def _T():
    return C.cargar(os.path.join(RAIZ, "camara_robot.json"))["T_mundo_cam"]


def _rayos(T):
    v, u = np.mgrid[0:H, 0:W_]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    return d.reshape(-1, 3) @ T[:3, :3].T, d.reshape(-1, 3)


def _caja(o, dw, caja):
    lo, hi = np.array([c[0] for c in caja]), np.array([c[1] for c in caja])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / dw, (hi - o) / dw
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((tmax >= tmin) & (tmax > 0), np.maximum(tmin, 0), np.inf)


def profundidad(T, rng, ruido=0.010, voladores=True):
    """Imagen de profundidad (mm) con ruido k·z² y píxeles voladores en los bordes."""
    dw, dc = _rayos(T)
    t = np.minimum(_caja(T[:3, 3], dw, CAJONERA), _caja(T[:3, 3], dw, PARED))
    z = (t * dc[:, 2]).reshape(H, W_)
    z = np.where(np.isfinite(z), z, 0.0)
    from scipy.ndimage import gaussian_filter
    campo = gaussian_filter(rng.normal(0, 1, z.shape), 15)          # error que se extiende por zonas
    campo /= max(float(campo.std()), 1e-9)
    z = z + (0.4 * rng.normal(0, 1, z.shape) + campo) * ruido * z * z
    if voladores:                                   # bordes: mezcla de lo de delante y lo de detrás
        dz = np.abs(np.diff(z, axis=1, prepend=z[:, :1])) > 0.2
        borde = np.zeros_like(dz)
        for k in range(-3, 4):
            borde |= np.roll(dz, k, axis=1)
        mezcla = rng.random(z.shape)
        izq, der = np.roll(z, 4, axis=1), np.roll(z, -4, axis=1)
        z = np.where(borde & (rng.random(z.shape) < 0.6), izq * mezcla + der * (1 - mezcla), z)
    return (np.clip(z, 0, 10) * 1000).astype(np.uint16)


def _mapa(metodo, T, n=12, **kw):
    m = MapaOcupacion(LIM, 0.02, z_suelo=-0.975, metodo=metodo, motor="numpy", **kw)
    rng = np.random.default_rng(0)
    for i in range(n):
        m.actualizar(profundidad(T, rng), K, T, ahora=1000.0 + i)
    return m


def _metricas(m, T):
    O = m._centros[(m.estado(1012.0) == OCUPADO).reshape(-1)]
    (x0, x1), (y0, y1), _ = CAJONERA
    dentro = (O[:, 0] > x0 + 0.04) & (O[:, 0] < x1 - 0.04) & (O[:, 1] > y0 + 0.04) & (O[:, 1] < y1 - 0.04)
    encima = dentro & (O[:, 2] > Z_TAPA + 0.03)                 # ruido por encima de la tapa
    fuera = ~((O[:, 0] > x0 - 0.04) & (O[:, 0] < x1 + 0.04) & (O[:, 1] > y0 - 0.04) & (O[:, 1] < y1 + 0.04))
    lejos = np.linalg.norm(O - T[:3, 3], axis=1) > 1.8
    # grosor de la tapa: cubos ocupados por columna, sobre la cajonera, en la franja de la tapa (±10 cm)
    tapa = dentro & (np.abs(O[:, 2] - Z_TAPA) < 0.10)
    _, n_col = np.unique(np.round(O[tapa, :2], 3), axis=0, return_counts=True)
    return dict(total=len(O), encima=int(encima.sum()), flotando=int((fuera & ~lejos).sum()), lejos=int(lejos.sum()),
                grosor_cm=round(float(np.mean(n_col)) * 2, 1) if len(n_col) else 0.0)


def test_tsdf_limpia_la_tapa_y_quita_lo_lejano():
    T = _T()
    viejo = _metricas(_mapa("impactos", T), T)                  # como antes: sin distancia máxima
    nuevo = _metricas(_mapa("tsdf", T, dist_max=1.7), T)
    print("\nimpactos:", viejo, "\ntsdf    :", nuevo)
    assert viejo["lejos"] > 0 and nuevo["lejos"] == 0          # lo de más de 1,7 m ya no está
    assert nuevo["encima"] <= viejo["encima"]                  # no más ruido sobre la tapa
    assert nuevo["flotando"] <= viejo["flotando"] + 5          # ni flotando (algún cubo en un borde, por el truncado)


def test_tsdf_ve_la_tapa_donde_esta():
    T = _T()
    m = _mapa("tsdf", T, dist_max=1.7)
    O = m._centros[(m.estado(1012.0) == OCUPADO).reshape(-1)]
    (x0, x1), (y0, y1), _ = CAJONERA
    sobre = (O[:, 0] > x0 + 0.04) & (O[:, 0] < x1 - 0.04) & (O[:, 1] > y0 + 0.04) & (O[:, 1] < -0.90)
    assert sobre.sum() > 50
    assert abs(float(O[sobre, 2].max()) + 0.01 - Z_TAPA) < 0.025    # la parte de arriba del cubo más alto


def test_las_mesas_salen_como_planos_y_se_publican():
    T = _T()
    m = _mapa("tsdf", T, n=8, dist_max=1.7, superficies=True)
    sup = m.superficies()
    assert sup, "no ha encontrado la tapa"
    z, poli = max(sup, key=lambda s_: len(s_[1]))
    assert abs(z - Z_TAPA) < 0.015
    assert dentro_poligono(poli, np.array([[0.35, -0.95]]))[0]
    v = desempaquetar_visto(empaquetar_visto(m, 1008.0))
    assert v["superficies"] and abs(v["superficies"][0][0] - z) < 1e-3
    # encima de la tapa, nada (la capa de ruido se limpia)
    O = m._centros[(m.estado(1008.0) == OCUPADO).reshape(-1)]
    en = dentro_poligono(poli, O[:, :2])
    assert not np.any(en & (O[:, 2] > z + 0.03))


def test_planos_horizontales_con_ruido():
    rng = np.random.default_rng(1)
    tapa = np.column_stack([rng.uniform(0.1, 0.6, 3000), rng.uniform(-1.2, -0.85, 3000), Z_TAPA + rng.normal(0, 0.004, 3000)])
    basura = rng.uniform([-0.5, -1.5, -0.5], [0.8, -0.5, 0.6], (1500, 3))
    planos = planos_horizontales(np.vstack([tapa, basura]))
    assert planos and abs(planos[0][0] - Z_TAPA) < 0.01 and planos[0][2] > 0.1


def test_gpu_igual_que_numpy():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("sin GPU con CUDA")
    from skill_shield.autonomo.tsdf import ObservadorTorch
    T = _T()
    m = MapaOcupacion(LIM, 0.02)
    rng = np.random.default_rng(0)
    d = profundidad(T, rng).astype(np.float32) * 0.001
    args = (m._centros, np.linalg.inv(T), d, d, K, 0.04, 0.02)
    a, b = observar_numpy(*args), ObservadorTorch(m._centros)(*args)
    assert set(a[0].tolist()) == set(b[0].tolist())
    ia, ib = np.argsort(a[0]), np.argsort(b[0])
    assert np.allclose(a[1][ia], b[1][ib], atol=1e-4)


def test_la_mesa_debajo_y_alrededor_de_la_pelota_no_desaparece():
    """(17-nov) Con las MISMAS exclusiones que el modo autónomo (la caja de la pelota elegida, sus cajas
    de halo y los píxeles del halo), la tapa de la cajonera debajo y alrededor de la pelota sigue en el
    mapa (antes quedaba un agujero justo donde hay que cogerla)."""
    import time as _t
    from skill_shield.autonomo.mapa import ConstructorMapa
    from skill_shield.autonomo.pelotas import DetectorPelotas, _Pista
    T = _T()
    r = 0.035
    c = np.array([0.35, -0.95, Z_TAPA + r])
    dw, dc = _rayos(T)
    o = T[:3, 3]
    oc = o - c
    b = dw @ oc
    a = np.sum(dw ** 2, -1)
    disc = b ** 2 - a * (oc @ oc - r * r)
    t_b = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
    t = np.minimum(_caja(o, dw, CAJONERA), t_b)
    z = np.where(np.isfinite(t), t * dc[:, 2], 0.0).reshape(H, W_)
    rng = np.random.default_rng(0)
    cc = np.linalg.inv(T) @ np.r_[c, 1.0]
    u, v, rp = cc[0] / cc[2] * K[0, 0] + K[0, 2], cc[1] / cc[2] * K[1, 1] + K[1, 2], r * K[0, 0] / cc[2]
    det = DetectorPelotas({}, RAIZ)
    det.T_mundo_cam = T
    e = dict(nombre="Pelota roja", color_gemelo=(1, 0, 0))
    det.ultimas = [(e, cc[:3], 2 * r, (u, v, rp))]
    det.pistas[100] = _Pista(100, e, cc[:3], 2 * r, _t.time())
    # como en el modo autónomo (AUTONOMO_MAPA_MESAS = True): la tapa como plano. Justo DEBAJO de la pelota la
    # cámara no ve la mesa (la tapa la propia pelota): ahí la pone el plano
    m = MapaOcupacion(LIM, 0.02, z_suelo=-0.975, metodo="tsdf", motor="numpy", dist_max=1.7, superficies=True)
    m.excluir = [tuple(np.asarray(x, float) for x in det.caja_mundo(100))] + det.cajas_mundo()
    for i in range(10):
        prof = (np.clip(z + rng.normal(0, 0.003, z.shape) * z * z, 0, 10) * 1000).astype(np.uint16)
        prof = ConstructorMapa.quitar_pixeles(prof, det.pixeles_halo(), K, T)
        m.actualizar(prof, K, T, ahora=1000.0 + i)
    O = m._centros[(m.estado(1010.0) == OCUPADO).reshape(-1)]
    tapa = (O[:, 2] > Z_TAPA - 0.03) & (O[:, 2] < Z_TAPA + 0.016)          # la capa de la tapa (z = 0,07)
    dh = np.linalg.norm(O[:, :2] - c[:2], axis=1)
    debajo = tapa & (dh < 0.03)
    anillo = tapa & (dh >= 0.04) & (dh < 0.10)
    print(f"\ndebajo de la pelota: {debajo.sum()} cubos de mesa · alrededor: {anillo.sum()}")
    assert debajo.sum() >= 2, f"debajo de la pelota solo {debajo.sum()} cubos de mesa"
    # el anillo alrededor: casi completo (lo que tapa la pelota a la cámara puede faltar, pero no un agujero)
    from skill_shield.autonomo.mapa import MapaOcupacion as _M
    total = sum(1 for x in np.arange(-0.10, 0.101, 0.02) for y in np.arange(-0.10, 0.101, 0.02)
                if 0.04 <= np.hypot(x, y) < 0.10)
    assert anillo.sum() >= 0.6 * total, f"alrededor de la pelota solo {anillo.sum()}/{total} cubos de mesa"
