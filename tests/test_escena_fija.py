"""
(17-nov) Escena FIJA: escanear lo que no se mueve (la cajonera de la escena sintética de test_mapa_tsdf.py,
con la calibración real de la cámara de cabeza y ruido del estéreo), guardarlo como cajas y comprobar que:
  · el ajuste encuentra la cajonera con su tapa a la altura buena y su planta;
  · con la escena fija puesta, la cámara ya no pinta la tapa (ni su ruido) pero SÍ lo que se pone encima;
  · el escaneo desde la web (ConstructorMapa) escribe el archivo y lo aplica;
  · el planificador usa sus caras como obstáculo, su tapa como suelo y apoya la pelota a su altura.

    python -m pytest tests/test_escena_fija.py -q
"""
import math
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                        # noqa: E402
import test_mapa_tsdf as TM                                          # noqa: E402
from skill_shield.autonomo import escena_fija as EF                  # noqa: E402
from skill_shield.autonomo.mapa import OCUPADO, MapaOcupacion        # noqa: E402

G = {k: getattr(config, k) for k in dir(config) if k.isupper()}
(X0, X1), (Y0, Y1), _ = TM.CAJONERA


@pytest.fixture
def con_fija(monkeypatch, tmp_path):
    """Escena fija activa con un archivo propio (no el del laboratorio)."""
    monkeypatch.setenv("SKILL_SHIELD_ESCENA_FIJA", "1")
    g = dict(G, AUTONOMO_ESCENA_FIJA_ARCHIVO=str(tmp_path / "escena_fija.json"), AUTONOMO_ESCENA_FIJA_CAJAS=[])
    return g


def profundidad(T, rng, cajas, ruido=0.010):
    """Como TM.profundidad, con las cajas que se quieran (sin píxeles voladores)."""
    dw, dc = TM._rayos(T)
    t = np.full(len(dw), np.inf)
    for c in cajas:
        t = np.minimum(t, TM._caja(T[:3, 3], dw, c))
    z = (t * dc[:, 2]).reshape(TM.H, TM.W_)
    z = np.where(np.isfinite(z), z, 0.0)
    from scipy.ndimage import gaussian_filter
    campo = gaussian_filter(rng.normal(0, 1, z.shape), 15)
    campo /= max(float(campo.std()), 1e-9)
    z = z + (0.4 * rng.normal(0, 1, z.shape) + campo) * ruido * z * z
    return (np.clip(z, 0, 10) * 1000).astype(np.uint16)


def _escanear(T, n=12):
    m = MapaOcupacion(TM.LIM, 0.02, z_suelo=-0.975, metodo="tsdf", motor="numpy", dist_max=1.7, superficies=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        m.actualizar(profundidad(T, rng, [TM.CAJONERA, TM.PARED]), TM.K, T, ahora=1000.0 + i)
    return m


def _cajonera(cajas):
    """La caja ajustada que está sobre el centro de la cajonera."""
    for c in cajas:
        if EF.en_cajas([[0.5 * (X0 + X1), 0.5 * (Y0 + Y1), TM.Z_TAPA - 0.05]], [c])[0]:
            return c
    return None


# ══════════════════════ geometría ══════════════════════
def test_rectangulo_minimo_girado():
    rng = np.random.default_rng(2)
    L = rng.uniform([-0.30, -0.10], [0.30, 0.10], (400, 2))
    a = math.radians(25.0)
    P = L @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) + [0.4, -0.7]
    ctr, semi, yaw = EF.rect_minimo(P)
    assert np.allclose(ctr, [0.4, -0.7], atol=0.01)
    assert np.allclose(sorted(semi), [0.10, 0.30], atol=0.01)
    assert abs(((yaw - 25.0 + 45.0) % 90.0) - 45.0) < 1.0


def test_superficie_y_dentro():
    c = EF.caja("x", (0.0, 0.0, 0.5), (0.2, 0.1, 0.5), yaw_deg=30.0)
    P = EF.superficie([c], 0.02)
    assert np.all(EF.en_cajas(P, [c], margen=1e-6))           # todos en la caja (en su borde)
    assert abs(P[:, 2].max() - 1.0) < 1e-9 and P[:, 2].min() >= -1e-9
    assert not EF.en_cajas([[0.0, 0.0, 1.05]], [c])[0] and EF.en_cajas([[0.0, 0.0, 1.05]], [c], margen=0.06)[0]


# ══════════════════════ escaneo y ajuste ══════════════════════
def test_el_escaneo_encuentra_la_cajonera():
    T = TM._T()
    m = _escanear(T)
    esc = EF.ajustar(m.centros_ocupados(1012.0), m.res, m.superficies(), m.z_suelo)
    c = _cajonera(esc["cajas"])
    print("\n", [(k["nombre"], k["origen"], np.round(k["centro"], 3).tolist(), np.round(k["semi"], 3).tolist(),
                  k["yaw_deg"]) for k in esc["cajas"]])
    assert c is not None, "no hay caja sobre la cajonera"
    zt, poli = EF.tapa(c)
    assert abs(zt - TM.Z_TAPA) < 0.015                          # la tapa, a su altura
    assert abs(c["centro"][2] - c["semi"][2] - (-0.975)) < 0.01  # maciza hasta el suelo
    lo, hi = poli.min(0), poli.max(0)                           # la planta, ±4 cm
    assert abs(lo[0] - X0) < 0.04 and abs(hi[0] - X1) < 0.04 and abs(hi[1] - Y1) < 0.04 and abs(lo[1] - Y0) < 0.06


def test_con_la_escena_fija_solo_se_ve_lo_que_cambia():
    T = TM._T()
    esc = EF.ajustar(*(lambda m: (m.centros_ocupados(1012.0), m.res, m.superficies(), m.z_suelo))(_escanear(T)))
    pelota_caja = ((0.30, 0.38), (-1.00, -0.92), (TM.Z_TAPA, TM.Z_TAPA + 0.10))   # algo de 10 cm encima de la tapa
    m = MapaOcupacion(TM.LIM, 0.02, z_suelo=-0.975, metodo="tsdf", motor="numpy", dist_max=1.7, superficies=True)
    m.poner_fijo(esc["cajas"], 0.03)
    rng = np.random.default_rng(5)
    for i in range(12):
        m.actualizar(profundidad(T, rng, [TM.CAJONERA, TM.PARED, pelota_caja]), TM.K, T, ahora=2000.0 + i)
    O = m.centros_ocupados(2012.0)
    sobre = (O[:, 0] > X0 + 0.04) & (O[:, 0] < X1 - 0.04) & (O[:, 1] > Y0 + 0.04) & (O[:, 1] < Y1 - 0.04)
    nuevo = sobre & (O[:, 0] > 0.28) & (O[:, 0] < 0.40) & (O[:, 1] > -1.02) & (O[:, 1] < -0.90)
    print(f"\nocupados sobre la cajonera: {int(sobre.sum())} (del objeto nuevo: {int(nuevo.sum())})")
    assert nuevo.sum() >= 4                                    # lo nuevo sí sale
    assert (sobre & ~nuevo).sum() <= 3                         # la tapa y su ruido, no
    assert not any(abs(z - TM.Z_TAPA) < 0.03 for z, _ in m.superficies())   # ni se busca como plano


def test_escanear_desde_la_web(con_fija):
    from skill_shield.autonomo.mapa import ConstructorMapa
    g = dict(con_fija, AUTONOMO_MAPA_HZ=1000.0, AUTONOMO_AUTOCALIB=False, AUTONOMO_MAPA_MOTOR="numpy",
             AUTONOMO_MAPA_LIMITES=TM.LIM)
    cm = ConstructorMapa(g, RAIZ)
    assert cm.mapa.fijo == [] and cm.estado_fijo()["texto"] == "sin escena fija"
    cm.escanear_fijo()
    assert cm.estado_fijo()["escaneando"]
    rng = np.random.default_rng(0)
    T = TM._T()
    import time
    for _ in range(12):
        time.sleep(0.002)
        assert cm.paso(profundidad(T, rng, [TM.CAJONERA, TM.PARED]), TM.K)
    esc = cm.terminar_escaneo()
    assert os.path.exists(con_fija["AUTONOMO_ESCENA_FIJA_ARCHIVO"])
    assert _cajonera(esc["cajas"]) is not None
    assert cm.mapa.fijo and cm.mapa._mascara_fija is not None and cm.estado_fijo()["cajas"] == len(esc["cajas"])
    # otro proceso (el planificador) lo lee igual
    e2 = EF.EscenaFija(con_fija)
    assert len(e2.cajas) == len(esc["cajas"])
    cm.olvidar_fijo()
    assert cm.mapa.fijo == [] and os.path.exists(con_fija["AUTONOMO_ESCENA_FIJA_ARCHIVO"] + ".bak")


def test_las_cajas_de_config_cuentan(con_fija):
    g = dict(con_fija, AUTONOMO_ESCENA_FIJA_CAJAS=[dict(nombre="mesa", xy=(0.3, -1.0), semi_xy=(0.2, 0.3), z_sup=0.07,
                                                        desde_suelo=True)])
    e = EF.EscenaFija(g)
    assert len(e.cajas) == 1 and abs(EF.tapa(e.cajas[0])[0] - 0.07) < 1e-9


# ══════════════════════ planificador ══════════════════════
def test_planificador_usa_la_escena_fija(con_fija):
    from skill_shield.autonomo import tarea as TA
    from skill_shield.teleop_fr5.cinematica import OFFSET_PALMA_ORCA, cargar_cadena
    cadena = cargar_cadena(config.RUTA_XML, verbose=False)
    q0 = np.radians([93.0, -17.0, 53.0, 43.0, 105.0, 25.0])
    Tb = cadena.fk_marcos(q0)[6]
    palma = Tb[:3, :3] @ OFFSET_PALMA_ORCA + Tb[:3, 3]
    # una mesa cuya tapa queda 4 cm por ENCIMA de la palma: la mano estaría dentro
    zt = float(palma[2]) + 0.04
    EF.guardar(con_fija["AUTONOMO_ESCENA_FIJA_ARCHIVO"],
               dict(cajas=[EF.caja("mesa", (palma[0], palma[1], 0.5 * (zt + palma[2] - 0.30)), (0.15, 0.15, 0.5 * 0.34))],
                    voxeles=[], res=0.02))
    pl = TA.Planificador(cadena, con_fija)
    pl.poner_mapa(None, 0.02)
    if pl.mundo_simple:                                        # (18-nov) la caja es una LOSA (tapa maciza)
        assert len(pl._losas) == 1 and abs(pl._losas[0].z - zt) < 2e-4
        assert "mesa" in str(pl.motivo(q0))
    else:
        assert len(pl._arbol.data) > 100                       # sus caras, como cubos
    assert any(abs(z - zt) < 2e-4 for z, _ in pl._superficies)  # su tapa, como suelo
    assert pl.motivo(q0) is not None                           # la mano, dentro: no vale
    # sin la escena fija, la misma postura sí vale
    os.remove(con_fija["AUTONOMO_ESCENA_FIJA_ARCHIVO"])
    pl.poner_mapa(None, 0.02)
    assert pl._arbol is None and pl.motivo(q0) is None and not pl._losas
    # la pelota sobre una tapa fija se apoya a su altura (aunque el estéreo la vea 2 cm alta)
    EF.guardar(con_fija["AUTONOMO_ESCENA_FIJA_ARCHIVO"],
               dict(cajas=[EF.caja("cajonera", (0.35, -1.04, 0.5 * (0.07 - 0.975)), (0.25, 0.21, 0.5 * 1.045))],
                    voxeles=[], res=0.02))
    pl.poner_mapa(None, 0.02)
    Tp = np.eye(4)
    Tp[:3, 3] = (0.35, -1.0, 0.07 + 0.0335 + 0.02)
    Ta = pl.apoyar(Tp, 0.067, apoyo_z=None)
    assert abs(Ta[2, 3] - (0.07 + 0.0335)) < 1e-6 and pl._apoyo_ok
