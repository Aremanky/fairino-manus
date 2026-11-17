"""
(17-nov) Migración a cuRobo: lo que se puede probar SIN GPU ni cuRobo (URDF, mundo/ESDF y cómo el
planificador del proyecto verifica lo que propone cuRobo). Con cuRobo instalado y la GPU, además,
probar_curobo.py en la raíz del proyecto.

    python -m pytest tests/test_curobo.py -q
"""
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tests"))

import config                                                        # noqa: E402
from skill_shield.autonomo import tarea as TA                        # noqa: E402
from skill_shield.autonomo.curobo_fr5 import mundo as M              # noqa: E402
from skill_shield.autonomo.curobo_fr5.urdf import (CADENA, HERRAMIENTA, fk_urdf, generar_urdf,   # noqa: E402
                                                   posturas_por_defecto)
from skill_shield.teleop_fr5.cinematica import cargar_cadena         # noqa: E402

G = {k: getattr(config, k) for k in dir(config) if k.isupper()}


@pytest.fixture(scope="module")
def cadena():
    return cargar_cadena(config.RUTA_XML, verbose=False)


def test_urdf_tiene_la_misma_cinematica_que_el_proyecto(cadena, tmp_path):
    ruta = generar_urdf(config.RUTA_XML, str(tmp_path / "fr5_orca.urdf"), raiz=RAIZ, posturas=posturas_por_defecto(config))
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-3, 3, 6)
        Mk, T = cadena.fk_marcos(q), fk_urdf(ruta, q)
        for k, link in enumerate(CADENA[1:] + (HERRAMIENTA,)):
            assert np.allclose(T[link], Mk[k], atol=1e-6), link
    import xml.etree.ElementTree as ET
    mallas = [m.get("filename") for m in ET.parse(ruta).getroot().iter("mesh")]
    assert mallas and all(os.path.exists(os.path.join(RAIZ, m)) for m in mallas)


def test_esdf_signo_y_distancias():
    occ = np.zeros((20, 20, 20), bool)
    occ[8:12, 8:12, 8:12] = True
    d = M.esdf(occ, 0.02)
    assert d[10, 10, 10] < 0 and d[0, 0, 0] > 0
    assert d[13, 10, 10] == pytest.approx(0.02 * 2 - 0.01, abs=1e-6)   # 2 cubos fuera: 3 cm del borde
    assert np.all(d[occ] < 0) and np.all(d[~occ] > 0)


def test_cajas_del_entorno_con_su_giro():
    from skill_shield.teleop_fr5.ik_teleop import Caja, Entorno
    from skill_shield.autonomo.choque_mano import _quat
    R = _quat([0.9, 0.1, -0.3, 0.2])
    ent = Entorno(z_suelo=-0.975, cajas=[Caja("mesa", (0.3, -0.6, -0.4), (0.2, 0.3, 0.4), R)])
    caps, cajas = M.obstaculos_entorno(ent)
    q = cajas[0]["pose"][3:]
    assert np.allclose(_quat(q), R, atol=1e-6) and cajas[0]["dims"] == pytest.approx([0.4, 0.6, 0.8])
    assert cajas[-1]["name"] == "suelo"


def test_mundo_del_planificador_lleva_cubos_y_suelo(cadena):
    from test_tarea import _cajonera
    pl = TA.Planificador(cadena, G)
    pl.poner_mapa(_cajonera(), 0.02)
    pl._empezar(np.radians([93.0, -17.0, 53.0, 43.0, 105.0, 25.0]))
    pl.poner_mesa(-0.25, [[0.1, -0.75], [0.6, -0.75], [0.6, -0.3], [0.1, -0.3]])
    pl.zonas_apoyo(())
    lim = ((-1.0, 0.6), (-1.6, 0.4), (-0.975, 0.9))
    sdf, centro, dims = M.mundo_del_planificador(pl, lim)
    origen, n = M.rejilla(lim, 0.02)
    def en(p):
        i = np.floor((np.asarray(p) - origen) / 0.02).astype(int)
        return sdf[i[0], i[1], i[2]]
    assert en([0.35, -0.5, -0.30]) < 0          # dentro de la cajonera (macizo / suelo de la mesa)
    assert en([0.35, -0.5, 0.10]) > 0.25        # encima, libre
    assert np.allclose(centro, origen + 0.5 * n * 0.02)
    pl.poner_mapa(None, 0.02)


class CuRoboFalso:
    """Lo que hace PlanificadorCuRobo, sin GPU: devuelve el tramo que se le diga."""

    def __init__(self, tramo):
        self.tramo, self.mundos, self.planes = tramo, 0, 0

    def actualizar_mundo(self, pl, clave=None):
        self.mundos += 1

    def plan(self, q0, q1):
        self.planes += 1
        return self.tramo(np.asarray(q0, float), np.asarray(q1, float))


def test_lo_que_propone_curobo_se_verifica(cadena):
    from test_tarea import _cajonera, _dentro_cajonera
    pl = TA.Planificador(cadena, G)
    pl.poner_mapa(_cajonera(), 0.02)
    q0 = pl._empezar(np.radians([93.0, -17.0, 53.0, 43.0, 105.0, 25.0]))
    q1 = q0 + np.radians([12.0, 0, 0, 0, 0, 0])
    # 1) cuRobo propone un tramo limpio (recto en joints): se usa tal cual (densificado)
    pl.curobo = CuRoboFalso(lambda a, b: np.linspace(a, b, 4))
    qs = pl.conectar(q0, q1, None, "prueba")
    assert pl.tramos_curobo == 1 and np.allclose(qs[0], q0) and np.allclose(qs[-1], q1)
    assert float(np.max(np.abs(np.diff(qs, axis=0)))) <= pl.paso + 1e-9
    # 2) cuRobo propone un tramo que mete la mano en la cajonera: se rechaza y se usa el propio
    bajar = None
    rng = np.random.default_rng(3)
    for _ in range(2000):
        q = q0 + rng.normal(0, 0.6, 6)
        if _dentro_cajonera(pl._puntos_robot(q)).any() and pl.motivo(q) is not None:
            bajar = q
            break
    assert bajar is not None
    pl.curobo = CuRoboFalso(lambda a, b: np.array([a, bajar, b]))
    qs = pl.conectar(q0, q1, None, "prueba")
    assert pl.curobo_rechazos == 1
    assert not any(_dentro_cajonera(pl._puntos_robot(q)).any() for q in qs)
    # 3) si cuRobo falla (excepción de la GPU), se desactiva y se sigue con el propio
    def roto(a, b):
        raise RuntimeError("CUDA error")
    pl.curobo = CuRoboFalso(roto)
    qs = pl.conectar(q0, q1, None, "prueba")
    assert pl.curobo is None and np.allclose(qs[-1], q1)
    pl.poner_mapa(None, 0.02)


def test_sin_curobo_instalado_sigue_el_propio(cadena):
    pl = TA.Planificador(cadena, dict(G, AUTONOMO_TAREA_PLANIFICADOR="curobo"))
    try:
        import curobo  # noqa: F401
        pytest.skip("cuRobo está instalado: esto se prueba con probar_curobo.py")
    except ImportError:
        pass
    assert pl.curobo is None
