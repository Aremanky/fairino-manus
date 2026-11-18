"""
(17-nov) Comprobación del mapa en lote (comprobador_lote.py): mapa de distancias + GPU.
Es un ATAJO conservador: nunca puede dar por libre algo que la búsqueda exacta da por chocando, y el
planificador tiene que decidir exactamente lo mismo con él que sin él.

    python -m pytest tests/test_comprobador_lote.py -q
"""
import os
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tests"))

import config                                                        # noqa: E402
from skill_shield.autonomo import tarea as TA                        # noqa: E402
from skill_shield.autonomo.comprobador_lote import (MapaDistancias, crear_motor, exacto_en_lote,   # noqa: E402
                                                    libres_en_lote)
from skill_shield.teleop_fr5.cinematica import cargar_cadena         # noqa: E402

G = {k: getattr(config, k) for k in dir(config) if k.isupper()}


@pytest.fixture(scope="module")
def cadena():
    return cargar_cadena(config.RUTA_XML, verbose=False)


def _nube(rng, res=0.02):
    return np.unique(np.round(rng.uniform(-0.3, 0.3, (400, 3)) / res), axis=0) * res


@pytest.mark.parametrize("desplazado", [0.0, 0.007])
def test_la_distancia_de_la_rejilla_es_cota_inferior(desplazado):
    rng = np.random.default_rng(1)
    C = _nube(rng) + desplazado                        # también con cubos fuera de la rejilla alineada
    md = MapaDistancias(C, 0.02)
    P = rng.uniform(-0.6, 0.6, (20000, 3))
    real, _ = cKDTree(C).query(P)
    est = md.leer(P) - md.h
    assert np.all(est <= real + 1e-6)                  # nunca dice que está más lejos de lo que está
    assert np.mean(real - est) < 0.03                  # y no se pasa de prudente


def test_en_lote_nunca_da_por_libre_lo_que_choca():
    rng = np.random.default_rng(2)
    C = _nube(rng)
    md, arbol = MapaDistancias(C, 0.02), cKDTree(C)
    P = rng.uniform(-0.6, 0.6, (300, 1, 3)) + rng.uniform(-0.05, 0.05, (300, 40, 3))   # «posturas» compactas
    U = rng.uniform(0.01, 0.06, 40)
    libres = libres_en_lote(md, P, U)
    d, _ = arbol.query(P.reshape(-1, 3))
    choca = np.any(d.reshape(300, 40) < U[None, :], axis=1)
    assert not np.any(libres & choca)
    assert libres.sum() > 30                           # y sirve: deja pasar bastantes sin búsqueda exacta


def test_exacto_en_lote_igual_que_el_arbol():
    rng = np.random.default_rng(5)
    C = _nube(rng)
    md, arbol = MapaDistancias(C, 0.02), cKDTree(C)
    P = rng.uniform(-0.6, 0.6, (400, 1, 3)) + rng.uniform(-0.05, 0.05, (400, 30, 3))
    U = rng.uniform(0.01, 0.08, 30)
    d, _ = arbol.query(P.reshape(-1, 3))
    esperado = ~np.any(d.reshape(400, 30) < U[None, :], axis=1)
    assert np.array_equal(exacto_en_lote(md, arbol, P, U), esperado)
    assert 20 < esperado.sum() < 380                   # hay de los dos


def test_mano_brazo_en_lote_igual_que_postura_a_postura(cadena):
    pl = TA.Planificador(cadena, G)
    q0 = np.radians([93.0, -17.0, 53.0, 43.0, 105.0, 25.0])
    pl._empezar(q0)
    rng = np.random.default_rng(6)
    Qs = q0 + rng.uniform(-1.0, 1.0, (300, 6))
    ms = [cadena.fk_marcos(q) for q in Qs]
    esperado = np.array([pl.choque.ok(q) for q in Qs])
    assert np.array_equal(pl.choque.cumple_lote(ms), esperado)
    assert np.array_equal(np.array([pl.choque.ok_rapido(m) for m in ms]), esperado)
    assert 20 < esperado.sum() < 280


def test_gpu_igual_que_numpy():
    torch, dev = crear_motor("gpu")
    if torch is None:
        pytest.skip("sin PyTorch con CUDA")
    rng = np.random.default_rng(3)
    C = _nube(rng)
    md, arbol = MapaDistancias(C, 0.02), cKDTree(C)
    P = rng.uniform(-0.5, 0.5, (500, 60, 3))
    U = rng.uniform(0.01, 0.12, 60)
    assert np.array_equal(libres_en_lote(md, P, U), libres_en_lote(md, P, U, torch, dev))
    assert np.array_equal(exacto_en_lote(md, arbol, P, U), exacto_en_lote(md, arbol, P, U, torch, dev))


def test_el_planificador_decide_lo_mismo_con_y_sin_lote(cadena):
    from test_tarea import _cajonera
    rng = np.random.default_rng(4)
    pl = TA.Planificador(cadena, G)
    pl.poner_mapa(_cajonera(), 0.02)
    pl._empezar(np.radians([93.0, -17.0, 53.0, 43.0, 105.0, 25.0]))
    pl.zonas_apoyo(())
    q0 = np.radians([93.0, -17.0, 53.0, 43.0, 105.0, 25.0])
    Qs = q0 + rng.uniform(-1.2, 1.2, (400, 6))
    try:
        pl.comprobar_lote = False
        sin = [pl.motivo(q) for q in Qs]
        h_sin = [pl._holgura(q)[0] for q in Qs[:60]]
        pl.comprobar_lote = True
        con = [m for _, m in pl.motivos(Qs)]
        h_con = [pl._holgura(q)[0] for q in Qs[:60]]
        assert sin == con
        assert pl.lote_aciertos > 50                   # el atajo se usa
        assert np.allclose(np.minimum(h_sin, 0.25), np.minimum(h_con, 0.25), atol=1e-6)
    finally:
        pl.poner_mapa(None, 0.02)
