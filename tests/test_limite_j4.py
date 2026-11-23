"""
(23-nov) Límite del J4 del modo «Gemelo con sliders» (skill_shield/interfaz/limite_j4.py):

    python -m pytest tests/test_limite_j4.py -q

  · Con la muñeca recta pasa; doblada hasta que la mano se acerca al giro del J4, no.
  · Se comprueba TODO el camino (el rectángulo J5 × J6), no solo el destino.
  · Si ya está cerca, puede alejarse pero no acercarse más.
"""
import math
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import config                                                        # noqa: E402
from skill_shield.interfaz.limite_j4 import LimiteJ4                 # noqa: E402


@pytest.fixture(scope="module")
def lim():
    L = LimiteJ4.desde_config({k: getattr(config, k) for k in dir(config) if k.isupper()})
    assert L.preparar(), L.error
    return L


def _q(j5, j6=0.0):
    q = np.array(config.POSICION_STANDBY_BRAZO, float)
    q[4], q[5] = math.radians(j5), math.radians(j6)
    return q


def test_recta_pasa_y_doblada_no(lim):
    assert lim.permitido(_q(90), _q(100))[0]
    ok, h = lim.permitido(_q(90), _q(150))
    assert not ok and h < lim.holgura


def test_se_mira_el_camino_entero(lim):
    # acabar en la zona mala no vale; cruzar de un lado a otro por la muñeca recta, sí
    assert not lim.permitido(_q(0), _q(160))[0]
    assert lim.permitido(_q(-100), _q(100))[0]


def test_si_ya_esta_cerca_solo_puede_alejarse(lim):
    assert lim.permitido(_q(140), _q(100))[0]           # se aleja
    assert not lim.permitido(_q(140), _q(160))[0]       # se acercaría más


def test_sin_tabla_no_deja_mover():
    L = LimiteJ4.desde_config({k: getattr(config, k) for k in dir(config) if k.isupper()})
    assert not L.listo and not L.permitido(_q(0), _q(10))[0]
