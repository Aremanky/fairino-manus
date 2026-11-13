"""
Mapa de obstáculos contra el RUIDO de la profundidad (11-nov), sin cámara:

    python -m pytest tests/test_mapa_ruido.py -q

Escena sintética: pared a 4 m, mesa, una caja a 0,8 m… y lo que estropeaba el mapa: una ventana
saturada con profundidades inventadas y píxeles sueltos a 0,4-1,2 m. Los objetos de verdad se
quedan; el ruido (que antes salía como cubos flotando) no.
"""
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del repo (tests/..)
sys.path.insert(0, RAIZ)

from skill_shield.autonomo.mapa import MapaOcupacion, pixeles_fiables   # noqa: E402

K = np.array([[470.0, 0, 320], [0, 470.0, 240], [0, 0, 1]])


def _escena(rng):
    prof = np.full((480, 640), 4000.0)
    img = np.full((480, 640, 3), 180, np.uint8)
    v = np.arange(480)[:, None] * np.ones((1, 640))
    mesa = v > 300
    prof[mesa] = 1400 - (v[mesa] - 300) * 4.4
    prof[200:280, 280:360] = 800                                   # caja
    img[40:160, 420:600] = 255                                     # ventana saturada
    prof[40:160, 420:600] = rng.uniform(500, 1100, (120, 180))      # con profundidad inventada
    suelto = rng.random((480, 640)) < 0.01
    prof[suelto] = rng.uniform(400, 1200, suelto.sum())            # píxeles sueltos
    prof[:20, :20] = 250                                           # demasiado cerca: no mide
    return prof.astype(np.uint16), img


def test_el_ruido_no_sale_como_obstaculo_y_la_caja_si():
    rng = np.random.default_rng(0)
    m = MapaOcupacion(((-1.5, 1.5), (-1.2, 1.2), (0.2, 2.0)), 0.02)
    for i in range(6):
        prof, img = _escena(rng)
        m.actualizar(prof, K, np.eye(4), ahora=100 + i, img_bgr=img)
    C = m.centros_ocupados(105)
    u, v = C[:, 0] / C[:, 2] * 470 + 320, C[:, 1] / C[:, 2] * 470 + 240
    caja = (abs(C[:, 2] - 0.8) < 0.05) & (u > 275) & (u < 365) & (v > 195) & (v < 285)
    mesa = v > 295
    assert caja.sum() > 40 and mesa.sum() > 1000
    assert (~caja & ~mesa).sum() == 0, "cubos flotando donde no hay nada"


def test_pixeles_fiables():
    prof = np.full((50, 50), 1000, np.uint16)
    prof[25, 25] = 600                                             # volador
    img = np.full((50, 50, 3), 100, np.uint8)
    img[:10, :10] = 255                                            # saturado
    ok = pixeles_fiables(prof, img)
    assert ok[30, 30] and not ok[25, 25] and not ok[5, 5]
