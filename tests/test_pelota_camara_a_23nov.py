"""
(23-nov) Pelota sobre la caja blanca, medida con la CÁMARA A; la caja, escaneada con la de cabeza.

Caso real (plan_fallido_20261123_115901 + la escena fija de ese momento): «no encuentro cómo llegar a la pelota
(…) bajar a la pelota: la mano tocaría la mesa (holgura 2 cm)». Tres cosas:
  · la tapa escaneada de la caja (mesa_3) salía más pequeña que la caja: la pelota quedaba 1-2 cm fuera, no se
    apoyaba en ella y se usaba la mesa medida en vivo (3 cm más baja): pelota «metida» en la tapa;
  · la cámara A ve la pelota ~2 cm más baja que la tapa de la cámara de cabeza: manda la A (y la tapa baja en el
    plan). Antes, cuando se apoyaba en la tapa, la mano se quedaba ~2 cm por encima de la pelota;
  · la dejada DE PIE (delante de la hoja) pedía 2 cm de holgura con «la mesa» que es la propia hoja: con
    6,6 cm de pelota no salía y con 6,7 sí.

    python -m pytest tests/test_pelota_camara_a_23nov.py -q
"""
import json
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "diagnosticos"))

DATOS = os.path.join(RAIZ, "tests", "datos")
CASO = os.path.join(DATOS, "caso_pelota_camara_a_23nov.npz")


@pytest.fixture(scope="module")
def planificador():
    os.environ["SKILL_SHIELD_ESCENA_FIJA"] = "1"
    try:
        import config
        from skill_shield.autonomo import tarea as TA
        from skill_shield.autonomo.calibracion import cargar
        from skill_shield.teleop_fr5.cinematica import cargar_cadena
        G = {k: getattr(config, k) for k in dir(config) if k.isupper()}
        G["AUTONOMO_ESCENA_FIJA_ARCHIVO"] = os.path.join(DATOS, "escena_fija_23nov.json")
        G["AUTONOMO_TAREA_PLANIFICADOR"] = "propio"
        T_cab = cargar(os.path.join(RAIZ, "camara_robot.json"))["T_mundo_cam"]
        yield TA, TA.Planificador(cargar_cadena(G["RUTA_XML"], verbose=False), G, T_cab)
    finally:
        os.environ["SKILL_SHIELD_ESCENA_FIJA"] = "0"


def _caso(diametro=None):
    d = dict(np.load(CASO, allow_pickle=False))
    info = json.loads(str(d["info"]))
    if diametro is not None:
        info["diametro"] = diametro
    return d, info


@pytest.mark.parametrize("diametro", [None, 0.06])
def test_sale_el_plan_y_coge_a_la_altura_de_la_camara_a(planificador, diametro):
    TA, pl = planificador
    from banco_planes import planificar
    d, info = _caso(diametro)
    plan = planificar(pl, d, info)
    assert plan is not None
    # manda la cámara A: la pelota se queda a su altura (no se sube 2 cm a la tapa de la cámara de cabeza)
    z_medida = float(np.asarray(d["T"])[0][2, 3])
    assert abs(float(pl._objetivo_curobo["c"][2]) - z_medida) < 1e-6
    assert any("cámara A" in a for a in pl.avisos) or pl._tapas_bajadas


def test_sin_camara_a_se_apoya_en_la_tapa_ampliada(planificador):
    """Si la pelota la mide la cámara de cabeza (la misma que la escena fija), se apoya en la tapa aunque quede
    1-2 cm fuera de la tapa escaneada."""
    TA, pl = planificador
    d, info = _caso()
    T = np.asarray(d["T"])[0]
    r = info["diametro"] / 2
    pl._T_cam_medida = None
    pl.apoyar(T, info["diametro"], info["apoyo_z"])
    assert pl._z_fijo_usado == pytest.approx(0.1992, abs=1e-3)
