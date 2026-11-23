"""
(23-nov) diagnosticos/medir_punto_pinza.py con poses SINTÉTICAS: una pelota fija y la mano real = la del gemelo girada
-51° en la brida (lo que dicen las calibraciones); el script tiene que encontrar ese punto de pinza y ese giro.
"""
import math
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "diagnosticos"))

import config                                                        # noqa: E402
import medir_punto_pinza as MPP                                      # noqa: E402
from skill_shield.teleop_fr5.cinematica import cargar_cadena         # noqa: E402


def _rot(eje, ang):
    x, y, z = np.asarray(eje, float) / np.linalg.norm(eje)
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * K @ K


@pytest.fixture(scope="module")
def datos():
    cad = cargar_cadena(config.RUTA_XML, verbose=False)
    _, p_mod = MPP.postura_contacto(config, 0.070, config.RUTA_XML)
    p_real = _rot([0, 0, 1], math.radians(-51.0)) @ p_mod + np.array([0.002, -0.001, 0.003])
    q_ref = np.radians([70.0, -33.0, 71.0, -17.0, 92.0, -138.0])
    T5 = cad.fk_marcos(q_ref)[5]
    bola = T5[:3, :3] @ p_real + T5[:3, 3]                         # la pelota fija, donde cae el punto de pinza real
    m0 = cad.fk_marcos(np.zeros(6))
    T_w3_mm = np.linalg.inv(m0[5]) @ m0[6]
    rng = np.random.default_rng(0)
    poses = []
    for eje, g in (((0, 0, 1), 0), ((0, 0, 1), 35), ((0, 0, 1), -35), ((1, 0, 0), 20), ((0, 1, 0), -20)):
        R = T5[:3, :3] @ _rot(eje, math.radians(g))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, bola - R @ p_real
        sols = cad.ik_analitica(T @ T_w3_mm, q_ref)
        assert sols, "sin IK para una pose de prueba"
        q = min(sols, key=lambda s: np.linalg.norm(s - q_ref))
        poses.append(np.degrees(q) + rng.normal(0.0, 0.05, 6))     # lectura de la controladora con algo de ruido
    return cad, p_mod, p_real, bola, np.array(poses)


def test_leer_poses(tmp_path):
    ruta = tmp_path / "poses.txt"
    ruta.write_text("# cabecera\npose1: 120.0, -43.0, 65.0, -27.0, 77.0, 158.0    # FR5 real\n"
                    "100, -49.5, 60, -38, 83, 170\nbasura\n", encoding="utf-8")
    Q = MPP.leer_poses(str(ruta))
    assert Q.shape == (2, 6) and Q[0, 0] == 120.0 and Q[1, 1] == -49.5


def test_encuentra_el_punto_de_pinza_y_el_giro(datos):
    cad, p_mod, p_real, bola, poses = datos
    Ts = [cad.fk_marcos(MPP.a_mjcf(q, config))[5] for q in poses]
    p, b, res, cond = MPP.resolver(Ts)
    assert np.linalg.norm(p - p_real) < 0.004 and np.linalg.norm(b - bola) < 0.004
    assert res.max() < 0.004 and cond > 1e-3
    th = math.degrees(math.atan2(p[1], p[0]) - math.atan2(p_mod[1], p_mod[0]))
    assert (th + 180.0) % 360.0 - 180.0 == pytest.approx(-51.0, abs=3.0)


def test_linea_de_mano_mount_gira_alrededor_del_eje_de_la_brida():
    from skill_shield.autonomo.choque_mano import _T_local
    import xml.etree.ElementTree as ET
    el = next(b for b in ET.parse(config.RUTA_XML).getroot().iter("body") if b.get("name") == "mano_mount")
    T0 = _T_local(el)
    linea = MPP.linea_mano_mount(config.RUTA_XML, 30.0)
    pos = np.array([float(v) for v in linea.split('pos="')[1].split('"')[0].split()])
    quat = [float(v) for v in linea.split('quat="')[1].split('"')[0].split()]
    from skill_shield.autonomo.choque_mano import _quat
    Rz = _rot([0, 0, 1], math.radians(30.0))
    assert np.allclose(pos, Rz @ T0[:3, 3], atol=1e-5)
    assert np.allclose(_quat(quat), Rz @ T0[:3, :3], atol=1e-5)
