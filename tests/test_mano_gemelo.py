"""
Pruebas de la postura real de la OrcaHand en el modo autónomo (11-nov), sin mano, sin MuJoCo y
sin cámara:

    python -m pytest tests/test_mano_gemelo.py -q

  · TraductorMano: motor → ángulo del sim es la conversión de la teleoperación al revés
    (ida y vuelta con una calibración inventada, incluida una vuelta del encoder de la muñeca).
  · MonitorMano: con un bus falso publica la postura y la enseña en la web.
  · PiezasMano: las cápsulas envuelven las cajas de todas las piezas, giradas como sea.
  · Canal «qbrazo»: ida y vuelta, y sigue leyendo el mensaje antiguo de 7 números.
  · Mapa: la nube de la mano no deja cubos si se pasan sus piezas; un objeto al lado, sí.
"""
import math
import os
import shutil
import sys
import time
import types

import numpy as np
import pytest
import yaml

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del repo (tests/..)
sys.path.insert(0, RAIZ)

if "mujoco" not in sys.modules:                                    # PiezasMano solo usa mj_name2id
    _mj = types.ModuleType("mujoco")
    _mj.mjtObj = types.SimpleNamespace(mjOBJ_BODY=1)
    _mj.mj_name2id = lambda m, t, n: m.nombres.index(n) if n in m.nombres else -1
    sys.modules["mujoco"] = _mj

from orca_hardware import _CONFIG_PATH                             # noqa: E402
from skill_shield.autonomo.mano import ESCALA_POS, MonitorMano, TraductorMano   # noqa: E402
from skill_shield.autonomo.mapa import (MapaOcupacion, PiezasMano, desempaquetar_pose,   # noqa: E402
                                        empaquetar_pose)


# ── mano con una calibración inventada ────────────────────────────────────────────────────────
@pytest.fixture
def config_mano(tmp_path):
    shutil.copy(_CONFIG_PATH, tmp_path / "config.yaml")
    c = yaml.safe_load(open(_CONFIG_PATH, encoding="utf-8"))
    lim, rat = {}, {}
    for j, m in c["joint_to_motor_map"].items():
        m = abs(int(m))
        lo, hi = c["joint_roms"][j]
        rat[m] = 0.03                                               # rad de motor por grado
        lim[m] = [0.3 * m, 0.3 * m + 0.03 * (hi - lo)]
    lim[1] = [lim[1][0] + 6.0, lim[1][1] + 6.0]                     # muñeca multivuelta
    with open(tmp_path / "calibration.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(calibrated=True, wrist_calibrated=True, motor_limits=lim,
                            joint_to_motor_ratios=rat), f)
    return str(tmp_path / "config.yaml")


def _pasos_de(tr, q_sim, vuelta_muneca=False):
    """Lo que haría la teleoperación: q del sim → grados de orca_core → motor → pasos."""
    grados = tr._w._a_sdk(q_sim)
    tr._hand._wrap_offsets_dict = {m: 0.0 for m in tr.ids}
    mot = tr._hand._joint_to_motor_pos(grados)
    if vuelta_muneca:
        mot[0] -= 2 * np.pi
    return {m: int(round(p / ESCALA_POS)) for m, p in zip(tr.ids, mot)}


def _q_al_azar(tr, rng):
    g = {}
    for j in tr.joints:
        lo, hi = tr._w._rango_sdk[j]
        g[j] = rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo))
    return tr._w._desde_sdk(g)


def test_traductor_es_la_teleoperacion_al_reves(config_mano):
    tr = TraductorMano(config_mano)
    assert tr.calibrada and len(tr.joints) == 17
    rng = np.random.default_rng(1)
    for i in range(20):
        q = _q_al_azar(tr, rng)
        tr.reiniciar()
        back = tr.a_sim(_pasos_de(tr, q, vuelta_muneca=(i == 0)))
        for k, j in enumerate(tr.joints):
            assert abs(back[j] - q[k]) < math.radians(0.1), j


def test_traductor_sin_todos_los_motores_no_inventa(config_mano):
    tr = TraductorMano(config_mano)
    assert tr.a_sim({m: 0 for m in tr.ids[1:]}) is None


class _BusFalso:
    pasos = {}

    def __init__(self, *_):
        pass

    def leer1(self, mid, addr):
        return {146: 35, 70: 0, 64: 0}[addr]

    def leer_posiciones(self, ids):
        return dict(self.pasos)

    def cerrar(self):
        pass


def test_monitor_publica_la_postura(config_mano, tmp_path):
    tr = TraductorMano(config_mano)
    q = _q_al_azar(tr, np.random.default_rng(2))
    _BusFalso.pasos = _pasos_de(tr, q)
    puerto = tmp_path / "ttyFALSO"
    puerto.touch()
    mon = MonitorMano(fabrica_bus=_BusFalso, config_mano=(str(puerto), 1000000, tr.ids),
                      periodo_postura_s=0.05, fabrica_traductor=lambda: TraductorMano(config_mano)).iniciar()
    try:
        t0 = time.monotonic()
        while mon.postura() is None and time.monotonic() - t0 < 3.0:
            time.sleep(0.05)
        p = mon.postura()
        assert p is not None and abs(p["wrist"] - q[tr.joints.index("wrist")]) < 1e-3
        assert "postura en el gemelo" in mon.estado()["texto"]
    finally:
        mon.cerrar()


# ── piezas de la mano ─────────────────────────────────────────────────────────────────────────
def _modelo_cajas(n=25, semilla=0):
    """Modelo/datos falsos: n cajas (geoms) colgando de mano_mount, cada una con su marco."""
    rng = np.random.default_rng(semilla)
    nombres = ["world", "mano_mount", "otro"] + [f"b{i}" for i in range(n)]
    padre = [0, 0, 0] + [1] * n
    aabb, xpos, xmat, gb = [], [], [], []
    for i in range(n):
        aabb.append(np.r_[rng.uniform(-0.01, 0.01, 3), rng.uniform(0.003, 0.04, 3)])
        xpos.append(rng.uniform(-0.2, 0.2, 3))
        xmat.append(np.linalg.qr(rng.normal(size=(3, 3)))[0].ravel())
        gb.append(3 + i)
    aabb.append(np.r_[0, 0, 0, 0.5, 0.5, 0.5])                      # geom de otro cuerpo: fuera
    xpos.append(np.zeros(3))
    xmat.append(np.eye(3).ravel())
    gb.append(2)
    m = types.SimpleNamespace(nombres=nombres, nbody=len(nombres), body_parentid=np.array(padre),
                              ngeom=n + 1, geom_bodyid=np.array(gb), geom_aabb=np.array(aabb),
                              geom_rbound=np.ones(n + 1))
    d = types.SimpleNamespace(geom_xpos=np.array(xpos), geom_xmat=np.array(xmat))
    return m, d


def _esquinas(m, d, g):
    c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return (c + s * h) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def test_piezas_envuelven_cada_caja():
    m, d = _modelo_cajas()
    pm = PiezasMano(m)
    assert len(pm) == m.ngeom - 1                                   # sin el geom del otro cuerpo
    for k, (g, cap) in enumerate(zip(pm.geoms, pm.capsulas(d))):
        a, b, r = cap[:3], cap[3:6], cap[6]
        for p in _esquinas(m, d, g):
            t = np.clip((p - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0, 1)
            assert np.linalg.norm(p - (a + t * (b - a))) <= r + 1e-9


# ── canal ─────────────────────────────────────────────────────────────────────────────────────
def test_canal_pose_ida_y_vuelta():
    caps = np.random.default_rng(3).normal(size=(30, 7))
    q, real, c = desempaquetar_pose(empaquetar_pose(np.arange(6.0), True, caps))
    assert real and np.allclose(q, np.arange(6)) and np.allclose(c, caps)
    q, real, c = desempaquetar_pose(empaquetar_pose(np.arange(6.0), False))
    assert not real and c is None
    q, real, c = desempaquetar_pose(np.r_[np.arange(6.0), 1.0].tobytes())   # mensaje antiguo
    assert real and c is None


# ── mapa ──────────────────────────────────────────────────────────────────────────────────────
def test_mapa_quita_la_mano_y_deja_lo_demas():
    m, d = _modelo_cajas()
    d.geom_xpos = d.geom_xpos + np.array([0.0, 0.0, 0.8])          # la mano a 0,8 m de la cámara
    pm = PiezasMano(m)
    caps = pm.capsulas(d)
    K = np.array([[450.0, 0, 320], [0, 450.0, 240], [0, 0, 1]])
    prof = np.zeros((480, 640), np.uint16)
    pts = np.vstack([_esquinas(m, d, g) for g in pm.geoms]
                    + [np.array([0.0, 0.0, 0.8]) + np.random.default_rng(4).uniform(-0.15, 0.15, (4000, 3))
                       * np.array([1, 1, 0])])                      # «palma»: plano a 0,8 m
    caja = np.array([0.35, 0.0, 0.8]) + np.random.default_rng(5).uniform(-0.03, 0.03, (500, 3))
    for P in (pts, caja):
        u = np.round(P[:, 0] / P[:, 2] * 450 + 320).astype(int)
        v = np.round(P[:, 1] / P[:, 2] * 450 + 240).astype(int)
        ok = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
        prof[v[ok], u[ok]] = (P[ok, 2] * 1000).astype(np.uint16)
    # «palma» dentro de las piezas: se añade como pieza más (plano fino)
    palma = np.array([[-0.15, 0, 0.8, 0.15, 0, 0.8, 0.16]])
    caps = np.vstack([caps, palma])
    lim = ((-0.6, 0.6), (-0.6, 0.6), (0.4, 1.2))
    import cv2
    prof = cv2.dilate(prof, np.ones((5, 5), np.uint8))           # superficies, no puntos sueltos
    # (aquí se prueba quitar la mano: sin los filtros de ruido de cubos sueltos)
    sin, con = (MapaOcupacion(lim, 0.02, min_puntos=1, min_grupo=1, z_min=0.2) for _ in range(2))
    for i in range(4):
        sin.actualizar(prof, K, np.eye(4), (), ahora=100 + i)
        con.actualizar(prof, K, np.eye(4), (), ahora=100 + i, capsulas_mano=caps)
    oc_sin, oc_con = sin.centros_ocupados(103), con.centros_ocupados(103)
    assert len(oc_sin) > len(oc_con)
    lejos = np.linalg.norm(oc_con - np.array([0.35, 0.0, 0.8]), axis=1) < 0.06
    assert lejos.all() and len(oc_con) > 0                          # solo queda la caja de al lado
