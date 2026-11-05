"""
Pruebas del armado con pedal y la entrada lenta del FR5 (5-nov), sin hardware ni MuJoCo:
calibrar no mueve el robot, sin armar no sale ningún comando y, al armar, cada joint entra
como mucho a v_entrada hasta alcanzar tu postura.

    python -m pytest tests_armado_fr5.py -q          # o simplemente:  python tests_armado_fr5.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skill_shield.teleop_fr5.diagnostico import TrayectoriaLimpia, construir_teleop
from skill_shield.teleop_fr5.pedal import Pedal

DT = 0.1            # el bucle real va a ~10 FPS
V_ENTRADA = 0.15


def _teleop(requiere_armado=True, v_entrada=V_ENTRADA):
    teleop, q0, _ = construir_teleop(usar_config=True)
    teleop.cfg.requiere_armado = requiere_armado
    teleop.cfg.v_entrada = v_entrada
    teleop.ik.q = None
    return teleop, q0


def _medida(tray, t=0.0):
    S, E, W, _ = tray.pose(t)
    f_h = (W - E) / np.linalg.norm(W - E)
    l_h = np.cross(f_h, [0.0, -1.0, 0.0])
    l_h /= np.linalg.norm(l_h)
    res = dict(hombro=S, codo=E, muneca=W, torso=tray.torso, codo_reconstruido=False)
    return res, (f_h, l_h)


def _paso(teleop, q_robot, t, tray):
    res, mano = _medida(tray)              # postura humana quieta
    return teleop.actualizar_desde_medida(res, t, mano=mano, q_actual=q_robot, dt=DT)


def test_sin_armar_no_hay_comando():
    teleop, q0 = _teleop()
    tray = TrayectoriaLimpia()
    for k in range(20):
        q, info = _paso(teleop, q0, k * DT, tray)
        assert q is None and info["estado"] == "desarmado"


def test_al_armar_entra_despacio_y_luego_sigue():
    teleop, q0 = _teleop()
    tray = TrayectoriaLimpia()
    for k in range(5):
        _paso(teleop, q0, k * DT, tray)
    teleop.armar()
    q_robot = q0.copy()
    entradas, k = 0, 5
    while k < 2000:
        q, info = _paso(teleop, q_robot, k * DT, tray)
        assert q is not None
        dq = np.abs(np.asarray(q) - q_robot).max()
        if info["entrada"]:
            entradas += 1
            assert dq <= V_ENTRADA * DT + 1e-9, dq      # nunca más rápido que v_entrada
        else:
            break
        q_robot = np.asarray(q, dtype=float)
        k += 1
    assert entradas > 10                                 # la postura estaba lejos: hubo entrada
    assert not info["entrada"]                           # y terminó (alcanzó tu postura)
    # ya siguiendo: el siguiente comando está cerca del último (no hay salto al acabar)
    assert np.abs(np.asarray(q) - q_robot).max() <= teleop.cfg.entrada_fin_rad + 1e-9


def test_desarmar_para_y_al_rearmar_vuelve_a_entrar_despacio():
    teleop, q0 = _teleop()
    tray = TrayectoriaLimpia()
    teleop.armar()
    q_robot = q0.copy()
    for k in range(20):
        q, info = _paso(teleop, q_robot, k * DT, tray)
        q_robot = np.asarray(q, dtype=float)
    teleop.desarmar()
    q, info = _paso(teleop, q_robot, 20 * DT, tray)
    assert q is None and info["estado"] == "desarmado"
    teleop.armar()
    q, info = _paso(teleop, q_robot, 21 * DT, tray)
    assert info["entrada"]
    assert np.abs(np.asarray(q) - q_robot).max() <= V_ENTRADA * DT + 1e-9


def test_calibrar_desarma():
    teleop, q0 = _teleop()
    teleop.armar()
    teleop.iniciar_calibracion()
    assert not teleop.armado


def test_sin_pedal_tambien_entra_despacio():
    teleop, q0 = _teleop(requiere_armado=False)
    tray = TrayectoriaLimpia()
    q, info = _paso(teleop, q0, 0.0, tray)
    assert info["entrada"]
    assert np.abs(np.asarray(q) - q0).max() <= V_ENTRADA * DT + 1e-9


def test_pedal_tecla_alterna():
    p = Pedal(modo="tecla", teclas=(ord("p"),))
    assert not p.armado
    assert p.tecla(ord("x")) is None
    assert p.tecla(-1) is None
    assert p.tecla(ord("p")) == "armar" and p.armado
    assert p.tecla(ord("p")) == "desarmar" and not p.armado


class _Ev:
    def __init__(self, code, value):
        self.type, self.code, self.value = 1, code, value


class _DevFalso:
    path = "/dev/input/falso"

    def __init__(self):
        self.cola = []

    def read(self):
        if not self.cola:
            raise BlockingIOError
        ev, self.cola = self.cola, []
        return ev


def test_pedal_evdev_hombre_muerto():
    p = Pedal(modo="tecla")                      # se construye sin evdev y se cambia a mano
    p.modo, p.hombre_muerto, p._dev = "evdev", True, _DevFalso()
    assert p.actualizar() is None and not p.armado
    p._dev.cola = [_Ev(30, 1)]
    assert p.actualizar() == "armar" and p.armado
    p._dev.cola = [_Ev(30, 2)]                   # repetición: sigue armado
    assert p.actualizar() is None and p.armado
    p._dev.cola = [_Ev(30, 0)]                   # soltar → quieto
    assert p.actualizar() == "desarmar" and not p.armado


def test_pedal_evdev_desconectado_desarma():
    class _Roto(_DevFalso):
        def read(self):
            raise OSError("No such device")
    p = Pedal(modo="tecla")
    p.modo, p.hombre_muerto, p._dev = "evdev", True, _DevFalso()
    p._dev.cola = [_Ev(30, 1)]
    p.actualizar()
    assert p.armado
    p._dev = _Roto()
    assert p.actualizar() == "desarmar" and not p.armado


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_") and callable(f):
            f()
            print(f"✓ {nombre}")
