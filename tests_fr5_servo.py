"""
Pruebas del proceso fr5_servo (skill_shield/hardware/fr5_servo.py) SIN robot: se usa un
SDK de FAIRINO falso que apunta cada llamada.

    python -m pytest tests_fr5_servo.py -q          # o simplemente:  python tests_fr5_servo.py
"""
import json
import math
import os
import sys
import tempfile
import textwrap
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skill_shield.hardware.fr5_servo import (
    ENVIANDO, ERROR, ESPERA, FR5Servo, paso_limitado,
)

POSE_REAL = [10.0, -90.0, 90.0, -90.0, 90.0, 0.0]   # grados, la que "tiene" el robot falso

_SDK_FALSO = textwrap.dedent('''
    import json, os, time
    class RPC:
        def __init__(self, ip):
            self.ip = ip
            self.llamadas = []
        def GetActualJointPosDegree(self, flag=0):
            return 0, %r
        def ServoMoveStart(self, cmdType=0):
            self.llamadas.append(("start", time.perf_counter())); return 0
        def ServoMoveEnd(self, cmdType=0):
            self.llamadas.append(("end", time.perf_counter())); return 0
        def ServoJ(self, joint_pos, axisPos, acc=0.0, vel=0.0, cmdT=0.008, filterT=0.0,
                   gain=0.0, id=0, cmdType=0):
            time.sleep(float(os.environ.get("FR5_FALSO_RETARDO", "0.001")))   # ~latencia RPC
            self.llamadas.append(("servoj", time.perf_counter(), list(joint_pos), cmdT, id))
            return 0
        def CloseRPC(self):
            with open(os.environ["FR5_FALSO_LOG"], "w") as f:
                json.dump(self.llamadas, f)
''' % (POSE_REAL,))


def _sdk_falso():
    d = tempfile.mkdtemp(prefix="fairino_falso_")
    os.makedirs(os.path.join(d, "fairino"))
    open(os.path.join(d, "fairino", "__init__.py"), "w").close()
    with open(os.path.join(d, "fairino", "Robot.py"), "w") as f:
        f.write(_SDK_FALSO)
    log = os.path.join(d, "llamadas.json")
    os.environ["FR5_FALSO_LOG"] = log
    return d, log


def _servo(**kw):
    ruta, log = _sdk_falso()
    par = dict(FR5_SDK_RUTA=ruta, FR5_SIMULACRO=False, FR5_PRIORIDAD_RT=0)
    par.update(kw)
    return FR5Servo(**par), log


def test_paso_limitado_no_se_pasa():
    q, v = np.zeros(6), np.zeros(6)
    obj = np.array([30.0, -30.0, 5.0, 0.0, 90.0, -1.0])
    vmax, amax, dt = 60.0, 300.0, 0.008
    for _ in range(1000):
        q_ant, v_ant = q, v
        q, v = paso_limitado(q, v, obj, vmax, amax, dt)
        assert np.all(np.abs(v) <= vmax + 1e-9)
        assert np.all(np.abs(v - v_ant) <= amax * dt + 1e-9)
        assert np.all(np.sign(obj - q_ant) * (q - obj) <= 1e-6)    # nunca rebasa el objetivo
    assert np.allclose(q, obj, atol=1e-3)


def test_envia_servoj_cada_8_ms_desde_la_pose_real():
    fr5, log = _servo()
    assert fr5.iniciar()
    assert fr5.estado()["estado"] == ESPERA
    q_sim = np.radians(np.array(POSE_REAL) + [8.0, 0, 0, 0, 0, 0])   # 8° de desfase en joint1
    fr5.publicar(q_sim)
    fr5.habilitar(True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        fr5.publicar(q_sim)
        time.sleep(0.002)
    assert fr5.estado()["estado"] == ENVIANDO
    fr5.habilitar(False)
    time.sleep(0.05)
    fr5.parar()

    llamadas = json.load(open(log))
    tipos = [c[0] for c in llamadas]
    assert tipos[0] == "start" and tipos[-1] == "end"
    sj = [c for c in llamadas if c[0] == "servoj"]
    assert 110 <= len(sj) <= 130, len(sj)                         # 1 s / 8 ms = 125
    per = np.diff([c[1] for c in sj]) * 1000
    assert abs(np.median(per) - 8.0) < 0.3, np.median(per)
    assert np.percentile(per, 95) < 10.0
    assert all(abs(c[3] - 0.008) < 1e-12 for c in sj)              # cmdT = 8 ms
    assert np.allclose(sj[0][2], POSE_REAL, atol=0.1)              # arranca donde está el robot
    assert abs(sj[-1][2][0] - (POSE_REAL[0] + 8.0)) < 0.05         # y llega al objetivo
    pasos = np.abs(np.diff([c[2][0] for c in sj]))
    assert pasos[:30].max() <= 10.0 * 0.008 + 1e-6                 # acercamiento a 10 °/s
    assert pasos.max() <= 60.0 * 0.008 + 1e-6                      # luego, tope normal


def test_watchdog_mantiene_la_pose():
    fr5, log = _servo(FR5_WATCHDOG_S=0.05)
    assert fr5.iniciar()
    fr5.habilitar(True)                     # nunca se publica nada → debe quedarse quieto
    time.sleep(0.3)
    fr5.habilitar(False)
    time.sleep(0.05)
    fr5.parar()
    sj = [c for c in json.load(open(log)) if c[0] == "servoj"]
    assert len(sj) > 20
    assert all(np.allclose(c[2], POSE_REAL) for c in sj)


def test_limites_del_fr5():
    fr5, log = _servo(FR5_VEL_APROX_DEG_S=1000.0, FR5_VEL_MAX_DEG_S=1000.0,
                      FR5_ACEL_MAX_DEG_S2=1e6)
    assert fr5.iniciar()
    q_fuera = np.radians([400.0, -400.0, 400.0, 400.0, -400.0, 400.0])
    fr5.publicar(q_fuera)
    fr5.habilitar(True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.5:
        fr5.publicar(q_fuera)
        time.sleep(0.005)
    fr5.habilitar(False)
    time.sleep(0.05)
    fr5.parar()
    ult = [c for c in json.load(open(log)) if c[0] == "servoj"][-1][2]
    assert np.allclose(ult, [172.0, -262.0, 157.0, 82.0, -172.0, 172.0], atol=1e-6)


def test_simulacro_no_envia():
    fr5, log = _servo(FR5_SIMULACRO=True)
    assert fr5.iniciar()
    fr5.publicar(np.radians(POSE_REAL))
    fr5.habilitar(True)
    time.sleep(0.2)
    assert fr5.estado()["ciclos"] > 15
    fr5.parar()
    assert json.load(open(log)) == []


def test_sin_sdk_da_error_y_no_se_cuelga():
    fr5 = FR5Servo(FR5_SDK_RUTA="/no/existe", FR5_PRIORIDAD_RT=0)
    sys_mod = sys.modules.pop("fairino", None)
    try:
        t0 = time.monotonic()
        assert not fr5.iniciar()
        assert time.monotonic() - t0 < 2.0
        assert fr5.estado()["estado"] == ERROR
    finally:
        fr5.parar()
        if sys_mod is not None:
            sys.modules["fairino"] = sys_mod


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_") and callable(f):
            f()
            print(f"✓ {nombre}")
