"""
Pruebas de la reconexión automática de fr5_servo (FR5_RECONECTAR, 13-nov) SIN robot: SDK de
FAIRINO falso + una «controladora» XML-RPC falsa en 127.0.0.1:20003 que se apaga y se enciende.

    python -m pytest tests/test_fr5_reconexion.py -q
"""
import os
import sys
import tempfile
import textwrap
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from skill_shield.hardware.fr5_servo import (  # noqa: E402
    ENVIANDO, ESPERA, RECONECTANDO, FR5Servo,
)

POSE = [10.0, -90.0, 90.0, -90.0, 90.0, 0.0]

# Robot falso: «apagado» mientras exista el archivo FR5_FALSO_APAGADO
_SDK = textwrap.dedent('''
    import os, time
    def _apagado():
        return os.path.exists(os.environ["FR5_FALSO_APAGADO"])
    class RPC:
        _reconnect_enable = True
        def __init__(self, ip):
            self.ip = ip
            self.robot = None
            self._com_err_flag = [0]
            self.ok = not _apagado()
        def SetReConnectParam(self, enable, maxRetries, period):
            RPC._reconnect_enable = enable; return 0
        def GetActualJointPosDegree(self, flag=0):
            return (0, %r) if self.ok and not _apagado() else (-4, [0.0] * 6)
        def GetRobotRealTimeState(self):
            class Pkg: pass
            pkg = Pkg(); pkg.jt_cur_pos = %r
            pkg.main_code = pkg.sub_code = pkg.collisionState = 0; pkg.robot_state = 1
            return 0, pkg                     # como el SDK real: la última pose aunque se apague
        def ServoMoveStart(self, cmdType=0):
            return 0 if not _apagado() else -4
        def ServoMoveEnd(self, cmdType=0):
            return 0
        def ServoJ(self, joint_pos, axisPos, acc=0.0, vel=0.0, cmdT=0.008, filterT=0.0,
                   gain=0.0, id=0, cmdType=0):
            if _apagado():
                raise RuntimeError("sin respuesta")
            return 0
        def CloseRPC(self):
            pass
''' % (POSE, POSE))


class _Controladora:
    """XML-RPC falso en el puerto de la controladora (solo para el latido), en un subproceso
    (si fuera un hilo de este proceso, el fork de fr5_servo heredaría el socket y el puerto no
    se liberaría al «apagarla»)."""

    _SRV = (
        "from xmlrpc.server import SimpleXMLRPCServer\n"
        "s = SimpleXMLRPCServer(('127.0.0.1', 20003), logRequests=False)\n"
        "s.register_function(lambda: [0, '127.0.0.1'], 'GetControllerIP')\n"
        "s.serve_forever()\n")

    def __init__(self):
        self.proc = None

    def encender(self):
        import socket
        import subprocess
        self.proc = subprocess.Popen([sys.executable, "-c", self._SRV])
        for _ in range(100):                           # hasta que escuche
            try:
                socket.create_connection(("127.0.0.1", 20003), 0.2).close()
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("la controladora falsa no arrancó")

    def apagar(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None


def _preparar(apagado):
    d = tempfile.mkdtemp(prefix="fairino_recon_")
    os.makedirs(os.path.join(d, "fairino"))
    open(os.path.join(d, "fairino", "__init__.py"), "w").close()
    with open(os.path.join(d, "fairino", "Robot.py"), "w") as f:
        f.write(_SDK)
    marca = os.path.join(d, "apagado")
    os.environ["FR5_FALSO_APAGADO"] = marca
    if apagado:
        open(marca, "w").close()
    fr5 = FR5Servo(FR5_SDK_RUTA=d, FR5_IP="127.0.0.1", FR5_SIMULACRO=False, FR5_PRIORIDAD_RT=0,
                   FR5_RECONECTAR=True, FR5_RECONEXION_S=0.5, FR5_PERDIDA_S=1.0,
                   FR5_TIMEOUT_CONEXION_S=1.0)
    return fr5, marca


def _esperar(cond, t_max):
    t0 = time.monotonic()
    while time.monotonic() - t0 < t_max:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_se_pierde_y_se_recupera_sin_reiniciar():
    ctl = _Controladora()
    ctl.encender()
    fr5, marca = _preparar(apagado=False)
    try:
        assert fr5.iniciar()
        assert fr5.q_real_rad() is not None
        # armado y enviando…
        fr5.publicar(fr5.q_real_rad())
        fr5.habilitar(True)
        assert _esperar(lambda: fr5.estado()["estado"] == ENVIANDO, 2.0)
        # …se apaga el FR5
        open(marca, "w").close()
        ctl.apagar()
        assert _esperar(lambda: fr5.estado()["estado"] == RECONECTANDO, 4.0)
        assert not fr5.conectado and not fr5.en_linea
        assert fr5.q_real_rad() is None                # nada de pose vieja como si fuera real
        assert "reintentando" in fr5.estado()["texto"]
        # se vuelve a encender: reconecta solo, DESARMADO
        os.remove(marca)
        ctl.encender()
        assert _esperar(lambda: fr5.conectado, 6.0)
        assert fr5.estado()["estado"] == ESPERA        # no vuelve a enviar por su cuenta
        assert _esperar(lambda: fr5.q_real_rad() is not None, 1.0)
        # y se puede volver a armar
        fr5.publicar(fr5.q_real_rad())
        fr5.habilitar(True)
        assert _esperar(lambda: fr5.estado()["estado"] == ENVIANDO, 2.0)
    finally:
        fr5.parar()
        ctl.apagar()


def test_apagado_al_arrancar_conecta_cuando_se_enciende():
    ctl = _Controladora()
    fr5, marca = _preparar(apagado=True)
    try:
        assert not fr5.iniciar()                       # no conecta, pero el proceso sigue vivo
        assert _esperar(lambda: fr5.estado()["estado"] == RECONECTANDO, 3.0)
        os.remove(marca)
        ctl.encender()
        assert _esperar(lambda: fr5.conectado, 6.0)
        assert fr5.estado()["estado"] == ESPERA
    finally:
        fr5.parar()
        ctl.apagar()


def test_proxy_seguro_no_deja_colgado_al_sdk():
    """El SDK hace `while True: try: robot.ServoJ(...) except socket.error: reintentar`. Con
    el FR5 apagado eso no termina nunca; con _ProxySeguro el fallo sale como FR5SinRespuesta."""
    import socket
    from skill_shield.hardware.fr5_servo import FR5SinRespuesta, _ProxySeguro
    px = _ProxySeguro("127.0.0.1", 0.5)            # nadie escuchando en el 20003
    intentos, t0 = 0, time.monotonic()
    try:
        while intentos < 1000:                     # el patrón del SDK (Robot.ServoJ, cmdType 0)
            try:
                px.ServoJ([0.0] * 6, [0.0] * 4, 0.0, 0.0, 0.008, 0.0, 0.0, 0)
                break
            except socket.error:
                intentos += 1
        raise AssertionError("no lanzó FR5SinRespuesta")
    except FR5SinRespuesta:
        pass
    assert intentos == 0 and time.monotonic() - t0 < 2.0


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_"):
            _f()
            print("✓", _n)
