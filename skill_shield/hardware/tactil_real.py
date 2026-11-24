"""
Sensores táctiles FÍSICOS de la OrcaHand Touch, vistos desde el programa principal.

Lanza puente_tactil.py (proceso aparte, con el orca_core NUEVO: ver su cabecera) y recibe
por UDP local la fuerza de cada yema en newtons. No toca el bus de motores ni importa nada
del orca_core nuevo en este proceso.

    tactil = TactilReal()
    tactil.iniciar()                 # lanza el puente y escucha (no bloquea)
    tactil.fuerzas()                 # {"thumb": [fx, fy, fz], ...} N, o None si no hay datos frescos
    tactil.cero()                    # nuevo cero (la mano robótica no debe tocar nada)
    tactil.cerrar()
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time

import config as _cfg


def _p(nombre, defecto):
    return getattr(_cfg, nombre, defecto)


_RAIZ = os.path.dirname(os.path.abspath(_cfg.__file__))
ORCA_TACTIL_UDP        = str(_p("ORCA_TACTIL_UDP", "127.0.0.1:47311"))
ORCA_TACTIL_PUENTE     = str(_p("ORCA_TACTIL_PUENTE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "puente_tactil.py")))
ORCA_TACTIL_LANZAR     = bool(_p("ORCA_TACTIL_LANZAR_PUENTE", True))
ORCA_TACTIL_MAX_EDAD_S = float(_p("ORCA_TACTIL_MAX_EDAD_S", 0.15))   # datos más viejos = sin datos
ORCA_TACTIL_ARGS_PUENTE = list(_p("ORCA_TACTIL_ARGS_PUENTE", []))   # p. ej. ["--puerto", "/dev/ttyACM0"]
DEDOS = ("thumb", "index", "middle", "ring", "pinky")   # mismo orden que la vibración de MANUS


class TactilReal:
    def __init__(self, udp=ORCA_TACTIL_UDP, lanzar=ORCA_TACTIL_LANZAR, args_puente=None):
        host, puerto = udp.rsplit(":", 1)
        self.direccion = (host, int(puerto))
        self.lanzar = lanzar
        self.args_puente = list(ORCA_TACTIL_ARGS_PUENTE if args_puente is None else args_puente)
        self._sock = None
        self._proc = None
        self._hilo = None
        self._activo = False
        self._lock = threading.Lock()
        self._f = None                 # {dedo: [fx, fy, fz]}
        self._t = 0.0                  # perf_counter() de la última trama
        self._puente = None            # dirección del puente (para mandarle "cero")
        self.estado = "arrancando"
        self.msg = ""
        self.hz = 0.0
        self._n, self._t_hz = 0, time.perf_counter()

    # ── ciclo de vida ────────────────────────────────────────────────────────
    def iniciar(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(self.direccion)
            self._sock.settimeout(0.5)
        except OSError as e:
            self.estado, self.msg = "error", f"no puedo escuchar en {self.direccion}: {e}"
            print(f"❌ [TACTIL] {self.msg} (¿hay otro Skill Shield abierto?)")
            return False
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle, name="Tactil-UDP", daemon=True)
        self._hilo.start()
        if self.lanzar:
            if not os.path.exists(ORCA_TACTIL_PUENTE):
                self.estado, self.msg = "error", f"falta {ORCA_TACTIL_PUENTE}"
                print(f"❌ [TACTIL] {self.msg}")
                return False
            self._proc = subprocess.Popen(
                [sys.executable, "-u", ORCA_TACTIL_PUENTE, "--padre", str(os.getpid()),
                 "--udp", f"{self.direccion[0]}:{self.direccion[1]}", *self.args_puente],
                cwd=_RAIZ)
            print(f"🖐️  [TACTIL] Puente de sensores táctiles lanzado (PID {self._proc.pid})")
        return True

    def cerrar(self):
        self._activo = False
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # ── recepción ────────────────────────────────────────────────────────────
    def _bucle(self):
        while self._activo:
            try:
                datos, origen = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                d = json.loads(datos.decode())
            except ValueError:
                continue
            ahora = time.perf_counter()
            with self._lock:
                self._puente = origen
                if d.get("estado") == "ok" and isinstance(d.get("f"), dict):
                    if self._f is None:                       # primera trama: empieza la cuenta de Hz
                        self._n, self._t_hz = 0, ahora
                    self._f = {k: [float(x) for x in v] for k, v in d["f"].items()}
                    self._t = ahora - float(d.get("edad", 0.0))
                    self.estado = "ok"
                    self._n += 1
                    if ahora - self._t_hz >= 1.0:
                        self.hz = self._n / (ahora - self._t_hz)
                        self._n, self._t_hz = 0, ahora
                elif d.get("estado") in ("sin_sensor", "sin_core"):
                    self.estado, self.msg = d["estado"], d.get("msg", "")

    # ── lectura ──────────────────────────────────────────────────────────────
    def fuerzas(self, max_edad=ORCA_TACTIL_MAX_EDAD_S):
        """{dedo: [fx, fy, fz]} en N (fz = normal), o None si no hay datos frescos."""
        with self._lock:
            if self._f is None or time.perf_counter() - self._t > max_edad:
                return None
            return dict(self._f)

    def cero(self):
        """Pide al puente un nuevo cero (la mano robótica no debe tocar nada ~2 s)."""
        with self._lock:
            destino = self._puente
        if destino is None or self._sock is None:
            return False
        try:
            self._sock.sendto(b"cero", destino)
            return True
        except OSError:
            return False

    def estado_txt(self):
        if self.estado == "ok" and self.fuerzas() is not None:
            return f"TACTIL OK {self.hz:3.0f} Hz"
        if self.estado == "sin_core":
            return "TACTIL: falta orca_core_touch"
        if self.estado == "sin_sensor":
            return "TACTIL: sin sensores"
        return f"TACTIL: {self.estado}"

    def estado_corto(self):
        """Para el HUD del programa principal (sin 'OK' en mayúsculas: ese color es del guante)."""
        if self.estado == "ok" and self.fuerzas() is not None:
            return "T:ok"
        return {"sin_core": "T:falta core", "sin_sensor": "T:sin sensor",
                "error": "T:error"}.get(self.estado, "T:...")
