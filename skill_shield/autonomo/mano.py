"""
Estado de la OrcaHand y de sus sensores táctiles para el modo autónomo (10-nov).

MonitorMano — SOLO LECTURA del bus de motores: abre el puerto serie de la mano y pregunta a
cada servo su temperatura, su error de hardware y si tiene el par activado. No escribe nada
y NO activa el par: la mano no se mueve. (Ni orca_core.connect() ni init_joints(): los dos
activan el par y el segundo además lleva la mano a la posición neutra.)

Mientras el modo autónomo no mueva la mano, este monitor es el dueño del puerto. Cuando
la mueva (paso 7) se sustituye por OrcaHardwareWrapper, que lee lo mismo con su candado.

Los táctiles van con TactilReal (skill_shield/hardware/tactil_real.py), que lanza
puente_tactil.py y recibe las fuerzas por UDP; aquí solo se resume su estado.
"""
import os
import threading
import time

import config as _cfg

# Direcciones de la tabla de control de los Dynamixel X (protocolo 2.0)
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_PRESENT_TEMPERATURE = 146

DEDOS_FISICOS = ("pulgar", "indice", "corazon", "anular", "menique")
NOMBRE_DEDO = {"pulgar": "Pulgar", "indice": "Índice", "corazon": "Corazón",
               "anular": "Anular", "menique": "Meñique"}
_SENSOR_DE_DEDO_DEFECTO = {"pulgar": "index", "indice": "ring", "corazon": "thumb",
                           "anular": "middle", "menique": "pinky"}


def _config_mano():
    """(puerto, baudios, ids de motor) del config.yaml de la OrcaHand (el que usa la teleop)."""
    from orca_hardware import _CONFIG_PATH
    import yaml
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    return str(c["port"]), int(c.get("baudrate", 1000000)), [int(m) for m in c["motor_ids"]]


class _BusDynamixel:
    """Lo mínimo de dynamixel_sdk para leer registros (se puede sustituir en las pruebas)."""

    def __init__(self, puerto, baudios):
        import dynamixel_sdk as dxl
        self._dxl = dxl
        self._port = dxl.PortHandler(puerto)
        self._pk = dxl.PacketHandler(2.0)
        if not self._port.openPort():
            raise OSError(f"no se puede abrir {puerto}")
        if not self._port.setBaudRate(baudios):
            self._port.closePort()
            raise OSError(f"no se puede fijar {baudios} bps en {puerto}")

    def leer1(self, mid, addr):
        """Valor de 1 byte o None si el motor no responde."""
        v, com, _err = self._pk.read1ByteTxRx(self._port, mid, addr)
        return int(v) if com == self._dxl.COMM_SUCCESS else None

    def cerrar(self):
        try:
            self._port.closePort()
        except Exception:
            pass


class MonitorMano:
    """Hilo que vigila la OrcaHand sin moverla. estado() → dict para la web."""

    def __init__(self, periodo_s=0.5, reintento_s=2.0, fabrica_bus=None, config_mano=None):
        try:
            self.puerto, self.baudios, self.ids = config_mano or _config_mano()
            self._error_cfg = None
        except Exception as e:
            self.puerto, self.baudios, self.ids = getattr(_cfg, "PUERTO_HARDWARE", "?"), 1000000, []
            self._error_cfg = f"no se pudo leer el config.yaml de la mano: {e}"
        self.periodo_s, self.reintento_s = float(periodo_s), float(reintento_s)
        self._fabrica = fabrica_bus or _BusDynamixel
        self._bus = None
        self._lock = threading.Lock()
        self._estado = dict(estado="arrancando", texto="buscando la mano…", motores_ok=0,
                            motores_total=len(self.ids), puerto=self.puerto)
        self._activo = False
        self._hilo = None
        self._fallos = 0

    def iniciar(self):
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle, name="MonitorMano", daemon=True)
        self._hilo.start()
        return self

    def cerrar(self):
        self._activo = False
        if self._hilo is not None:
            self._hilo.join(timeout=2.0)
        self._soltar_bus()

    def estado(self):
        with self._lock:
            return dict(self._estado)

    # ── interno ──
    def _poner(self, **kw):
        kw.setdefault("motores_total", len(self.ids))
        kw.setdefault("puerto", self.puerto)
        with self._lock:
            self._estado = kw

    def _soltar_bus(self):
        if self._bus is not None:
            self._bus.cerrar()
            self._bus = None

    def _bucle(self):
        while self._activo:
            if self._error_cfg:
                self._poner(estado="error", texto=self._error_cfg, motores_ok=0)
                time.sleep(self.reintento_s)
                continue
            if self._bus is None:
                if not os.path.exists(self.puerto):
                    self._poner(estado="sin_usb", texto=f"sin USB ({self.puerto} no existe)", motores_ok=0)
                    time.sleep(self.reintento_s)
                    continue
                try:
                    self._bus = self._fabrica(self.puerto, self.baudios)
                except ImportError:
                    self._poner(estado="error", texto="falta dynamixel_sdk en este entorno", motores_ok=0)
                    time.sleep(self.reintento_s * 5)
                    continue
                except Exception as e:
                    self._poner(estado="error", texto=f"no se puede abrir el puerto: {e}", motores_ok=0)
                    time.sleep(self.reintento_s)
                    continue
            try:
                self._leer_motores()
            except Exception as e:                       # p. ej. cable sacado a mitad de lectura
                self._soltar_bus()
                self._poner(estado="error", texto=f"error leyendo el bus: {e}", motores_ok=0)
                time.sleep(self.reintento_s)
                continue
            time.sleep(self.periodo_s)

    def _leer_motores(self):
        temps, err_hw, par, mudos = [], [], 0, []
        for mid in self.ids:
            t = self._bus.leer1(mid, ADDR_PRESENT_TEMPERATURE)
            if t is None:
                mudos.append(mid)
                continue
            temps.append(t)
            e = self._bus.leer1(mid, ADDR_HARDWARE_ERROR)
            if e:
                err_hw.append(mid)
            if self._bus.leer1(mid, ADDR_TORQUE_ENABLE) == 1:
                par += 1
        n_ok, n = len(temps), len(self.ids)
        base = dict(motores_ok=n_ok, temp_max=max(temps) if temps else None, par_activo=par,
                    errores_hw=err_hw, sin_respuesta=mudos)
        if n_ok == 0:
            self._fallos += 1
            if self._fallos >= 3:                       # 3 vueltas seguidas sin nadie
                self._soltar_bus()                      # se reabre en la siguiente vuelta (cable
                if os.path.exists(self.puerto):         # sacado y vuelto a meter, etc.)
                    self._poner(estado="sin_motores",
                                texto="USB conectado, los motores no responden (¿sin alimentación?)", **base)
                else:
                    self._poner(estado="sin_usb", texto=f"sin USB ({self.puerto} no existe)", **base)
            return
        self._fallos = 0
        if err_hw:
            self._poner(estado="aviso", texto=f"error de hardware en los motores {err_hw}", **base)
        elif n_ok < n:
            self._poner(estado="aviso", texto=f"{n_ok}/{n} motores responden (faltan {mudos})", **base)
        else:
            self._poner(estado="ok", texto=f"conectada · {n}/{n} motores", **base)


def estado_tactil(tactil):
    """Resumen de TactilReal para la web: estado, Hz y fuerza normal (N) por dedo FÍSICO."""
    if tactil is None:
        return dict(estado="desactivado", texto="táctiles desactivados")
    f = tactil.fuerzas()
    mapa = dict(getattr(_cfg, "ORCA_TACTIL_SENSOR_DE_DEDO", _SENSOR_DE_DEDO_DEFECTO))
    if tactil.estado == "ok" and f is not None:
        dedos = []
        for d in DEDOS_FISICOS:
            v = f.get(mapa.get(d, ""))
            fz = abs(float(v[2])) if v is not None and len(v) >= 3 else None
            dedos.append(dict(dedo=NOMBRE_DEDO[d], fz=round(fz, 2) if fz is not None else None))
        return dict(estado="ok", texto=f"recibiendo · {tactil.hz:.0f} Hz", hz=round(tactil.hz, 1), dedos=dedos)
    texto = {"sin_core": "falta orca_core_touch", "sin_sensor": "sin sensores (¿USB de los táctiles?)",
             "error": tactil.msg or "error", "arrancando": "arrancando el puente…"}.get(tactil.estado,
                                                                                    tactil.estado)
    if tactil.estado == "ok":
        texto = "sin datos recientes"
    return dict(estado="error" if tactil.estado in ("error", "sin_core", "sin_sensor") else "aviso",
                texto=texto)
