"""
Estado de la OrcaHand y de sus sensores táctiles para el modo autónomo (10-nov).

MonitorMano — SOLO LECTURA del bus de motores: abre el puerto serie de la mano y pregunta a
cada servo su temperatura, su error de hardware y si tiene el par activado. No escribe nada
y NO activa el par: la mano no se mueve. (Ni orca_core.connect() ni init_joints(): los dos
activan el par y el segundo además lleva la mano a la posición neutra.)

Mientras el modo autónomo no mueva la mano, este monitor es el dueño del puerto. Cuando
la mueva (paso 7) se sustituye por OrcaHardwareWrapper, que lee lo mismo con su candado.

11-nov: también lee la POSTURA (posición actual de cada motor, registro 132, solo lectura) y
la convierte en ángulos del simulador con TraductorMano —la conversión de la teleoperación al
revés—, para que el gemelo ponga la muñeca de la Orca y los dedos como están de verdad y el
mapa de obstáculos quite la mano real (antes salían cubos falsos sobre ella).
Ojo: sin par, los tendones quedan algo flojos y la postura medida puede diferir unos grados
de la de los dedos.

Los táctiles van con TactilReal (skill_shield/hardware/tactil_real.py), que lanza
puente_tactil.py y recibe las fuerzas por UDP; aquí solo se resume su estado.
"""
import os
import threading
import time

import numpy as np

import config as _cfg

# Direcciones de la tabla de control de los Dynamixel X (protocolo 2.0)
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_PRESENT_POSITION = 132        # 4 bytes con signo (multivuelta en la muñeca)
ADDR_PRESENT_TEMPERATURE = 146
ESCALA_POS = 2.0 * np.pi / 4096    # pasos → rad (lo mismo que orca_core)

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

    def leer_posiciones(self, ids):
        """{id: posición en pasos, con signo} de los motores que responden. Una sola trama
        (GroupSyncRead); si falla (p. ej. un motor no contesta), motor a motor."""
        dxl = self._dxl
        out = {}
        try:
            g = dxl.GroupSyncRead(self._port, self._pk, ADDR_PRESENT_POSITION, 4)
            for mid in ids:
                g.addParam(mid)
            if g.txRxPacket() == dxl.COMM_SUCCESS:
                for mid in ids:
                    if g.isAvailable(mid, ADDR_PRESENT_POSITION, 4):
                        out[mid] = _con_signo(g.getData(mid, ADDR_PRESENT_POSITION, 4))
                if len(out) == len(ids):
                    return out
        except Exception:
            pass
        for mid in ids:
            if mid in out:
                continue
            v, com, _err = self._pk.read4ByteTxRx(self._port, mid, ADDR_PRESENT_POSITION)
            if com == dxl.COMM_SUCCESS:
                out[mid] = _con_signo(v)
        return out

    def cerrar(self):
        try:
            self._port.closePort()
        except Exception:
            pass


def _con_signo(v):
    v = int(v) & 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


class TraductorMano:
    """Posición de los motores de la OrcaHand → ángulo de cada articulación en el simulador
    (rad, convención del MJCF). Es la conversión de la teleoperación al revés:
    orca_core (motor → grados, con su calibration.yaml) y OrcaHardwareWrapper._desde_sdk
    (grados de orca_core → sim, con los mismos signos y tramos). No abre el puerto."""

    def __init__(self, config_path=None):
        import contextlib
        import io
        from orca_hardware import OrcaHardwareWrapper, _CONFIG_PATH
        from orca_core import OrcaHand
        ruta = config_path or _CONFIG_PATH
        with contextlib.redirect_stdout(io.StringIO()):     # sus avisos ya los da la teleoperación
            self._hand = OrcaHand(config_path=ruta)
            self._w = OrcaHardwareWrapper(config_path=ruta)
            self._w._hand = self._hand
            self._w._preparar_traduccion()
            self.calibrada = bool(self._hand.is_calibrated())
        self.ids = [int(m) for m in self._hand.config.motor_ids]
        self.joints = list(self._w._joints)
        self._vuelta = None

    def reiniciar(self):
        """Tras reabrir el puerto: se vuelven a mirar las vueltas de los encoders."""
        self._vuelta = None

    def _offsets_vuelta(self, pos):
        """Lo mismo que OrcaHand._compute_wrap_offsets_dict (al conectar), con la lectura dada."""
        lim = self._hand.motor_limits_dict
        off = {}
        for mid, p in zip(self.ids, pos):
            lo, hi = lim[mid]
            if lo is None or hi is None:
                off[mid] = 0.0
            elif p < lo - 0.25 * np.pi:
                off[mid] = -2 * np.pi
            elif p > hi + 0.25 * np.pi:
                off[mid] = 2 * np.pi
            else:
                off[mid] = 0.0
        return off

    def a_sim(self, pasos):
        """pasos: {id: posición en pasos}. → {joint de orca_core: rad en el sim}, o None si
        falta algún motor o la mano no está calibrada."""
        if not self.calibrada or any(m not in pasos for m in self.ids):
            return None
        pos = np.array([pasos[m] for m in self.ids], float) * ESCALA_POS
        if self._vuelta is None:
            self._vuelta = self._offsets_vuelta(pos)
        self._hand._wrap_offsets_dict = self._vuelta
        grados = self._hand._motor_to_joint_pos(pos)
        q = self._w._desde_sdk(grados)
        if q is None or not np.all(np.isfinite(q)):
            return None
        return dict(zip(self.joints, (float(v) for v in q)))


class MonitorMano:
    """Hilo que vigila la OrcaHand sin moverla. estado() → dict para la web; postura() → ángulos
    reales de la mano (sim, rad) para el gemelo y el mapa."""

    def __init__(self, periodo_s=0.5, reintento_s=2.0, fabrica_bus=None, config_mano=None,
                 periodo_postura_s=None, fabrica_traductor=None):
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
        # postura (11-nov)
        if periodo_postura_s is None:
            periodo_postura_s = getattr(_cfg, "AUTONOMO_MANO_POSTURA_PERIODO_S", 0.1)
        self.periodo_postura_s = float(periodo_postura_s) if periodo_postura_s else None   # None/0 = no leer
        self._fabrica_traductor = fabrica_traductor or TraductorMano
        self._traductor = None
        self._error_traductor = None
        self._postura, self._t_postura, self._aviso_postura = None, 0.0, None

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
            e = dict(self._estado)
            fresca = self._postura is not None and time.time() - self._t_postura < 1.0
            e["postura"] = fresca
            if self.periodo_postura_s and e.get("estado") in ("ok", "aviso"):
                nota = ("postura en el gemelo" if fresca
                        else "postura: " + (self._error_traductor or self._aviso_postura or "leyendo…"))
                e["texto"] = f"{e.get('texto', '')} · {nota}"
            return e

    def postura(self, max_edad_s=1.0):
        """{joint de orca_core: rad (sim)} leído hace menos de max_edad_s, o None."""
        with self._lock:
            if self._postura is None or time.time() - self._t_postura > max_edad_s:
                return None
            return dict(self._postura)

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
        if self._traductor is not None:
            self._traductor.reiniciar()

    def _bucle(self):
        t_estado = -1e9
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
                ahora = time.monotonic()
                if ahora - t_estado >= self.periodo_s:
                    t_estado = ahora
                    self._leer_motores()
                if self._bus is not None and self.periodo_postura_s:
                    self._leer_postura()
            except Exception as e:                       # p. ej. cable sacado a mitad de lectura
                self._soltar_bus()
                self._poner(estado="error", texto=f"error leyendo el bus: {e}", motores_ok=0)
                time.sleep(self.reintento_s)
                continue
            time.sleep(min(self.periodo_s, self.periodo_postura_s) if self.periodo_postura_s
                       and self._error_traductor is None else self.periodo_s)

    def _leer_postura(self):
        if self._traductor is None and self._error_traductor is None:
            try:
                self._traductor = self._fabrica_traductor()
                if not self._traductor.calibrada:
                    self._error_traductor = "la mano no está calibrada (calibration.yaml de orca_core)"
            except Exception as e:
                self._error_traductor = f"no se puede convertir ({e})"
        if self._error_traductor is not None:
            return
        ids = self._traductor.ids
        pasos = self._bus.leer_posiciones(ids)
        q = self._traductor.a_sim(pasos)
        with self._lock:
            if q is None:
                self._aviso_postura = f"incompleta ({len(pasos)}/{len(ids)} motores)"
            else:
                self._postura, self._t_postura, self._aviso_postura = q, time.time(), None

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
