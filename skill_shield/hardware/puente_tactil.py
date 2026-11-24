#!/usr/bin/env python3
"""
Puente táctil: sensores de las yemas de la OrcaHand TOUCH (físicos) → programa principal
========================================================================================
Proceso APARTE que lee los 5 sensores táctiles con la versión ACTUAL de orca_core (la que
trae OrcaHandTouch / TactileClient) y manda por UDP local, ~100 veces por segundo, la fuerza
de cada yema en newtons:  {"thumb": [fx, fy, fz], "index": [...], ...}   (fz = normal).

¿Por qué un proceso aparte? Tu orca_core (carpeta orca_core-main) controla los MOTORES y
funciona; la versión nueva (la de los táctiles) ha cambiado funciones internas que usa
orca_hardware.py y renombrado los modelos. Aquí la nueva se carga SOLO en este proceso y
SOLO para el puerto USB de los sensores: nunca abre el bus de motores.

Instalación (una vez), en la raíz del proyecto:
    git clone https://github.com/orcahand/orca_core.git orca_core_touch
(dependencias: pyserial, pyyaml, numpy ≥ 2.2 y dynamixel-sdk: las mismas que ya usas)

Uso normal: NO hace falta lanzarlo a mano. GuanteManus lo arranca solo
(config.ORCA_TACTIL_ACTIVO = True) y lo cierra al salir. A mano, para probar:
    python skill_shield/hardware/puente_tactil.py --ver                 # tabla en vivo de las fuerzas (N)
    python skill_shield/hardware/puente_tactil.py --ver --mock          # sin mano: sensores simulados
    python skill_shield/hardware/puente_tactil.py --puerto /dev/ttyACM0 # puerto fijo (si la detección falla)

Al conectar pone a CERO los sensores (la mano NO debe tocar nada esos ~2 s). El programa
principal puede pedir un nuevo cero en cualquier momento (mensaje "cero").

SEGURIDAD DEL BUS DE MOTORES: la detección automática de orca_core, si no encuentra la
placa de sensores, acaba sondeando los adaptadores FTDI (tu U2D2 de los motores lo es).
Aquí NO: solo se buscan el adaptador táctil (Paxini) y la placa ORCA, y los puertos de
--excluir (por defecto config.PUERTO_HARDWARE) no se tocan nunca.
"""
import argparse
import json
import os
import socket
import sys
import threading
import time

# Raíz del repo (config.py, orca_core_touch/): este archivo está en skill_shield/hardware/
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# Para la tabla de --ver: qué sensor lleva cada dedo FÍSICO (igual que en manus_guante.py;
# el puente manda SIEMPRE los nombres de orca_core: el reparto lo hace el programa principal)
DEDOS_FISICOS = ("pulgar", "indice", "corazon", "anular", "menique")
_SENSOR_DE_DEDO_DEFECTO = {"pulgar": "index", "indice": "ring", "corazon": "thumb",
                           "anular": "middle", "menique": "pinky"}


def _cfg(nombre, defecto):
    try:
        sys.path.insert(0, RAIZ)
        import config
        return getattr(config, nombre, defecto)
    except Exception:
        return defecto
    finally:
        if sys.path and sys.path[0] == RAIZ:
            sys.path.pop(0)


def argumentos():
    p = argparse.ArgumentParser(description="Sensores táctiles de la OrcaHand Touch → UDP")
    p.add_argument("--core", default=_cfg("ORCA_TACTIL_CORE_RUTA", os.path.join(RAIZ, "orca_core_touch")),
                   help="carpeta del orca_core NUEVO (git clone ... orca_core_touch)")
    p.add_argument("--puerto", default=_cfg("ORCA_TACTIL_PUERTO", "auto"),
                   help="'auto' o el puerto de los sensores (p. ej. /dev/ttyACM0)")
    p.add_argument("--baudios", default="auto")
    p.add_argument("--excluir", nargs="*", default=[_cfg("PUERTO_HARDWARE", "/dev/ttyUSB0")],
                   help="puertos que no se tocan nunca (el bus de motores)")
    p.add_argument("--udp", default=_cfg("ORCA_TACTIL_UDP", "127.0.0.1:47311"),
                   help="destino host:puerto de los datos")
    p.add_argument("--hz", type=float, default=100.0)
    p.add_argument("--sin-cero", action="store_true", help="no poner a cero al conectar")
    p.add_argument("--ver", action="store_true", help="imprimir las fuerzas en vivo")
    p.add_argument("--mock", action="store_true", help="sensores simulados (probar sin la mano)")
    p.add_argument("--padre", type=int, default=0, help="PID del programa principal (salir si muere)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# orca_core NUEVO (solo la parte táctil)
# ══════════════════════════════════════════════════════════════════════════════
def importar_core(ruta):
    if not os.path.isdir(os.path.join(ruta, "orca_core")):
        raise FileNotFoundError(
            f"no encuentro el orca_core nuevo en {ruta}. En la raíz del proyecto:\n"
            f"    git clone https://github.com/orcahand/orca_core.git orca_core_touch")
    sys.path.insert(0, ruta)
    import orca_core
    from orca_core.hardware.hand_serial_link import HandSerialLink
    from orca_core.hardware.tactile_client import TactileClient
    from orca_core.hardware.sensing import serial_discovery as sd
    from orca_core.hardware.sensing import constants as sc
    return orca_core, HandSerialLink, TactileClient, sd, sc


def buscar_puerto(sd, sc, excluir):
    """(puerto, baudios, descripción) de los sensores, o (None, None, motivo).
    Solo adaptador táctil (USB-VID de Paxini) o placa ORCA que responde 'sensor'.
    Nunca FTDI genéricos (el U2D2 de los motores) ni los puertos de 'excluir'."""
    import serial.tools.list_ports
    excluir = {os.path.realpath(p) for p in excluir if p}
    puertos = [p for p in serial.tools.list_ports.comports()
               if os.path.realpath(p.device) not in excluir]
    vids = getattr(sd, "KNOWN_VIDS", getattr(sc, "KNOWN_VIDS", {}))
    paxini = [p for p in puertos if p.vid in vids.get("tactile_sensor", ())]
    if len(paxini) == 1:
        return paxini[0].device, sc.DEFAULT_SENSOR_BAUDRATE, "adaptador táctil"
    if len(paxini) > 1:
        return None, None, f"hay {len(paxini)} adaptadores táctiles: indica --puerto"
    resp_sensor = getattr(sd, "ORCA_ID_RESP_SENSOR", None)
    sondear = getattr(sd, "_probe_orca_id", None)
    if resp_sensor is not None and sondear is not None:
        for p in puertos:
            if p.vid in vids.get("oh_board", ()) and sondear(p.device) == resp_sensor:
                return p.device, sc.DEFAULT_ENCODER_BAUDRATE, "placa ORCA (sensores)"
    return None, None, "no encuentro los sensores táctiles (¿USB de los sensores enchufado?)"


class Sensores:
    """Enlace serie + TactileClient de los táctiles (real o simulado)."""

    def __init__(self, args):
        (self.oc, self.HandSerialLink, self.TactileClient,
         self.sd, self.sc) = importar_core(args.core)
        self.args = args
        self.link = self.tac = None
        self.puerto = None
        self._mock_hilo = None
        self._mock_activo = False

    def conectar(self):
        a = self.args
        if a.mock:
            from orca_core.hardware.mock_hand_serial_link import MockHandSerialLink
            from orca_core.hardware.sensing.tactile_mock import TactileMockState, install_tactile_mock
            self.link = MockHandSerialLink()
            self._estado_mock = TactileMockState()
            install_tactile_mock(self.link, self._estado_mock)
            self.puerto, desc = "mock", "simulado"
        else:
            if a.puerto != "auto":
                self.puerto, desc = a.puerto, "fijado a mano"
                if os.path.realpath(a.puerto) in {os.path.realpath(p) for p in a.excluir if p}:
                    raise RuntimeError(f"{a.puerto} es el bus de MOTORES (--excluir): no lo abro")
                baud = int(a.baudios) if a.baudios != "auto" else self.sd.baud_for_port(a.puerto)
            else:
                self.puerto, baud, desc = buscar_puerto(self.sd, self.sc, a.excluir)
                if self.puerto is None:
                    raise RuntimeError(desc)
                if a.baudios != "auto":
                    baud = int(a.baudios)
            self.link = self.HandSerialLink(port=self.puerto, baudrate=baud)
        self.link.connect()
        if not a.mock:
            time.sleep(self.sc.TACTILE_PORT_OPEN_SETTLE_S)
        self.tac = self.TactileClient(self.link)
        self.tac.connect()
        conf = self.tac.get_tactile_configuration()
        if conf is None:
            raise RuntimeError(f"{self.puerto} se abre pero los sensores no responden")
        # taxels activados: hacen falta para poner a cero (y el cero del resultante sale
        # de las mismas tramas)
        self.tac.start_stream(resultant=True, taxels=True, min_sensors=1)
        if a.mock:
            self._arrancar_mock()
        self.tac.wait_for_first_frame(timeout=3.0)
        return f"{self.puerto} ({desc}) · sensores: {', '.join(conf.active_sensors)}"

    def poner_a_cero(self):
        self.tac.capture_taxel_offsets(num_samples=100)

    def fuerzas(self):
        """({dedo: [fx, fy, fz]}, edad en s) o (None, inf)."""
        r = self.tac.get_latest_forces()
        if r is None:
            return None, float("inf")
        edad = time.monotonic() - r.timestamp if r.timestamp is not None else 0.0
        return {f: [float(x) for x in r.forces[f]] for f in r.fingers}, edad

    def vivo(self):
        return self.link is not None and self.link.is_connected and not self.link.is_port_dead

    def cerrar(self):
        self._mock_activo = False
        for fn in ((self.tac.stop_stream if self.tac else None),
                   (self.tac.disconnect if self.tac else None),
                   (self.link.disconnect if self.link else None)):
            if fn is None:
                continue
            try:
                fn()
            except Exception:
                pass
        self.tac = self.link = None

    # ── sensores simulados: cada dedo aprieta y suelta por turnos, con desfase de cero ──
    def _arrancar_mock(self):
        import math
        from orca_core.hardware.sensing.tactile_mock import feed_combined_frame
        cuentas = dict(self.sc.DEFAULT_TAXEL_COUNTS)
        activos = self._estado_mock.active_sensors
        self._mock_activo = True

        def bucle():
            t0 = time.monotonic()
            while self._mock_activo:
                t = time.monotonic() - t0
                fuerzas, taxels = {}, {}
                for i, f in enumerate(FINGERS):
                    pulso = max(0.0, math.sin(2 * math.pi * (t / 6.0 - i / 5.0)))
                    fz = 0.4 + (3.0 * pulso if t > 3.0 else 0.0)       # 0.4 N de desfase
                    fuerzas[f] = [0.1 * pulso, 0.0, fz]
                    taxels[f] = [[0.0, 0.0, fz / cuentas[f]]] * cuentas[f]
                try:
                    feed_combined_frame(self.link, fuerzas, taxels, activos)
                except Exception:
                    pass
                time.sleep(0.01)
        self._mock_hilo = threading.Thread(target=bucle, daemon=True)
        self._mock_hilo.start()


# ══════════════════════════════════════════════════════════════════════════════
def padre_vivo(pid):
    if not pid:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    a = argumentos()
    host, puerto_udp = a.udp.rsplit(":", 1)
    destino = (host, int(puerto_udp))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))              # para recibir órdenes ("cero") del principal
    sock.settimeout(0.0)

    def enviar(d):
        try:
            sock.sendto(json.dumps(d).encode(), destino)
        except OSError:
            pass

    def log(msg):
        print(f"[TACTIL] {msg}", flush=True)
        enviar({"v": 1, "estado": "info", "msg": msg})

    try:
        sensores = Sensores(a)
    except Exception as e:
        log(f"❌ {e}")
        while padre_vivo(a.padre):              # avisa al principal y espera (no reintenta:
            enviar({"v": 1, "estado": "sin_core", "msg": str(e)})   # falta instalar)
            time.sleep(2.0)
            if not a.padre:
                return 1
        return 1

    periodo = 1.0 / max(a.hz, 1.0)
    ultimo_error = None
    mapa = dict(_cfg("ORCA_TACTIL_SENSOR_DE_DEDO", _SENSOR_DE_DEDO_DEFECTO))
    columnas = [(d, str(mapa.get(d, s))) for d, s in zip(DEDOS_FISICOS, FINGERS)]
    if a.ver:
        print("[TACTIL] tabla por dedo FÍSICO (entre paréntesis, el sensor de orca_core): "
              + "  ".join(f"{d}({s})" for d, s in columnas), flush=True)
    while padre_vivo(a.padre):
        # ── conectar (y reconectar si se desenchufa) ──
        try:
            desc = sensores.conectar()
            log(f"✅ conectado: {desc}")
            if not a.sin_cero:
                log("poniendo a cero: NO toques nada con la mano robótica (2 s)...")
                sensores.poner_a_cero()
                log("cero hecho")
        except Exception as e:
            sensores.cerrar()
            enviar({"v": 1, "estado": "sin_sensor", "msg": str(e)})
            if str(e) != ultimo_error:            # no llenar la consola: solo si cambia
                print(f"[TACTIL] ⏳ {e} — reintentando cada 3 s", flush=True)
                ultimo_error = str(e)
            time.sleep(3.0)
            continue
        ultimo_error = None
        # ── bucle de envío ──
        t_ver = 0.0
        while padre_vivo(a.padre) and sensores.vivo():
            t0 = time.monotonic()
            f, edad = sensores.fuerzas()
            if f is not None:
                enviar({"v": 1, "estado": "ok", "edad": round(edad, 4), "f": f})
                if a.ver and t0 - t_ver > 0.1:
                    t_ver = t0
                    print("\r" + "  ".join(f"{d[:5]:5s} {f[s][2]:5.2f} N" if s in f else f"{d[:5]:5s}  ---  "
                                           for d, s in columnas) + f"  ({1000 * edad:3.0f} ms)  ",
                          end="", flush=True)
            try:                                  # órdenes del principal
                orden, _ = sock.recvfrom(256)
                if orden.strip() == b"cero":
                    log("cero pedido: NO toques nada (2 s)...")
                    try:
                        sensores.poner_a_cero()
                        log("cero hecho")
                    except Exception as e:
                        log(f"⚠️  no se pudo poner a cero: {e}")
            except (BlockingIOError, socket.timeout):
                pass
            except OSError:
                pass
            espera = periodo - (time.monotonic() - t0)
            if espera > 0:
                time.sleep(espera)
        if a.ver:
            print()
        log("⚠️  se ha perdido el enlace de los sensores: reconectando...")
        sensores.cerrar()
        time.sleep(1.0)
    sensores.cerrar()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(0)
