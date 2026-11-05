"""
Pedal para ARMAR el brazo (5-nov).

Calibrar ([A] + T-pose) ya no mueve el robot: al acabar, el brazo queda "desarmado" (quieto)
hasta que el operador pisa el pedal. Entonces entra despacio (TELEOP_FR5_V_ENTRADA) hacia tu
postura y, al alcanzarla, te sigue a velocidad normal.

Dos formas de leerlo (config.TELEOP_FR5_PEDAL):
  · "tecla": pedal USB que se comporta como un teclado (lo normal) o una tecla cualquiera en la
    ventana de la cámara. Cada pisada ALTERNA armado / desarmado. OpenCV no ve cuándo se suelta
    una tecla, así que en este modo no hay "hombre muerto".
  · "evdev": el pedal se lee directamente de /dev/input (paquete `evdev`, el usuario en el grupo
    `input`). Con TELEOP_FR5_PEDAL_HOMBRE_MUERTO = True el brazo solo está armado MIENTRAS el
    pedal está pisado: al soltarlo se queda quieto. Si el pedal se desconecta → desarmado.
  · None: sin pedal; el brazo se arma solo al acabar de calibrar (solo para el simulador).

Para saber qué dispositivo y qué código manda tu pedal:
    python -m skill_shield.teleop_fr5.pedal            # lista los dispositivos de entrada
    python -m skill_shield.teleop_fr5.pedal /dev/input/eventX   # muestra lo que manda al pisar
"""


class Pedal:
    def __init__(self, modo="tecla", teclas=(ord("p"), ord("P")), dispositivo=None, codigo=None,
                 hombre_muerto=True):
        self.modo = modo
        self.teclas = tuple(int(k) for k in (teclas or ()))
        self.codigo = codigo
        self.hombre_muerto = bool(hombre_muerto) and modo == "evdev"
        self.armado = modo is None           # sin pedal: armado siempre (lo decide la calibración)
        self.pisado = False
        self._dev = None
        if modo == "evdev":
            self._dev = _abrir_evdev(dispositivo)
            if self._dev is None:
                print("⚠ PEDAL: no encuentro el dispositivo evdev → el brazo NO se podrá armar")
        elif modo not in ("tecla", None):
            raise ValueError(f"TELEOP_FR5_PEDAL desconocido: {modo!r} (\"tecla\", \"evdev\" o None)")

    @property
    def descripcion(self):
        if self.modo is None:
            return "sin pedal (se arma al calibrar)"
        if self.modo == "tecla":
            return "tecla " + "/".join(_nombre_tecla(k) for k in self.teclas) + " (alterna)"
        return f"evdev {getattr(self._dev, 'path', '?')}" + (" (hombre muerto)" if self.hombre_muerto
                                                            else " (alterna)")

    def tecla(self, key):
        """Pasa aquí la tecla de cv2.waitKeyEx. Devuelve "armar"/"desarmar" o None."""
        if self.modo != "tecla" or key is None or key < 0 or key not in self.teclas:
            return None
        self.armado = not self.armado
        return "armar" if self.armado else "desarmar"

    def actualizar(self):
        """Lee el pedal (modo evdev, sin bloquear). Devuelve "armar"/"desarmar" o None."""
        if self.modo != "evdev":
            return None
        antes = self.armado
        if self._dev is None:
            self.armado = self.pisado = False
        else:
            try:
                for ev in self._dev.read():
                    if ev.type != 1:                          # EV_KEY
                        continue
                    if self.codigo is not None and ev.code != self.codigo:
                        continue
                    if ev.value == 1:                         # pisado (2 = repetición)
                        self.pisado = True
                        if not self.hombre_muerto:
                            self.armado = not self.armado
                    elif ev.value == 0:                       # soltado
                        self.pisado = False
            except BlockingIOError:
                pass                                          # nada nuevo
            except OSError as e:                              # pedal desconectado → seguro
                print(f"⚠ PEDAL: error leyendo el dispositivo ({e}) → DESARMADO")
                self._dev = None
                self.pisado = False
                self.armado = False
            if self.hombre_muerto:
                self.armado = self.pisado
        if self.armado != antes:
            return "armar" if self.armado else "desarmar"
        return None

    def desarmar(self):
        """Fuerza desarmado ([A], [S]…). En hombre muerto hay que soltar y volver a pisar."""
        self.armado = False if self.modo is not None else self.armado
        if self.hombre_muerto:
            self.pisado = False          # aunque siga pisado, hace falta una pisada NUEVA

    def cerrar(self):
        if self._dev is not None:
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None


def _nombre_tecla(k):
    return chr(k) if 32 < k < 127 else str(k)


def _abrir_evdev(dispositivo):
    try:
        import evdev
    except ImportError:
        print("⚠ PEDAL: falta el paquete evdev (pip install evdev)")
        return None
    elegido = None
    if dispositivo and str(dispositivo).startswith("/dev/"):
        elegido = str(dispositivo)
    elif dispositivo:                                   # parte del nombre del dispositivo
        for r in evdev.list_devices():
            try:
                if str(dispositivo).lower() in evdev.InputDevice(r).name.lower():
                    elegido = r
                    break
            except OSError:
                continue
    if elegido is None:
        print(f"⚠ PEDAL: ningún dispositivo coincide con {dispositivo!r}. Disponibles:")
        _listar(evdev)
        return None
    try:
        d = evdev.InputDevice(elegido)
        print(f"🦶 PEDAL: {d.path} ({d.name})")
        return d
    except OSError as e:
        print(f"⚠ PEDAL: no puedo abrir {elegido} ({e}). ¿Usuario en el grupo 'input'?")
        return None


def _listar(evdev):
    for r in evdev.list_devices():
        try:
            d = evdev.InputDevice(r)
            print(f"   {d.path}: {d.name}")
        except OSError:
            print(f"   {r}: (sin permiso)")


if __name__ == "__main__":
    import sys
    import evdev as _ev
    if len(sys.argv) < 2:
        _listar(_ev)
        print("Pisa el pedal con:  python -m skill_shield.teleop_fr5.pedal /dev/input/eventX")
    else:
        dev = _ev.InputDevice(sys.argv[1])
        print(f"{dev.name}: pisa y suelta el pedal (Ctrl+C para salir)")
        for e in dev.read_loop():
            if e.type == _ev.ecodes.EV_KEY:
                print(f"código {e.code} ({_ev.ecodes.KEY.get(e.code, _ev.ecodes.BTN.get(e.code, '?'))})"
                      f"  valor {e.value} (1 = pisado, 0 = soltado)")
