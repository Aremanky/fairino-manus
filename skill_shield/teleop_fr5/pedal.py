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
  · "teclado" (9-nov): una tecla MANTENIDA hace de pedal de hombre muerto (por defecto la P).
    Se arma cuando la ventana de la cámara recibe la tecla (tiene el foco) y la tecla está
    sujeta según evdev; en cuanto se suelta, desarmado. Necesita evdev y el grupo `input`.
  · None: sin pedal; el brazo se arma solo al acabar de calibrar (solo para el simulador).

Para saber qué dispositivo y qué código manda tu pedal:
    python -m skill_shield.teleop_fr5.pedal            # lista los dispositivos de entrada
    python -m skill_shield.teleop_fr5.pedal /dev/input/eventX   # muestra lo que manda al pisar
"""


class Pedal:
    def __init__(self, modo="tecla", teclas=(ord("p"), ord("P")), dispositivo=None, codigo=None,
                 hombre_muerto=True, exclusivo=True, codigo_teclado=25):
        self.modo = modo
        self.teclas = tuple(int(k) for k in (teclas or ()))
        self.codigo = codigo
        self.hombre_muerto = (bool(hombre_muerto) and modo == "evdev") or modo == "teclado"
        self.armado = modo is None           # sin pedal: armado siempre (lo decide la calibración)
        self.pisado = False
        self._dev = None
        # modo "teclado" (9-nov): tecla MANTENIDA como pedal de hombre muerto
        self.codigo_teclado = int(codigo_teclado)
        self._teclados = []                  # dispositivos evdev con esa tecla
        self._sujeta = {}                    # ruta → ¿tecla sujeta en ese teclado?
        self._esperar_soltar = False         # tras [A]/[S] hay que soltar y volver a pulsar
        if modo == "evdev":
            self._dev = _abrir_evdev(dispositivo, exclusivo)
            if self._dev is None:
                print("⚠ PEDAL: no encuentro el dispositivo evdev → el brazo NO se podrá armar")
        elif modo == "teclado":
            self._teclados = _abrir_teclados(self.codigo_teclado)
            if not self._teclados:
                print("⚠ PEDAL: no puedo leer ningún teclado con evdev → el brazo NO se podrá armar "
                      "(¿pip install evdev? ¿usuario en el grupo 'input'?)")
        elif modo not in ("tecla", None):
            raise ValueError(f"TELEOP_FR5_PEDAL desconocido: {modo!r} "
                             "(\"teclado\", \"evdev\", \"tecla\" o None)")

    @property
    def descripcion(self):
        if self.modo is None:
            return "sin pedal (se arma al calibrar)"
        if self.modo == "tecla":
            return "tecla " + "/".join(_nombre_tecla(k) for k in self.teclas) + " (alterna)"
        if self.modo == "teclado":
            return ("tecla " + "/".join(sorted({_nombre_tecla(k).upper() for k in self.teclas}))
                    + " MANTENIDA en la ventana de la cámara (hombre muerto)")
        return f"evdev {getattr(self._dev, 'path', '?')}" + (" (hombre muerto)" if self.hombre_muerto
                                                            else " (alterna)")

    def tecla(self, key):
        """Pasa aquí la tecla de cv2.waitKeyEx. Devuelve "armar"/"desarmar" o None."""
        if self.modo == "teclado":
            # Se ARMA solo si la ventana de la cámara recibe la tecla (tiene el foco) y evdev
            # dice que está sujeta; se DESARMA en cuanto evdev ve que se suelta (actualizar).
            self._leer_teclados()
            if key is not None and key >= 0 and key in self.teclas and self._mantenida \
                    and not self.armado and not self._esperar_soltar:
                self.armado = True
                return "armar"
            return None
        if self.modo != "tecla" or key is None or key < 0 or key not in self.teclas:
            return None
        self.armado = not self.armado
        return "armar" if self.armado else "desarmar"

    def actualizar(self):
        """Lee el pedal (modo evdev, sin bloquear). Devuelve "armar"/"desarmar" o None."""
        if self.modo == "teclado":
            self._leer_teclados()
            if not self._mantenida:
                self._esperar_soltar = False
                if self.armado:
                    self.armado = False
                    return "desarmar"
            return None
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
        if self.modo == "teclado":
            self._esperar_soltar = self._mantenida
        elif self.hombre_muerto:
            self.pisado = False          # aunque siga pisado, hace falta una pisada NUEVA

    @property
    def _mantenida(self):
        return any(self._sujeta.values())

    def _leer_teclados(self):
        for d in list(self._teclados):
            try:
                for ev in d.read():
                    if ev.type == 1 and ev.code == self.codigo_teclado:      # EV_KEY
                        if ev.value in (1, 2):                              # pulsada / repetición
                            self._sujeta[d.path] = True
                        elif ev.value == 0:                                 # soltada
                            self._sujeta[d.path] = False
            except BlockingIOError:
                pass
            except OSError as e:                                            # teclado desconectado
                print(f"⚠ PEDAL: error leyendo {getattr(d, 'path', '?')} ({e})")
                self._teclados.remove(d)
                self._sujeta.pop(getattr(d, "path", None), None)

    def cerrar(self):
        for d in self._teclados:
            try:
                d.close()
            except Exception:
                pass
        self._teclados = []
        if self._dev is not None:
            try:
                self._dev.ungrab()
            except Exception:
                pass
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None


def _abrir_teclados(codigo):
    """Todos los dispositivos de entrada que tienen la tecla `codigo` (sin exclusiva: siguen
    escribiendo con normalidad)."""
    try:
        import evdev
    except ImportError:
        print("⚠ PEDAL: falta el paquete evdev (pip install evdev)")
        return []
    teclados = []
    for r in evdev.list_devices():
        try:
            d = evdev.InputDevice(r)
            if codigo in d.capabilities().get(1, []):          # EV_KEY
                teclados.append(d)
            else:
                d.close()
        except OSError:
            continue
    if teclados:
        print("⌨️  PEDAL (teclado): leyendo " + ", ".join(f"{d.path} ({d.name})" for d in teclados))
    return teclados


def _nombre_tecla(k):
    return chr(k) if 32 < k < 127 else str(k)


# 9-nov: con TELEOP_FR5_PEDAL_EVDEV = None se busca un dispositivo con alguno de estos nombres
_NOMBRES_PEDAL = ("pedal", "foot", "footswitch", "switch")


def _abrir_evdev(dispositivo, exclusivo=True):
    try:
        import evdev
    except ImportError:
        print("⚠ PEDAL: falta el paquete evdev (pip install evdev)")
        return None
    elegido = None
    if not dispositivo:                                 # 9-nov: autodetección por el nombre
        candidatos = []
        for r in evdev.list_devices():
            try:
                nombre = evdev.InputDevice(r).name.lower()
            except OSError:
                continue
            if any(k in nombre for k in _NOMBRES_PEDAL):
                candidatos.append((r, nombre))
        if len(candidatos) == 1:
            elegido = candidatos[0][0]
        elif candidatos:
            print("⚠ PEDAL: varios dispositivos parecen un pedal; pon el bueno en "
                  "TELEOP_FR5_PEDAL_EVDEV: " + ", ".join(f"{r} ({n})" for r, n in candidatos))
    elif str(dispositivo).startswith("/dev/"):
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
        if exclusivo:
            # 9-nov: acceso EXCLUSIVO → el pedal deja de escribir teclas en las ventanas
            # (si mandara p. ej. una "a" o una "s", activaría [A] o [S] sin querer)
            try:
                d.grab()
            except OSError as e:
                print(f"⚠ PEDAL: no puedo tomarlo en exclusiva ({e}); sus pulsaciones también "
                      "llegarán como teclas a las ventanas")
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
