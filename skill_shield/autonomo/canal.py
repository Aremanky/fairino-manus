"""
Canal de memoria compartida entre procesos del modo autónomo (10-nov).

Un bloque con nombre (multiprocessing.shared_memory) que guarda el ÚLTIMO mensaje (bytes):
la imagen JPEG de la cámara, el estado del robot en JSON, etc. Un solo escritor, varios
lectores; nadie espera a nadie. No hace falta que los procesos sean hijos unos de otros:
basta con conocer el nombre, así que la percepción puede ir en otro entorno virtual.

Lectura sin cortes a medias ("seqlock"): el escritor pone la secuencia impar mientras
escribe y par al terminar; el lector repite si la secuencia cambió o era impar.

Cabecera: secuencia (uint64) · tiempo de escritura time.time() (double) · tamaño (uint32).
"""
import struct
import time
from multiprocessing import shared_memory

_CABECERA = struct.Struct("<QdI")


def _abrir_sin_seguimiento(nombre):
    """Se engancha a un bloque existente SIN que el resource_tracker lo borre al salir.
    (Hasta Python 3.12 cualquier proceso que se engancha lo «adopta» y lo borra al terminar,
    aunque no sea el dueño; en 3.13 basta con track=False.)"""
    try:
        return shared_memory.SharedMemory(name=nombre, create=False, track=False)
    except TypeError:                                  # Python < 3.13
        shm = shared_memory.SharedMemory(name=nombre, create=False)
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        return shm


class Canal:
    """Último mensaje compartido. `crear=True` solo en el proceso dueño (el que lo borra)."""

    def __init__(self, nombre, capacidad=1 << 20, crear=False):
        self.nombre = nombre
        self._dueno = crear
        if crear:
            try:                                       # restos de una ejecución que se colgó
                viejo = _abrir_sin_seguimiento(nombre)
                viejo.close()
                viejo.unlink()
            except FileNotFoundError:
                pass
            self._shm = shared_memory.SharedMemory(
                name=nombre, create=True, size=_CABECERA.size + int(capacidad))
            _CABECERA.pack_into(self._shm.buf, 0, 0, 0.0, 0)
        else:
            self._shm = _abrir_sin_seguimiento(nombre)
        self.capacidad = self._shm.size - _CABECERA.size
        self._seq_escrita = 0
        self._seq_leida = -1

    @classmethod
    def esperar(cls, nombre, timeout=10.0):
        """Se engancha a un canal que crea otro proceso (reintenta hasta `timeout` s)."""
        t0 = time.monotonic()
        while True:
            try:
                return cls(nombre)
            except FileNotFoundError:
                if time.monotonic() - t0 > timeout:
                    raise
                time.sleep(0.1)

    # ── escritor ──
    def escribir(self, datos: bytes):
        n = len(datos)
        if n > self.capacidad:
            raise ValueError(f"Canal {self.nombre}: {n} bytes > capacidad {self.capacidad}")
        buf = self._shm.buf
        seq = struct.unpack_from("<Q", buf, 0)[0]
        seq = seq + 1 if seq % 2 == 0 else seq + 2     # impar = escribiendo
        struct.pack_into("<Q", buf, 0, seq)
        buf[_CABECERA.size:_CABECERA.size + n] = datos
        _CABECERA.pack_into(buf, 0, seq + 1, time.time(), n)   # par = listo
        self._seq_escrita = seq + 1

    # ── lectores ──
    def leer(self, solo_nuevo=False, intentos=20):
        """(bytes, tiempo de escritura) del último mensaje, o (None, None) si no hay nada
        (o nada nuevo con solo_nuevo=True)."""
        buf = self._shm.buf
        for _ in range(intentos):
            seq1, t, n = _CABECERA.unpack_from(buf, 0)
            if seq1 == 0:
                return None, None
            if seq1 % 2:
                time.sleep(0.0005)
                continue
            if solo_nuevo and seq1 == self._seq_leida:
                return None, None
            datos = bytes(buf[_CABECERA.size:_CABECERA.size + n])
            if struct.unpack_from("<Q", buf, 0)[0] == seq1:
                self._seq_leida = seq1
                return datos, t
        return None, None

    def secuencia(self):
        return struct.unpack_from("<Q", self._shm.buf, 0)[0]

    def cerrar(self):
        try:
            self._shm.close()
        except Exception:
            pass
        if self._dueno:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
