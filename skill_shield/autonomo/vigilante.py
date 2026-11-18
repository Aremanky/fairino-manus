"""
Vigilante de los procesos del modo autónomo (18-nov): cámara de cabeza, web y gemelo de MuJoCo para la web.

Si uno se cae (termina) o se cuelga (sigue vivo pero deja de dar señales), se reinicia solo, sin tocar el
resto: el gemelo, la tarea y el FR5 siguen en el proceso principal como si nada.

Señales de vida (lo que escribe cada uno en su canal de memoria compartida):
  · cámara → el canal de imagen (≈30 por segundo)
  · gemelo → el canal del render (lo que dibuja, ≈15-30 por segundo)
  · web    → el canal de mandos (lo reescribe cada 0,1 s) y además responde a GET /api/estado
Si un proceso lleva AUTONOMO_VIGILAR_SIN_DATOS_S sin señales (pasado el arranque, AUTONOMO_VIGILAR_GRACIA_S),
se da por colgado: se cierra (terminate → kill) y se vuelve a lanzar.

Si vuelve a caerse enseguida (p. ej. la cámara desenchufada), espera cada vez más entre intentos (2 s, 4 s, 8 s…
hasta AUTONOMO_VIGILAR_ESPERA_MAX_S) para no saturar; tras un minuto funcionando bien se olvida.

Todo en su propio hilo: cerrar un proceso colgado puede tardar segundos y el bucle principal (gemelo, tarea,
FR5) no debe esperar nunca por eso.
"""
import subprocess
import threading
import time
import urllib.request


class Vigilado:
    """Un proceso hijo: cómo lanzarlo y cómo saber si está vivo.
    senal(): time.time() de su última señal de vida o None. extra(): None si bien, o texto del problema."""

    def __init__(self, nombre, cmd, cwd, senal=None, extra=None, env=None, gracia_s=25.0, sin_datos_s=6.0,
                 espera_ini_s=2.0, espera_max_s=30.0, estable_s=60.0):
        self.nombre, self.cmd, self.cwd, self.env = nombre, list(cmd), cwd, env
        self.senal, self.extra = senal, extra
        self.gracia_s, self.sin_datos_s = float(gracia_s), float(sin_datos_s)
        self.espera_ini_s, self.espera_max_s, self.estable_s = float(espera_ini_s), float(espera_max_s), float(estable_s)
        self.proc = None
        self.t_lanzado = None
        self.reinicios = 0
        self.motivo = ""
        self._espera = self.espera_ini_s
        self._t_proximo = 0.0                     # time.monotonic() a partir del cual se puede relanzar
        self._t_problema = None                   # desde cuándo falla la comprobación extra

    # ── lanzar / cerrar ──
    def lanzar(self):
        try:
            self.proc = subprocess.Popen(self.cmd, cwd=self.cwd, env=self.env)
            self.t_lanzado = time.monotonic()
            self._t_problema = None
            print(f"   ▶ {self.nombre}: PID {self.proc.pid}")
        except OSError as e:
            self.proc = None
            self.motivo = f"no se pudo lanzar ({e})"
            print(f"⚠️  No se pudo lanzar {self.nombre} ({e}).")
        return self.proc

    def cerrar(self, timeout=3.0):
        p, self.proc = self.proc, None
        if p is None or p.poll() is not None:
            return
        p.terminate()
        try:
            p.wait(timeout)
        except subprocess.TimeoutExpired:
            print(f"⚠️  {self.nombre} no terminó a tiempo → kill")
            p.kill()
            try:
                p.wait(2.0)
            except subprocess.TimeoutExpired:
                pass

    # ── ¿está bien? ──
    def problema(self, ahora_mono=None, ahora=None):
        """None si va bien; si no, por qué (texto). Solo mira, no toca nada."""
        ahora_mono = time.monotonic() if ahora_mono is None else ahora_mono
        ahora = time.time() if ahora is None else ahora
        if self.proc is None:
            return "no está en marcha"
        rc = self.proc.poll()
        if rc is not None:
            return f"se ha cerrado (código {rc})"
        if self.t_lanzado is not None and ahora_mono - self.t_lanzado < self.gracia_s:
            return None                           # arrancando: aún no se le pide nada
        if self.senal is not None:
            t = self.senal()
            # sin señal o con una anterior a este arranque (la del proceso que se cayó) = sin datos
            desde = (ahora - (ahora_mono - self.t_lanzado)) if self.t_lanzado is not None else None
            if t is None or (desde is not None and t < desde) or ahora - t > self.sin_datos_s:
                edad = "nunca" if t is None or (desde is not None and t < desde) else f"hace {ahora - t:.0f} s"
                return f"colgado: sin datos ({edad})"
        if self.extra is not None:
            m = self.extra()
            if m:
                if self._t_problema is None:
                    self._t_problema = ahora_mono
                if ahora_mono - self._t_problema > self.sin_datos_s:
                    return f"colgado: {m}"
            else:
                self._t_problema = None
        return None

    def revisar(self, ahora_mono=None):
        """Comprueba y, si hace falta, reinicia. True si lo ha (re)lanzado en esta llamada."""
        ahora_mono = time.monotonic() if ahora_mono is None else ahora_mono
        m = self.problema(ahora_mono)
        if m is None:
            if self.t_lanzado is not None and ahora_mono - self.t_lanzado > self.estable_s:
                self._espera = self.espera_ini_s      # lleva un rato bien: si cae, se reintenta enseguida
            return False
        if self.proc is not None:                 # caído o colgado: fuera, y a esperar el turno
            if m != self.motivo:
                print(f"⚠️  {self.nombre}: {m} → lo reinicio en {self._espera:.0f} s")
            self.motivo = m
            self.cerrar()
            self._t_proximo = ahora_mono + self._espera
            self._espera = min(self._espera * 2.0, self.espera_max_s)
            return False
        if ahora_mono < self._t_proximo:
            return False
        self.reinicios += 1
        print(f"🔁 {self.nombre}: reinicio n.º {self.reinicios} ({self.motivo or 'no estaba en marcha'})")
        return self.lanzar() is not None

    def estado(self):
        p = self.proc                             # (lo puede cambiar el hilo del vigilante a la vez)
        vivo = p is not None and p.poll() is None
        return dict(vivo=vivo, reinicios=self.reinicios, motivo=self.motivo if (self.reinicios or not vivo) else "",
                    pid=p.pid if vivo else None)


def comprobar_http(url, timeout=2.0):
    """Función para Vigilado.extra: None si la URL responde 200, si no el error (texto)."""
    def f():
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return None if r.status == 200 else f"responde {r.status}"
        except Exception as e:
            return f"no responde ({e.__class__.__name__})"
    return f


class Vigilante:
    """Revisa los procesos cada 'cada' segundos en su propio hilo y reinicia los caídos o colgados."""

    def __init__(self, vigilados, cada=0.5, activo=True):
        self.vigilados = [v for v in vigilados if v is not None]
        self.cada, self.activo = float(cada), bool(activo)
        self._parar = threading.Event()
        self._hilo = None

    def arrancar(self):
        for v in self.vigilados:
            v.lanzar()
        if self.activo:
            self._hilo = threading.Thread(target=self._bucle, daemon=True, name="vigilante")
            self._hilo.start()
        return self

    def agregar(self, v):
        """Un proceso más (p. ej. el gemelo, que se lanza cuando el modelo ya está cargado)."""
        if v is not None:
            v.lanzar()
            self.vigilados.append(v)
        return v

    def _bucle(self):
        while not self._parar.wait(self.cada):
            for v in list(self.vigilados):
                if self._parar.is_set():
                    return
                try:
                    v.revisar()
                except Exception as e:            # el vigilante nunca debe caerse
                    print(f"⚠️  Vigilante ({v.nombre}): {e}")

    def avisos_sin_vigilar(self, avisados):
        """Con el vigilante apagado (AUTONOMO_VIGILAR = False): solo avisa una vez, como antes."""
        for v in self.vigilados:
            if v.proc is not None and v.proc.poll() is not None and v.nombre not in avisados:
                avisados.add(v.nombre)
                print(f"⚠️  {v.nombre} terminó (código {v.proc.returncode}).")

    def estado(self):
        return {v.nombre: v.estado() for v in self.vigilados}

    def parar(self):
        """Para el hilo y cierra todos los procesos (al salir del modo autónomo)."""
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(5.0)
        for v in self.vigilados:
            v.cerrar()
