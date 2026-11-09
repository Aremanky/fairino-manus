"""
Modo GEMELO_SLIDERS (9-nov): el BRAZO se mueve con una ventana de sliders, no con las cámaras.

La mano (guante MANUS → OrcaHand real + simulada) funciona igual que en HARDWARE_REAL; solo
cambia de dónde sale el objetivo del brazo. Los sliders NO hablan con el robot: publican un
objetivo en memoria compartida y el bucle principal se lo da al GeneradorSuave (hilo de
física) → MuJoCo → fr5_servo → ServoJ. Así se reutilizan los mismos topes de velocidad,
aceleración y límites que en la teleoperación, y hay UNA sola conexión con la controladora.

Reparto:
  · Proceso "sliders_brazo" (fork, como fr5_servo): ventana Tkinter.
      - DESARMADO: los sliders siguen la pose REAL (del FR5, o del simulador si no hay FR5)
        y no se pueden tocar.
      - ARMAR: los sliders arrancan EXACTAMENTE en la pose real y pasan a ser el objetivo.
        Moverlos mueve el brazo en directo (con la velocidad limitada).
      - STOP / Esc / Espacio / cerrar la ventana → desarmado: el FR5 deja de recibir ServoJ.
  · Bucle principal: SlidersBrazo.publicar_real(...), .armado, .objetivo_rad(), .desarmar().

Ángulos: los sliders están en GRADOS DE LA CONTROLADORA (los mismos que la WebApp del FR5).
Al pasar al simulador se usa la inversa exacta de fr5_servo (q_FR5 = SIGNO·grados(q_MJCF) +
OFFSET), así que el robot real recibe justo el valor del slider aunque SIGNO/OFFSET estén mal;
si lo estuvieran, lo verías porque el gemelo de MuJoCo no coincidiría con el robot.

Prueba aislada (sin robot, sin MuJoCo):  python -m skill_shield.interfaz.sliders_brazo
"""
import math
import multiprocessing as mp
import os
import time

# Estado del FR5 que el principal enseña en la ventana
SIN_FR5, FR5_ESPERA, FR5_ENVIANDO, FR5_ERROR = range(4)
TEXTO_FR5 = {SIN_FR5: "sin FR5: solo simulador", FR5_ESPERA: "FR5 conectado (quieto)",
             FR5_ENVIANDO: "FR5 recibiendo ServoJ", FR5_ERROR: "FR5 con ERROR: pulsa RESET errores FR5"}

_N_MSG = 256


def limites_sliders(cfg):
    """
    Rango de cada slider (grados de la controladora) a partir de config:
      SLIDERS_LIMITES_DEG si existe; si no, topes del FR5 ∩ límites blandos que fr5_servo
      escribe en la controladora (si FR5_FIJAR_LIMITES_BLANDOS), menos SLIDERS_MARGEN_DEG.
    """
    from skill_shield.hardware.fr5_servo import LIMITES_FR5_DEG
    propios = cfg.get("SLIDERS_LIMITES_DEG")
    if propios is not None:
        return [(float(a), float(b)) for a, b in propios]
    m = float(cfg.get("SLIDERS_MARGEN_DEG", 5.0))
    lim = [(a + m, b - m) for a, b in LIMITES_FR5_DEG]
    blandos = cfg.get("FR5_LIMITES_BLANDOS_DEG")
    if cfg.get("FR5_FIJAR_LIMITES_BLANDOS") and blandos is not None:
        lim = [(max(a, ba + m), min(b, bb - m)) for (a, b), (ba, bb) in zip(lim, blandos)]
    return lim


class SlidersBrazo:
    """Asa del proceso de sliders para el programa principal."""

    def __init__(self, limites_deg, signo=(1,) * 6, offset_deg=(0.0,) * 6, velocidad=0.3):
        self.lim = [(float(a), float(b)) for a, b in limites_deg]
        self.signo = [float(s) for s in signo]
        self.offset = [float(o) for o in offset_deg]
        self._ctx = mp.get_context("fork")
        self._obj = self._ctx.Array("d", 7)        # objetivo (°, controladora), nº de secuencia
        self._real = self._ctx.Array("d", 8)       # pose mostrada (°), ¿del FR5?, t
        self._armado = self._ctx.Value("i", 0)
        self._vel = self._ctx.Value("d", float(velocidad))
        self._fr5 = self._ctx.Value("i", SIN_FR5)
        self._reset = self._ctx.Value("i", 0)      # 9-nov: botón RESET errores FR5
        self._msg = self._ctx.Array("c", _N_MSG)
        self._salir = self._ctx.Event()
        self._proc = None
        self._armado_prev = False

    @classmethod
    def desde_config(cls, cfg):
        return cls(limites_sliders(cfg), cfg.get("FR5_SIGNO", (1,) * 6),
                   cfg.get("FR5_OFFSET_DEG", (0.0,) * 6),
                   float(cfg.get("SLIDERS_VELOCIDAD_INICIAL", 0.3)))

    # ── ciclo de vida ──
    def iniciar(self):
        """Abre la ventana en un proceso aparte. False si no se puede (p. ej. sin Tkinter)."""
        try:
            import tkinter  # noqa: F401  (comprobarlo aquí: el error del hijo se perdería)
        except ImportError as e:
            print(f"✗ Sliders del brazo: falta Tkinter ({e}). En el venv de Python 3.12: "
                  "sudo apt install python3.12-tk   (o python3-tk). El brazo se quedará QUIETO.")
            return False
        self._proc = self._ctx.Process(
            target=_proceso_sliders, name="sliders_brazo", daemon=True,
            args=(self.lim, self._obj, self._real, self._armado, self._vel, self._fr5,
                  self._msg, self._salir, self._reset))
        self._proc.start()
        print("🎚️  Sliders del brazo: ventana abierta (ARMAR para mover el brazo). Rangos (°): "
              + "  ".join(f"j{i + 1}[{a:.0f},{b:.0f}]" for i, (a, b) in enumerate(self.lim)))
        return True

    @property
    def vivo(self):
        return self._proc is not None and self._proc.is_alive()

    def parar(self, timeout=2.0):
        self.desarmar()
        self._salir.set()
        if self._proc is not None:
            self._proc.join(timeout)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None

    # ── conversiones grados (controladora) ↔ rad (MJCF) ──
    def a_grados(self, q_rad):
        return [s * math.degrees(float(q)) + o for q, s, o in zip(q_rad, self.signo, self.offset)]

    def a_rad(self, q_deg):
        return [math.radians((float(q) - o) * s) for q, s, o in zip(q_deg, self.signo, self.offset)]

    def fuera_de_rango(self, q_deg, tol=0.5):
        """Joints (1..6) cuya pose está fuera del rango de su slider."""
        return [i + 1 for i, (q, (a, b)) in enumerate(zip(q_deg, self.lim))
                if not (a - tol <= q <= b + tol)]

    # ── principal → ventana ──
    def publicar_real(self, q_deg, de_fr5):
        with self._real.get_lock():
            self._real[0:6] = [float(v) for v in q_deg[:6]]
            self._real[6] = 1.0 if de_fr5 else 0.0
            self._real[7] = time.monotonic()

    def estado_fr5(self, codigo):
        self._fr5.value = int(codigo)

    def mensaje(self, texto):
        b = texto.encode("utf-8")[:_N_MSG - 1]
        with self._msg.get_lock():
            self._msg.value = b

    def desarmar(self, motivo=None):
        self._armado.value = 0
        if motivo:
            self.mensaje(motivo)
            print(f"🎚️  Sliders: {motivo}")

    # ── ventana → principal ──
    @property
    def armado(self):
        return self.vivo and bool(self._armado.value)

    def pide_reset(self):
        """True (una vez) si se ha pulsado RESET errores FR5 en la ventana."""
        if self._reset.value:
            self._reset.value = 0
            return True
        return False

    def recien_armado(self):
        """True solo en la primera llamada tras pulsar ARMAR."""
        a = self.armado
        nuevo = a and not self._armado_prev
        self._armado_prev = a
        return nuevo

    @property
    def velocidad(self):
        return min(1.0, max(0.05, float(self._vel.value)))

    def objetivo_deg(self):
        with self._obj.get_lock():
            return list(self._obj[0:6])

    def objetivo_rad(self):
        return self.a_rad(self.objetivo_deg())


# ══════════════════════════════════════════════════════════════════════════════
# PROCESO DE LA VENTANA
# ══════════════════════════════════════════════════════════════════════════════

def _proceso_sliders(lim, obj, real, armado, vel, fr5, msg, salir, reset=None):
    ppid = os.getppid()
    try:
        import tkinter as tk
    except ImportError as e:
        print(f"✗ sliders_brazo: falta Tkinter ({e}). Instala python3-tk (o tk en conda).")
        return
    try:
        root = tk.Tk()
    except Exception as e:     # sin pantalla, etc.
        print(f"✗ sliders_brazo: no puedo abrir la ventana ({e})")
        return
    v = _VentanaSliders(root, lim, obj, real, armado, vel, fr5, msg, salir, ppid)
    v.reset_v = reset
    try:
        root.mainloop()
    finally:
        armado.value = 0


class _VentanaSliders:
    PERIODO_MS = 50

    def __init__(self, root, lim, obj, real, armado, vel, fr5, msg, salir, ppid):
        self.root, self.lim = root, lim
        self.obj, self.real, self.armado_v = obj, real, armado
        self.vel, self.fr5, self.msg, self.salir, self.ppid = vel, fr5, msg, salir, ppid
        self._seq = 0
        self._ignorar_trazas = False

        import tkinter as tk
        root.title("Gemelo digital - sliders FR5")
        root.protocol("WM_DELETE_WINDOW", self.cerrar)
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        f = tk.Frame(root, padx=10, pady=8)
        f.pack(fill="both", expand=True)
        self.vars, self.escalas, self.lbl_real, self.botones = [], [], [], []
        for i, (a, b) in enumerate(lim):
            tk.Label(f, text=f"J{i + 1}", width=3, font=("TkDefaultFont", 11, "bold")).grid(row=i, column=0)
            v = tk.DoubleVar(value=0.0)
            s = tk.Scale(f, from_=a, to=b, resolution=0.5, orient="horizontal", length=430,
                         variable=v, showvalue=True)
            s.grid(row=i, column=1, padx=4)
            v.trace_add("write", lambda *_: self._objetivo_cambiado())
            bs = []
            for c, d in enumerate((-5, -1, +1, +5)):
                bt = tk.Button(f, text=f"{d:+d}", width=3,
                               command=lambda vv=v, ss=s, dd=d: self._empujar(vv, ss, dd))
                bt.grid(row=i, column=2 + c)
                bs.append(bt)
            lr = tk.Label(f, text="real: ---", width=14, anchor="w", font=("TkFixedFont", 10))
            lr.grid(row=i, column=6, padx=6)
            self.vars.append(v); self.escalas.append(s); self.lbl_real.append(lr); self.botones += bs

        tk.Label(f, text="Vel %").grid(row=6, column=0, pady=(10, 0))
        self.v_vel = tk.DoubleVar(value=round(100 * vel.value))
        tk.Scale(f, from_=5, to=100, resolution=5, orient="horizontal", length=430,
                 variable=self.v_vel, command=lambda _v: self._vel_cambiada()).grid(row=6, column=1, pady=(10, 0))

        barra = tk.Frame(f)
        barra.grid(row=7, column=0, columnspan=7, pady=10, sticky="w")
        self.b_armar = tk.Button(barra, text="ARMAR", width=12, font=("TkDefaultFont", 12, "bold"),
                                 bg="#2e7d32", fg="white", command=self.alternar_armado)
        self.b_armar.pack(side="left", padx=4)
        tk.Button(barra, text="STOP", width=10, font=("TkDefaultFont", 12, "bold"),
                  bg="#c62828", fg="white", command=self.stop).pack(side="left", padx=10)
        tk.Button(barra, text="Objetivo = pose real", command=self.copiar_real).pack(side="left", padx=4)
        tk.Button(barra, text="Imprimir pose", command=self.imprimir_pose).pack(side="left", padx=4)
        tk.Button(barra, text="RESET errores FR5", command=self.pedir_reset).pack(side="left", padx=4)

        self.lbl_estado = tk.Label(f, text="", anchor="w", justify="left", font=("TkDefaultFont", 10))
        self.lbl_estado.grid(row=8, column=0, columnspan=7, sticky="w")
        self.lbl_msg = tk.Label(f, text="", anchor="w", justify="left", fg="#c62828",
                                wraplength=640, font=("TkDefaultFont", 10, "bold"))
        self.lbl_msg.grid(row=9, column=0, columnspan=7, sticky="w")

        root.bind("<Escape>", lambda e: self.stop())
        root.bind("<space>", lambda e: self.stop())
        self._aplicar_estado_widgets(False)
        root.after(self.PERIODO_MS, self.refrescar)

    # ── lectura de la memoria compartida ──
    def _pose_real(self):
        with self.real.get_lock():
            r = list(self.real[:])
        return r[0:6], bool(r[6]), r[7]

    def _escribir_objetivo(self, q):
        self._seq += 1
        with self.obj.get_lock():
            self.obj[0:6] = [float(v) for v in q]
            self.obj[6] = float(self._seq)

    def _poner_sliders(self, q):
        self._ignorar_trazas = True
        try:
            for v, s, x in zip(self.vars, self.escalas, q):
                a, b = float(s.cget("from")), float(s.cget("to"))
                v.set(min(b, max(a, round(float(x) * 2) / 2)))
        finally:
            self._ignorar_trazas = False

    # ── acciones ──
    def _objetivo_cambiado(self):
        if self._ignorar_trazas or not self.armado_v.value:
            return
        self._escribir_objetivo([v.get() for v in self.vars])

    def _empujar(self, v, s, d):
        if not self.armado_v.value:
            return
        a, b = float(s.cget("from")), float(s.cget("to"))
        v.set(min(b, max(a, v.get() + d)))

    def _vel_cambiada(self):
        self.vel.value = float(self.v_vel.get()) / 100.0

    def alternar_armado(self):
        if self.armado_v.value:
            self.stop()
            return
        q, de_fr5, t = self._pose_real()
        if not t:
            self._aviso("Aún no hay pose real: espera a que arranque la teleoperación.")
            return
        fuera = [i + 1 for i, (x, (a, b)) in enumerate(zip(q, self.lim)) if not (a - 0.5 <= x <= b + 0.5)]
        if fuera:
            self._aviso(f"No se arma: {', '.join(f'J{j}' for j in fuera)} fuera del rango de los sliders. "
                        "Llévalo dentro con la WebApp del FR5 (o ajusta SLIDERS_LIMITES_DEG).")
            return
        # el objetivo arranca EXACTAMENTE en la pose real (sin redondear): ni un salto
        self._escribir_objetivo(q)
        self._poner_sliders(q)
        self.armado_v.value = 1
        self._aviso("")
        self._aplicar_estado_widgets(True)

    def stop(self):
        self.armado_v.value = 0
        self._aplicar_estado_widgets(False)

    def pedir_reset(self):
        """9-nov: borra el error del FR5 (colisión, fallo) sin reiniciar nada. Queda DESARMADO."""
        self.stop()
        if getattr(self, "reset_v", None) is not None:
            self.reset_v.value = 1
            self._aviso("RESET pedido: cuando desaparezca el error, vuelve a ARMAR.")

    def imprimir_pose(self):
        """9-nov: escribe en la terminal la pose REAL como una línea de episodio_poses.txt."""
        q, de_fr5, t = self._pose_real()
        if not t:
            self._aviso("Aún no hay pose real que imprimir.")
            return
        self._n_impresas = getattr(self, "_n_impresas", 0) + 1
        linea = ", ".join(f"{v:.1f}" for v in q)
        print(f"\n📋 pose{self._n_impresas}: {linea}"
              f"    # {'FR5 real' if de_fr5 else 'simulador'} (cópiala en episodio_poses.txt)", flush=True)
        self._aviso(f"Pose {self._n_impresas} impresa en la terminal.")

    def copiar_real(self):
        q, _, t = self._pose_real()
        if t:
            if self.armado_v.value:
                self._escribir_objetivo(q)
            self._poner_sliders(q)

    def cerrar(self):
        self.armado_v.value = 0
        try:
            self.root.destroy()
        except Exception:
            pass

    # ── refresco periódico ──
    def _aviso(self, texto):
        with self.msg.get_lock():
            self.msg.value = texto.encode("utf-8")[:_N_MSG - 1]

    def _aplicar_estado_widgets(self, armado):
        est = "normal" if armado else "disabled"
        for s in self.escalas:
            s.configure(state=est)
        for b in self.botones:
            b.configure(state=est)
        if armado:
            self.b_armar.configure(text="DESARMAR", bg="#ef6c00")
        else:
            self.b_armar.configure(text="ARMAR", bg="#2e7d32")

    def refrescar(self):
        if self.salir.is_set() or os.getppid() != self.ppid:
            self.cerrar()
            return
        armado = bool(self.armado_v.value)
        if self.b_armar.cget("text") != ("DESARMAR" if armado else "ARMAR"):
            self._aplicar_estado_widgets(armado)        # el principal ha desarmado
        q, de_fr5, t = self._pose_real()
        fresco = t and (time.monotonic() - t) < 1.0
        for lbl, x in zip(self.lbl_real, q):
            lbl.configure(text=(f"real: {x:8.2f}" if fresco else "real: ---"))
        if not armado and fresco:
            self._poner_sliders(q)                       # desarmado: los sliders siguen al robot
        fuente = "FR5 real" if de_fr5 else "simulador"
        self.lbl_estado.configure(
            text=f"{'ARMADO' if armado else 'DESARMADO'}  ·  pose de: {fuente}  ·  "
                 f"{TEXTO_FR5.get(int(self.fr5.value), '?')}  ·  "
                 "Esc / Espacio = STOP (la seta física es la parada de verdad)")
        with self.msg.get_lock():
            texto = self.msg.value.decode("utf-8", errors="ignore")
        self.lbl_msg.configure(text=texto)
        self.root.after(self.PERIODO_MS, self.refrescar)


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBA AISLADA: un "robot" de juguete que sigue al objetivo despacio
# ══════════════════════════════════════════════════════════════════════════════

def _main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        import config as cfg
        espacio = vars(cfg)
    except Exception:
        espacio = {}
    s = SlidersBrazo.desde_config(espacio)
    s.iniciar()
    q = [0.0, -90.0, 90.0, -90.0, 90.0, 0.0]
    q = [min(b, max(a, x)) for x, (a, b) in zip(q, s.lim)]
    try:
        while s.vivo:
            if s.armado:
                obj = s.objetivo_deg()
                paso = 60.0 * s.velocidad * 0.02
                q = [x + max(-paso, min(paso, o - x)) for x, o in zip(q, obj)]
            s.publicar_real(q, de_fr5=False)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        s.parar()


if __name__ == "__main__":
    _main()
