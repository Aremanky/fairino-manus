"""
Ejecución de un plan de la tarea en el tiempo (18-nov: separado de tarea.py).

  · Ejecutor: recorre las fases del plan. avanzar(dt) solo cuando se puede mover (pedal); q() y
    mano() dan el comando de ESTE instante.
      - Tiempos de cada tramo del brazo: cada paso entre dos posturas tarda lo que pida el joint más
        rápido a la velocidad máxima y, en los tramos RECTOS (fase.vel_cart), también lo que pida la
        PALMA a su velocidad cartesiana (más despacio aún en los últimos fase.lento_m: la llegada a
        la pelota). Sobre ese «tiempo a velocidad de crucero» se aplica un trapecio (arranca y para).
      - Cierre con los TÁCTILES (fase.cierre_tactil): cada dedo se cierra hasta que su yema nota la
        pelota y ahí se queda; al final, «apretar» y «aflojar» se calculan desde donde tocó cada
        dedo (no desde la postura del modelo). Sin táctiles, el cierre de siempre (por tiempo).
  · ManoActiva: la OrcaHand con par.
"""
import math
import threading
import time

import numpy as np

# Índices de ctrl de la OrcaHand (orca_hardware.OrcaHardwareWrapper.JOINT_A_CTRL) de cada dedo, con el
# nombre con el que salen en estado_tactil() (mano.NOMBRE_DEDO)
DEDOS_CTRL = {"Pulgar": (13, 14, 15, 16), "Índice": (10, 11, 12), "Corazón": (7, 8, 9),
              "Anular": (4, 5, 6), "Meñique": (1, 2, 3)}


def _trapecio(L, v, a):
    """(duración, s(t)) para recorrer L con velocidad máx v y aceleración a, parando al final."""
    if L <= 1e-9:
        return 0.0, lambda t: L
    if L <= v * v / a:                                     # no llega a velocidad máxima
        T = 2.0 * math.sqrt(L / a)
        return T, lambda t: (0.5 * a * t * t if t <= T / 2 else L - 0.5 * a * (T - t) ** 2) if t < T else L
    ta = v / a
    T = L / v + ta

    def s(t):
        if t >= T:
            return L
        if t <= ta:
            return 0.5 * a * t * t
        if t >= T - ta:
            return L - 0.5 * a * (T - t) ** 2
        return 0.5 * a * ta * ta + v * (t - ta)
    return T, s


def _minjerk(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


def tiempos_crucero(qs, v_joint, fk_palma=None, vel_cart=None, lento_m=0.0, vel_lenta=None):
    """Tiempo (s) de cada paso de qs a velocidad de crucero: el máximo entre lo que pide el joint más
    rápido (a v_joint) y, si hay vel_cart y fk_palma, lo que pide la palma (a vel_cart; a vel_lenta en
    los últimos lento_m del tramo)."""
    qs = np.asarray(qs, float).reshape(-1, 6)
    if len(qs) < 2:
        return np.zeros(0)
    t = np.max(np.abs(np.diff(qs, axis=0)), axis=1) / float(v_joint)
    if vel_cart and fk_palma is not None:
        P = np.array([fk_palma(q) for q in qs])
        dp = np.linalg.norm(np.diff(P, axis=0), axis=1)
        tras = np.cumsum(dp[::-1])[::-1] - dp                # lo que queda del tramo al acabar cada paso
        v = np.full(len(dp), float(vel_cart))
        if lento_m and vel_lenta:
            v[tras < float(lento_m) - 1e-9] = float(vel_lenta)
        t = np.maximum(t, dp / v)
    return t


_AVISO_TOPPRA = None


def toppra_disponible():
    try:
        import toppra  # noqa: F401
        return True
    except Exception:
        return False


def tiempos_toppra(qs, v_joint, a_joint, tol_rad=math.radians(0.5)):
    """(18-nov) Tramo ARTICULAR con TOPP-RA (pip install toppra): el más rápido que respeta la velocidad y la
    aceleración de CADA joint, también en las esquinas del camino (el trapecio de antes solo miraba el joint más
    rápido y en las esquinas cambiaba de dirección de golpe). El camino es un spline que pasa por las posturas
    comprobadas; si se separa de la línea entre ellas más de tol_rad (esquinas muy cerradas), no se usa.
    Devuelve (duración, trayectoria q(t)) o None (sin toppra, o no ha salido: entonces, el trapecio)."""
    global _AVISO_TOPPRA
    try:
        import toppra as ta
        import toppra.algorithm as talg
        import toppra.constraint as tcon
    except Exception as e:
        if _AVISO_TOPPRA is None:
            _AVISO_TOPPRA = str(e)
            print(f"ℹ️  TOPP-RA no está instalado ({e}): tramos articulares con el trapecio (./arrancar_nivel6.sh lo instala).")
        return None
    qs = np.asarray(qs, float).reshape(-1, 6)
    if len(qs) < 2:
        return None
    d = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    qs = qs[np.r_[True, d > 1e-7]]                      # sin posturas repetidas (el spline las necesita distintas)
    if len(qs) < 2:
        return None
    ss = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(qs, axis=0), axis=1))]
    try:
        camino = ta.SplineInterpolator(ss, qs) if len(qs) > 2 else ta.SplineInterpolator(ss, qs, bc_type="clamped")
        fino = np.linspace(0.0, ss[-1], max(50, 4 * len(qs)))
        recta = np.column_stack([np.interp(fino, ss, qs[:, j]) for j in range(qs.shape[1])])
        if float(np.max(np.abs(np.asarray(camino(fino)) - recta))) > tol_rad:
            return None                                  # el spline se sale del camino comprobado
        v = np.full(qs.shape[1], float(v_joint))
        a = np.full(qs.shape[1], float(a_joint))
        inst = talg.TOPPRA([tcon.JointVelocityConstraint(np.column_stack([-v, v])),
                            tcon.JointAccelerationConstraint(np.column_stack([-a, a]))],
                           camino, parametrizer="ParametrizeConstAccel")
        tr = inst.compute_trajectory(0.0, 0.0)
    except Exception as e:
        print(f"⚠️  TOPP-RA falló en un tramo ({e}): uso el trapecio.")
        return None
    if tr is None:
        return None
    return float(tr.duration), tr


class Ejecutor:
    """Recorre el plan en el tiempo. avanzar(dt) solo cuando se puede mover; q() y mano() dan el
    comando de ESTE instante (si no se avanza, el mismo: el robot espera donde está).

    fk_palma (18-nov): q → centro de la palma (mundo), para los tramos rectos a velocidad cartesiana.
    contactos (18-nov): función → {dedo: fuerza normal (N) o None} de los táctiles, o None si no hay
    táctiles (entonces el cierre es por tiempo, como antes)."""

    def __init__(self, plan, posturas, vel_rad_s, acel_rad_s2, mano_ini=None, t_mano=(1.0, 1.5),
                 fk_palma=None, contactos=None, toppra=False, toppra_tol_rad=math.radians(0.5)):
        self.plan = plan
        self.posturas = dict(posturas, **(getattr(plan, "posturas", None) or {}))
        self.v, self.a = float(vel_rad_s), float(acel_rad_s2)
        self.t_abrir, self.t_cerrar = t_mano
        self.fk = fk_palma
        self.contactos = contactos
        self.toppra, self.toppra_tol = bool(toppra), float(toppra_tol_rad)
        self.tramos_toppra = 0
        self.i, self.t = 0, 0.0
        self._q = plan.q0.copy()
        self._mano = dict(mano_ini if mano_ini is not None else posturas["abierta"])
        self._mano_ini = dict(self._mano)
        self._iniciada = -1
        self._cierre = None                    # cierre con táctiles en curso
        self.cierre_info = None                # cómo fue el último (dedos que tocaron), para el texto
        self._prep = [self._preparar(f) for f in plan.fases]
        self.duracion = sum(p[0] for p in self._prep)

    def _preparar(self, f):
        if f.qs is not None and self.toppra and not getattr(f, "vel_cart", None):
            r = tiempos_toppra(f.qs, self.v, self.a, self.toppra_tol)    # (18-nov) tramos articulares
            if r is not None:
                self.tramos_toppra += 1
                return r[0], "toppra", r[1]
        if f.qs is not None:
            tc = tiempos_crucero(f.qs, self.v, self.fk, getattr(f, "vel_cart", None),
                                 getattr(f, "lento_m", 0.0), getattr(f, "vel_lenta", None))
            s_acum = np.r_[0.0, np.cumsum(tc)]
            # s = «segundos a velocidad de crucero»: velocidad 1, aceleración la de los joints normalizada
            T, s = _trapecio(float(s_acum[-1]), 1.0, self.a / self.v)
            return T, s_acum, s
        T = f.dur if f.dur is not None else (self.t_cerrar if f.mano in ("agarre", "pelota_cerrada")
                                             else self.t_abrir)
        return float(T), None, None

    @property
    def terminado(self):
        return self.i >= len(self.plan.fases)

    @property
    def fase(self):
        return None if self.terminado else self.plan.fases[self.i]

    def progreso(self):
        hecho = sum(p[0] for p in self._prep[:self.i]) + (self.t if not self.terminado else 0.0)
        return min(1.0, hecho / self.duracion) if self.duracion > 0 else 1.0

    # ── cierre con los táctiles ──
    def _leer_contactos(self):
        if self.contactos is None:
            return None
        try:
            return self.contactos()
        except Exception:
            return None

    def _empezar_fase(self, f):
        self._cierre = None
        ct = getattr(f, "cierre_tactil", None)
        if not ct or f.mano is None:
            return
        fz = self._leer_contactos()
        if not fz:
            return                                       # sin táctiles: por tiempo, a la postura del modelo
        dest = dict(self.posturas[ct["destino"]])
        claves = set(dest) | set(self._mano_ini)
        activos = {d: [k for k in ks if k in claves and
                       abs(dest.get(k, 0.0) - self._mano_ini.get(k, 0.0)) > 0.02]
                   for d, ks in DEDOS_CTRL.items()}
        activos = {d: ks for d, ks in activos.items() if ks}
        self._cierre = dict(dest=dest, activos=activos, congelado={}, tocados=[],
                            base={d: float(v) for d, v in fz.items() if v is not None},
                            umbral=float(ct.get("umbral_n", 0.2)))
        T = float(ct.get("dur", 2.5))
        self._prep[self.i] = (T, None, None)
        self.duracion = sum(p[0] for p in self._prep)

    def _paso_cierre(self, u):
        c = self._cierre
        fz = self._leer_contactos() or {}
        claves = set(c["dest"]) | set(self._mano_ini)
        nuevo = {k: self._mano_ini.get(k, 0.0) + (c["dest"].get(k, 0.0) - self._mano_ini.get(k, 0.0)) * u
                 for k in claves}
        for d, ks in c["activos"].items():
            if d in c["tocados"]:
                continue
            v, b = fz.get(d), c["base"].get(d)
            if v is not None and b is not None and float(v) - b >= c["umbral"]:
                c["tocados"].append(d)
                for k in ks:
                    c["congelado"][k] = self._mano.get(k, nuevo[k])
        nuevo.update(c["congelado"])
        self._mano = nuevo
        return len(c["tocados"]) == len(c["activos"])

    def _acabar_cierre(self, f):
        c, ct = self._cierre, f.cierre_tactil
        contacto = dict(self._mano)
        ini = self._mano_ini

        def mas(frac):
            out = dict(contacto)
            for d, ks in c["activos"].items():
                for k in ks:
                    out[k] = contacto.get(k, 0.0) + float(frac) * (c["dest"].get(k, 0.0) - ini.get(k, 0.0))
            return out
        if ct.get("apretar"):
            self.posturas[ct["apretar"]] = mas(ct.get("apriete", 0.15))
        if ct.get("aflojar"):
            self.posturas[ct["aflojar"]] = mas(ct.get("aflojado", 0.05))
        self.cierre_info = dict(tocados=list(c["tocados"]), activos=sorted(c["activos"]))
        self._cierre = None

    def avanzar(self, dt):
        while dt > 0 and not self.terminado:
            f = self.plan.fases[self.i]
            if self._iniciada != self.i:
                self._iniciada = self.i
                self._empezar_fase(f)
            T, s_acum, s = self._prep[self.i]
            usar = min(dt, max(T - self.t, 0.0))
            fin = self.t + usar >= T - 1e-9
            dt = dt - usar if fin else 0.0           # (sin restos de coma flotante que no avanzan)
            if fin and f.comprobar == "afinar":
                dt = 0.0                             # se para aquí: la Tarea puede rehacer el siguiente tramo
            t = self.t = T if fin else self.t + usar
            if f.qs is not None and isinstance(s_acum, str):        # TOPP-RA: la trayectoria ya va en el tiempo
                self._q = f.qs[-1].copy() if t >= T else np.asarray(s(min(t, T)), float).reshape(-1)[:6].copy()
            elif f.qs is not None:
                si = s(t)
                k = int(np.clip(np.searchsorted(s_acum, si, side="right") - 1, 0, max(len(f.qs) - 2, 0)))
                if len(f.qs) == 1 or s_acum[k + 1] <= s_acum[k]:
                    self._q = f.qs[min(k + 1, len(f.qs) - 1)].copy() if t >= T else f.qs[k].copy()
                else:
                    u = (si - s_acum[k]) / (s_acum[k + 1] - s_acum[k])
                    self._q = f.qs[k] + (f.qs[k + 1] - f.qs[k]) * min(max(u, 0.0), 1.0)
            elif f.mano is not None:
                u = _minjerk(t / T) if T > 0 else 1.0
                if self._cierre is not None:
                    if self._paso_cierre(u) and not fin:
                        self.t = T                   # todos los dedos tocan: no hace falta seguir cerrando
                        dt = 0.0
                else:
                    dest = self.posturas[f.mano]
                    claves = set(dest) | set(self._mano_ini)
                    self._mano = {k: self._mano_ini.get(k, 0.0) + (dest.get(k, 0.0) - self._mano_ini.get(k, 0.0)) * u
                                  for k in claves}
            if self.t >= T:
                if f.qs is not None:
                    self._q = f.qs[-1].copy()
                if self._cierre is not None:
                    self._acabar_cierre(f)
                self.i += 1
                self.t = 0.0
                self._mano_ini = dict(self._mano)

    def q(self):
        return self._q.copy()

    def mano(self):
        return dict(self._mano)

    def cambiar_tramo(self, i, qs):
        """Sustituye las posturas de un tramo que aún no ha empezado (o justo empieza)."""
        f = self.plan.fases[i]
        f.qs = np.asarray(qs, float).reshape(-1, 6)
        self._prep[i] = self._preparar(f)
        self.duracion = sum(p[0] for p in self._prep)


class ManoActiva:
    """La OrcaHand con par (OrcaHardwareWrapper), para la tarea. Misma cara que MonitorMano
    (estado() y postura()) para que modo.py la use igual. Al conectar se queda donde está y arranca
    despacio hacia el primer objetivo; al cerrar el modo, va a su neutra y quita el par (desconectar()).

    23-nov: con iniciar(), un hilo la conecta y la VIGILA: si el wrapper deja de poder enviarle (cable
    USB, alimentación), lista pasa a False (la tarea en marcha se aborta), estado() lo dice y se
    reintenta cada reintento_s hasta que vuelve. conectar() sigue siendo un único intento."""

    def __init__(self, puerto="/dev/ttyUSB0", fabrica=None, reintento_s=2.0):
        self.puerto = puerto
        self._fabrica = fabrica
        self.reintento_s = max(0.2, float(reintento_s))
        self.w = None
        self.lista = False
        self.error = None
        self.perdidas = 0                    # veces que se ha caído estando lista
        self.reconecta = False               # True con iniciar()
        self._parar = threading.Event()
        self._hilo = None
        self._postura, self._t_postura, self._t_leida = None, 0.0, 0.0

    @staticmethod
    def _soltar(w):
        if w is None:
            return
        try:
            w.desconectar()
        except Exception:
            pass

    def conectar(self):
        """Un intento. True si queda conectada y con par."""
        w = None
        try:
            if self._fabrica is not None:
                w = self._fabrica()
            else:
                from orca_hardware import OrcaHardwareWrapper
                w = OrcaHardwareWrapper(puerto=self.puerto)
            if not w.conectar():
                self.error = "no se pudo conectar la mano con par (mira la terminal)"
                self._soltar(w)
                return False
            if self._parar.is_set():                    # se cerró mientras conectaba
                self._soltar(w)
                return False
            self.w, self.error, self.lista = w, None, True
            return True
        except Exception as e:
            self.error = f"error al conectar la mano: {e}"
            self._soltar(w)
            return False

    def iniciar(self):
        """Conecta y vigila en un hilo (reconexión automática)."""
        self.reconecta = True
        self._parar.clear()
        self._hilo = threading.Thread(target=self._bucle, daemon=True, name="ManoActiva")
        self._hilo.start()
        return self

    def _viva(self):
        w = self.w
        return w is not None and bool(getattr(w, "conectado", True))

    def _bucle(self):
        avisado = False
        while not self._parar.is_set():
            if not self.lista:
                if self.conectar():
                    if self.perdidas:
                        print("✅ OrcaHand con par: reconectada.")
                    avisado = False
                    continue
                if not avisado:
                    print(f"⚠️  OrcaHand con par: {self.error}. Reintento cada {self.reintento_s:.0f} s.")
                    avisado = True
                self._parar.wait(self.reintento_s)
            elif not self._viva():
                self.lista = False
                self.perdidas += 1
                self.error = "se ha desconectado (¿cable USB o alimentación?); reconectando"
                print(f"❌ OrcaHand con par: {self.error}…")
                w, self.w = self.w, None
                self._soltar(w)
                avisado = True
            else:
                self._parar.wait(0.2)

    def enviar(self, postura_ctrl):
        """postura_ctrl: {índice de ctrl de la mano 0..16: rad}."""
        w = self.w
        if not self.lista or w is None:
            return False
        arr = [float(postura_ctrl.get(w.JOINT_A_CTRL[j], 0.0)) for j in w._joints]
        return w.enviar_posiciones(arr)

    def postura(self, max_edad_s=1.0):
        """{joint de orca_core: rad (sim)} real, leída como mucho cada 0,1 s."""
        w = self.w
        if not self.lista or w is None:
            return None
        ahora = time.time()
        if ahora - self._t_leida >= 0.1:
            self._t_leida = ahora
            try:
                e = w.estado_real()
                q = w._desde_sdk({j: v[1] for j, v in e.items()})
                if q is not None and np.all(np.isfinite(q)):
                    self._postura, self._t_postura = dict(zip(w._joints, (float(v) for v in q))), ahora
            except Exception:
                pass
        if self._postura is None or ahora - self._t_postura > max_edad_s:
            return None
        return dict(self._postura)

    def estado(self):
        if self.lista and self._viva():
            n = len(getattr(self.w, "_joints", []) or [])
            return dict(estado="ok", texto="con par · la mueve la tarea", motores_ok=n, motores_total=n,
                        par_activo=n, puerto=self.puerto, postura=self._postura is not None)
        if self.error:
            return dict(estado="error", texto=self.error + (" · reintentando" if self.reconecta else ""),
                        puerto=self.puerto)
        return dict(estado="aviso", texto="conectando con par…", puerto=self.puerto)

    def cerrar(self):
        self._parar.set()
        h = self._hilo
        if h is not None and h.is_alive() and h is not threading.current_thread():
            h.join(timeout=5.0)
        self._hilo = None
        w, self.w = self.w, None
        self.lista = False
        self._soltar(w)
