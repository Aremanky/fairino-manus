"""
(23-nov) Carga que lleva el FR5 en la brida, MEDIDA con el par real de los joints (para la web).

No resta la carga de config (FR5_CARGA_KG): sale TODO lo que cuelga de la brida (mano + adaptador + lo
que tenga cogido). El FR5 no tiene célula de carga: el par de cada joint lo da la controladora a partir
de la corriente de los motores (jt_cur_tor), así que es una ESTIMACIÓN.

Cómo se calcula (en cada lectura, sin mover nada a propósito):
  · Con el robot quieto, el par de cada joint es el de la gravedad: el de los eslabones del FR5 (masas y
    centros de masas del MJCF) más el de la carga de la brida.
  · La carga se describe con 4 incógnitas: x = [m, m·cx, m·cy, m·cz] (masa y centro de masas en el mundo,
    respecto al origen de wrist3). El par que hace en el joint j es lineal en x:
        par_j = m · a_j·((o5 − o_j) × g) + (m·c)·(g × a_j)
  · Con los 6 joints hay 6 ecuaciones para 4 incógnitas → mínimos cuadrados (con un poco de
    regularización en m·c para que un error del modelo no se vaya al centro de masas). El centro de masas
    NO hace falta conocerlo: sale del propio ajuste.
  · El signo del par de la controladora (motor que sujeta la gravedad = −gravedad, o al revés) se detecta
    solo, comparando el par medido con el del modelo en los joints que más cargan; o se fija en config
    (FR5_PAR_SIGNO_GRAVEDAD = +1 / −1).

(23-nov, 2ª) Si con el brazo habilitado y quieto el par sale ≈ 0 (mucho menor que la gravedad del modelo), la
controladora ya está restando la gravedad del brazo y de SU carga configurada (GetTargetPayload): el par es
«neto». Entonces carga = la de la controladora + lo que explique ese par neto (lo que se coja cambia la
lectura). Se elige solo; con FR5_PAR_SIGNO_GRAVEDAD se fija el signo en los dos modos.

Cuándo es de fiar:
  · Quieto o despacio: «estable». Moviéndose, la inercia y el rozamiento se cuelan: se enseña el último
    valor estable marcado «en movimiento».
  · Siempre se calcula. Solo si la CONTROLADORA dice que el FR5 está deshabilitado (rbtEnableState = 0) los
    frenos sujetan el brazo y los motores no hacen par: entonces se enseña el último valor medido, en gris.
  · El rozamiento de las reductoras y los errores de las masas del MJCF dan un error de unos cientos de
    gramos. Para saber cuánto: comparar con un peso conocido (sin nada cogido debería salir ~FR5_CARGA_KG).
"""
import time

import numpy as np

G_MUNDO = np.array([0.0, 0.0, -9.81])
# cuerpo del MJCF que gira con cada joint (marco k de CadenaFR5.fk_marcos = cuerpo k ya girado)
ESLABONES = ("shoulder_Link", "upperarm_Link", "forearm_Link", "wrist1_Link", "wrist2_Link", "wrist3_Link")
CARGA_MAX_FR5_KG = 5.0


def inerciales(ruta_xml, nombres=ESLABONES):
    """[(masa kg, centro de masas en el marco del cuerpo)] de los 6 eslabones del FR5, del MJCF."""
    import xml.etree.ElementTree as ET
    d = {}
    for b in ET.parse(ruta_xml).getroot().iter("body"):
        if b.get("name") in nombres:
            i = b.find("inertial")
            if i is not None:
                d[b.get("name")] = (float(i.get("mass", 0.0)),
                                    np.array([float(v) for v in i.get("pos", "0 0 0").split()]))
    return [d.get(n, (0.0, np.zeros(3))) for n in nombres]


class EstimadorCarga:
    """est = EstimadorCarga(cfg["RUTA_XML"]); est.actualizar(q_rad, par_nm) → dict para la web."""

    def __init__(self, ruta_xml, signo=None, pesos=(0.5, 0.7, 0.8, 1.0, 1.0, 1.0), regul=2.0,
                 vel_quieto_deg_s=3.0, quieto_s=0.4, tau_filtro_s=0.6, cadena=None):
        from skill_shield.teleop_fr5.cinematica import CadenaFR5
        self.cadena = cadena or CadenaFR5.desde_mjcf(ruta_xml)
        self.eslabones = inerciales(ruta_xml)
        self.signo_fijo = None if signo is None else float(np.sign(signo))
        self.pesos = np.asarray(pesos, float)
        self.regul = float(regul)              # N·m por kg·m: cuánto «cuesta» alejar el centro de masas de wrist3
        self.vel_quieto = np.radians(float(vel_quieto_deg_s))
        self.quieto_s = float(quieto_s)
        self.tau_filtro = float(tau_filtro_s)
        self._corr = []                        # productos medido·modelo para detectar el signo
        self.signo = self.signo_fijo
        self._q_ant = self._t_ant = None
        self._quieto_desde = None
        self._par = None                       # último par recibido (N·m), para la web
        self._diag = None                      # datos para el diagnóstico de la web
        self._ratios = []                      # |par medido| / |par de gravedad del modelo|, con el brazo quieto
        self.modo = None                       # «absoluto» (el par trae la gravedad) o «neto» (ya restada)
        self.masa = None                       # filtrada (kg)
        self._t_masa = None

    # ── modelo ──────────────────────────────────────────────────────────────
    def sistema(self, q):
        """(A 6x4, conocido 6): par de gravedad = A·x + conocido (convenio: fuerza generalizada de la
        gravedad, N·m) con x = [m, m·c] de la carga respecto al origen de wrist3."""
        m = self.cadena.fk_marcos(np.asarray(q, float))
        o5 = m[5][:3, 3]
        A, conocido = np.zeros((6, 4)), np.zeros(6)
        g = G_MUNDO
        pb = [m[k][:3, :3] @ cb + m[k][:3, 3] for k, (_, cb) in enumerate(self.eslabones)]
        for j in range(6):
            a = m[j][:3, :3] @ self.cadena._ejes[j]
            oj = m[j][:3, 3]
            A[j, 0] = a @ np.cross(o5 - oj, g)
            A[j, 1:] = np.cross(g, a)
            for k in range(j, 6):
                mb = self.eslabones[k][0]
                conocido[j] += mb * (a @ np.cross(pb[k] - oj, g))
        return A, conocido

    def resolver(self, q, par, signo):
        """x = [m, m·c] por mínimos cuadrados ponderados (con regularización en m·c). Devuelve (x, residuo)."""
        A, k = self.sistema(q)
        y = signo * np.asarray(par, float) - k
        W = self.pesos[:, None]
        M = np.vstack([A * W, np.c_[np.zeros(3), self.regul * np.eye(3)]])
        b = np.r_[y * self.pesos, np.zeros(3)]
        x, *_ = np.linalg.lstsq(M, b, rcond=None)
        res = float(np.sqrt(np.mean((A @ x - y) ** 2)))
        return x, res

    def _detectar_signo(self, q, par):
        """El par medido y el del modelo (eslabones + FR5_CARGA_KG nominal no hace falta: los eslabones
        mandan) deben ir en el mismo sentido en los joints con más par. Necesita unas cuantas lecturas."""
        A, k = self.sistema(q)
        par = np.asarray(par, float)
        fuerte = np.abs(k) > 3.0                          # N·m: joints donde la gravedad se nota bien
        if fuerte.sum() == 0:
            return
        self._corr.append(float(np.sum(par[fuerte] * k[fuerte])))
        self._corr = self._corr[-50:]
        if len(self._corr) >= 10:
            s = np.sign(np.sum(self._corr))
            if s != 0:
                self.signo = float(s)

    # ── lectura ─────────────────────────────────────────────────────────────
    def actualizar(self, q, par, ahora=None, habilitado=None, carga_ctrl=None, par_tgt=None):
        """q (rad, MJCF) y par (N·m, sentido MJCF) del FR5, o None si no hay lectura; habilitado = lo que dice
        la controladora (True / False / None si no se sabe); carga_ctrl = kg que tiene puestos la controladora
        (para el modo «neto»); par_tgt = par de consigna (solo diagnóstico). Devuelve dict."""
        ahora = time.monotonic() if ahora is None else ahora
        self._diag = dict(habilitado=habilitado, carga_ctrl=carga_ctrl,
                          par_tgt=None if par_tgt is None else [round(float(v), 2) for v in par_tgt])
        if q is None or par is None:
            self._q_ant = None
            return self.estado("sin lectura del FR5")
        q, par = np.asarray(q, float), np.asarray(par, float)
        self._par = par
        if self._q_ant is not None and ahora > self._t_ant:
            vel = np.max(np.abs(q - self._q_ant)) / (ahora - self._t_ant)
            if vel > self.vel_quieto:
                self._quieto_desde = None
            elif self._quieto_desde is None:
                self._quieto_desde = ahora
        self._q_ant, self._t_ant = q, ahora
        A, k = self.sistema(q)
        self._diag["par_brazo_modelo"] = [round(float(v), 1) for v in k]
        if habilitado is False:                 # frenos puestos: los motores no sujetan el brazo
            return self.estado("FR5 deshabilitado (frenos): último valor medido" if self.masa is not None
                               else "FR5 deshabilitado (frenos): habilítalo para medir", frenado=True)
        quieto = self._quieto_desde is not None and ahora - self._quieto_desde >= self.quieto_s
        if not quieto:
            return self.estado("en movimiento", moviendo=True)
        # ¿el par trae la gravedad («absoluto») o la controladora ya la ha restado con su modelo («neto»)?
        fuerte = np.abs(k) > 3.0
        if fuerte.any():
            self._ratios.append(float(np.sum(np.abs(par[fuerte])) / np.sum(np.abs(k[fuerte]))))
            self._ratios = self._ratios[-30:]
        if len(self._ratios) < 5:
            return self.estado("comprobando el par del FR5… (deja el brazo quieto)")
        ratio = float(np.median(self._ratios))
        self._diag["ratio"] = round(ratio, 2)
        if ratio >= 0.25:
            self.modo = "absoluto"
            if self.signo_fijo is None:
                self._detectar_signo(q, par)
            if self.signo is None:
                return self.estado("calibrando el signo del par… (deja el brazo quieto)")
            x, res = self.resolver(q, par, self.signo)
            m_inst = float(x[0])
        else:
            # par ≈ 0 con el brazo sujeto: la controladora ya resta gravedad + SU carga configurada
            self.modo = "neto"
            if habilitado is None:
                return self.estado("el FR5 no da par: ¿está deshabilitado? (la controladora no dice si lo está)")
            if carga_ctrl is None:
                return self.estado("par neto de la controladora, pero no se pudo leer su carga configurada")
            s_neto = self.signo_fijo if self.signo_fijo is not None else -1.0
            W = self.pesos[:, None]
            M = np.vstack([A * W, np.c_[np.zeros(3), self.regul * np.eye(3)]])
            b = np.r_[s_neto * par * self.pesos, np.zeros(3)]
            dx, *_ = np.linalg.lstsq(M, b, rcond=None)
            res = float(np.sqrt(np.mean((A @ dx - s_neto * par) ** 2)))
            m_inst = float(carga_ctrl) + float(dx[0])
        if not (-0.3 <= m_inst <= 7.0):
            return self.estado(f"medida no válida ({m_inst:.1f} kg): revisa el diagnóstico", invalida=True)
        dt = 0.0 if self._t_masa is None else ahora - self._t_masa
        a = 1.0 if self.masa is None else min(1.0, dt / max(self.tau_filtro, 1e-3))
        self.masa = m_inst if self.masa is None else self.masa + a * (m_inst - self.masa)
        self._t_masa = ahora
        return self.estado("estable", residuo=res)

    def estado(self, texto, frenado=False, moviendo=False, residuo=None, invalida=False):
        m = None if invalida else self.masa
        d = dict(self._diag or {}, modo=self.modo, signo=self.signo,
                 par=None if self._par is None else [round(float(v), 2) for v in self._par],
                 residuo_nm=None if residuo is None else round(residuo, 2))
        return dict(
            texto=texto, estable=texto == "estable", frenado=frenado, moviendo=moviendo, invalida=invalida,
            kg=None if m is None else round(m, 2),
            pct=None if m is None else round(100.0 * m / CARGA_MAX_FR5_KG, 1),
            max_kg=CARGA_MAX_FR5_KG, diag=d)
