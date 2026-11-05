"""
Separar MEDIDA de IK (paso 1 del análisis del vídeo de teleoperación, 28-oct).

En el vídeo se suman tres problemas (medida 3D, IK, latencia) y no se sabe cuánto pesa cada
uno. Este módulo los separa con dos pruebas que no dependen la una de la otra:

1) IK CON UNA TRAYECTORIA LIMPIA (sin cámaras ni hardware)
   Un brazo humano sintético hace los movimientos del vídeo (T-pose, colgando, arriba, al
   frente, codo flexionado, mano a la cara, cruzado...) y su hombro/codo/muñeca EXACTOS entran
   directamente en el retargeting + ControlIKFR5, con la MISMA configuración que la app
   (config.py). Se cuentan los fotogramas exacta / aprox / transición / bloqueado, los motivos
   y el error. Después se repite metiendo ruido controlado en la medida (σ por eje, en cm, y
   saltos al cambiar de cámara) para ver a partir de cuánto ruido se rompe la IK.

       python -m skill_shield.teleop_fr5.diagnostico ik
       python -m skill_shield.teleop_fr5.diagnostico ik --ruido 0,0.5,1,2,4 --saltos 0,3,6 --fps 10
       python -m skill_shield.teleop_fr5.diagnostico ik --detalle          # tabla por pose

   Con ruido 0 la IK debería ir "exacta" casi siempre: lo que no lo sea es culpa de la IK.

2) MEDIDA QUIETO EN T-POSE (con las cámaras reales)
   En Skill_Shield_Nivel_6.py, con la imitación activa ([A]) y TELEOP_FR5_V2 = True, la tecla
   [R] graba TELEOP_FR5_DIAG_S segundos (10 por defecto) de la salida de Fusor2Cam y de lo que
   midió cada cámara. Al terminar imprime el informe y guarda diagnosticos/<etiqueta>_<fecha>.npz:
     · cuánto se mueven hombro, codo y muñeca estando quieto (σ por eje → comparable con el
       --ruido de la prueba 1)
     · huesos (brazo / antebrazo) y ángulos azimut / elevación / codo en el marco del torso
     · A contra B: sesgo sistemático (extrínseca) y dispersión, % de fotogramas con "!",
       cambios de cámara y tamaño típico del salto (→ --saltos de la prueba 1)
     · inclinación de la vertical de la cámara A respecto al tronco (si se ven las caderas)
   Repetir el informe de un archivo guardado:

       python -m skill_shield.teleop_fr5.diagnostico informe diagnosticos/tpose_20261028_101500.npz

Cruzando las dos: si con el σ y los saltos medidos en (2) la prueba (1) ya da muchos
aprox/transición, el problema principal es la medida; si la (1) falla incluso con ruido 0,
es la IK.
"""
import ast
import json
import math
import os
import sys
import time

import numpy as np

from skill_shield.teleop_fr5.retargeting import Anatomia, Retargeting, az_elev, dir_desde_az_elev

RAIZ_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARTS = ("hombro", "codo", "muneca")


# ══════════════════════════════════════════════════════════════════════════════
# 1) trayectoria limpia → retargeting → IK
# ══════════════════════════════════════════════════════════════════════════════

def _d(az, elev):
    return dir_desde_az_elev(az, elev)


def _n(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# Poses (direcciones en el marco del torso: fuera, arriba, delante). Elevación: 0 = colgando,
# 90 = horizontal, 180 = arriba. Azimut: 0 = delante, 90 = fuera (lado del brazo).
POSES = {
    "T-pose":         (_d(90, 90), _d(90, 90)),
    "colgando":       (_d(90, 10), _d(90, 10)),
    "lateral 45 abajo": (_d(90, 45), _d(90, 45)),
    "arriba":         (_d(90, 170), _d(90, 170)),
    "al frente":      (_d(0, 90), _d(0, 90)),
    "frente codo 90": (_d(0, 90), _d(0, 180)),
    "frente abajo codo 60": (_d(20, 45), _d(10, 105)),
    "mano a la cara": (_n((0.6, -0.5, 0.6)), _n((-0.7, 0.6, 0.3))),
    "cruzado":        (_d(-30, 80), _d(-40, 95)),
}

# Recorrido parecido al del vídeo; la primera T-pose es el arranque (calibración, la IK sale
# de la postura de reposo del robot) y no cuenta en las estadísticas.
SECUENCIA = ["T-pose", "colgando", "T-pose", "arriba", "T-pose", "lateral 45 abajo", "al frente",
             "frente codo 90", "frente abajo codo 60", "mano a la cara", "cruzado", "al frente", "T-pose"]


def _slerp(a, b, s):
    a, b = _n(a), _n(b)
    c = float(np.clip(a @ b, -1.0, 1.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return a.copy()
    return _n((math.sin((1 - s) * ang) * a + math.sin(s * ang) * b) / math.sin(ang))


class TrayectoriaLimpia:
    """Hombro fijo; el brazo va de pose en pose (transición suave t_trans s + quieto t_hold s)."""

    def __init__(self, secuencia=SECUENCIA, poses=POSES, L=(0.30, 0.26), t_arranque=3.0,
                 t_trans=1.5, t_hold=1.0, hombro=(-0.15, -0.35, 2.2), lado=1.0):
        self.sec, self.poses = list(secuencia), poses
        self.L1, self.L2 = L
        self.t_arranque, self.t_trans, self.t_hold = t_arranque, t_trans, t_hold
        self.S = np.array(hombro, dtype=float)
        # de cara a la cámara A (como simulador.BrazoHumanoSintetico)
        self.derecha = np.array([-1.0, 0.0, 0.0])
        self.arriba = np.array([0.0, -1.0, 0.0])
        self.delante = np.array([0.0, 0.0, -1.0])
        self.lado = lado
        self.torso = (self.derecha, self.arriba, self.delante, lado)
        self.duracion = t_arranque + (len(self.sec) - 1) * (t_trans + t_hold)

    def a_camara(self, d):
        """(fuera, arriba, delante) → marco de la cámara A."""
        return self.lado * d[0] * self.derecha + d[1] * self.arriba + d[2] * self.delante

    def pose(self, t):
        """(S, E, W, fase) en el marco A. fase = nombre de la pose a la que se va / en la que está."""
        if t < self.t_arranque:
            db, da = self.poses[self.sec[0]]
            fase = "arranque"
        else:
            k = int((t - self.t_arranque) // (self.t_trans + self.t_hold))
            k = min(k, len(self.sec) - 2)
            tl = t - self.t_arranque - k * (self.t_trans + self.t_hold)
            a, b = self.poses[self.sec[k]], self.poses[self.sec[k + 1]]
            s = min(tl / self.t_trans, 1.0)
            s = 0.5 - 0.5 * math.cos(math.pi * s)
            db, da = _slerp(a[0], b[0], s), _slerp(a[1], b[1], s)
            fase = self.sec[k + 1]
        E = self.S + self.L1 * self.a_camara(db)
        W = E + self.L2 * self.a_camara(da)
        return self.S.copy(), E, W, fase


def construir_teleop(L=(0.30, 0.26), usar_config=True, verbose=False, obstaculos=False):
    """TeleopBrazoFR5 con la configuración de la app (config.py, igual que en
    Skill_Shield_Nivel_6.py) y longitudes fijas. Devuelve (teleop, q_reposo, origen)."""
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    from skill_shield.teleop_fr5.ik_teleop import Entorno, ParametrosIK, entorno_soporte_hombros
    from skill_shield.teleop_fr5.retargeting import ConfigRetargeting, LimitesAnatomicos
    from skill_shield.teleop_fr5.teleop_brazo import ConfigTeleop, TeleopBrazoFR5

    C = None
    if usar_config:
        try:
            if RAIZ_REPO not in sys.path:
                sys.path.insert(0, RAIZ_REPO)
            import config as C                                    # noqa: N812
        except Exception as e:                                    # pragma: no cover
            print(f"⚠️  No se pudo importar config.py ({e}); uso valores por defecto")
            C = None
    g = (lambda n, d=None: getattr(C, n, d)) if C is not None else (lambda n, d=None: d)
    ruta = g("RUTA_XML") or os.path.join(RAIZ_REPO, "fairino_description", "mjcf", "fairino_mjcf.xml")
    cadena = cargar_cadena(ruta, verbose=verbose)
    inclinada = not np.allclose(cadena.T_base[:3, :3], np.eye(3))
    soporte = g("SOPORTE_HOMBROS", inclinada)
    if soporte:
        z_suelo = g("SOPORTE_SUELO_Z", -0.975)
        ent, z_min = entorno_soporte_hombros(z_suelo), z_suelo + 0.25
    else:
        ent, z_min = Entorno(z_suelo=0.0), g("ARM_Z_MINIMA", 0.10)
    if g("TELEOP_FR5_Z_MIN") is not None:
        z_min = g("TELEOP_FR5_Z_MIN")
    if C is not None and obstaculos:                    # 5-nov: mesa, peana, cámaras, zonas
        from skill_shield.teleop_fr5.obstaculos import cajas_desde_config
        ent.cajas = cajas_desde_config(C, ent.z_suelo, verbose=verbose)
    muneca = g("TELEOP_FR5_MUNECA_DEG", (80.0, 60.0))
    teleop = TeleopBrazoFR5(
        cadena=cadena, camaras={},
        cfg=ConfigTeleop(brazo=g("BRAZO_HUMANO", "derecho"), longitudes_manual=tuple(L),
                         ik=str(g("TELEOP_FR5_IK", "diferencial"))),
        cfg_retarget=ConfigRetargeting(escala=g("TELEOP_FR5_ESCALA"),
                                       fraccion_alcance=float(g("TELEOP_FR5_FRACCION", 0.85)),
                                       z_min=z_min, prono_escala_z=g("PRONO_Z_ESCALA", 1.6),
                                       roll_rango_deg=float(g("TELEOP_FR5_ROLL_RANGO_DEG", 120.0)),
                                       alcance_max=float(g("TELEOP_FR5_ALCANCE_MAX", 0.88))),
        lim_anatomicos=LimitesAnatomicos(hombro_az_min=g("HOMBRO_AZ_MIN", -45.0),
                                         hombro_az_max=g("HOMBRO_AZ_MAX", 130.0),
                                         hombro_ext_elev=g("HOMBRO_EXT_ELEV", 55.0),
                                         hombro_az_libre_elev=g("HOMBRO_AZ_LIBRE_ELEV", 20.0),
                                         codo_flex_max=g("CODO_FLEX_MAX", 145.0),
                                         humero_rot_ext=g("HUMERO_ROT_EXT", 95.0),
                                         humero_rot_int=g("HUMERO_ROT_INT", 110.0),
                                         activos=g("LIMITES_HUMANOS", True)),
        par_ik=ParametrosIK(limites_extra=dict(g("TELEOP_FR5_LIMITES", {2: (-2.53, 2.53)})),
                            v_seguimiento=float(g("TELEOP_FR5_V_MAX", 3.0)),
                            postura_humana=bool(g("TELEOP_FR5_POSTURA_HUMANA", True)),
                            codo_signo=g("TELEOP_FR5_CODO_SIGNO"),
                            muneca_q4_deg=float(muneca[0]), muneca_q5_deg=float(muneca[1])),
        entorno=ent, verbose=verbose)
    q0 = np.array(g("POSICION_STANDBY_BRAZO", [0.0, -1.5708, 1.35, 0.0, 1.5708, 0.0])[:6], dtype=float)
    return teleop, q0, ("config.py" if C is not None else "valores por defecto")


def _categoria(info):
    """Motivo legible de un fotograma que no es 'exacta'."""
    est = info.get("estado", "?")
    m = str(info.get("motivo") or "")
    if est == "transicion":
        return m or "(sigue la transición)"
    if m.startswith("sin solucion exacta"):
        try:
            d = ast.literal_eval(m[m.index("{"):])
            return "sin solución exacta: " + "+".join(sorted(d))
        except Exception:
            return "sin solución exacta"
    return m or est


def diagnosticar_ik(ruido_cm=0.0, saltos_cm=0.0, p_salto=0.15, fps=10.0, semilla=0, L=(0.30, 0.26),
                    usar_config=True, trayectoria=None, verbose=True, detalle=False, teleop=None, q0=None):
    """Mete la trayectoria (limpia + ruido) en retargeting + IK. Devuelve (resumen, serie).

    ruido_cm: σ POR EJE (cm) de ruido blanco en hombro, codo y muñeca, cada fotograma.
    saltos_cm: cada articulación alterna entre dos medidas separadas saltos_cm (como cuando la
               fusión pasa de la cámara A a la B); cambia con probabilidad p_salto por fotograma.
    fps: el bucle real va a 9-11 FPS (dt ≈ 0,1 s), que es lo que ve el límite de velocidad.
    """
    if teleop is None:
        teleop, q0, origen = construir_teleop(L, usar_config)
    else:
        origen = "teleop dado"
    lado = 1.0 if teleop.cfg.brazo == "derecho" else -1.0
    tray = trayectoria or TrayectoriaLimpia(L=L, lado=lado)
    ref = Retargeting(teleop.cadena, teleop.retarget.cfg, Anatomia(teleop.retarget.anatomia.lim))
    from skill_shield.teleop_fr5.cinematica import OFFSET_MUNECA_ORCA
    rng = np.random.default_rng(semilla)
    sig = ruido_cm / 100.0
    off = {a: _n(rng.normal(size=3)) * (saltos_cm / 200.0) for a in ARTS}   # ±mitad del salto
    lado_salto = {a: 1.0 for a in ARTS}
    dt = 1.0 / fps
    n = int(tray.duracion * fps) + 1
    serie = []
    l_h = None
    q_prev = None
    for k in range(n):
        t = k * dt
        S, E, W, fase = tray.pose(t)
        verdad = dict(hombro=S, codo=E, muneca=W)
        med = {}
        for a, p in verdad.items():
            if saltos_cm > 0 and rng.random() < p_salto:
                lado_salto[a] = -lado_salto[a]
            med[a] = p + (rng.normal(0.0, sig, 3) if sig > 0 else 0.0) + lado_salto[a] * off[a]
        # mano: adelante = antebrazo, lateral continuo (como simulador.simular_teleop)
        f_h = _n(W - E)
        if l_h is None:
            l_h = np.cross(f_h, np.array([0.0, -1.0, 0.0]))
        l_h = _n(l_h - float(l_h @ f_h) * f_h)
        res = dict(hombro=med["hombro"], codo=med["codo"], muneca=med["muneca"], torso=tray.torso,
                   codo_reconstruido=False)
        q, info = teleop.actualizar_desde_medida(res, t, mano=(f_h, l_h), q_actual=q0, dt=dt)
        obj_v = ref.objetivo(dict(hombro=S, codo=E, muneca=W, torso=tray.torso), L, mano=(f_h, l_h), t=t)
        fila = dict(t=t, fase=fase, estado=info.get("estado"), cat=_categoria(info),
                    inicio_tr=(info.get("estado") == "transicion" and info.get("paso") == 0))
        if q is not None and teleop.obj_prev is not None:
            fk = teleop.cadena.fk(q)
            pw = fk["T_tool"][:3, :3] @ OFFSET_MUNECA_ORCA + fk["T_tool"][:3, 3]
            ad = fk["adelante"]
            fila.update(
                err_obj=float(np.linalg.norm(pw - teleop.obj_prev["p_muneca"])),
                err_real=float(np.linalg.norm(pw - obj_v["p_muneca"])),
                ang_obj=math.degrees(math.acos(float(np.clip(ad @ teleop.obj_prev["R_mano"][:, 1], -1, 1)))),
                ang_real=math.degrees(math.acos(float(np.clip(ad @ obj_v["R_mano"][:, 1], -1, 1)))),
                dq=float(np.abs(q - q_prev).max()) if q_prev is not None else 0.0)
            q_prev = q.copy()
        serie.append(fila)
    res = resumir_ik(serie, fps)
    res.update(ruido_cm=ruido_cm, saltos_cm=saltos_cm, fps=fps, origen=origen,
               escala=teleop.obj_prev["escala"] if teleop.obj_prev else None)
    if verbose:
        imprimir_ik(res, detalle)
    return res, serie


def resumir_ik(serie, fps):
    util = [x for x in serie if x["fase"] != "arranque"]
    n = len(util)
    est = {}
    cats = {}
    for x in util:
        est[x["estado"]] = est.get(x["estado"], 0) + 1
        if x["estado"] != "exacta":
            cats[x["cat"]] = cats.get(x["cat"], 0) + 1
    def pct(v, q):
        v = [x for x in v if x is not None and not math.isnan(x)]
        return float(np.percentile(v, q)) if v else float("nan")
    sig = [x for x in util if "err_obj" in x and x["estado"] in ("exacta", "aprox")]
    todos = [x for x in util if "err_real" in x]
    por_fase = {}
    for x in util:
        f = por_fase.setdefault(x["fase"], dict(n=0, estados={}, err=[], ang=[]))
        f["n"] += 1
        f["estados"][x["estado"]] = f["estados"].get(x["estado"], 0) + 1
        if "err_real" in x:
            f["err"].append(x["err_real"])
            f["ang"].append(x["ang_real"])
    return dict(
        n=n, estados=est, motivos=dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        pct={k: 100.0 * v / max(n, 1) for k, v in est.items()},
        n_transiciones=sum(1 for x in util if x["inicio_tr"]),
        s_sin_seguir=sum(1 for x in util if x["estado"] in ("transicion", "bloqueado")) / fps,
        err_obj_med=pct([x["err_obj"] for x in sig], 50), err_obj_p95=pct([x["err_obj"] for x in sig], 95),
        ang_obj_p95=pct([x["ang_obj"] for x in sig], 95),
        err_real_med=pct([x["err_real"] for x in todos], 50), err_real_p95=pct([x["err_real"] for x in todos], 95),
        ang_real_med=pct([x["ang_real"] for x in todos], 50), ang_real_p95=pct([x["ang_real"] for x in todos], 95),
        dq_max=max([x["dq"] for x in todos], default=0.0),
        por_fase=por_fase,
    )


def imprimir_ik(r, detalle=False):
    p = r["pct"]
    print(f"\n── IK con trayectoria {'LIMPIA' if r['ruido_cm'] == 0 and r['saltos_cm'] == 0 else 'con ruido'}: "
          f"σ={r['ruido_cm']:.1f} cm/eje, saltos={r['saltos_cm']:.1f} cm, {r['fps']:.0f} FPS "
          f"({r['n']} fotogramas, {r['origen']}, escala {r['escala']:.2f})")
    print(f"   exacta {p.get('exacta', 0):5.1f} %   aprox {p.get('aprox', 0):5.1f} %   "
          f"transición {p.get('transicion', 0):5.1f} %   bloqueado {p.get('bloqueado', 0):5.1f} %   "
          f"(transiciones: {r['n_transiciones']}, {r['s_sin_seguir']:.1f} s sin seguirte)")
    print(f"   error a lo pedido (exacta+aprox): muñeca med {100 * r['err_obj_med']:.1f} cm, "
          f"p95 {100 * r['err_obj_p95']:.1f} cm · orientación p95 {r['ang_obj_p95']:.0f}°")
    print(f"   error a la VERDAD (sin ruido):    muñeca med {100 * r['err_real_med']:.1f} cm, "
          f"p95 {100 * r['err_real_p95']:.1f} cm · orientación med {r['ang_real_med']:.0f}° p95 {r['ang_real_p95']:.0f}°")
    if r["motivos"]:
        print("   motivos (fotogramas): " + " · ".join(f"{k} ×{v}" for k, v in list(r["motivos"].items())[:6]))
    if detalle:
        for f, d in r["por_fase"].items():
            e = np.array(d["err"]) if d["err"] else np.array([np.nan])
            a = np.array(d["ang"]) if d["ang"] else np.array([np.nan])
            ests = " ".join(f"{k}:{v}" for k, v in d["estados"].items())
            print(f"     → {f:22s} {ests:40s} err med {100 * np.nanmedian(e):4.1f} cm  "
                  f"máx {100 * np.nanmax(e):5.1f} cm  ang p95 {np.nanpercentile(a, 95):4.0f}°")


def barrido_ruido(ruidos=(0.0, 0.5, 1.0, 2.0, 4.0), saltos=(0.0,), fps=10.0, semilla=0, detalle=False,
                  usar_config=True, obstaculos=False):
    """La misma trayectoria con varios niveles de ruido. Imprime una tabla y devuelve la lista."""
    filas = []
    for s in saltos:
        for r in ruidos:
            teleop, q0, _ = construir_teleop(usar_config=usar_config, obstaculos=obstaculos)
            res, _ = diagnosticar_ik(r, s, fps=fps, semilla=semilla, teleop=teleop, q0=q0,
                                     verbose=detalle or (r == 0 and s == 0), detalle=detalle)
            filas.append(res)
    print("\n── Barrido de ruido (misma trayectoria; σ por eje en la salida de la fusión)")
    print("  σ cm  salto cm │ exacta  aprox  trans  bloq │ trans.  s sin seguir │ err verdad med/p95 cm │ ori p95")
    for r in filas:
        p = r["pct"]
        print(f"  {r['ruido_cm']:4.1f}  {r['saltos_cm']:6.1f}   │ {p.get('exacta', 0):5.1f}  {p.get('aprox', 0):5.1f}  "
              f"{p.get('transicion', 0):5.1f}  {p.get('bloqueado', 0):4.1f} │ {r['n_transiciones']:5d}  {r['s_sin_seguir']:9.1f}    │ "
              f"{100 * r['err_real_med']:7.1f} / {100 * r['err_real_p95']:5.1f}       │ {r['ang_real_p95']:4.0f}°")
    print("  (σ y salto reales: los da el registro quieto en T-pose, tecla [R] en la app)")
    return filas


# ══════════════════════════════════════════════════════════════════════════════
# 2) registro quieto (cámaras reales) → informe
# ══════════════════════════════════════════════════════════════════════════════

class RegistroQuieto:
    """Graba la salida de Fusor2Cam (y lo que midió cada cámara) durante duracion_s.

        reg = RegistroQuieto(10.0)
        reg.iniciar(t)
        ...cada fotograma, después de teleop.actualizar(...):
        if reg.activo and reg.alimentar(t, teleop):   # True al terminar (ya guardado e impreso)
            ...
    """

    def __init__(self, duracion_s=10.0, carpeta=None, etiqueta="tpose", longitudes_manual=None):
        self.duracion = float(duracion_s)
        self.carpeta = carpeta or os.path.join(RAIZ_REPO, "diagnosticos")
        self.etiqueta = etiqueta
        self.longitudes_manual = longitudes_manual
        self.activo = False
        self.t0 = None
        self.filas = []
        self.ruta = None
        self.informe = None

    def iniciar(self, t=None):
        self.t0 = time.perf_counter() if t is None else float(t)
        self.filas = []
        self.activo = True
        self.ruta = None
        self.informe = None

    def restante(self, t):
        return max(0.0, self.duracion - (float(t) - self.t0)) if self.activo else 0.0

    def alimentar(self, t, teleop):
        if not self.activo:
            return False
        t = float(t)
        f = teleop.fusor
        res = getattr(teleop, "ultima_medida", None)
        nan3 = np.full(3, np.nan)
        fila = dict(t=t - self.t0,
                    S=res["hombro"] if res else nan3, E=res["codo"] if res else nan3,
                    W=res["muneca"] if res else nan3,
                    rec=bool(res["codo_reconstruido"]) if res else False,
                    fuentes=str(f.resumen), estado=str(teleop.info.get("estado", "")))
        if res:
            der, arr, dela, _ = res["torso"]
            fila.update(derecha=np.asarray(der, float), arriba=np.asarray(arr, float))
        else:
            fila.update(derecha=nan3, arriba=nan3)
        for k in ("hombro2", "cadera", "cadera2"):
            kf = f.kf.get(k)
            fila[k] = kf.p.copy() if (kf is not None and kf.p is not None) else nan3
        diag = getattr(f, "diag", {}) or {}
        for a in ARTS:
            d = diag.get(a, {})
            meds = d.get("meds", {})
            for cam in ("A", "B"):
                m = meds.get(cam)
                fila[f"{a}_{cam}"] = np.asarray(m[0], float) if m else nan3
                fila[f"{a}_{cam}_z"] = bool(m[1]) if m else False
                fila[f"{a}_{cam}_sosp"] = bool(m[2]) if m else False
            fila[f"{a}_d2"] = float(d["d2"]) if d.get("d2") is not None else np.nan
            fila[f"{a}_tri"] = float(d["tri"]) if d.get("tri") is not None else np.nan
            fila[f"{a}_eleg"] = d.get("elegida") or ""
        self.filas.append(fila)
        if t - self.t0 >= self.duracion:
            self.activo = False
            datos = self.datos()
            try:
                self.ruta = guardar(datos, self.carpeta, self.etiqueta)
            except Exception as e:                                # pragma: no cover
                print(f"⚠️  No se pudo guardar el registro: {e}")
            self.informe = informe_quieto(datos, imprimir=True, ruta=self.ruta)
            return True
        return False

    def datos(self):
        """Dict de arrays (lo que se guarda en el .npz)."""
        if not self.filas:
            return {}
        out = {}
        for k in self.filas[0]:
            v = [r[k] for r in self.filas]
            out[k] = np.array(v) if not isinstance(v[0], str) else np.array(v, dtype=str)
        out["meta"] = np.array(json.dumps(dict(duracion_s=self.duracion, etiqueta=self.etiqueta,
                                               longitudes_manual=self.longitudes_manual,
                                               fecha=time.strftime("%Y-%m-%d %H:%M:%S"))))
        return out


def guardar(datos, carpeta, etiqueta="tpose"):
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"{etiqueta}_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    np.savez_compressed(ruta, **datos)
    return ruta


def cargar(ruta):
    with np.load(ruta, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _angulos(S, E, W, der, arr, lado=1.0):
    """az/elev del brazo y flexión del codo (°) por fotograma, en el marco del torso medido."""
    az, el, fl = [], [], []
    for s, e, w, d, a in zip(S, E, W, der, arr):
        if np.any(np.isnan(s)) or np.any(np.isnan(e)) or np.any(np.isnan(w)) or np.any(np.isnan(d)):
            continue
        dl = np.cross(a, d)
        tors = (d, a, dl, lado)
        db = Retargeting.a_torso(e - s, tors)
        da = Retargeting.a_torso(w - e, tors)
        db, da = db / np.linalg.norm(db), da / np.linalg.norm(da)
        x, y = az_elev(db)
        az.append(x)
        el.append(y)
        fl.append(math.degrees(math.acos(float(np.clip(db @ da, -1.0, 1.0)))))
    return np.array(az), np.array(el), np.array(fl)


def _nota(valor, bien, regular):
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "  ?"
    return "✅" if valor <= bien else ("🟡" if valor <= regular else "❌")


# umbrales (mm, °) — los del análisis: A y B deben coincidir con < 2 cm
UMBRAL_RMS_MM = {"hombro": (5.0, 10.0), "codo": (10.0, 20.0), "muneca": (10.0, 20.0)}
UMBRAL_AB_MM = (20.0, 40.0)
UMBRAL_ANG_STD = (3.0, 6.0)


def informe_quieto(d, imprimir=True, ruta=None, lado=1.0):
    """Estadísticas de un registro quieto. d = RegistroQuieto.datos() o cargar(ruta)."""
    if not d or "t" not in d or len(d["t"]) == 0:
        if imprimir:
            print("⚠️  Registro vacío")
        return {}
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    t = d["t"]
    n = len(t)
    dur = float(t[-1] - t[0]) if n > 1 else 0.0
    ok = ~np.isnan(d["W"][:, 0]) & ~np.isnan(d["S"][:, 0]) & ~np.isnan(d["E"][:, 0])
    out = dict(n=n, duracion=dur, fps=(n - 1) / dur if dur > 0 else float("nan"),
               frac_medida=float(ok.mean()), arts={}, ab={}, ruta=ruta, meta=meta)
    L = []
    pr = []
    pr.append(f"\n══ REGISTRO QUIETO ({meta.get('etiqueta', '?')}) — {n} fotogramas en {dur:.1f} s "
              f"({out['fps']:.1f} FPS), con medida fusionada el {100 * out['frac_medida']:.0f} %")
    if ruta:
        pr.append(f"   guardado en {ruta}")
    if ok.sum() < 5:
        pr.append("   ⚠️  Casi sin medida fusionada: ¿estabas delante de las dos cámaras?")
        if imprimir:
            print("\n".join(pr))
        return out

    # ── 1) ruido estando quieto (salida de Fusor2Cam) ──────────────────────────
    pr.append("\n1) Cuánto se mueve cada articulación estando quieto (salida de la fusión, marco cámara A)")
    pr.append("   articulación   σx   σy   σz(prof. A)  │ RMS 3D │ σ por eje  │ p95 desv. │ máx")
    for a, k in (("hombro", "S"), ("codo", "E"), ("muneca", "W")):
        X = d[k][ok]
        dev = X - np.median(X, axis=0)
        r = np.linalg.norm(dev, axis=1)
        sd = dev.std(axis=0) * 1000
        rms = float(np.sqrt(np.mean(r ** 2)) * 1000)
        out["arts"][a] = dict(std_mm=sd.tolist(), rms_mm=rms, sigma_eje_cm=rms / math.sqrt(3) / 10,
                              p95_mm=float(np.percentile(r, 95) * 1000), max_mm=float(r.max() * 1000))
        b, rg = UMBRAL_RMS_MM[a]
        pr.append(f"   {a:12s} {sd[0]:5.0f} {sd[1]:5.0f} {sd[2]:7.0f} mm    │ {rms:4.0f} mm│ {rms / math.sqrt(3) / 10:5.2f} cm   │"
                  f" {out['arts'][a]['p95_mm']:5.0f} mm  │ {out['arts'][a]['max_mm']:4.0f} mm  {_nota(rms, b, rg)}")
    if "muneca_tri" in d:
        fr = {a: float((~np.isnan(d[f"{a}_tri"][ok])).mean()) for a in ARTS}
        out["frac_triangulado"] = fr
        pr.append("   triangulado con A+B: " + " · ".join(f"{a} {100 * v:.0f} %" for a, v in fr.items()))
    rec = d["rec"][ok]
    if rec.any():
        pr.append(f"   (codo reconstruido con los huesos en el {100 * rec.mean():.0f} % de los fotogramas)")
    sig_max = max(out["arts"][a]["sigma_eje_cm"] for a in ARTS)
    out["sigma_eje_cm"] = sig_max

    # ── 2) huesos y ángulos ────────────────────────────────────────────────────
    S, E, W = d["S"][ok], d["E"][ok], d["W"][ok]
    l1, l2 = np.linalg.norm(E - S, axis=1), np.linalg.norm(W - E, axis=1)
    out["huesos"] = dict(brazo=(float(np.median(l1)), float(l1.std())), antebrazo=(float(np.median(l2)), float(l2.std())))
    pr.append("\n2) Huesos y ángulos (marco del torso de la v2)")
    txt = (f"   brazo {100 * np.median(l1):.1f} ± {100 * l1.std():.1f} cm · antebrazo {100 * np.median(l2):.1f} ± "
           f"{100 * l2.std():.1f} cm")
    if np.median(l1) < np.median(l2):
        txt += "   ❌ brazo más corto que el antebrazo (no anatómico: codo desplazado)"
    man = meta.get("longitudes_manual")
    if man:
        txt += f"   (medidas con cinta: {100 * man[0]:.1f} / {100 * man[1]:.1f} cm)"
    pr.append(txt)
    az, el, fl = _angulos(d["S"][ok], d["E"][ok], d["W"][ok], d["derecha"][ok], d["arriba"][ok], lado)
    if len(az):
        out["angulos"] = {}
        for nom, v in (("azimut", az), ("elevación", el), ("codo", fl)):
            out["angulos"][nom] = dict(med=float(np.median(v)), p5=float(np.percentile(v, 5)),
                                       p95=float(np.percentile(v, 95)), std=float(v.std()))
            o = out["angulos"][nom]
            pr.append(f"   {nom:9s} med {o['med']:6.1f}°   rango p5–p95 {o['p5']:6.1f}° … {o['p95']:6.1f}°   "
                      f"σ {o['std']:4.1f}°  {_nota(o['std'], *UMBRAL_ANG_STD)}")
        pr.append("   (en T-pose real: azimut ≈ 90°, elevación ≈ 90°, codo ≈ 0°. Graba también colgando (≈10°) "
                  "y arriba (≈170°) para ver la compresión)")

    # ── 3) A contra B ─────────────────────────────────────────────────────────
    pr.append("\n3) Cámara A contra cámara B (medidas con profundidad de las dos, antes de fusionar)")
    hay_b = any((~np.isnan(d[f"{a}_B"][:, 0])).any() for a in ARTS if f"{a}_B" in d)
    saltos = []
    if not hay_b:
        pr.append("   (no hay medidas de la cámara B: ¿B sin extrínsecos o sin pose?)")
    else:
        pr.append("   articulación  frames A+B │ sesgo (media A−B) │ dispersión │ dist. med / p95 │ con '!' │ cambios A↔B")
        for a in ARTS:
            pa, pb = d[f"{a}_A"], d[f"{a}_B"]
            ambos = (~np.isnan(pa[:, 0]) & ~np.isnan(pb[:, 0]) & d[f"{a}_A_z"].astype(bool) & d[f"{a}_B_z"].astype(bool))
            el_ = d[f"{a}_eleg"]
            con_dos = ~np.isnan(d[f"{a}_d2"])
            frac_ex = float((el_[con_dos] != "").mean()) if con_dos.any() else float("nan")
            seq = [x for x in el_ if x]
            cambios = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
            if ambos.sum() < 3:
                pr.append(f"   {a:12s} {int(ambos.sum()):6d}     │ (pocas medidas con profundidad en las dos)")
                out["ab"][a] = dict(n=int(ambos.sum()), frac_exclam=frac_ex, cambios=cambios)
                continue
            diff = pa[ambos] - pb[ambos]
            dist = np.linalg.norm(diff, axis=1) * 1000
            sesgo = diff.mean(axis=0)
            disp = float(np.sqrt(np.mean(np.linalg.norm(diff - sesgo, axis=1) ** 2)) * 1000)
            # salto al cambiar de cámara: lo que separa A y B en los fotogramas en que la fusión
            # se quedó con una sola ('!'); sin '!' la fusión promedia y no hay saltos
            con_ex = ambos & (el_ != "")
            salto = float(np.median(np.linalg.norm(pa[con_ex] - pb[con_ex], axis=1)) * 1000) if con_ex.sum() >= 3 else 0.0
            out["ab"][a] = dict(n=int(ambos.sum()), sesgo_mm=float(np.linalg.norm(sesgo) * 1000),
                                sesgo_vec_mm=(sesgo * 1000).tolist(), dispersion_mm=disp,
                                dist_med_mm=float(np.median(dist)), dist_p95_mm=float(np.percentile(dist, 95)),
                                frac_exclam=frac_ex, cambios=cambios, salto_mm=salto)
            if cambios > 0:
                saltos.append(salto)
            o = out["ab"][a]
            pr.append(f"   {a:12s} {o['n']:6d}     │ {o['sesgo_mm']:8.0f} mm       │ {disp:6.0f} mm  │ "
                      f"{o['dist_med_mm']:5.0f} / {o['dist_p95_mm']:4.0f} mm │ {100 * frac_ex:5.0f} % │ {cambios:5d}  "
                      f"{_nota(o['dist_med_mm'], *UMBRAL_AB_MM)}")
        out["salto_cm"] = float(np.max(saltos)) / 10 if saltos else 0.0
        if saltos:
            pr.append(f"   salto típico al cambiar de cámara (A↔B con '!'): {out['salto_cm']:.1f} cm")
        if any("sesgo_mm" in v for v in out["ab"].values()):
            pr.append("   sesgo grande y dispersión pequeña → extrínseca mal (rehacer calibración);")
            pr.append("   dispersión grande → ruido de la medida (estéreo sobre ropa oscura, landmarks).")

    # ── 4) vertical ───────────────────────────────────────────────────────────
    h2, c1, c2 = d["hombro2"], d["cadera"], d["cadera2"]
    vok = ok & ~np.isnan(h2[:, 0]) & ~np.isnan(c1[:, 0]) & ~np.isnan(c2[:, 0])
    if vok.sum() >= 5:
        tron = (d["S"][vok] + h2[vok]) / 2 - (c1[vok] + c2[vok]) / 2
        tron /= np.linalg.norm(tron, axis=1, keepdims=True)
        arr = d["arriba"][vok]
        ang = np.degrees(np.arccos(np.clip(np.sum(tron * arr, axis=1), -1, 1)))
        out["vertical_deg"] = float(np.median(ang))
        pr.append(f"\n4) Vertical: la 'arriba' de la v2 (eje Y de la cámara A) se desvía {np.median(ang):.1f}° del "
                  f"tronco (caderas→hombros)  {_nota(float(np.median(ang)), 5.0, 10.0)}")
    else:
        pr.append("\n4) Vertical: no se vieron las caderas en las dos ventanas; no se puede comparar")

    # ── resumen para el diagnóstico de IK ─────────────────────────────────────
    s_cm = out.get("salto_cm", 0.0)
    pr.append(f"\n→ Para la prueba de IK con el ruido real:  python -m skill_shield.teleop_fr5.diagnostico ik "
              f"--ruido 0,{sig_max:.1f} --saltos 0,{s_cm:.0f}")
    if imprimir:
        print("\n".join(pr))
    out["texto"] = "\n".join(pr)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# línea de órdenes
# ══════════════════════════════════════════════════════════════════════════════

def _lista(s):
    return tuple(float(x) for x in s.split(",") if x.strip())


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Separar medida de IK (teleop FR5 v2)")
    sub = ap.add_subparsers(dest="cmd")
    pi = sub.add_parser("ik", help="trayectoria limpia (+ ruido) por retargeting + IK")
    pi.add_argument("--ruido", default="0,0.5,1,2,4", help="σ por eje en cm, separados por comas")
    pi.add_argument("--saltos", default="0", help="salto A↔B en cm, separados por comas")
    pi.add_argument("--fps", type=float, default=10.0)
    pi.add_argument("--semilla", type=int, default=0)
    pi.add_argument("--detalle", action="store_true", help="tabla por pose")
    pi.add_argument("--sin-config", action="store_true", help="no leer config.py")
    pi.add_argument("--obstaculos", action="store_true",
                    help="con la mesa, peana, cámaras y zonas prohibidas de config.py")
    pf = sub.add_parser("informe", help="informe de un registro quieto guardado (.npz)")
    pf.add_argument("ruta", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "ik":
        barrido_ruido(_lista(a.ruido), _lista(a.saltos), fps=a.fps, semilla=a.semilla, detalle=a.detalle,
                      usar_config=not a.sin_config, obstaculos=a.obstaculos)
    elif a.cmd == "informe":
        for r in a.ruta:
            informe_quieto(cargar(r), ruta=r)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
