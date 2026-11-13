"""
Simulador sin hardware: un brazo humano sintético visto por dos cámaras sintéticas con los
MISMOS defectos que las reales cuando la mano apunta a una cámara (codo "inventado" dentro
de la mano, profundidad de la muñeca = profundidad de la palma, escorzo). Sirve para probar
fusion_muneca_codo, retargeting e ik_teleop sin la sala.

    python -m skill_shield.teleop_fr5.simulador            # informe por fases
    python -m skill_shield.teleop_fr5.simulador --plot     # además dibuja los errores (matplotlib)
"""
import json
import math
import os

import numpy as np

from skill_shield.teleop_fr5.fusion_muneca_codo import (
    Camara, Fusor2Cam, Intrinsecos, ObservacionCamara, Parametros, CalibradorHuesos,
    PS_HOMBRO_DER, PS_HOMBRO_IZQ, PS_CODO_DER, PS_MUNECA_DER, PS_CADERA_DER, PS_CADERA_IZQ,
)


class BrazoHumanoSintetico:
    """Hombro fijo en el marco de la cámara A; el brazo recorre fases de movimiento."""

    def __init__(self, L1=0.30, L2=0.26, hombro=(-0.15, -0.35, 2.2)):
        # (hombro: posición en el marco de la cámara A tal que las dos cámaras reales
        #  —extrínsecos de camaras_extrinsecas.json— ven el brazo entero)
        self.L1, self.L2 = L1, L2
        self.S = np.array(hombro, dtype=float)
        self.S2 = self.S + np.array([0.36, 0.0, 0.0])          # hombro izquierdo (a la derecha en la imagen)
        self.C = self.S + np.array([0.02, 0.50, 0.0])          # cadera
        self.C2 = self.S2 + np.array([-0.02, 0.50, 0.0])
        # marco del torso (de cara a la cámara A): derecha = −x, arriba = −y, delante = −z
        self.derecha = np.array([-1.0, 0.0, 0.0])
        self.arriba = np.array([0.0, -1.0, 0.0])
        self.delante = np.array([0.0, 0.0, -1.0])

    def dir_torso(self, fuera, arriba, delante):
        return fuera * self.derecha + arriba * self.arriba + delante * self.delante

    def pose(self, t, mezcla_s=1.0):
        """(S, E, W, fase) en el marco A. Fases de 3 s cada una; en los primeros 'mezcla_s'
        segundos de cada fase el brazo va de forma continua desde donde acabó la anterior."""
        fase = int(t // 3.0) % 7
        s_fase = t % 3.0
        if fase > 0 and s_fase < mezcla_s and mezcla_s > 0:
            S, E0, W0, _ = self._pose_fase(fase - 1, 1.0)
            _, E1, W1, _ = self._pose_fase(fase, s_fase / 3.0)
            a = 0.5 - 0.5 * math.cos(math.pi * s_fase / mezcla_s)
            db = (1 - a) * (E0 - S) + a * (E1 - S)
            db /= np.linalg.norm(db)
            da = (1 - a) * (W0 - E0) + a * (W1 - E1)
            da /= np.linalg.norm(da)
            E = S + self.L1 * db
            return S, E, E + self.L2 * da, fase
        return self._pose_fase(fase, s_fase / 3.0)

    def _pose_fase(self, fase, s):
        sm = 0.5 - 0.5 * math.cos(math.pi * s)               # suave
        if fase == 0:        # T-pose (calibración)
            db = self.dir_torso(1.0, 0.0, 0.0)
            da = db
        elif fase == 1:      # de T-pose a brazo al frente estirado (hacia la cámara A)
            az = (math.pi / 2) * (1.0 - sm)
            db = self.dir_torso(math.sin(az), 0.0, math.cos(az))
            da = db
        elif fase == 2:      # brazo al frente, codo flexionando (la palma sigue hacia A)
            flex = 1.2 * sm
            db = self.dir_torso(math.sin(0.35 * flex), -math.sin(0.6 * flex), math.cos(0.7 * flex))
            db /= np.linalg.norm(db)
            da = self.dir_torso(0.0, 0.0, 1.0)              # antebrazo apuntando a la cámara A
        elif fase == 3:      # girar hacia la cámara B (a la izquierda de A): brazo hacia fuera-delante
            az = 0.2 + 1.1 * sm
            db = self.dir_torso(math.sin(az) * 0.9, -0.3, math.cos(az) * 0.9)
            db /= np.linalg.norm(db)
            da = self.dir_torso(math.sin(az + 0.4), 0.1, math.cos(az + 0.4))
            da /= np.linalg.norm(da)
        elif fase == 4:      # mano a la cara (codo muy doblado)
            db = self.dir_torso(0.6, -0.5 - 0.3 * sm, 0.6)
            db /= np.linalg.norm(db)
            da = self.dir_torso(-0.7, 0.6, 0.3)
            da /= np.linalg.norm(da)
        elif fase == 5:      # círculo con el brazo estirado
            a = 2 * math.pi * s
            db = self.dir_torso(0.5 + 0.4 * math.cos(a), 0.4 * math.sin(a), 0.6)
            db /= np.linalg.norm(db)
            da = db
        else:                # brazo colgando y subiendo
            db = self.dir_torso(0.15, -1.0 + 1.3 * sm, 0.2)
            db /= np.linalg.norm(db)
            da = self.dir_torso(0.1, -0.8 + 1.3 * sm, 0.6)
            da /= np.linalg.norm(da)
        E = self.S + self.L1 * db
        W = E + self.L2 * da
        return self.S, E, W, fase


class CamaraSintetica:
    """Proyecta puntos y simula los fallos de MediaPipe/estéreo con la mano hacia la cámara."""

    def __init__(self, nombre, R=None, t=None, f_px=465.0, ancho=640, alto=480, rng=None,
                 sigma_pix=2.0, sigma_z_k=0.006, p_hueco=0.05, ang_oclusion=28.0):
        self.cam = Camara(nombre, R, t)
        self.intr = Intrinsecos.desde_focal(f_px, ancho, alto)
        self.rng = rng or np.random.default_rng(0)
        self.sigma_pix, self.sigma_z_k, self.p_hueco = sigma_pix, sigma_z_k, p_hueco
        self.ang_oclusion = ang_oclusion
        self.ultimo = {}

    def observar(self, S, E, W, S2, C, C2, t, con_mano=True):
        """ObservacionCamara con los defectos simulados; también guarda 'ultimo' para diagnóstico."""
        pts = {PS_HOMBRO_DER: S, PS_HOMBRO_IZQ: S2, PS_CODO_DER: E, PS_MUNECA_DER: W,
               PS_CADERA_DER: C, PS_CADERA_IZQ: C2}
        pc = {k: self.cam.de_A(p) for k, p in pts.items()}            # marco de esta cámara
        # ¿la mano apunta a la cámara? ángulo antebrazo / rayo de la muñeca
        da = pc[PS_MUNECA_DER] - pc[PS_CODO_DER]
        da /= np.linalg.norm(da)
        rw = pc[PS_MUNECA_DER] / np.linalg.norm(pc[PS_MUNECA_DER])
        ang = math.degrees(math.acos(min(1.0, abs(float(da @ rw)))))
        ocluido = ang < self.ang_oclusion and float(da @ rw) < 0     # el antebrazo viene hacia la cámara
        # mano: 0.19 m desde la muñeca en la dirección del antebrazo, 0.09 de ancho
        lat = np.cross(da, np.array([0.0, 1.0, 0.0]))
        lat /= max(np.linalg.norm(lat), 1e-9)
        mano3d = [pc[PS_MUNECA_DER] + da * (0.19 * i / 4) + lat * (0.045 * j) for i in range(5) for j in (-1, 0, 1)]
        mano3d += [pc[PS_MUNECA_DER] + da * 0.19 + lat * 0.02 * j for j in (-2, 2)]
        mano_px = []
        for p in mano3d:
            px = self.intr.proyectar(p)
            if px is not None:
                mano_px.append(px + self.rng.normal(0, 1.5, 2))
        lm = {}
        for k, p in pc.items():
            px = self.intr.proyectar(p)
            if px is None:
                continue
            vis = 0.95
            px = px + self.rng.normal(0, self.sigma_pix, 2)
            if ocluido and k == PS_CODO_DER:
                # MediaPipe "inventa" el codo cerca de la muñeca (dentro de la mano), con vis. alta
                pw = self.intr.proyectar(pc[PS_MUNECA_DER])
                px = pw + self.rng.normal(0, 12.0, 2)
                vis = 0.75
            lm[k] = (px[0] / self.intr.ancho, px[1] / self.intr.alto, vis)
        z_true = {k: float(p[2]) for k, p in pc.items()}
        pix_true = {k: self.intr.proyectar(p) for k, p in pc.items()}

        def z_func(u, v):
            # profundidad en (u, v): la del punto más cercano en píxeles, con ruido y huecos
            if self.rng.random() < self.p_hueco:
                return None
            mejor, zbest = 1e9, None
            for k, px in pix_true.items():
                if px is None:
                    continue
                d = np.linalg.norm(np.array([u, v]) - px)
                if d < mejor:
                    mejor, zbest = d, z_true[k]
            if ocluido:
                # dentro de la silueta de la mano (que está DELANTE) el estéreo mide la palma
                pw = pix_true[PS_MUNECA_DER]
                if pw is not None and np.linalg.norm(np.array([u, v]) - pw) < 0.12 * self.intr.fx / max(z_true[PS_MUNECA_DER], 0.3):
                    zbest = min(zbest, z_true[PS_MUNECA_DER] - 0.14)
            if zbest is None:
                return None
            return zbest + self.rng.normal(0, self.sigma_z_k * zbest * zbest)

        self.ultimo = dict(ocluido=ocluido, ang=ang)
        return ObservacionCamara(lm, None, self.intr, mano_lm=np.array(mano_px) if (con_mano and mano_px) else None,
                                 z_func=z_func, t=t)


def cargar_extrinsecas_repo():
    """R_ab, t_ab de camaras_extrinsecas.json del repo (p_A = R·p_B + t); si no existe, unas
    cámaras separadas 90° a 2 m."""
    import config as _cfg                          # (sin multicamara: no hace falta depthai)
    ruta = _cfg.EXTRINSECAS_ARCHIVO
    if not os.path.isabs(ruta):
        ruta = os.path.join(os.path.dirname(os.path.abspath(_cfg.__file__)), ruta)
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        return np.array(d["R_ab"], dtype=float), np.array(d["t_ab"], dtype=float)
    R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    return R, np.array([-2.0, 0.0, 2.0])


class FusionIngenua:
    """Referencia: lo que hacía la fusión anterior (fusion_brazo.py) en esencia — punto 3D de
    cada cámara con la profundidad estéreo en el píxel del landmark, media ponderada por la
    incertidumbre si las dos cámaras 'se parecen' (χ²) y, si no, la más cercana al punto
    anterior. Sin tests de oclusión ni filtro temporal."""

    def __init__(self, cams, par=None):
        self.cams = cams
        self.par = par or Parametros()
        self.prev = {}

    def actualizar(self, obs: dict):
        from skill_shield.teleop_fr5.fusion_muneca_codo import _ids
        ids = _ids("derecho")
        out = {}
        for art, i in (("hombro", ids["h"]), ("codo", ids["c"]), ("muneca", ids["m"])):
            meds = []
            for nombre, o in obs.items():
                if o is None:
                    continue
                pix, vis = o.pixel(i)
                if pix is None or vis < self.par.vis_min:
                    continue
                z = o.z_en(pix[0], pix[1], self.par)
                if z is None:
                    continue
                cam = self.cams[nombre]
                p = cam.a_A(o.intr.rayo(*pix) * z)
                s_lat = self.par.sigma_pix * z / o.intr.f
                s_z = self.par.sigma_z_k * z * z
                rz = o.intr.rayo(*pix); rz /= np.linalg.norm(rz)
                tmp = np.array([1.0, 0, 0]) if abs(rz[0]) < 0.9 else np.array([0, 1.0, 0])
                rx = np.cross(rz, tmp); rx /= np.linalg.norm(rx); ry = np.cross(rz, rx)
                B = np.column_stack([rx, ry, rz])
                S = cam.R @ (B @ np.diag([s_lat ** 2, s_lat ** 2, s_z ** 2]) @ B.T) @ cam.R.T
                meds.append((p, S))
            if not meds:
                continue
            if len(meds) == 2:
                (pa, Sa), (pb, Sb) = meds
                d = pa - pb
                if float(d @ np.linalg.solve(Sa + Sb, d)) <= 16.0:
                    Wa, Wb = np.linalg.inv(Sa), np.linalg.inv(Sb)
                    out[art] = np.linalg.solve(Wa + Wb, Wa @ pa + Wb @ pb)
                else:
                    pv = self.prev.get(art)
                    if pv is not None:
                        out[art] = pa if np.linalg.norm(pa - pv) <= np.linalg.norm(pb - pv) else pb
                    else:
                        out[art] = pa if np.trace(Sa) <= np.trace(Sb) else pb
            else:
                out[art] = meds[0][0]
        self.prev.update(out)
        return out


def simular_fusion(duracion=21.0, fps=25.0, semilla=0, con_mano=True, verbose=True):
    """Corre el brazo sintético por las 7 fases y mide el error de la fusión (y el de la fusión
    ingenua de referencia). Devuelve el dict de resultados (errores por fase) y la serie."""
    rng = np.random.default_rng(semilla)
    brazo = BrazoHumanoSintetico()
    R_ab, t_ab = cargar_extrinsecas_repo()
    cam_a = CamaraSintetica("A", rng=rng)
    cam_b = CamaraSintetica("B", R_ab, t_ab, rng=rng)
    fusor = Fusor2Cam({"A": cam_a.cam, "B": cam_b.cam}, "derecho", Parametros())
    ingenua = FusionIngenua({"A": cam_a.cam, "B": cam_b.cam})
    calib = CalibradorHuesos(n=30)
    calib.iniciar()
    serie = []
    n = int(duracion * fps)
    for k in range(n):
        t = k / fps
        S, E, W, fase = brazo.pose(t)
        oa = cam_a.observar(S, E, W, brazo.S2, brazo.C, brazo.C2, t, con_mano)
        ob = cam_b.observar(S, E, W, brazo.S2, brazo.C, brazo.C2, t, con_mano)
        ing = ingenua.actualizar({"A": oa, "B": ob})
        eW_ing = float(np.linalg.norm(ing["muneca"] - W)) if "muneca" in ing else np.nan
        eE_ing = float(np.linalg.norm(ing["codo"] - E)) if "codo" in ing else np.nan
        res = fusor.actualizar({"A": oa, "B": ob}, t)
        if res is not None and calib.calibrando:
            calib.alimentar(res["hombro"], res["codo"], res["muneca"])
            if calib.longitudes is not None:
                fusor.longitudes = calib.longitudes
        if res is None:
            serie.append(dict(t=t, fase=fase, ok=False, eW=np.nan, eE=np.nan, eS=np.nan,
                              eW_ing=eW_ing, eE_ing=eE_ing,
                              oclA=cam_a.ultimo["ocluido"], oclB=cam_b.ultimo["ocluido"], fuentes=fusor.resumen))
            continue
        serie.append(dict(t=t, fase=fase, ok=True,
                          eW=float(np.linalg.norm(res["muneca"] - W)),
                          eE=float(np.linalg.norm(res["codo"] - E)),
                          eS=float(np.linalg.norm(res["hombro"] - S)),
                          eW_ing=eW_ing, eE_ing=eE_ing,
                          oclA=cam_a.ultimo["ocluido"], oclB=cam_b.ultimo["ocluido"],
                          rec=res["codo_reconstruido"], fuentes=res["fuentes"]))
    # informe
    nombres = ["T-pose", "al frente → cámara A", "codo flexiona (mano hacia A)", "hacia la cámara B",
               "mano a la cara", "círculo estirado", "colgando → subiendo"]
    resumen = {}
    for f in range(7):
        s = [x for x in serie if x["fase"] == f]
        ok = [x for x in s if x["ok"]]
        eW = np.array([x["eW"] for x in ok]) if ok else np.array([np.nan])
        eE = np.array([x["eE"] for x in ok]) if ok else np.array([np.nan])
        eWi = np.array([x["eW_ing"] for x in s])
        eEi = np.array([x["eE_ing"] for x in s])
        resumen[f] = dict(nombre=nombres[f], n=len(s), ok=len(ok), eW_med=float(np.nanmedian(eW)),
                          eW_p95=float(np.nanpercentile(eW, 95)) if ok else np.nan,
                          eE_med=float(np.nanmedian(eE)), eE_p95=float(np.nanpercentile(eE, 95)) if ok else np.nan,
                          eW_ing_med=float(np.nanmedian(eWi)), eW_ing_p95=float(np.nanpercentile(eWi, 95)),
                          eE_ing_med=float(np.nanmedian(eEi)), eE_ing_p95=float(np.nanpercentile(eEi, 95)),
                          oclA=sum(x["oclA"] for x in s), oclB=sum(x["oclB"] for x in s))
        if verbose:
            r = resumen[f]
            print(f"fase {f} {r['nombre']:30s} válidos {r['ok']:3d}/{r['n']:3d}  "
                  f"muñeca {100*r['eW_med']:4.1f} cm (p95 {100*r['eW_p95']:4.1f})  "
                  f"codo {100*r['eE_med']:4.1f} cm (p95 {100*r['eE_p95']:4.1f})   "
                  f"| ingenua: muñeca {100*r['eW_ing_med']:4.1f} (p95 {100*r['eW_ing_p95']:5.1f})  "
                  f"codo {100*r['eE_ing_med']:4.1f} (p95 {100*r['eE_ing_p95']:5.1f})   "
                  f"[oclusión A {r['oclA']}, B {r['oclB']}]")
    if verbose:
        print("huesos calibrados:", calib.longitudes, " (reales 0.30 / 0.26)")
    return resumen, serie


def simular_teleop(ruta_mjcf=None, duracion=21.0, fps=25.0, semilla=0, verbose=True):
    """Cadena completa: brazo sintético → cámaras sintéticas → fusión → retargeting → IK.
    Comprueba que el robot sigue la muñeca sin salirse de límites ni chocar."""
    from skill_shield.teleop_fr5.cinematica import OFFSET_MUNECA_ORCA, cargar_cadena
    from skill_shield.teleop_fr5.ik_teleop import entorno_soporte_hombros
    from skill_shield.teleop_fr5.retargeting import ConfigRetargeting
    from skill_shield.teleop_fr5.teleop_brazo import TeleopBrazoFR5

    if ruta_mjcf is None:
        from config import RUTA_XML as ruta_mjcf
    cadena = cargar_cadena(ruta_mjcf, verbose=verbose)
    rng = np.random.default_rng(semilla)
    brazo = BrazoHumanoSintetico()
    R_ab, t_ab = cargar_extrinsecas_repo()
    cam_a = CamaraSintetica("A", rng=rng)
    cam_b = CamaraSintetica("B", R_ab, t_ab, rng=rng)
    inclinada = not np.allclose(cadena.T_base[:3, :3], np.eye(3))
    entorno = entorno_soporte_hombros() if inclinada else None
    cfg_r = ConfigRetargeting(z_min=(-0.975 + 0.25) if inclinada else 0.10)
    teleop = TeleopBrazoFR5(cadena=cadena, camaras={"A": cam_a.cam, "B": cam_b.cam},
                            cfg_retarget=cfg_r, entorno=entorno, verbose=verbose)
    teleop.iniciar_calibracion()
    q0 = np.array([0.0, -1.5708, 1.35, 0.0, 1.5708, 0.0])        # POSICION_STANDBY_BRAZO (soporte)
    serie = []
    q_prev, t_prev = None, None
    l_h = None
    n = int(duracion * fps)
    for k in range(n):
        t = k / fps
        S, E, W, fase = brazo.pose(t)
        oa = cam_a.observar(S, E, W, brazo.S2, brazo.C, brazo.C2, t)
        ob = cam_b.observar(S, E, W, brazo.S2, brazo.C, brazo.C2, t)
        # mano sintética: adelante = antebrazo; lateral continuo (transporte paralelo del
        # lateral anterior ⟂ al antebrazo; al principio, horizontal = pronación neutra)
        f_h = (W - E) / np.linalg.norm(W - E)
        if l_h is None:
            l_h = np.cross(f_h, np.array([0.0, -1.0, 0.0]))
        l_h = l_h - float(l_h @ f_h) * f_h
        l_h /= max(np.linalg.norm(l_h), 1e-9)
        q, info = teleop.actualizar({"A": oa, "B": ob}, t, mano=(f_h, l_h), q_actual=q0, dt=1.0 / fps)
        fila = dict(t=t, fase=fase, estado=info["estado"])
        if q is not None and teleop.obj_prev is not None:
            fk = cadena.fk(q)
            pw = fk["T_tool"][:3, :3] @ OFFSET_MUNECA_ORCA + fk["T_tool"][:3, 3]
            fila.update(err=float(np.linalg.norm(pw - teleop.obj_prev["p_muneca"])),
                        ang=float(math.degrees(math.acos(np.clip(fk["adelante"] @ teleop.obj_prev["R_mano"][:, 1], -1, 1)))),
                        dentro=teleop.ik.en_limites(q), col=len(teleop.ik.col.colisiones(q)) == 0,
                        vmax=float(np.abs(q - q_prev).max() * fps) if q_prev is not None else 0.0,
                        q=q.copy(), rama=info.get("rama"))
            q_prev = q.copy()
        serie.append(fila)
    con_q = [x for x in serie if "err" in x]
    estados = {}
    for x in serie:
        estados[x["estado"]] = estados.get(x["estado"], 0) + 1
    err = np.array([x["err"] for x in con_q if x["estado"] in ("exacta", "aprox")])
    ang = np.array([x["ang"] for x in con_q if x["estado"] in ("exacta", "aprox")])
    vmax = np.array([x["vmax"] for x in con_q])
    res = dict(estados=estados, n=len(serie),
               err_med=float(np.median(err)) if len(err) else np.nan,
               err_p95=float(np.percentile(err, 95)) if len(err) else np.nan,
               err_max=float(err.max()) if len(err) else np.nan,
               ang_med=float(np.median(ang)) if len(ang) else np.nan,
               ang_p95=float(np.percentile(ang, 95)) if len(ang) else np.nan,
               fuera_limites=sum(not x["dentro"] for x in con_q),
               colisiones=sum(not x["col"] for x in con_q),
               v_max=float(vmax.max()) if len(vmax) else 0.0,
               cambios_rama=teleop.ik.n_cambios_rama,
               anatomia=(teleop.retarget.anatomia.n_hombro, teleop.retarget.anatomia.n_codo),
               escala=teleop.obj_prev["escala"] if teleop.obj_prev else None)
    if verbose:
        print(f"estados: {estados}")
        print(f"error muñeca (exacta+aprox): med {100*res['err_med']:.2f} cm  p95 {100*res['err_p95']:.2f}  máx {100*res['err_max']:.2f}")
        print(f"error orientación 'adelante': med {res['ang_med']:.1f}°  p95 {res['ang_p95']:.1f}°")
        print(f"fuera de límites: {res['fuera_limites']}   colisiones: {res['colisiones']}   "
              f"vel. articular máx {res['v_max']:.2f} rad/s   cambios de rama: {res['cambios_rama']}   "
              f"límites humanos actuaron (hombro, codo): {res['anatomia']}   escala {res['escala']:.2f}")
        for f in range(7):
            s = [x for x in serie if x["fase"] == f and "err" in x]
            if s:
                e = np.array([x["err"] for x in s]); ests = {}
                for x in s:
                    ests[x["estado"]] = ests.get(x["estado"], 0) + 1
                print(f"   fase {f}: err med {100*np.median(e):.1f} cm máx {100*e.max():.1f}   {ests}")
    return res, serie


if __name__ == "__main__":
    import sys
    if "--teleop" in sys.argv:
        simular_teleop()
        sys.exit(0)
    resumen, serie = simular_fusion()
    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt
        t = [x["t"] for x in serie]
        plt.figure(figsize=(11, 4))
        plt.plot(t, [100 * x["eW"] for x in serie], label="error muñeca (cm)")
        plt.plot(t, [100 * x["eE"] for x in serie], label="error codo (cm)")
        for x in serie:
            if x["oclA"]:
                plt.axvspan(x["t"], x["t"] + 0.04, color="red", alpha=0.08, lw=0)
            if x["oclB"]:
                plt.axvspan(x["t"], x["t"] + 0.04, color="blue", alpha=0.08, lw=0)
        plt.xlabel("t (s)"); plt.ylabel("cm"); plt.legend(); plt.grid(alpha=0.3)
        plt.title("Fusión 2 cámaras — rojo: mano hacia A, azul: mano hacia B")
        plt.tight_layout(); plt.savefig("simulacion_fusion.png", dpi=120)
        print("→ simulacion_fusion.png")
