"""
Orquestador por fotograma: cámaras → fusión → retargeting (límites humanos) → IK → 6 joints.

Uso mínimo (ver la integración en Skill_Shield_Nivel_6.py y el simulador):

    teleop = TeleopBrazoFR5(ruta_mjcf, camaras={"A": Camara("A"), "B": Camara("B", R_ab, t_ab)})
    teleop.iniciar_calibracion()                # al pulsar [A]
    ...
    q, info = teleop.actualizar({"A": obs_a, "B": obs_b}, t, mano=(f_h, l_h), q_actual=q_robot)
    if q is not None: enviar q (6 joints) al generador suave / al robot

Calibración al pulsar [A] (28-oct: TODO automático, sin medir nada a mano):
  1. "camaras": se comprueba la extrínseca A↔B con los puntos del cuerpo; si no vale, se
     recalcula con el propio cuerpo (autocalib.AutoExtrinseca) — mueve un poco los brazos.
  2. "huesos": T-pose con el brazo estirado: longitudes de brazo y antebrazo (solo fotogramas
     con hombro/codo/muñeca triangulados por las dos cámaras y brazo recto) y VERTICAL del
     tronco (caderas → hombros), que sustituye a la vertical de la cámara A.
Durante la teleoperación se vigila la extrínseca y, si A y B dejan de coincidir (alguien movió
una cámara), se recalibra sola en segundo plano.

Estados (info["estado"]): "calibrando", "desarmado", "sin_medida", "hold", "exacta", "aprox",
"transicion", "bloqueado".

ARMADO Y ENTRADA LENTA (5-nov): calibrar NO mueve el robot.
  · Con cfg.requiere_armado, al acabar la calibración el estado es "desarmado": no se devuelve
    ningún q (el robot se queda donde está) hasta que alguien llame a armar() (el pedal).
    desarmar() lo vuelve a dejar quieto; iniciar_calibracion() siempre desarma.
  · Cada vez que la IK arranca desde la postura real del robot (al armar, o al acabar de calibrar
    si no hace falta armar), la salida va como mucho a cfg.v_entrada rad/s por joint hasta estar a
    menos de cfg.entrada_fin_rad del objetivo; luego, seguimiento normal (TELEOP_FR5_V_MAX).
    Antes ese arranque iba a v_transicion = 1,2 rad/s y TELEOP_FR5_V_MAX no lo limitaba.
    info["entrada"] = True mientras dura.
"""
import math
import time
from dataclasses import dataclass

import numpy as np

from skill_shield.teleop_fr5.autocalib import AutoExtrinseca, ParametrosAutoExt
from skill_shield.teleop_fr5.cinematica import CadenaFR5, cargar_cadena
from skill_shield.teleop_fr5.fusion_muneca_codo import CalibradorHuesos, Fusor2Cam, Parametros
from skill_shield.teleop_fr5.ik_diferencial import ControlIKDiferencial, ParametrosIKDif
from skill_shield.teleop_fr5.ik_teleop import ControlIKFR5, Entorno, ModeloColision, ParametrosIK
from skill_shield.teleop_fr5.retargeting import Anatomia, ConfigRetargeting, LimitesAnatomicos, Retargeting


@dataclass
class ConfigTeleop:
    brazo: str = "derecho"
    zona_muerta_m: float = 0.004          # la muñeca del robot no se mueve por temblores menores
    hold_s: float = 0.5                   # s que se mantiene el último objetivo sin medida nueva
    longitudes_manual: tuple = None       # (brazo, antebrazo) m si no se quiere calibrar
    n_calibracion: int = 30
    ik: str = "diferencial"               # "diferencial" (ik_diferencial, 28-oct) | "analitica" (ik_teleop)
    auto_extrinseca: bool = True          # comprobar / recalcular la extrínseca A↔B con el cuerpo
    vigilar_extrinseca: bool = True       # ... y vigilarla durante la teleoperación
    vertical_auto: bool = True            # vertical del tronco medida en la T-pose
    huesos_triangulados: bool = True      # huesos solo con las dos cámaras (si hay dos)
    requiere_armado: bool = False         # 5-nov: tras calibrar, quieto hasta armar() (pedal)
    v_entrada: float = None               # 5-nov: rad/s máx. al arrancar desde la postura real (None = sin tope)
    entrada_fin_rad: float = 0.05         # a menos de esto (rad, cada joint) del objetivo acaba la entrada


class TeleopBrazoFR5:
    def __init__(self, ruta_mjcf=None, camaras=None, cfg: ConfigTeleop = None,
                 par_fusion: Parametros = None, cfg_retarget: ConfigRetargeting = None,
                 lim_anatomicos: LimitesAnatomicos = None, par_ik: ParametrosIK = None,
                 entorno: Entorno = None, cadena: CadenaFR5 = None, verbose=True,
                 par_ik_dif: ParametrosIKDif = None, par_autoext: ParametrosAutoExt = None,
                 al_calibrar_extrinseca=None):
        self.cfg = cfg or ConfigTeleop()
        self.verbose = verbose
        self.cadena = cadena if cadena is not None else cargar_cadena(ruta_mjcf, verbose=verbose)
        self.fusor = Fusor2Cam(camaras or {}, self.cfg.brazo, par_fusion)
        self.calib = CalibradorHuesos(self.cfg.n_calibracion, self.cfg.longitudes_manual)
        self.retarget = Retargeting(self.cadena, cfg_retarget, Anatomia(lim_anatomicos))
        # la postura humana del robot usa los mismos ejes y el mismo cono de hombro que el retargeting
        par_ik = par_ik or ParametrosIK()
        rc, la = self.retarget.cfg, self.retarget.anatomia.lim
        par_ik.ejes = (tuple(rc.fuera), tuple(rc.arriba), tuple(rc.delante))
        par_ik.hombro_az_min, par_ik.hombro_az_max = la.hombro_az_min, la.hombro_az_max
        par_ik.hombro_ext_elev = la.hombro_ext_elev
        par_ik.codo_flex_max_deg = la.codo_flex_max
        if self.cfg.ik == "analitica":
            self.ik = ControlIKFR5(self.cadena, ModeloColision(self.cadena, entorno), par_ik)
        else:
            self.ik = ControlIKDiferencial(self.cadena, ModeloColision(self.cadena, entorno), par_ik, par_ik_dif)
        if verbose and self.ik.postura is not None:
            if self.cfg.ik == "analitica":
                lo, hi = np.degrees(self.ik.lo), np.degrees(self.ik.hi)
                print("🧍 IK analítica. Postura humana del FR5: codo joint3 "
                      f"{'>' if self.ik.postura.codo_signo > 0 else '<'} 0 (0…{par_ik.codo_flex_max_deg:.0f}°), "
                      f"muñeca |q4|≤{par_ik.muneca_q4_deg:.0f}° |q5−90°|≤{par_ik.muneca_q5_deg:.0f}°, "
                      f"hombro q1 [{lo[0]:.0f}, {hi[0]:.0f}]° q2 [{lo[1]:.0f}, {hi[1]:.0f}]° + cono")
            else:
                lo, hi = np.degrees(self.ik.lo_h), np.degrees(self.ik.hi_h)
                print("🧍 IK diferencial (una sola rama, sin transiciones; límites humanos como coste): "
                      f"codo joint3 {'>' if self.ik.postura.codo_signo > 0 else '<'} 0, "
                      f"muñeca |q4|≤{par_ik.muneca_q4_deg:.0f}° |q5−90°|≤{par_ik.muneca_q5_deg:.0f}°, "
                      f"banda hombro q1 [{lo[0]:.0f}, {hi[0]:.0f}]° q2 [{lo[1]:.0f}, {hi[1]:.0f}]°")
        # extrínseca automática (solo con dos cámaras)
        self.autoext = None
        if "B" in self.fusor.cams and (self.cfg.auto_extrinseca or self.cfg.vigilar_extrinseca):
            self.autoext = AutoExtrinseca(self.fusor.par, par_autoext)
        self.al_calibrar_extrinseca = al_calibrar_extrinseca
        self.fase_cal = None                  # "camaras" | "huesos" | None
        self.sin_b = False                    # B descartada (extrínseca imposible de calcular)
        self.t_fase = None
        self.troncos = []
        self.aviso = ""                       # texto para el HUD (recalibración, etc.)
        self.recal_t0 = None
        self.n_pares_ext = 0                  # puntos con los que se calculó la extrínseca en uso
        self.base_res = 0.05                  # mediana A↔B "normal" con la extrínseca en uso (m)
        self.t_refino = -1e9
        self.d_antes = None
        self.q = None
        self.obj_prev = None
        self.t_obj = None
        self.p_w_filtrada = None
        self.info = {"estado": "sin_medida"}
        self.ultima_medida = None             # salida de la fusión del último fotograma (diagnóstico)
        self.armado = False                   # 5-nov: pedal (solo cuenta con cfg.requiere_armado)
        self._entrando = False
        self._q_entrada = None
        if self.calib.longitudes is not None:
            self.fusor.longitudes = self.calib.longitudes

    # ── calibración (al pulsar [A]) ──────────────────────────────────────────
    # ── armado (pedal) ────────────────────────────────────────────────────────
    def armar(self):
        """El robot empieza a seguirte, entrando despacio desde donde esté."""
        self.armado = True

    def desarmar(self):
        """El robot se queda quieto; al volver a armar, entra otra vez despacio."""
        self.armado = False

    def iniciar_calibracion(self):
        self.armado = False                   # calibrar nunca mueve el robot
        self.calib.iniciar()
        self.fusor.reiniciar()
        self.retarget.reiniciar_roll()
        self.ik.q = None                      # la IK arranca de la postura real del robot (q_actual)
        self.troncos = []
        self.sin_b = False
        self.t_fase = None
        if self.autoext is not None and self.cfg.auto_extrinseca:
            self.autoext.reiniciar()
            self.fase_cal = "camaras"
        else:
            self.fase_cal = "huesos"

    @property
    def calibrado(self):
        return self.calib.longitudes is not None and self.fase_cal is None

    def _aplicar_extrinseca(self, R, t, info, motivo, reiniciar=True):
        cam = self.fusor.cams["B"]
        t_ant = cam.t.copy()
        cam.R, cam.t = np.asarray(R, float), np.asarray(t, float)
        self.n_pares_ext = int(info.get("n_total", info.get("inliers", 0)))
        self.base_res = float(info.get("residuo_todos_m", info.get("residuo_m", 0.05)))
        if reiniciar or float(np.linalg.norm(cam.t - t_ant)) > 0.03:
            self.fusor.reiniciar()
        if self.verbose:
            ang = math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))
            print(f"📐 Extrínseca A↔B {motivo}: residuo {1000 * self.base_res:.0f} mm con "
                  f"{info.get('n_total', info['inliers'])} puntos del cuerpo ({info['fotogramas']} fotogramas); "
                  f"B a {np.linalg.norm(t):.2f} m de A, girada {ang:.0f}°")
        if self.al_calibrar_extrinseca is not None:
            try:
                self.al_calibrar_extrinseca(cam.R, cam.t, info)
            except Exception as e:                         # pragma: no cover
                print(f"⚠️  No se pudo guardar la extrínseca: {e}")

    def _umbral_vigilancia(self):
        par = self.autoext.par
        return max(par.umbral_mal_m, par.factor_mal * self.base_res + par.margen_mal_m)

    def _fase_camaras(self, t):
        """Devuelve texto para el HUD mientras dura; pasa a 'huesos' al terminar. El ajuste corre
        en un hilo aparte: el bucle de la cámara no se para."""
        ae, par = self.autoext, self.autoext.par
        if self.t_fase is None:
            self.t_fase = t
        cam = self.fusor.cams["B"]
        dt = t - self.t_fase
        sol = ae.resultado()
        if sol is not None and sol[0] is not None and sol[2]["aceptada"]:
            d = ae.discrepancia(cam.R, cam.t)
            if d is not None and d <= sol[2]["residuo_todos_m"] + 0.015:
                self.base_res = d
                if self.verbose:
                    print(f"📐 Extrínseca A↔B en uso correcta ({1000 * d:.0f} mm, el mejor ajuste "
                          f"posible da {1000 * sol[2]['residuo_todos_m']:.0f} mm)")
            else:
                self._aplicar_extrinseca(sol[0], sol[1], sol[2],
                                         f"recalculada con el cuerpo (antes no coincidían: {1000 * (d or 0):.0f} mm)")
            self.fase_cal, self.t_fase = "huesos", None
            return None
        if dt >= par.t_comprobar_s:
            d = ae.discrepancia(cam.R, cam.t)
            if d is not None and d <= par.umbral_ok_m:
                self.base_res = d
                if self.verbose:
                    print(f"📐 Extrínseca A↔B en uso correcta (A y B coinciden en {1000 * d:.0f} mm)")
                self.fase_cal, self.t_fase = "huesos", None
                return None
            if not ae.en_curso and ae.listo():
                ae.resolver_async()
        if dt >= par.t_max_calib_s:
            print("⚠️  No se pudo calcular la extrínseca A↔B con el cuerpo (¿B no te ve entero?): "
                  "sigo SOLO con la cámara A.")
            self.sin_b = True
            self.fase_cal, self.t_fase = "huesos", None
            return None
        return (f"CALIBRANDO CAMARAS {int(100 * ae.progreso())}% - de frente a las dos camaras, "
                "mueve los brazos despacio")

    def _vigilar(self, t):
        """Durante la teleoperación (coste fijo por fotograma; los ajustes van en otro hilo):
          · refinado: cada refinar_cada_s, ajuste con todo lo acumulado; se aplica si usa ≥1,5×
            más puntos que el que está en uso y no empeora el residuo;
          · vigilancia: si la mediana A↔B reciente supera el umbral (adaptado al residuo real de
            la sala) durante t_mal_s, se recalibra desde cero (alguien movió una cámara)."""
        ae = self.autoext
        cam = self.fusor.cams["B"]
        sol = ae.resultado()
        if sol is not None and sol[0] is not None:
            if self.recal_t0 is not None:
                if sol[2]["aceptada"] and sol[2]["residuo_todos_m"] < 0.7 * (self.d_antes or 1.0):
                    self._aplicar_extrinseca(sol[0], sol[1], sol[2], "recalibrada en marcha")
                    self.recal_t0, self.aviso = None, ""
                    ae.t_mal = None
            else:
                actual = ae.discrepancia(cam.R, cam.t)
                if sol[2]["aceptada"] and actual is not None and ae.t_mal is None \
                        and sol[2]["n_total"] >= 1.5 * self.n_pares_ext \
                        and sol[2]["residuo_todos_m"] <= actual + 0.005:
                    self._aplicar_extrinseca(sol[0], sol[1], sol[2],
                                             f"refinada en marcha ({1000 * actual:.0f} → "
                                             f"{1000 * sol[2]['residuo_todos_m']:.0f} mm)", reiniciar=False)
        if self.recal_t0 is None:
            if not ae.en_curso and ae.t_mal is None and t - self.t_refino >= ae.par.refinar_cada_s \
                    and ae.n_pares >= 1.5 * max(self.n_pares_ext, 1) and ae.listo():
                self.t_refino = t
                ae.resolver_async()
            if ae.vigilar(t, self._umbral_vigilancia()):
                d = ae.discrepancia_reciente(t)
                print(f"⚠️  A y B ya no coinciden ({1000 * (d or 0):.0f} mm, lo normal aquí es "
                      f"{1000 * self.base_res:.0f} mm): recalibrando la extrínseca con el cuerpo")
                self.recal_t0 = t
                self.d_antes = d
                ae.reiniciar(t)
            self.aviso = ""
            return
        self.aviso = f"RECALIBRANDO CAMARAS {int(100 * ae.progreso())}%"
        if not ae.en_curso and ae.listo():
            ae.resolver_async()
        if t - self.recal_t0 > ae.par.t_max_calib_s:
            self.recal_t0, self.aviso = None, ""
            ae.t_mal = None

    def _tronco(self, t):
        """Dirección caderas→hombros del filtro (marco A) o None."""
        kf = self.fusor.kf
        ps = []
        for k in ("hombro", "hombro2", "cadera", "cadera2"):
            f = kf[k]
            if f.p is None or f.t_ult_medida is None or (t - f.t_ult_medida) > 0.3:
                return None
            ps.append(f.p)
        v = (ps[0] + ps[1]) / 2 - (ps[2] + ps[3]) / 2
        n = float(np.linalg.norm(v))
        return v / n if n > 0.2 else None

    def _fijar_vertical(self):
        if not self.troncos:
            if self.verbose:
                print("📏 Vertical: no se vieron las caderas en la T-pose; uso la vertical de la cámara A")
            return
        v = np.median(np.array(self.troncos), axis=0)
        v /= np.linalg.norm(v)
        ang = math.degrees(math.acos(float(np.clip(v @ np.array([0.0, -1.0, 0.0]), -1, 1))))
        if ang > 50.0:
            print(f"⚠️  Vertical del tronco a {ang:.0f}° de la de la cámara A: demasiado, la ignoro")
            return
        self.fusor.par.arriba_fija = tuple(float(x) for x in v)
        if self.fusor.par.arriba_ref == "camara":
            self.fusor.par.arriba_ref = "calibrada"
        if self.verbose:
            print(f"📏 Vertical del tronco calibrada ({len(self.troncos)} fotogramas): la cámara A está "
                  f"inclinada {ang:.1f}° respecto a ti")

    # ── fotograma ────────────────────────────────────────────────────────────
    def actualizar(self, observaciones: dict, t=None, mano=None, roll=None, q_actual=None, dt=None):
        """
        observaciones: {nombre_cam: ObservacionCamara | None}
        mano: (f_h, l_h) vectores de la mano en ejes de la cámara A (MediaPipe Hands) o None
        roll: pronosupinación (rad) de otra fuente (IMU del guante) o None
        q_actual: joints reales del robot (6) para arrancar la IK sin saltos
        Devuelve (q o None, info).
        """
        t = time.perf_counter() if t is None else float(t)
        obs = dict(observaciones)
        if self.autoext is not None and not self.sin_b:
            # calibrando: todos los fotogramas; teleoperando: 1 de cada N (coste por fotograma)
            self._n_fot = getattr(self, "_n_fot", 0) + 1
            if self.fase_cal is not None or self.recal_t0 is not None \
                    or self._n_fot % max(1, self.autoext.par.cada_n_teleop) == 0:
                cb = self.fusor.cams["B"]
                vig = self.fase_cal is None and self.recal_t0 is None
                self.autoext.alimentar(obs.get("A"), obs.get("B"), t,
                                       cb.R if vig else None, cb.t if vig else None)
        if self.sin_b:
            obs.pop("B", None)
        if self.fase_cal == "camaras":
            txt = self._fase_camaras(t)
            if txt is not None:
                self.ultima_medida = None
                self.info = dict(estado="calibrando", fase="camaras", texto=txt, progreso=0)
                return None, self.info
        elif self.fase_cal is None and self.autoext is not None and self.cfg.vigilar_extrinseca and not self.sin_b:
            self._vigilar(t)
        res = self.fusor.actualizar(obs, t)
        self.ultima_medida = res
        return self.actualizar_desde_medida(res, t, mano=mano, roll=roll, q_actual=q_actual, dt=dt,
                                            fuentes=self.fusor.resumen)

    def actualizar_desde_medida(self, res, t, mano=None, roll=None, q_actual=None, dt=None, fuentes="-"):
        """La parte de actualizar() que va DESPUÉS de la fusión: calibración de huesos y vertical,
        retargeting e IK. 'res' tiene la forma de la salida de Fusor2Cam.actualizar (hombro,
        codo, muñeca y torso en el marco A) o es None. Sirve para meter en la IK una medida que
        no viene de las cámaras (trayectoria limpia, ver diagnostico.py)."""
        t = float(t)
        if self.ik.q is None and q_actual is not None:
            self.ik.q = np.asarray(q_actual, dtype=float).copy()
            self.ik.rama = None
            if hasattr(self.ik, "arrancando"):
                self.ik.arrancando = True
            self.retarget.reiniciar_roll(self.cadena.fk(self.ik.q)["T_tool"][:3, :3])
            self._entrando = self.cfg.v_entrada is not None   # 5-nov: arranque lento
            self._q_entrada = self.ik.q.copy()
        if self.fase_cal == "camaras":            # (solo si se llama directamente sin cámaras)
            self.fase_cal = "huesos"
        if self.fase_cal == "huesos" or (self.fase_cal is None and self.calib.longitudes is None):
            if self.fase_cal == "huesos" and self.t_fase is None:
                self.t_fase = t
            if res is not None:
                dos = len([c for c in self.fusor.cams if not (self.sin_b and c == "B")]) >= 2
                tri = res.get("triangulado", frozenset())
                fiable = (not dos or not self.cfg.huesos_triangulados
                          or {"hombro", "codo", "muneca"} <= set(tri))
                if self.calib.calibrando:
                    self.calib.alimentar(res["hombro"], res["codo"], res["muneca"], fiable=fiable)
                if fiable and self.cfg.vertical_auto and self.fase_cal == "huesos":
                    v = self._tronco(t)
                    if v is not None:
                        self.troncos.append(v)
            if self.calib.longitudes is not None:
                self.fusor.longitudes = self.calib.longitudes
                if self.fase_cal == "huesos":
                    vertical_lista = (not self.cfg.vertical_auto or len(self.troncos) >= 15
                                      or (t - self.t_fase) > 3.0)
                    if vertical_lista:
                        if self.verbose and self.calib.manual is None:
                            print(f"🦴 Huesos calibrados: brazo {100 * self.calib.longitudes[0]:.1f} cm, "
                                  f"antebrazo {100 * self.calib.longitudes[1]:.1f} cm")
                        if self.cfg.vertical_auto:
                            self._fijar_vertical()
                        self.fase_cal, self.t_fase = None, None
        if not self.calibrado:
            self.info = dict(estado="calibrando", fase="huesos", progreso=self.calib.progreso, fuentes=fuentes,
                             texto=f"CALIBRANDO HUESOS {self.calib.progreso}/{self.calib.n} - T-pose, brazo estirado")
            return None, self.info

        armado = self.armado                  # (lo cambia el bucle principal: se lee una vez)
        if self.cfg.requiere_armado and not armado:
            # Calibrado pero SIN ARMAR: no se manda nada (el robot se queda donde está) y la IK
            # se reinicia para arrancar, al armar, desde la postura real del robot.
            self.ik.q = None
            self.q = None
            self.info = dict(estado="desarmado", fuentes=fuentes, aviso=self.aviso,
                             texto="CALIBRADO - pisa el PEDAL para armar el brazo")
            return None, self.info

        obj = None
        if res is not None:
            obj = self.retarget.objetivo(res, self.calib.longitudes, mano=mano, roll=roll, t=t)
        if obj is not None:
            # zona muerta sobre la muñeca
            p = obj["p_muneca"]
            if self.p_w_filtrada is not None and np.linalg.norm(p - self.p_w_filtrada) < self.cfg.zona_muerta_m:
                obj["p_muneca"] = self.p_w_filtrada.copy()
            else:
                self.p_w_filtrada = p.copy()
            self.obj_prev, self.t_obj = obj, t
        elif self.obj_prev is not None and (t - self.t_obj) <= self.cfg.hold_s:
            obj = self.obj_prev                     # sin medida nueva: se mantiene el objetivo
        if obj is None:
            self.info = dict(estado="sin_medida", fuentes=fuentes, aviso=self.aviso)
            return self.q, self.info

        q, info_ik = self.ik.resolver(obj, t, dt)
        entrando = False
        if self._entrando:
            # 5-nov: entrada lenta. La salida va hacia la de la IK a v_entrada como mucho, y la
            # IK sigue desde la postura limitada (es por donde va el robot de verdad).
            q = np.asarray(q, dtype=float)
            d = q - self._q_entrada
            if float(np.abs(d).max()) <= self.cfg.entrada_fin_rad:
                self._entrando = False
                if self.verbose:
                    print("🟢 TELEOP v2: brazo alcanzado → seguimiento normal")
            else:
                paso = self.cfg.v_entrada * float(dt) if dt else 0.0
                q = self._q_entrada + np.clip(d, -paso, paso)
                self.ik.q = q.copy()
                entrando = True
            self._q_entrada = q.copy()
        self.q = q
        self.info = dict(info_ik)
        self.info["entrada"] = entrando
        self.info.update(fuentes=fuentes, codo_reconstruido=res.get("codo_reconstruido") if res else None,
                         hold=res is None, escala=obj["escala"], roll=obj["roll"], aviso=self.aviso,
                         anatomia=(self.retarget.anatomia.n_hombro, self.retarget.anatomia.n_codo))
        return q, self.info


class TeleopEnParalelo:
    """
    Ejecuta TeleopBrazoFR5.actualizar en un hilo aparte (28-oct, fluidez).

    El bucle principal entrega el fotograma N (lanzar) y, en la vuelta siguiente, recoge su
    resultado (recoger) antes de entregar el N+1. Mientras tanto espera a la cámara y corre
    MediaPipe (C++, sin el GIL), así que la fusión + IK quedan casi gratis en el tiempo de vuelta.
    Coste: el comando del brazo llega UN fotograma más tarde (~35 ms a 28 FPS).

    Reglas: todo lo que toque el teleop (iniciar_calibracion, registro de diagnóstico...) se hace
    después de recoger() y antes de lanzar(): así el hilo nunca está a mitad de un fotograma.
    """

    def __init__(self, teleop: TeleopBrazoFR5):
        from concurrent.futures import ThreadPoolExecutor
        self.teleop = teleop
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TeleopV2")
        self._fut = None
        self.ultimo = (None, {"estado": "sin_medida"})

    def recoger(self):
        """Espera (si hace falta) al fotograma lanzado y devuelve su (q, info). Si no hay ninguno
        pendiente, devuelve el último resultado."""
        if self._fut is not None:
            fut, self._fut = self._fut, None
            try:
                q, info = fut.result()
                self.ultimo = (None if q is None else np.array(q, dtype=float), dict(info))
            except Exception as e:                              # pragma: no cover
                print(f"⚠️  [TELEOP v2] error en el hilo: {e}")
        return self.ultimo

    def lanzar(self, observaciones, t=None, **kw):
        if self._fut is not None:
            self.recoger()
        self._fut = self._pool.submit(self.teleop.actualizar, observaciones, t, **kw)

    def cerrar(self):
        self.recoger()
        self._pool.shutdown(wait=True)
