"""
Pruebas del módulo skill_shield/teleop_fr5 (sin hardware ni MuJoCo):

    python -m pytest tests/test_teleop_fr5.py -q          # o simplemente:  python tests/test_teleop_fr5.py
"""
import math
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del repo (tests/..)
sys.path.insert(0, RAIZ)

from skill_shield.teleop_fr5.cinematica import CadenaFR5, OFFSET_MUNECA_ORCA, T_de, cargar_cadena
from skill_shield.teleop_fr5.ik_teleop import ControlIKFR5, ModeloColision, ParametrosIK, entorno_soporte_hombros
from skill_shield.teleop_fr5.retargeting import Retargeting, ConfigRetargeting

RUTA_MJCF = os.path.join(RAIZ,
                         "fairino_description", "mjcf", "fairino_mjcf.xml")


def _cadena():
    return CadenaFR5.desde_mjcf(RUTA_MJCF) if os.path.exists(RUTA_MJCF) else CadenaFR5.desde_dh()


def test_fk_equivale_a_dh():
    c = _cadena()
    dh = c.dh_equivalente()
    ref = CadenaFR5.desde_dh(params=dh["params"], T_base=c.T_base @ dh["T_0"], T_flange_tool=np.eye(4))
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = rng.uniform(c.limites[:, 0], c.limites[:, 1])
        Ta = c.fk_marcos(q)[6]
        Tb = ref.fk_marcos(q * dh["signos"])[6] @ dh["T_6tool"]
        assert np.abs(Ta - Tb).max() < 1e-9


def test_ik_analitica_ida_y_vuelta():
    c = _cadena()
    rng = np.random.default_rng(2)
    recuperadas = 0
    for _ in range(300):
        q = rng.uniform(c.limites[:, 0], c.limites[:, 1])
        T = c.fk_marcos(q)[6]
        sols = c.ik_analitica(T, q_ref=q)
        assert sols, "sin soluciones para una pose alcanzable"
        for s in sols:
            Ts = c.fk_marcos(s)[6]
            assert np.linalg.norm(Ts[:3, 3] - T[:3, 3]) < 1e-6
            assert np.abs(Ts[:3, :3] - T[:3, :3]).max() < 1e-6
        if min(np.abs(np.asarray(s) - q).max() for s in sols) < 1e-6:
            recuperadas += 1
    assert recuperadas == 300


def test_jacobiana_numerica():
    c = _cadena()
    q = np.array([0.3, -1.1, 1.4, -0.7, 1.2, 0.5])
    Jl, Ja = c.jacobiana(q, OFFSET_MUNECA_ORCA)
    p0 = c.punto(q, OFFSET_MUNECA_ORCA)
    h = 1e-6
    for i in range(6):
        dq = np.zeros(6); dq[i] = h
        num = (c.punto(q + dq, OFFSET_MUNECA_ORCA) - p0) / h
        assert np.abs(num - Jl[:, i]).max() < 1e-4


def test_fusion_robusta_a_oclusion():
    from skill_shield.teleop_fr5.simulador import simular_fusion
    resumen, serie = simular_fusion(verbose=False)
    for f, r in resumen.items():
        assert r["ok"] == r["n"], f"fase {f}: fotogramas sin salida"
        assert r["eW_med"] < 0.02, f"fase {f}: muñeca med {r['eW_med']:.3f} m"
        assert r["eE_med"] < 0.02, f"fase {f}: codo med {r['eE_med']:.3f} m"
        assert r["eW_p95"] < 0.04, f"fase {f}: muñeca p95 {r['eW_p95']:.3f} m"
        assert r["eE_p95"] < 0.06, f"fase {f}: codo p95 {r['eE_p95']:.3f} m"
    # la fase 2 (mano hacia la cámara A todo el rato) es la que rompía la fusión antigua
    assert resumen[2]["eE_ing_p95"] > resumen[2]["eE_p95"] * 3


def test_teleop_sin_colisiones_ni_limites():
    from skill_shield.teleop_fr5.simulador import simular_teleop
    res, serie = simular_teleop(verbose=False)
    assert res["fuera_limites"] == 0
    assert res["colisiones"] == 0
    assert res["v_max"] <= 3.0 + 1e-6
    assert res["estados"].get("exacta", 0) + res["estados"].get("aprox", 0) > 380
    assert res["err_med"] < 0.005 and res["err_p95"] < 0.03
    # postura humana: codo de un lado, muñeca sin voltear, sin "vueltas"
    Q = np.array([x["q"] for x in serie if "q" in x])
    assert np.all(Q[:, 4] > 0) and np.all(np.abs(Q[:, 3]) <= math.radians(80) + 1e-6)
    assert np.all(Q[:, 2] >= -math.radians(5) - 1e-6)
    assert np.abs(np.diff(Q, axis=0)).max() < 0.2           # ningún salto articular entre fotogramas


def test_otro_camino_cambio_de_rama():
    """Objetivo alcanzable solo desde otra rama de la IK → transición planificada, sin
    colisiones, a velocidad limitada, que termina en la rama nueva con error ~0."""
    c = _cadena()
    inclinada = not np.allclose(c.T_base[:3, :3], np.eye(3))
    col = ModeloColision(c, entorno_soporte_hombros() if inclinada else None)
    par = ParametrosIK(limites_extra={2: (-2.53, 2.53)}, postura_humana=False)   # todas las ramas
    ctl = ControlIKFR5(c, col, par)
    rng = np.random.default_rng(5)

    def objetivo_de(q):
        fk = c.fk(q)
        return dict(p_muneca=fk["muneca"], R_mano=fk["T_tool"][:3, :3], p_codo_sugerido=fk["codo"])

    # arranque en una postura válida
    while True:
        q_a = rng.uniform(ctl.lo, ctl.hi)
        if ctl._valida(q_a)[0]:
            break
    ctl.q = q_a.copy()
    q, info = ctl.resolver(objetivo_de(q_a), 0.0, 0.04)
    assert info["estado"] == "exacta"
    rama_a = info["rama"]
    # buscar un objetivo cuyas soluciones válidas NO incluyan la rama actual y estén lejos
    encontrado = None
    for _ in range(2000):
        q_b = rng.uniform(ctl.lo, ctl.hi)
        if not ctl._valida(q_b)[0]:
            continue
        T = c.fk_marcos(q_b)[6]
        validas = [(s, r) for s, r in c.ik_analitica(T, q_ref=ctl.q, con_rama=True) if ctl._valida(s)[0]]
        if validas and all(r != rama_a for _, r in validas) and ctl._dist_q(validas[0][0], ctl.q) > 2.0:
            encontrado = q_b
            break
    assert encontrado is not None
    obj = objetivo_de(encontrado)
    t, dt = 0.0, 0.04
    estados = []
    for k in range(400):
        t += dt
        q, info = ctl.resolver(obj, t, dt)
        estados.append(info["estado"])
        assert ctl.en_limites(q)
        assert col.libre(q), f"colisión durante '{info['estado']}' en el paso {k}"
        if k > 0:
            assert np.abs(q - q_prev).max() <= max(par.v_transicion, par.v_seguimiento) * dt + 1e-9
        q_prev = q.copy()
        if info["estado"] == "exacta" and info.get("error", 1.0) < 1e-6:
            break
    assert "transicion" in estados, estados[:10]
    assert info["estado"] == "exacta" and info["rama"] != rama_a
    assert np.linalg.norm(c.punto(q, OFFSET_MUNECA_ORCA) - obj["p_muneca"]) < 1e-6


def test_postura_humana_descarta_munecas_volteadas():
    """Con la postura humana activa, ninguna solución elegida tiene la muñeca volteada ni el
    codo al revés, y las cajas articulares salen de los límites anatómicos."""
    c = _cadena()
    ctl = ControlIKFR5(c, ModeloColision(c), ParametrosIK())
    assert ctl.postura is not None and ctl.postura.codo_signo in (1, -1)
    assert math.degrees(ctl.hi[3]) <= 80.0 + 1e-6 and math.degrees(ctl.lo[4]) >= 30.0 - 1e-6
    rng = np.random.default_rng(9)
    n = 0
    for _ in range(300):
        q = rng.uniform(ctl.lo, ctl.hi)
        if not ctl._valida(q)[0]:
            continue
        fk = c.fk(q)
        obj = dict(p_muneca=fk["muneca"], R_mano=fk["T_tool"][:3, :3], p_codo_sugerido=fk["codo"])
        ctl.q, ctl.rama = None, None
        qs, info = ctl.resolver(obj, 0.0, None)
        assert qs[4] > 0 and abs(qs[3]) <= math.radians(80) + 1e-6
        assert ctl.postura.codo_signo * qs[2] >= -math.radians(5) - 1e-6
        n += 1
    assert n > 50


def test_objetivo_imposible_no_choca():
    """Muñeca pedida dentro de la columna del soporte / bajo el suelo: el robot se acerca lo
    que puede ('aprox' o 'bloqueado') sin chocar ni salirse de límites."""
    c = _cadena()
    if np.allclose(c.T_base[:3, :3], np.eye(3)):
        return                                    # el entorno de columna es del soporte inclinado
    col = ModeloColision(c, entorno_soporte_hombros())
    ctl = ControlIKFR5(c, col, ParametrosIK())
    q0 = np.array([0.0, -1.5708, 1.35, 0.0, 1.5708, 0.0])
    assert col.libre(q0)
    ctl.q = q0.copy()
    rt = Retargeting(c, ConfigRetargeting())
    t = 0.0
    for p_obj in (np.array([0.26, 0.0, -0.2]), np.array([-0.4, -0.3, -1.1]), np.array([-0.8, -0.9, 0.2])):
        R = rt.R_mano(np.array([0.0, -1.0, 0.0]), 0.0)
        for _ in range(60):
            t += 0.04
            q, info = ctl.resolver(dict(p_muneca=p_obj, R_mano=R), t, 0.04)
            assert ctl.en_limites(q)
            assert col.libre(q), info
        assert info["estado"] in ("aprox", "bloqueado", "exacta", "transicion")


def test_diagnostico_ik_trayectoria_limpia():
    """Separar medida de IK: la trayectoria limpia entra directamente en retargeting + IK (sin
    cámaras) con la config de la app. Comprueba que el diagnóstico corre, que todo fotograma
    tiene estado, que el robot no se sale de límites ni salta, y que donde hay solución exacta
    el error frente a la verdad es nulo (sin ruido, el error solo puede venir de la IK)."""
    from skill_shield.teleop_fr5.diagnostico import construir_teleop, diagnosticar_ik
    teleop, q0, _ = construir_teleop()
    res, serie = diagnosticar_ik(0.0, 0.0, fps=10.0, teleop=teleop, q0=q0, verbose=False)
    assert res["n"] > 250
    assert sum(res["estados"].values()) == res["n"]
    assert set(res["estados"]) <= {"exacta", "aprox", "transicion", "bloqueado", "sin_medida"}
    assert res["dq_max"] <= 3.0 * 0.1 + 1e-6               # límite de velocidad a 10 FPS
    ex = [x for x in serie if x["estado"] == "exacta" and x["fase"] != "arranque" and "err_real" in x]
    assert len(ex) > 0.5 * res["n"]
    assert np.median([x["err_real"] for x in ex]) < 1e-3


def test_diagnostico_ruido_empeora_la_ik():
    """Con ruido en la medida, el error frente a la verdad crece (lo que mide el barrido)."""
    from skill_shield.teleop_fr5.diagnostico import construir_teleop, diagnosticar_ik
    t0, q0, _ = construir_teleop()
    limpio, _ = diagnosticar_ik(0.0, 0.0, teleop=t0, q0=q0, verbose=False)
    t1, q1, _ = construir_teleop()
    ruidoso, _ = diagnosticar_ik(2.0, 4.0, teleop=t1, q0=q1, verbose=False)
    assert ruidoso["err_real_med"] > limpio["err_real_med"] + 0.01


def test_registro_quieto_con_camaras_sinteticas():
    """[R] de la app: 10 s quieto en T-pose con las cámaras sintéticas → informe coherente
    (huesos ≈ reales, elevación ≈ 90°, A y B comparadas) y archivo .npz que se puede releer."""
    import tempfile
    from skill_shield.teleop_fr5.diagnostico import RegistroQuieto, cargar, construir_teleop, informe_quieto
    from skill_shield.teleop_fr5.simulador import BrazoHumanoSintetico, CamaraSintetica, cargar_extrinsecas_repo
    rng = np.random.default_rng(3)
    brazo = BrazoHumanoSintetico()
    R_ab, t_ab = cargar_extrinsecas_repo()
    ca, cb = CamaraSintetica("A", rng=rng), CamaraSintetica("B", R_ab, t_ab, rng=rng)
    teleop, q0, _ = construir_teleop()
    teleop.fusor.cams = {"A": ca.cam, "B": cb.cam}
    S, E, W, _ = brazo.pose(0.5)                                  # T-pose
    with tempfile.TemporaryDirectory() as tmp:
        reg = RegistroQuieto(10.0, carpeta=tmp)
        reg.iniciar(0.0)
        k, fin = 0, False
        while not fin:
            t = k * 0.1
            obs = {"A": ca.observar(S, E, W, brazo.S2, brazo.C, brazo.C2, t),
                   "B": cb.observar(S, E, W, brazo.S2, brazo.C, brazo.C2, t)}
            teleop.actualizar(obs, t, q_actual=q0, dt=0.1)
            fin = reg.alimentar(t, teleop)
            k += 1
        assert reg.ruta and os.path.exists(reg.ruta)
        inf = informe_quieto(cargar(reg.ruta), imprimir=False)
    assert inf["n"] >= 100 and inf["frac_medida"] > 0.9
    assert abs(inf["huesos"]["brazo"][0] - 0.30) < 0.03 and abs(inf["huesos"]["antebrazo"][0] - 0.26) < 0.03
    assert abs(inf["angulos"]["elevación"]["med"] - 90.0) < 8.0
    assert abs(inf["angulos"]["azimut"]["med"] - 90.0) < 8.0
    assert all(a in inf["ab"] for a in ("hombro", "codo", "muneca"))
    assert inf["vertical_deg"] < 5.0
    for a in ("hombro", "codo", "muneca"):
        assert inf["arts"][a]["rms_mm"] < 60.0


def test_ik_diferencial_una_rama_sin_transiciones():
    """IK diferencial (28-oct): con la trayectoria limpia y con ruido + saltos A↔B, el robot
    nunca hace transiciones (salvo el arranque) ni se bloquea, no salta entre fotogramas y la
    orientación no se dispara (con la IK analítica, 2 cm de ruido ya daban ~30 transiciones)."""
    from skill_shield.teleop_fr5.diagnostico import construir_teleop, diagnosticar_ik
    for ruido, saltos, min_exacta in ((0.0, 0.0, 70.0), (2.0, 5.0, 40.0)):
        teleop, q0, _ = construir_teleop()
        assert teleop.cfg.ik == "diferencial"
        res, serie = diagnosticar_ik(ruido, saltos, teleop=teleop, q0=q0, verbose=False)
        tras = [x for x in serie if x["fase"] != "arranque" and x["estado"] == "transicion"]
        assert len(tras) <= 3, f"σ={ruido}: transiciones fuera del arranque"     # solo el final del arranque
        assert res["estados"].get("bloqueado", 0) == 0
        assert res["pct"].get("exacta", 0) >= min_exacta, res["pct"]
        assert res["ang_real_p95"] < 45.0
        assert res["dq_max"] <= 3.0 * 0.1 + 1e-6


def test_triangulacion_mejora_la_fusion():
    """Con las dos cámaras, triangular los rayos da menos error que el estéreo de cada una."""
    import skill_shield.teleop_fr5.simulador as sim
    from skill_shield.teleop_fr5.fusion_muneca_codo import Parametros
    orig = sim.Parametros
    try:
        err = {}
        for tri in (False, True):
            sim.Parametros = lambda **k: Parametros(triangular=tri, **k)
            r, _ = sim.simular_fusion(verbose=False)
            err[tri] = np.mean([r[f]["eW_p95"] for f in range(7)])
    finally:
        sim.Parametros = orig
    assert err[True] < 0.9 * err[False], err


def test_extrinseca_automatica_con_el_cuerpo():
    """Extrínseca A↔B equivocada ~42 cm / ~10° (como la del registro del 28-oct): al pulsar [A]
    se recalcula sola con los puntos del cuerpo, sin tablero, y se avisa para guardarla."""
    import math as _m
    from skill_shield.teleop_fr5.autocalib import _rotvec
    from skill_shield.teleop_fr5.diagnostico import construir_teleop
    from skill_shield.teleop_fr5.fusion_muneca_codo import Camara
    from skill_shield.teleop_fr5.simulador import BrazoHumanoSintetico, CamaraSintetica, cargar_extrinsecas_repo
    from skill_shield.teleop_fr5.teleop_brazo import ConfigTeleop, TeleopBrazoFR5
    rng = np.random.default_rng(0)
    brazo = BrazoHumanoSintetico()
    R, t = cargar_extrinsecas_repo()
    ca, cb = CamaraSintetica("A", rng=rng), CamaraSintetica("B", R, t, rng=rng)
    R_mal = _rotvec(np.radians([3, 8, -4])) @ R
    t_mal = t + np.array([-0.40, 0.03, 0.12])
    base, q0, _ = construir_teleop()
    guardadas = []
    tel = TeleopBrazoFR5(cadena=base.cadena, camaras={"A": Camara("A"), "B": Camara("B", R_mal, t_mal)},
                         cfg=ConfigTeleop(), cfg_retarget=base.retarget.cfg, entorno=base.ik.col.entorno,
                         verbose=False, al_calibrar_extrinseca=lambda R_, t_, i: guardadas.append(i))
    tel.iniciar_calibracion()
    fases = []
    for k in range(170):                  # cámaras (moviendo los brazos) → T-pose a los 12 s
        tt = k / 10.0
        S, E, W, _ = brazo.pose((tt + 9.0) % 21.0)
        obs = {"A": ca.observar(S, E, W, brazo.S2, brazo.C, brazo.C2, tt),
               "B": cb.observar(S, E, W, brazo.S2, brazo.C, brazo.C2, tt)}
        q, info = tel.actualizar(obs, tt, q_actual=q0, dt=0.1)
        fases.append(info.get("fase"))
    cam = tel.fusor.cams["B"]
    ang = _m.degrees(_m.acos(float(np.clip((np.trace(cam.R.T @ R) - 1) / 2, -1, 1))))
    assert guardadas, "no se recalculó la extrínseca"
    assert np.linalg.norm(cam.t - t) < 0.08 and ang < 3.0, (np.linalg.norm(cam.t - t), ang)
    assert tel.calibrado and "camaras" in fases and "huesos" in fases
    assert abs(tel.calib.longitudes[0] - 0.30) < 0.03 and abs(tel.calib.longitudes[1] - 0.26) < 0.03


def test_teleop_en_paralelo_da_lo_mismo_un_fotograma_despues():
    """TeleopEnParalelo (hilo aparte, fluidez): el resultado del fotograma N es exactamente el
    que daría la versión en el bucle, solo que se recoge en la vuelta N+1."""
    from skill_shield.teleop_fr5.diagnostico import construir_teleop
    from skill_shield.teleop_fr5.fusion_muneca_codo import Camara, Parametros
    from skill_shield.teleop_fr5.simulador import BrazoHumanoSintetico, CamaraSintetica, cargar_extrinsecas_repo
    from skill_shield.teleop_fr5.teleop_brazo import ConfigTeleop, TeleopBrazoFR5, TeleopEnParalelo
    R, t = cargar_extrinsecas_repo()
    brazo = BrazoHumanoSintetico()

    def observaciones():
        # las cámaras sintéticas sortean el ruido de profundidad en cada lectura: cada versión
        # recibe su propia secuencia, con la misma semilla
        rng = np.random.default_rng(4)
        ca, cb = CamaraSintetica("A", rng=rng), CamaraSintetica("B", R, t, rng=rng)
        out = []
        for k in range(80):
            tt = k / 10.0
            S, E, W, _ = brazo.pose(tt)
            out.append((tt, ca.observar(S, E, W, brazo.S2, brazo.C, brazo.C2, tt),
                        cb.observar(S, E, W, brazo.S2, brazo.C, brazo.C2, tt)))
        return out
    base, q0, _ = construir_teleop()

    def nuevo():
        # sin autocalibración (su ajuste va en otro hilo y no es determinista en el tiempo)
        return TeleopBrazoFR5(cadena=base.cadena, camaras={"A": Camara("A"), "B": Camara("B", R, t)},
                              cfg=ConfigTeleop(longitudes_manual=(0.30, 0.26), auto_extrinseca=False,
                                               vigilar_extrinseca=False),
                              cfg_retarget=base.retarget.cfg, entorno=base.ik.col.entorno, verbose=False)
    a, b = nuevo(), nuevo()
    par = TeleopEnParalelo(b)
    qa, qb = [], []
    for tt, oa, ob in observaciones():
        q, _ = a.actualizar({"A": oa, "B": ob}, tt, q_actual=q0, dt=0.1)
        qa.append(None if q is None else q.copy())
    for tt, oa, ob in observaciones():
        q2, _ = par.recoger()
        qb.append(q2)
        par.lanzar({"A": oa, "B": ob}, tt, q_actual=q0, dt=0.1)
    par.cerrar()
    for k in range(1, len(qa)):
        if qa[k - 1] is None:
            assert qb[k] is None
        else:
            assert np.abs(qa[k - 1] - qb[k]).max() < 1e-12


if __name__ == "__main__":
    import inspect
    fallos = 0
    for nombre, fn in list(globals().items()):
        if nombre.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"✅ {nombre}")
            except AssertionError as e:
                fallos += 1
                print(f"❌ {nombre}: {e}")
    sys.exit(1 if fallos else 0)
