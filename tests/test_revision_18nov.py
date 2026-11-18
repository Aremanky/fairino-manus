"""
Pruebas de la revisión del modo autónomo (18-nov), sin robot, sin cámara y sin MuJoCo:

    python -m pytest tests/test_revision_18nov.py -q

  · Ejecutor: tramos rectos a velocidad de la palma (más despacio al llegar); cierre con los táctiles
    (cada dedo para al tocar; apretar y aflojar desde donde tocó).
  · Tarea: agarre fallido → abre, sube y vuelve a planificar (con el pedal), hasta REINTENTOS; un robot que
    va algo por detrás de lo mandado no aborta, uno que no sigue sí; la autocorrección se congela con tarea.
  · Planificador: pelota deslizada por el rayo de la cámara; huecos de la mano fijos (no dependen del inicio);
    IK en lote = IK una a una; la pinza se eleva si los dedos bajan demasiado; un fallo de cuRobo no lo
    apaga para siempre; el brazo no tapa la pelota al afinar si hay otra postura.
  · cuRobo: la mano con el margen y la pelota cogida; el mundo inflado con el margen.
  · Pelotas: centro por la silueta aunque el lado en sombra no salga en la máscara.
  · Escala de la profundidad: el cociente tablero/estéreo de una imagen sintética.
"""
import math
import os
import sys
import time

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import config                                                        # noqa: E402
from skill_shield.autonomo import tarea as TA                        # noqa: E402
from skill_shield.autonomo import ejecucion as EJ                    # noqa: E402
from skill_shield.hardware.fr5_servo import ACERCANDO, ENVIANDO      # noqa: E402
from skill_shield.teleop_fr5.cinematica import cargar_cadena         # noqa: E402

G = {k: getattr(config, k) for k in dir(config) if k.isupper()}
Q0 = np.radians([93.0, -17.0, 53.0, 43.0, 105.0, 25.0])


@pytest.fixture(scope="module")
def cadena():
    return cargar_cadena(config.RUTA_XML, verbose=False)


@pytest.fixture(scope="module")
def pl(cadena):
    return TA.Planificador(cadena, G)


def _plano(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


# ══════════════════════ Ejecutor ══════════════════════
def test_recta_a_velocidad_de_la_palma_y_llegada_lenta():
    qs = np.zeros((101, 6))
    qs[:, 0] = np.linspace(0.0, math.radians(10), 101)             # J1: 0,1° por paso
    fk = lambda q: np.array([q[0], 0.0, 0.0])                      # noqa: E731  palma: 1 rad = 1 m (de juguete)
    f = TA.Fase("Bajar", qs)
    f.vel_cart, f.lento_m, f.vel_lenta = 0.05, 0.03, 0.01
    plan = TA.Plan(qs[0], [f])
    ej = TA.Ejecutor(plan, dict(abierta={}), math.radians(30), math.radians(300), fk_palma=fk)
    v_max_lejos = v_max_cerca = 0.0
    p_ant, dt = fk(ej.q()), 1 / 200
    while not ej.terminado:
        ej.avanzar(dt)
        p = fk(ej.q())
        v = float(np.linalg.norm(p - p_ant)) / dt
        quedan = float(np.linalg.norm(fk(qs[-1]) - p))
        if quedan > 0.035:
            v_max_lejos = max(v_max_lejos, v)
        elif quedan < 0.025:
            v_max_cerca = max(v_max_cerca, v)
        p_ant = p
    assert v_max_lejos <= 0.05 + 1e-3 and v_max_lejos > 0.04       # a la velocidad de la palma
    assert v_max_cerca <= 0.01 + 1e-3                               # y despacio en los últimos 3 cm
    assert np.allclose(ej.q(), qs[-1])


def test_cierre_con_los_tactiles_para_cada_dedo_al_tocar():
    abierta = {0: 0.0, 8: 0.0, 9: 0.0, 11: 0.0, 12: 0.0, 15: -0.2, 16: -0.2}
    cerrada = {0: 0.0, 8: 1.0, 9: 0.8, 11: 1.0, 12: 0.8, 15: 0.4, 16: 0.3}
    estado = dict(fz={"Pulgar": 0.05, "Índice": 0.05, "Corazón": 0.05, "Anular": 0.0, "Meñique": 0.0})
    f = TA.Fase("Cerrar la mano", mano="cerrada_modelo")
    f.cierre_tactil = dict(destino="max", apretar="apretar", aflojar="cerrada_modelo", apriete=0.3, aflojado=0.1,
                           umbral_n=0.2, dur=2.0)
    plan = TA.Plan(np.zeros(6), [f, TA.Fase("Apretar", mano="apretar"), TA.Fase("Aflojar", mano="cerrada_modelo")])
    plan.posturas = dict(max=cerrada, cerrada_modelo=cerrada, apretar=cerrada)
    ej = TA.Ejecutor(plan, dict(abierta=abierta), 1.0, 1.0, mano_ini=abierta, contactos=lambda: estado["fz"])
    toque_indice = toque_pulgar = None
    while not ej.terminado and ej.i == 0:
        ej.avanzar(1 / 100)
        m = ej.mano()
        if toque_indice is None and m[11] > 0.35:                   # el índice toca la pelota aquí
            estado["fz"] = dict(estado["fz"], **{"Índice": 0.6})
            toque_indice = m[11]
        if toque_pulgar is None and m[15] > 0.0:                    # el pulgar, aquí
            estado["fz"] = dict(estado["fz"], **{"Pulgar": 0.9, "Corazón": 0.5})
            toque_pulgar = m[15]
    assert ej.cierre_info is not None and set(ej.cierre_info["tocados"]) == {"Pulgar", "Índice", "Corazón"}
    m = ej.mano()
    assert m[11] < 0.45                                             # el índice se quedó donde tocó (no a 1.0)
    # apretar = donde tocó + 0,3 del recorrido; aflojar = donde tocó + 0,1
    assert ej.posturas["apretar"][11] == pytest.approx(m[11] + 0.3 * (1.0 - 0.0))
    assert ej.posturas["cerrada_modelo"][11] == pytest.approx(m[11] + 0.1)
    assert ej.posturas["apretar"][0] == pytest.approx(0.0)          # la muñeca no se toca


def test_sin_tactiles_cierra_como_antes():
    f = TA.Fase("Cerrar la mano", mano="cerrada")
    f.cierre_tactil = dict(destino="max", apretar="ap", aflojar="cerrada", umbral_n=0.2, dur=2.0)
    plan = TA.Plan(np.zeros(6), [f])
    ej = TA.Ejecutor(plan, dict(abierta={8: 0.0}, cerrada={8: 0.7}, max={8: 1.0}), 1.0, 1.0,
                     t_mano=(1.0, 1.5), contactos=lambda: None)
    ej.avanzar(5.0)
    assert ej.terminado and ej.mano()[8] == pytest.approx(0.7) and ej.cierre_info is None


# ══════════════════════ Tarea ══════════════════════
class FR5Falso:
    def __init__(self, sigue=True, retraso=0):
        self.conectado, self.en_error = True, False
        self.q = np.zeros(6)
        self._estado = ACERCANDO
        self.sigue, self.retraso, self._cola = sigue, retraso, []
        self.hab = []

    def publicar(self, q):
        self._cola.append(np.array(q))
        if self.sigue and len(self._cola) > self.retraso:
            self.q = self._cola[-1 - self.retraso].copy()

    def habilitar(self, si=True):
        self.hab.append(bool(si))
        self._estado = ENVIANDO if si else 1

    def estado(self):
        return dict(estado=self._estado)


def _plan_pelota_falso(q0):
    qs = q0 + np.radians([0, 0, 0, 0, 0, 10]) * np.linspace(0, 1, 12)[:, None]
    fases = [TA.Fase("Bajar a la pelota", qs), TA.Fase("Cerrar la mano", mano="pelota_cerrada"),
             TA.Fase("Comprobar que la tengo", dur=0.2, comprobar="agarre"), TA.Fase("Subir con la pelota", qs[::-1])]
    p = TA.Plan(q0, fases, nombre="coger, apretar y dejar la pelota")
    p.pelota = dict(c=np.zeros(3), r=0.035, i_bajar=0, i_subir=3, k_abierta="pelota_abierta")
    return p


HERR = [dict(id=100, nombre="Pelota roja", tipo="pelota", estable=True, diametro_m=0.07, T_mundo=_plano((0.3, -0.5, 0)).tolist()),
        dict(id=22, nombre="Punto de dejada", tipo="punto_dejada", estable=True, T_mundo=np.eye(4).tolist())]


def _tarea(tactil, reintentos=2, **kw):
    fr5 = FR5Falso(**kw)
    llamadas = []

    class Pl:
        def poner_mapa(self, *a, **k):
            return 0

        def planificar_pelota(self, q0, T, d, Td, **k):
            llamadas.append(np.array(q0))
            return _plan_pelota_falso(np.asarray(q0, float))

        def plan_volver(self, q, qi):
            return TA.Plan(q, [TA.Fase("Volver", np.vstack([q, qi]))], nombre="volver al inicio")
    t = TA.Tarea(dict(G, AUTONOMO_TAREA_MOVER_MANO=False, AUTONOMO_TAREA_REINTENTOS=reintentos,
                      AUTONOMO_TAREA_PRECALENTAR=False), ".", fr5, simulacro=False, fabrica_planificador=Pl,
                 tactil=tactil)
    return t, fr5, llamadas


def _correr(t, fr5, n=6000):
    ctx = lambda: dict(q_real=fr5.q, herramientas=HERR, objetivo_id=100, mapa=None)   # noqa: E731
    t.orden({"tarea": dict(accion="planificar", id="r1")}, ctx)
    for _ in range(n):
        if t.estado == "planificando":
            time.sleep(0.002)
        if t.estado == "lista" and t.ejec is None:
            t.orden({"tarea": dict(accion="ejecutar", id=f"r{time.time()}")}, ctx)
        t.paso(1 / 60, fr5.q, True, ctx)
        if t.estado in ("terminada", "error") and t.plan is not None and not getattr(t.plan, "replanificar", False):
            break
    return ctx


def test_agarre_fallido_reintenta_y_al_final_para():
    t, fr5, llamadas = _tarea(lambda: dict(estado="ok", dedos=[dict(dedo="Pulgar", fz=0.0)] * 5), reintentos=2)
    _correr(t, fr5)
    assert t.estado == "error" and "No noto la pelota" in t.texto and "2 reintentos" in t.texto
    assert len(llamadas) == 3                                     # el plan y dos reintentos
    assert fr5.hab[-1] is False


def test_agarre_que_sale_al_reintentar_termina():
    n = {"c": 0}

    def tactil():
        n["c"] += 1
        return dict(estado="ok", dedos=[dict(dedo="Pulgar", fz=1.0 if n["c"] > 1 else 0.0)] * 5)
    t, fr5, llamadas = _tarea(tactil)
    _correr(t, fr5)
    assert t.estado == "terminada" and len(llamadas) == 2
    assert t._q_inicio is not None and np.allclose(t._q_inicio, 0.0)   # el inicio, el de antes del reintento


def test_robot_algo_por_detras_no_aborta_y_el_que_no_sigue_si():
    t, fr5, _ = _tarea(lambda: dict(estado="ok", dedos=[dict(dedo="Pulgar", fz=1.0)] * 5), retraso=12)
    _correr(t, fr5)
    assert t.estado == "terminada", t.texto                      # 0,2 s por detrás: dentro de la ventana
    t, fr5, _ = _tarea(lambda: dict(estado="ok", dedos=[dict(dedo="Pulgar", fz=1.0)] * 5), sigue=False)
    _correr(t, fr5)
    assert t.estado == "error" and "no sigue" in t.texto


def test_autocorreccion_congelada_mientras_hay_tarea():
    t, fr5, _ = _tarea(lambda: None)
    assert not t.congelar_autocalib
    t.estado = "ejecutando"
    assert t.congelar_autocalib
    t.estado = "terminada"
    assert not t.congelar_autocalib


def test_constructor_del_mapa_no_aplica_la_correccion_congelada():
    import json

    class CanalFalso:
        def __init__(self, d):
            self.d = d

        def leer(self):
            return json.dumps(self.d).encode(), time.time()
    from skill_shield.autonomo.mapa import ConstructorMapa
    c = ConstructorMapa.__new__(ConstructorMapa)
    c.canal_congelar, c.congelada, c.medida_congelada, c._t_aviso_congelada = CanalFalso(dict(congelar=True)), False, None, 0.0

    class Corr:
        residuo_max = 0.012
    c.correccion = Corr()
    assert c._congelada()
    c._medida_sin_aplicar(dict(t=np.array([0.01, 0.0, 0.0]), residuo=0.004))
    assert np.allclose(c.medida_congelada, (0.01, 0, 0))
    c.canal_congelar = CanalFalso(dict(congelar=False))
    assert not c._congelada()


# ══════════════════════ Planificador ══════════════════════
def test_pelota_se_desliza_por_el_rayo_de_la_camara(cadena):
    Tc = np.eye(4)
    Tc[:3, 3] = (0.0, 0.0, 0.5)                                   # cámara 50 cm por encima de la mesa (z = 0)
    pl = TA.Planificador(cadena, G, Tc)
    real = np.array([0.6, 0.0, 0.035])                            # pelota de 7 cm apoyada en z = 0
    d = real - Tc[:3, 3]
    medida = real + d / np.linalg.norm(d) * 0.02                  # el estéreo la ve 2 cm más lejos POR EL RAYO
    T = pl.apoyar(_plano(medida), 0.07, 0.0)
    assert np.linalg.norm(T[:3, 3] - real) < 1e-6                 # vuelve a su sitio (también en horizontal)
    pl.pelota_por_rayo = False
    T2 = pl.apoyar(_plano(medida), 0.07, 0.0)
    assert abs(T2[2, 3] - 0.035) < 1e-9 and np.linalg.norm(T2[:3, 3] - real) > 0.015   # en vertical: sigue lejos


def test_huecos_de_la_mano_no_dependen_de_la_postura_de_inicio(cadena):
    pl = TA.Planificador(cadena, G)
    rng = np.random.default_rng(4)
    qs = [Q0 + rng.normal(0, 0.3, 6) for _ in range(40)]
    qs = [q for q in qs if pl.motivo(q) is None][:10]
    ref = [pl.motivo(q) for q in qs]
    for q_ini in qs[:3]:
        pl._empezar(q_ini)
        assert [pl.motivo(q) for q in qs] == ref                  # misma respuesta empiece donde empiece
    assert pl.choque.holgura == pytest.approx(G["AUTONOMO_TAREA_MANO_HOLGURA"])


def test_ik_en_lote_da_lo_mismo_que_una_a_una(cadena, pl):
    from collections import Counter
    rng = np.random.default_rng(9)
    poses = [cadena.fk_marcos(Q0 + rng.normal(0, 0.4, 6))[6] for _ in range(25)]
    pl.poner_mapa(np.array([[0.3, -0.5, -0.1], [0.32, -0.5, -0.1]]), 0.02)
    pl._empezar(Q0)
    r1, r2 = Counter(), Counter()
    uno = [pl.ik(T, Q0, 2, r1) for T in poses]
    lote = pl.ik_lote(poses, Q0, 2, r2)
    assert len(uno) == len(lote)
    for a, b in zip(uno, lote):
        assert len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b))
    assert r1 == r2
    pl.poner_mapa(None, 0.02)


def test_la_pinza_se_eleva_si_los_dedos_bajan_demasiado(cadena, monkeypatch):
    pl = TA.Planificador(cadena, G)
    monkeypatch.setattr(TA.Planificador, "_caida_pinza", lambda self, pz: 0.025)
    pz = pl.pinza(0.07)
    e = pl._elevar_pinza(pz, 0.035)
    # hace falta 2,5 + 2 + 0,3 − 3,5 = 1,3 cm; como mucho 0,35 · 3,5 = 1,225 cm
    assert e == pytest.approx(0.35 * 0.035) and "probablemente no salga" in pl.avisos[-1]
    monkeypatch.setattr(TA.Planificador, "_caida_pinza", lambda self, pz: 0.005)
    assert pl._elevar_pinza(pz, 0.035) == 0.0
    b = pl._bola_en_pinza(pz, 0.01)
    assert np.linalg.norm(b - pz["punto"]) == pytest.approx(0.01)


def test_fallo_de_curobo_no_lo_apaga_para_siempre(cadena):
    pl = TA.Planificador(cadena, G)

    class CuRoboQueFalla:
        def actualizar_mundo(self, *a, **k):
            pass

        def plan(self, a, b):
            raise RuntimeError("CUDA out of memory")
    falso = CuRoboQueFalla()
    pl.curobo = falso
    pl._empezar(Q0)
    with pytest.raises(TA.ErrorPlan):
        pl.tramo_curobo(Q0, Q0 + 0.01)
    assert pl.curobo is None and any("cuRobo falló" in a for a in pl.avisos)
    pl._empezar(Q0)                                               # el siguiente plan lo vuelve a probar
    assert pl.curobo is falso
    pl.curobo_fallos = pl.gpu_fallos_max
    pl._pausar_curobo(RuntimeError("otra vez"))
    pl._empezar(Q0)
    assert pl.curobo is None and any("apagado" in a for a in pl.avisos)


def test_al_afinar_prefiere_una_postura_en_la_que_el_brazo_no_tapa(cadena, monkeypatch):
    """Si hay posturas de encima de la pelota igual de buenas, se queda con la que no tapa."""
    pl = TA.Planificador(cadena, G, np.eye(4))
    pl._mirar_obj = (np.zeros(3), 0.035)
    tapa = {}
    monkeypatch.setattr(TA.Planificador, "fraccion_tapada", lambda self, q, c, r, n=9: tapa.get(q.tobytes(), 0.0))
    q1, q2 = Q0.copy(), Q0 + 0.001
    tapa[q1.tobytes()] = 0.9
    assert pl._coste_tapada(q1) > 100 and pl._coste_tapada(q2) == 0.0
    pl._mirar_obj = None
    assert pl._coste_tapada(q1) == 0.0


# ══════════════════════ cuRobo (sin cuRobo) ══════════════════════
def test_curobo_mano_con_margen_y_pelota_cogida():
    from skill_shield.autonomo.curobo_fr5.planificador import ajustar_mano
    robot = {"kinematics": {"collision_spheres": {"mano_mount": [dict(center=[0, 0, 0], radius=0.02)],
                                                  "forearm_Link": [dict(center=[0, 0, 0], radius=0.05)]}}}
    n = ajustar_mano(robot, "mano_mount", 0.01, [((0.1, 0.0, 0.05), 0.035)])
    esf = robot["kinematics"]["collision_spheres"]
    assert n == 1 and esf["mano_mount"][0]["radius"] == pytest.approx(0.03)
    assert esf["mano_mount"][1]["radius"] == pytest.approx(0.045) and esf["forearm_Link"][0]["radius"] == 0.05
    assert ajustar_mano({"kinematics": {}}, "mano_mount", 0.01) is None


def test_curobo_mundo_inflado_con_el_margen():
    from skill_shield.autonomo.curobo_fr5 import mundo as M

    class Pl:
        _res, _suelo, _discos, _visto, _losas = 0.02, None, [], None, []

        class _arbol:
            data = np.array([[0.0, 0.0, 0.0]])
    lim = ((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))
    s0, _, _ = M.mundo_del_planificador(Pl, lim, 0.02)
    s1, _, _ = M.mundo_del_planificador(Pl, lim, 0.02, margen=0.03)
    assert np.allclose(s0 - s1, 0.03, atol=1e-6)


# ══════════════════════ pelotas ══════════════════════
def test_centro_por_la_silueta_aunque_falte_el_lado_en_sombra():
    import cv2
    from skill_shield.autonomo.pelotas import ajustar_circulo
    m = np.zeros((200, 200), np.uint8)
    cv2.circle(m, (100, 100), 40, 255, -1)
    m[:, 85:] = 0                                                # más de media pelota en sombra: no sale
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (u0, v0), r0 = cv2.minEnclosingCircle(cs[0])
    assert abs(u0 - 100) > 5                                     # el círculo mínimo se va hacia la luz
    u, v, r = ajustar_circulo(cs[0].reshape(-1, 2), r0)
    assert abs(u - 100) < 1.0 and abs(v - 100) < 1.0 and abs(r - 40) < 1.0


# ══════════════════════ escala de la profundidad ══════════════════════
def test_cociente_tablero_estereo_de_una_imagen():
    from calibrar_escala_profundidad import medir_imagen
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    T = np.eye(4)
    T[:3, 3] = (-0.08, -0.06, 0.9)                               # tablero de frente a 90 cm
    prof = np.full((480, 640), 900.0 / 0.98, np.float64).astype(np.uint16)   # el estéreo mide un 2 % de más
    r = medir_imagen(T, (7, 5), 0.025, prof, K, None)
    assert r is not None and r[1] == pytest.approx(0.98, abs=0.002)


# ══════════════════════ (18-nov, 2ª parte) TOPP-RA, goalset de cuRobo, carga de la brida ══════════════════════
def _toppra_falso(monkeypatch, desvio=0.0):
    """Módulos toppra de mentira (misma API que toppra 0.6): camino lineal a velocidad constante."""
    import types
    ta, talg, tcon = types.ModuleType("toppra"), types.ModuleType("toppra.algorithm"), types.ModuleType("toppra.constraint")

    class Spline:
        def __init__(self, ss, qs, bc_type="not-a-knot"):
            self.ss, self.qs = np.asarray(ss, float), np.asarray(qs, float)

        def __call__(self, s, order=0):
            s = np.atleast_1d(np.asarray(s, float))
            out = np.column_stack([np.interp(s, self.ss, self.qs[:, j]) for j in range(self.qs.shape[1])]) + desvio
            return out

    class Traj:
        def __init__(self, path, v):
            self.path, self.duration = path, float(path.ss[-1]) / v

        def __call__(self, t, order=0):
            return self.path(float(t) / self.duration * self.path.ss[-1])[0]

    class TOPPRA:
        def __init__(self, cons, path, parametrizer=None, **kw):
            self.path, self.v = path, float(cons[0].v[0, 1])

        def compute_trajectory(self, a=0.0, b=0.0):
            return Traj(self.path, self.v)

    class Vel:
        def __init__(self, v):
            self.v = np.asarray(v)

    ta.SplineInterpolator, talg.TOPPRA = Spline, TOPPRA
    tcon.JointVelocityConstraint, tcon.JointAccelerationConstraint = Vel, Vel
    ta.algorithm, ta.constraint = talg, tcon
    for nombre, m in (("toppra", ta), ("toppra.algorithm", talg), ("toppra.constraint", tcon)):
        monkeypatch.setitem(sys.modules, nombre, m)


def test_tramos_articulares_con_toppra_si_esta(monkeypatch):
    _toppra_falso(monkeypatch)
    qs = np.zeros((20, 6))
    qs[:, 1] = np.linspace(0, 0.5, 20)
    plan = TA.Plan(qs[0], [TA.Fase("Ir", qs)])
    ej = TA.Ejecutor(plan, dict(abierta={}), 0.25, 1.0, toppra=True)
    assert ej.tramos_toppra == 1 and ej._prep[0][1] == "toppra" and ej.duracion == pytest.approx(2.0)
    ej.avanzar(1.0)
    assert ej.q()[1] == pytest.approx(0.25, abs=1e-6)
    ej.avanzar(5.0)
    assert ej.terminado and np.allclose(ej.q(), qs[-1])


def test_toppra_no_se_usa_si_el_spline_se_sale_del_camino(monkeypatch):
    _toppra_falso(monkeypatch, desvio=math.radians(2.0))
    qs = np.zeros((20, 6))
    qs[:, 1] = np.linspace(0, 0.5, 20)
    ej = TA.Ejecutor(TA.Plan(qs[0], [TA.Fase("Ir", qs)]), dict(abierta={}), 0.25, 1.0, toppra=True)
    assert ej.tramos_toppra == 0 and not isinstance(ej._prep[0][1], str)   # el trapecio de siempre


def test_toppra_de_verdad_respeta_los_limites():
    pytest.importorskip("toppra")
    from skill_shield.autonomo.ejecucion import tiempos_toppra
    qs = np.radians([30, -10, 5, 0, 20, 60]) * np.linspace(0, 1, 40)[:, None]
    T, tr = tiempos_toppra(qs, math.radians(15), math.radians(30))
    ts = np.linspace(0, T, 500)
    Q = np.array([tr(t) for t in ts])
    assert np.abs(np.diff(Q, axis=0)).max() / (ts[1] - ts[0]) <= math.radians(15) * 1.03
    assert np.allclose(Q[-1], qs[-1], atol=1e-3)


def test_goalset_de_curobo_elige_la_opcion_y_su_camino(cadena):
    pl = TA.Planificador(cadena, G)
    pl._empezar(Q0)
    rng = np.random.default_rng(11)
    ops = []
    while len(ops) < 3:
        q = Q0 + rng.normal(0, 0.15, 6)
        if pl.motivo(q) is None and all(pl.motivo(Q0 + (q - Q0) * u) is None for u in np.linspace(0, 1, 15)):
            ops.append((q, q, None, 0.0))

    class CuRoboFalso:
        llamadas = 0

        def actualizar_mundo(self, *a, **k):
            pass

        def plan_goalset(self, q0, Ts):
            CuRoboFalso.llamadas += 1
            assert len(Ts) == 3 and np.allclose(Ts[2], cadena.fk_marcos(ops[2][0])[6])
            return 2, q0 + (ops[2][0] - q0) * np.linspace(0, 1, 12)[:, None]

        def plan(self, a, b):
            return a + (b - a) * np.linspace(0, 1, 12)[:, None]
    pl.curobo = CuRoboFalso()
    (elegida,), (ir,), vuelta = pl._combinar(Q0, [("ir", ops)])
    assert CuRoboFalso.llamadas == 1 and np.allclose(elegida[0], ops[2][0])
    assert np.allclose(ir[0], Q0) and np.allclose(ir[-1], ops[2][0]) and pl.goalsets_curobo == 1


def test_goalset_que_acaba_en_otra_configuracion_no_se_usa(cadena):
    pl = TA.Planificador(cadena, G)
    pl._empezar(Q0)
    q1 = Q0 + np.radians([0, 0, 0, 0, 0, 10])
    ops = [(Q0, Q0, None, 0.0), (q1, q1, None, 0.0)]

    class CuRoboFalso:
        def actualizar_mundo(self, *a, **k):
            pass

        def plan_goalset(self, q0, Ts):
            return 1, np.vstack([q0, q1 + np.radians([0, 0, 0, 0, 0, 40])])    # misma pose, otra muñeca
    pl.curobo = CuRoboFalso()
    assert pl._goalset(Q0, ops, "ir") is None and pl.curobo_rechazos == 1


def test_carga_de_la_brida_con_los_pares_de_la_muneca(cadena):
    sys.path.insert(0, os.path.join(RAIZ, "diagnosticos"))
    import identificar_carga as IC
    eslab = IC.inerciales_muneca(config.RUTA_XML)
    assert set(eslab) == set(IC.MUNECA)
    m_real, c_real = 2.2, np.array([0.004, 0.012, 0.11])       # en el marco de wrist3
    rng = np.random.default_rng(2)
    Qs, Pares = [], []
    for d in IC.DELTAS_DEG[:13]:
        q = Q0 + np.radians([0, 0, 0, *d])
        A, k = IC.filas(cadena, q, eslab)
        tau = -(A @ np.r_[m_real, m_real * c_real] + k) + np.array([0.05, -0.08, 0.02])   # signo del motor + desfase
        par = np.zeros(6)
        par[3:] = tau + rng.normal(0, 0.01, 3)
        Qs.append(q)
        Pares.append(par)
    r = IC.ajustar(cadena, Qs, Pares, eslab, 0.100)
    assert r is not None and r["signo"] == -1.0
    assert r["masa_kg"] == pytest.approx(m_real, rel=0.03)
    assert np.allclose(r["cdm_wrist3_m"], c_real, atol=0.004)
    assert r["cdm_brida_mm"][2] == pytest.approx((c_real[2] - 0.100) * 1000, abs=4)
