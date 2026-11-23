"""
Pruebas de la tarea coger → usar → dejar (11-nov, paso 10), sin robot, sin mano y sin MuJoCo:

    python -m pytest tests/test_tarea.py -q

  · Planificador (con la cadena y las mallas reales del MJCF): con marcadores alcanzables hace un
    plan en el que TODAS las posturas pasan las comprobaciones, la punta llega a su sitio, el giro
    de J6 no la mueve y se vuelve a la postura de inicio.
  · Ejecutor: nunca va más rápido que la velocidad pedida y acaba donde acaba el plan.
  · Tarea: sin pedal no se manda nada; al soltarlo se para (ServoMoveEnd); se aborta si la
    controladora da error o si el robot no sigue; la mano se conecta con par solo con el pedal.
"""
import math
import os
import sys
import time

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del repo (tests/..)
sys.path.insert(0, RAIZ)

import config                                                        # noqa: E402
from skill_shield.autonomo import tarea as TA                        # noqa: E402
from skill_shield.hardware.fr5_servo import ENVIANDO, ACERCANDO      # noqa: E402
from skill_shield.teleop_fr5.cinematica import OFFSET_PALMA_ORCA, cargar_cadena   # noqa: E402

G = {k: getattr(config, k) for k in dir(config) if k.isupper()}
G["AUTONOMO_CAM_A_OBLIGATORIA"] = False     # (23-nov) aquí, el camino de la cabeza; la A obligatoria, en test_camara_a.py
Q0 = np.radians([93.0, -17.0, 53.0, 43.0, 105.0, 25.0])     # mano delante de la cámara (tests de calibración)
POSTURAS = dict(abierta=TA.MANO_ABIERTA, agarre=TA.MANO_AGARRE)


# ══════════════════════ planificador con el robot de verdad ══════════════════════
@pytest.fixture(scope="module")
def cadena():
    return cargar_cadena(config.RUTA_XML, verbose=False)


@pytest.fixture(scope="module")
def pl(cadena):
    return TA.Planificador(cadena, G)


def _marcador(z, p):
    z = np.asarray(z, float) / np.linalg.norm(z)
    u = np.cross([0.0, 0.0, 1.0], z)
    u /= np.linalg.norm(u)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([u, np.cross(z, u), z])
    T[:3, 3] = p
    return T


def _escenario(cadena, pl):
    """Marcadores colocados donde el robot SÍ llega (desde posturas válidas cerca de Q0)."""
    rng = np.random.default_rng(3)

    def q_ok():
        while True:
            q = Q0 + rng.normal(0, 0.3, 6)
            if pl.motivo(q) is None:
                return q
    qh, qu, qd = q_ok(), q_ok(), q_ok()
    Th, Tu, Td = (cadena.fk_marcos(q)[6] for q in (qh, qu, qd))
    palma = lambda T: T[:3, :3] @ OFFSET_PALMA_ORCA + T[:3, 3]
    m_h = _marcador(Th[:3, 0], palma(Th) - Th[:3, 0] * pl.d_previa)
    m_d = _marcador(Td[:3, 0], palma(Td) - Td[:3, 0] * pl.d_previa)
    punta, eje = Tu[:3, :3] @ pl.punta + Tu[:3, 3], Tu[:3, :3] @ pl.eje
    m_u = _marcador(-eje, punta + eje * pl.uso_previa)
    return m_h, m_u, m_d


def test_plan_completo_y_valido(cadena, pl):
    m_h, m_u, m_d = None, None, None
    for intento in range(6):                    # algunos sitios al azar no tienen camino: se prueba otro
        try:
            m_h, m_u, m_d = _escenario(cadena, pl) if intento == 0 else _escenario_semilla(cadena, pl, intento)
            plan = pl.planificar(Q0, m_h, m_u, m_d)
            break
        except TA.ErrorPlan:
            plan = None
    assert plan is not None, "ningún escenario alcanzable dio plan"
    nombres = [f.nombre for f in plan.fases]
    assert nombres[0] == "Abrir la mano" and "Cerrar la mano (coger)" in nombres and "Abrir la mano (dejar)" in nombres
    assert sum(n.startswith("Girar") for n in nombres) == pl.uso_veces
    assert np.allclose(plan.fases[1].qs[0], Q0) and np.allclose(plan.q_final, Q0)
    q_ant = None
    for f in plan.fases:
        if f.qs is None:
            continue
        if q_ant is not None:
            assert np.allclose(f.qs[0], q_ant, atol=1e-6), f"salto entre tramos antes de «{f.nombre}»"
        for q in f.qs:
            assert pl.motivo(q) is None, f.nombre
        assert np.max(np.abs(np.diff(f.qs, axis=0))) <= max(pl.salto_max, pl.paso) + 1e-9, f.nombre
        q_ant = f.qs[-1]
    # la punta llega a 2 cm del punto de uso, apuntándolo; al girar no se mueve
    fu = next(f for f in plan.fases if f.nombre == "Acercar la punta")
    T = cadena.fk_marcos(fu.qs[-1])[6]
    punta = T[:3, :3] @ pl.punta + T[:3, 3]
    hacia = m_u[:3, 3] - punta                              # (puede llegar inclinada: pl.inclinaciones)
    assert abs(np.linalg.norm(hacia) - pl.uso_cerca) < 0.002
    assert (T[:3, :3] @ pl.eje) @ (hacia / np.linalg.norm(hacia)) > 0.999          # apunta al marcador
    assert (T[:3, :3] @ pl.eje) @ -m_u[:3, 2] >= math.cos(math.radians(pl.inclinaciones[1])) - 1e-6
    fg = next(f for f in plan.fases if f.nombre.startswith("Girar"))
    T1, T2 = cadena.fk_marcos(fg.qs[0])[6], cadena.fk_marcos(fg.qs[-1])[6]
    assert np.linalg.norm((T1[:3, :3] @ pl.punta + T1[:3, 3]) - (T2[:3, :3] @ pl.punta + T2[:3, 3])) < 0.002
    e1 = T1[:3, :3] @ pl.eje
    assert np.allclose(e1, T2[:3, :3] @ pl.eje, atol=1e-3)                # gira alrededor de la varilla
    Rrel = T2[:3, :3] @ T1[:3, :3].T
    assert abs(math.acos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)) - pl.uso_giro) < 1e-3
    # la palma, a d_agarre del marcador de la herramienta y mirándolo
    fa = plan.fases[2]
    T = cadena.fk_marcos(fa.qs[-1])[6]
    hacia = m_h[:3, 3] - (T[:3, :3] @ OFFSET_PALMA_ORCA + T[:3, 3])
    assert abs(np.linalg.norm(hacia) - pl.d_agarre) < 0.002
    assert -T[:3, 0] @ (hacia / np.linalg.norm(hacia)) > 0.999      # la palma (−X) mira al marcador


def _escenario_semilla(cadena, pl, s):
    rng = np.random.default_rng(100 + s)

    def q_ok():
        while True:
            q = Q0 + rng.normal(0, 0.3, 6)
            if pl.motivo(q) is None:
                return q
    qh, qu, qd = q_ok(), q_ok(), q_ok()
    Th, Tu, Td = (cadena.fk_marcos(q)[6] for q in (qh, qu, qd))
    palma = lambda T: T[:3, :3] @ OFFSET_PALMA_ORCA + T[:3, 3]
    punta, eje = Tu[:3, :3] @ pl.punta + Tu[:3, 3], Tu[:3, :3] @ pl.eje
    return (_marcador(Th[:3, 0], palma(Th) - Th[:3, 0] * pl.d_previa),
            _marcador(-eje, punta + eje * pl.uso_previa),
            _marcador(Td[:3, 0], palma(Td) - Td[:3, 0] * pl.d_previa))


def test_obstaculo_del_mapa_en_el_camino_no_da_plan(cadena, pl):
    m_h, m_u, m_d = _escenario(cadena, pl)
    T = cadena.fk_marcos(Q0)[6]
    palma = T[:3, :3] @ OFFSET_PALMA_ORCA + T[:3, 3]
    caja = palma + np.stack(np.meshgrid(*[np.arange(-0.10, 0.11, 0.02)] * 3), -1).reshape(-1, 3)
    pl.poner_mapa(caja, 0.02)                              # un bloque justo donde está la mano
    try:
        with pytest.raises(TA.ErrorPlan, match="mapa"):
            pl.planificar(Q0, m_h, m_u, m_d)
        pl.poner_mapa(caja, 0.02, excluir=[(palma, 0.30)])  # ...salvo que esté excluido
        assert pl.motivo(Q0) is None
    finally:
        pl.poner_mapa(None, 0.02)


def test_marcador_inalcanzable_explica_por_que(pl):
    lejos = _marcador((0.0, 1.0, 0.0), (0.2, -2.0, 0.4))
    with pytest.raises(TA.ErrorPlan, match="herramienta"):
        pl.planificar(Q0, lejos, lejos, lejos)


# ══════════════════════ ejecutor ══════════════════════
def _plan_sencillo():
    qa = np.zeros(6)
    qb = qa + np.radians([30, -10, 5, 0, 0, 60])
    qs = qa + (qb - qa) * np.linspace(0, 1, 40)[:, None]
    return TA.Plan(qa, [TA.Fase("Abrir", mano="abierta"), TA.Fase("Ir", qs),
                        TA.Fase("Cerrar", mano="agarre"), TA.Fase("Volver", qs[::-1])])


def test_ejecutor_respeta_la_velocidad_y_acaba_en_el_final():
    plan = _plan_sencillo()
    v, a, dt = math.radians(15), math.radians(30), 1 / 60
    ej = TA.Ejecutor(plan, POSTURAS, v, a)
    q_ant, t = ej.q(), 0.0
    while not ej.terminado:
        ej.avanzar(dt)
        t += dt
        assert np.max(np.abs(ej.q() - q_ant)) <= v * dt + 1e-9
        q_ant = ej.q()
        assert t < 60
    assert np.allclose(ej.q(), plan.q0)
    assert abs(ej.mano()[2] - TA.MANO_AGARRE[2]) < 1e-9          # acabó con la mano cerrada
    assert abs(t - ej.duracion) < 0.1


def test_ejecutor_sin_avanzar_no_se_mueve():
    ej = TA.Ejecutor(_plan_sencillo(), POSTURAS, math.radians(15), math.radians(30))
    ej.avanzar(2.0)
    q = ej.q()
    for _ in range(50):
        assert np.allclose(ej.q(), q)


# ══════════════════════ tarea (pedal, FR5, mano) ══════════════════════
class FR5Falso:
    def __init__(self, sigue=True):
        self.conectado, self.en_error = True, False
        self.hab, self.pub, self.sigue = [], [], sigue
        self.q = np.zeros(6)
        self._estado = ACERCANDO

    def publicar(self, q):
        self.pub.append(np.array(q))
        if self.sigue:
            self.q = np.array(q)

    def habilitar(self, si=True):
        self.hab.append(bool(si))
        self._estado = ENVIANDO if si else 1

    def estado(self):
        return dict(estado=self._estado)


class PlanFalso:
    def poner_mapa(self, *a, **k):
        return 0

    def planificar(self, q0, *T):
        p = _plan_sencillo()
        return TA.Plan(q0, p.fases)

    def plan_volver(self, q, q_ini):
        return TA.Plan(q, [TA.Fase("Volver", np.vstack([q, q_ini]))], nombre="volver al inicio")


class ManoFalsa:
    def __init__(self):
        self.lista, self.error, self.env = False, None, []

    def conectar(self):
        time.sleep(0.05)
        self.lista = True

    def enviar(self, p):
        self.env.append(p)

    def postura(self, *a):
        return None

    def cerrar(self):
        self.lista = False


class MonitorFalso:
    puerto = "/dev/falso"

    def __init__(self):
        self.cerrado = False

    def cerrar(self):
        self.cerrado = True

    def postura(self, *a):
        return None


HERR = [dict(id=20, nombre="Destornillador", tipo="herramienta", estable=True, T_mundo=np.eye(4).tolist()),
        dict(id=21, nombre="Punto de uso", tipo="punto_uso", estable=True, T_mundo=np.eye(4).tolist()),
        dict(id=22, nombre="Punto de dejada", tipo="punto_dejada", estable=True, T_mundo=np.eye(4).tolist())]


def _tarea(mover_mano=False, sigue=True):
    fr5 = FR5Falso(sigue)
    mano = ManoFalsa()
    mon = MonitorFalso()
    t = TA.Tarea(dict(G, AUTONOMO_TAREA_MOVER_MANO=mover_mano), ".", fr5, simulacro=False, monitor_mano=mon,
                 fabrica_planificador=PlanFalso, fabrica_mano=lambda: mano)
    return t, fr5, mano, mon


def _orden(t, accion, n, fr5):
    t.orden({"tarea": dict(accion=accion, id=f"x-{n}")},
            lambda: dict(q_real=fr5.q, herramientas=HERR, objetivo_id=20, mapa=None))


def _esperar_plan(t, fr5):
    t0 = time.time()
    while t.estado == "planificando" and time.time() - t0 < 5:
        time.sleep(0.01)
        t.paso(0.0, fr5.q, False)


def test_sin_pedal_no_se_manda_nada_y_al_soltar_se_para():
    t, fr5, _, _ = _tarea()
    _orden(t, "planificar", 1, fr5)
    _esperar_plan(t, fr5)
    assert t.estado == "lista"
    _orden(t, "ejecutar", 2, fr5)
    assert t.estado == "pausada"
    for _ in range(30):
        t.paso(1 / 60, fr5.q, False)
    assert fr5.pub == [] and fr5.hab == []                  # sin pedal: nada
    for _ in range(60):
        t.paso(1 / 60, fr5.q, True)
    assert fr5.hab == [True] and t.estado == "ejecutando"
    q_pausa = fr5.q.copy()
    t.paso(1 / 60, fr5.q, False)
    assert fr5.hab == [True, False] and t.estado == "pausada"
    n = len(fr5.pub)
    for _ in range(30):
        t.paso(1 / 60, fr5.q, False)
    assert len(fr5.pub) == n and np.allclose(fr5.q, q_pausa)   # quieto mientras no se pisa
    for _ in range(3000):
        t.paso(1 / 60, fr5.q, True)
        if t.estado == "terminada":
            break
    assert t.estado == "terminada" and fr5.hab[-1] is False
    assert np.allclose(fr5.q, 0.0)


def test_error_de_la_controladora_aborta():
    t, fr5, _, _ = _tarea()
    _orden(t, "planificar", 1, fr5)
    _esperar_plan(t, fr5)
    _orden(t, "ejecutar", 2, fr5)
    for _ in range(40):
        t.paso(1 / 60, fr5.q, True)
    fr5.en_error = True
    t.paso(1 / 60, fr5.q, True)
    assert t.estado == "error" and fr5.hab[-1] is False


def test_robot_que_no_sigue_aborta():
    t, fr5, _, _ = _tarea(sigue=False)
    _orden(t, "planificar", 1, fr5)
    _esperar_plan(t, fr5)
    _orden(t, "ejecutar", 2, fr5)
    for _ in range(600):
        t.paso(1 / 60, fr5.q, True)
        if t.estado == "error":
            break
    assert t.estado == "error" and "no sigue" in t.texto and fr5.hab[-1] is False


def test_no_ejecuta_si_el_robot_se_movio_desde_el_plan():
    t, fr5, _, _ = _tarea()
    _orden(t, "planificar", 1, fr5)
    _esperar_plan(t, fr5)
    fr5.q = fr5.q + np.radians(10)
    _orden(t, "ejecutar", 2, fr5)
    assert t.estado == "error" and "se ha movido" in t.texto


def test_la_mano_solo_se_conecta_con_el_pedal():
    t, fr5, mano, mon = _tarea(mover_mano=True)
    _orden(t, "planificar", 1, fr5)
    _esperar_plan(t, fr5)
    _orden(t, "ejecutar", 2, fr5)
    for _ in range(10):
        t.paso(1 / 60, fr5.q, False)
    assert not mon.cerrado and t.mano_activa is None
    t0 = time.time()
    while not mano.lista and time.time() - t0 < 2:
        t.paso(1 / 60, fr5.q, True)
        time.sleep(0.01)
    assert mon.cerrado and t.mano_activa is mano and fr5.hab == []   # el brazo esperó a la mano
    for _ in range(30):
        t.paso(1 / 60, fr5.q, True)
    assert fr5.hab == [True] and mano.env                            # ahora sí: brazo y mano


def test_cancelar_mientras_planifica_descarta_el_plan():
    t, fr5, _, _ = _tarea()
    _orden(t, "planificar", 1, fr5)
    _orden(t, "cancelar", 2, fr5)
    time.sleep(0.2)
    t.paso(0.0, fr5.q, False)
    assert t.estado == "libre" and t.plan is None


@pytest.mark.parametrize("pos", [((0.30, -0.44, 0.47), (0.44, -0.36, 0.53), (0.15, -0.46, 0.40)),    # 1ª captura
                                 ((0.26, -0.38, 0.54), (0.36, -0.39, 0.43), (0.16, -0.46, 0.40))])   # 2ª captura
def test_marcadores_cerca_mirando_a_la_camara_llega_de_lado(cadena, pl, pos):
    """Caso real del 11-nov: marcadores a 35-50 cm de la cámara de cabeza, mirándola. De frente la
    muñeca no da (J4/J5 fuera de límites); inclinándose sí."""
    from skill_shield.autonomo.calibracion import cargar
    cam = cargar(os.path.join(RAIZ, "camara_robot.json"))["T_mundo_cam"][:3, 3]
    m = [_marcador(cam - np.array(p), p) for p in pos]
    plan = pl.planificar(Q0, *m)
    assert np.allclose(plan.q_final, Q0) and max(pl.inclinaciones) > 0
    for f in plan.fases:
        if f.qs is not None:
            assert all(pl.motivo(q) is None for q in f.qs[::5])


# ══════════════════════ pelotas y rodeos (11-nov) ══════════════════════
def _plano(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


BOLA, DEJAR = (0.35, -0.45, -0.15), (0.15, -0.45, -0.15)


def test_plan_pelota_desde_arriba(cadena, pl, monkeypatch):
    monkeypatch.setattr(pl, "pel_modos", ("rodeándola", "con los dedos"))   # con toda la mano
    plan = pl.planificar_pelota(Q0, _plano(BOLA), 0.067, _plano(DEJAR))
    nombres = [f.nombre for f in plan.fases]
    for n in ("Bajar a la pelota", "Cerrar la mano", "Apretar un poco", "Comprobar que la tengo", "Bajar a dejarla"):
        assert n in nombres
    assert "usar" not in " ".join(nombres).lower() and np.allclose(plan.q_final, Q0)
    fb = plan.fases[nombres.index("Bajar a la pelota")]
    T = cadena.fk_marcos(fb.qs[-1])[6]
    palma = T[:3, :3] @ OFFSET_PALMA_ORCA + T[:3, 3]
    hacia = np.array(BOLA) - palma
    assert abs(np.linalg.norm(hacia) - (0.0335 + pl.pel_holgura)) < 0.002
    assert -T[:3, 0] @ (hacia / np.linalg.norm(hacia)) > 0.999           # la palma mira a la pelota
    if pl.inclinaciones[0] == 0:
        assert palma[2] > BOLA[2]                                         # desde arriba
    for f in plan.fases:
        if f.qs is not None:
            assert all(pl.valida(f, q) is None for q in f.qs[::3])


def test_rodea_un_obstaculo_en_medio(cadena, pl):
    plan = pl.planificar_pelota(Q0, _plano(BOLA), 0.067, _plano(DEJAR))
    ir = plan.fases[1].qs
    T = cadena.fk_marcos(ir[len(ir) // 2])[6]
    medio = T[:3, :3] @ OFFSET_PALMA_ORCA + T[:3, 3]                     # un bloque donde pasaba la mano
    g_ = np.arange(-0.08, 0.081, 0.02)
    C = medio + np.stack(np.meshgrid(g_, g_, g_), -1).reshape(-1, 3)
    C = C[np.linalg.norm(C - medio, axis=1) <= 0.08]
    try:
        pl.poner_mapa(C, 0.02, [(np.array(BOLA), 0.0535)])
        assert pl.motivo(ir[len(ir) // 2]) is not None                   # el camino de antes ya choca
        plan2 = pl.planificar_pelota(Q0, _plano(BOLA), 0.067, _plano(DEJAR))
        assert pl.rodeos >= 1
        for f in plan2.fases:
            if f.qs is not None:
                assert all(pl.valida(f, q) is None for q in f.qs), f.nombre
                assert np.max(np.abs(np.diff(f.qs, axis=0))) <= max(pl.salto_max, pl.paso) + 1e-9
        pl.rodeo = False                                                  # sin rodeos: no hay plan
        with pytest.raises(TA.ErrorPlan):
            pl.planificar_pelota(Q0, _plano(BOLA), 0.067, _plano(DEJAR))
    finally:
        pl.rodeo = True
        pl.poner_mapa(None, 0.02)


def _tarea_pelota(tactil):
    fr5 = FR5Falso()
    capturado = {}

    class PlanPelota(PlanFalso):
        def poner_mapa(self, centros, res, excluir=()):
            capturado["excluir"] = list(excluir)
            return 0

        def planificar_pelota(self, q0, T, d, Td):
            capturado["pelota"] = (T, d, Td)
            p = _plano_pelota_falso()
            return TA.Plan(q0, p.fases, nombre="coger, apretar y dejar la pelota")
    t = TA.Tarea(dict(G, AUTONOMO_TAREA_MOVER_MANO=False), ".", fr5, simulacro=False,
                 fabrica_planificador=PlanPelota, tactil=tactil)
    return t, fr5, capturado


def _plano_pelota_falso():
    qa = np.zeros(6)
    qs = qa + np.radians([10, 0, 0, 0, 0, 0]) * np.linspace(0, 1, 10)[:, None]
    return TA.Plan(qa, [TA.Fase("Bajar", qs), TA.Fase("Cerrar", mano="pelota_cerrada"),
                        TA.Fase("Comprobar", dur=0.3, comprobar="agarre"), TA.Fase("Subir", qs[::-1])])


HERR_PELOTA = [dict(id=100, nombre="Pelota roja", tipo="pelota", estable=True, diametro_m=0.08,
                    T_mundo=_plano((0.3, -0.5, 0.0)).tolist()),
               dict(id=101, nombre="Pelota verde", tipo="pelota", estable=True, diametro_m=0.045,
                    T_mundo=_plano((0.42, -0.5, 0.0)).tolist()),
               dict(id=22, nombre="Punto de dejada", tipo="punto_dejada", estable=True, T_mundo=np.eye(4).tolist())]


def _ejecutar_pelota(t, fr5):
    t.orden({"tarea": dict(accion="planificar", id="p1")},
            lambda: dict(q_real=fr5.q, herramientas=HERR_PELOTA, objetivo_id=100, mapa=None))
    _esperar_plan(t, fr5)
    t.orden({"tarea": dict(accion="ejecutar", id="p2")}, lambda: dict(q_real=fr5.q))
    for _ in range(2000):
        t.paso(1 / 60, fr5.q, True)
        if t.estado in ("terminada", "error"):
            break


def test_pelota_no_necesita_punto_de_uso_y_solo_excluye_la_pelota():
    t, fr5, cap = _tarea_pelota(None)
    _ejecutar_pelota(t, fr5)
    assert t.estado == "terminada"
    T, d, _ = cap["pelota"]
    assert d == 0.08 and np.allclose(T[:3, 3], (0.3, -0.5, 0.0))
    exc = [(np.asarray(c, float), float(r)) for c, r in cap["excluir"]]
    marcador = [(c, r) for c, r in exc if np.linalg.norm(c) < 1e-9]
    roja = [(c, r) for c, r in exc if np.linalg.norm(c) >= 1e-9]
    assert len(marcador) == 1 and marcador[0][1] == pytest.approx(G["AUTONOMO_TAREA_EXCLUIR_RADIO"])
    # (18-nov, tarde) la roja: su zona (pelotas.zona_propia) la cubre entera, y nada más que alrededor de ella
    assert roja and any(np.linalg.norm(c - (0.3, -0.5, 0.0)) + 0.04 <= r for c, r in roja)
    assert all(np.linalg.norm(c - (0.3, -0.5, 0.0)) < 0.05 for c, _ in roja)
    # la verde está a 9,75 cm de la roja (antes se ignoraba todo a 15 cm): sigue siendo obstáculo
    assert all(np.linalg.norm(np.array((0.42, -0.5, 0.0)) - c) - 0.0225 > r for c, r in roja)


def test_si_los_tactiles_no_notan_la_pelota_se_para():
    t, fr5, _ = _tarea_pelota(lambda: dict(estado="ok", dedos=[dict(fz=0.0)] * 5))
    _ejecutar_pelota(t, fr5)
    assert t.estado == "error" and "No noto la pelota" in t.texto and fr5.hab[-1] is False


def test_si_los_tactiles_la_notan_sigue():
    t, fr5, _ = _tarea_pelota(lambda: dict(estado="ok", dedos=[dict(fz=1.2)] * 3 + [dict(fz=0.0)] * 2))
    _ejecutar_pelota(t, fr5)
    assert t.estado == "terminada"


# ══════════════════════ pelota sobre una mesa (12-nov) ══════════════════════
def test_nube_de_la_mano_sigue_la_postura():
    """Con los dedos a 0 el pulgar sobresale ~10 cm por el lado de la palma: así no se puede bajar
    sobre una pelota de una mesa. Con la postura de coger pelotas sobresale bastante menos."""
    from skill_shield.autonomo.choque_mano import nube_mano
    m = cargar_cadena(config.RUTA_XML, verbose=False).fk_marcos(np.zeros(6))
    T_w_mm = np.linalg.inv(m[5]) @ m[6]          # (23-nov) mano_mount en la brida, del MJCF (antes, el montaje fijo)
    Ti = np.linalg.inv(T_w_mm)

    def sale(post):
        P = nube_mano(config.RUTA_XML, tuple(sorted(post.items())))
        return float(np.max(OFFSET_PALMA_ORCA[0] - (P @ Ti[:3, :3].T + Ti[:3, 3])[:, 0]))
    assert sale({}) > 0.09
    assert sale(TA.MANO_PELOTA_ABIERTA) < 0.06


def test_pinza_con_el_punto_medido(cadena):
    """(23-nov) Con AUTONOMO_TAREA_PINZA_PUNTO_BRIDA_M (medido con diagnosticos/medir_punto_pinza.py), el punto que
    se lleva a la pelota es ese (marco de la brida), no el del modelo de la mano; el resto de la pinza, igual."""
    p_brida = np.array([-0.070, 0.026, 0.366])
    normal = TA.Planificador(cadena, G).pinza(0.07)
    medida = TA.Planificador(cadena, dict(G, AUTONOMO_TAREA_PINZA_PUNTO_BRIDA_M=tuple(p_brida))).pinza(0.07)
    m = cadena.fk_marcos(np.zeros(6))
    T_w_mm = np.linalg.inv(m[5]) @ m[6]                              # mano_mount en la brida
    assert np.allclose(T_w_mm[:3, :3] @ medida["punto"] + T_w_mm[:3, 3], p_brida, atol=1e-9)
    assert not np.allclose(medida["punto"], normal["punto"])
    assert medida["s"] == pytest.approx(normal["s"]) and np.allclose(medida["linea"], normal["linea"])


def _mesa(zt, y0=-0.30):
    X, Y = np.meshgrid(np.arange(-0.25, 0.71, 0.02), np.arange(-0.85, y0, 0.02))
    return np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, zt - 0.01)])


@pytest.mark.parametrize("bxy", [(0.35, -0.45), (0.20, -0.50)])
def test_pelota_encima_de_una_mesa_del_mapa(cadena, pl, bxy):
    """Caso del 12-nov: la pelota en una mesa que la cámara ve. Antes no había plan («colisión de
    la mano»): la mano se comprobaba con el pulgar estirado hacia la mesa."""
    zt, r = -0.25, 0.035
    bola = np.array([*bxy, zt + r])
    dej = np.array([0.40 if bxy[0] < 0.3 else 0.10, -0.50, zt + 0.002])
    try:
        pl.poner_mapa(_mesa(zt), 0.02, [(bola, r + 0.02), (dej, 0.15)])
        plan = pl.planificar_pelota(Q0, _plano(bola), 2 * r, _plano(dej))
        nombres = [f.nombre for f in plan.fases]
        fb = plan.fases[nombres.index("Bajar a la pelota")]
        # la última postura (con la mano como la lleva el plan al subir: cerrada y con la pelota) sigue sin
        # tocar la mesa, y TODA la mano (el pulgar también, 17-nov) le deja la holgura pedida
        fs = plan.fases[nombres.index("Subir con la pelota")]
        with pl.con_mano(*fs.mano_chk):
            assert pl.motivo(fb.qs[-1]) is None
            Tb = cadena.fk_marcos(fb.qs[-1])[5]
            P = pl._mano_pts @ Tb[:3, :3].T + Tb[:3, 3]
            assert P[:, 2].min() > zt + pl.holgura_mesa_mano - 0.002
    finally:
        pl.poner_mapa(None, 0.02)


def test_mensaje_dice_el_motivo_mas_util(pl):
    from collections import Counter
    razones = Counter({"la muñeca tendría que doblarse más de lo que permiten sus límites (J4-J6)": 40,
                       "la mano tocaría la mesa u otro obstáculo que ve la cámara": 3})
    e = pl._no_encuentro("llegar a la pelota", (razones, Counter()))
    assert "cámara" in str(e) and "muñeca" not in str(e)


def test_plan_fallido_se_guarda(tmp_path):
    t = TA.Tarea(dict(G), str(tmp_path), FR5Falso(), simulacro=False)
    txt = t._guardar_fallo(dict(q=np.zeros(6), T=[np.eye(4)], mapa=(0.02, np.ones((5, 3))),
                                excluir=[(np.zeros(3), 0.1)], nombres=["Pelota roja"], pelota=True,
                                diametro=0.07), "no hay camino")
    f = list((tmp_path / "diagnosticos").glob("plan_fallido_*.npz"))
    assert f and "guardado" in txt
    d = np.load(f[0])
    assert d["mapa"].shape == (5, 3) and d["T"].shape == (1, 4, 4) and "no hay camino" in str(d["info"])


def test_zona_de_agarre_ignora_el_ruido_y_pone_suelo(cadena, pl):
    """Lejos de la cámara la mesa sale como una capa gruesa (ruido) que llega a la altura de la
    pelota: en la zona de agarre no cuenta, pero la mano no puede bajar de la parte de abajo de la
    pelota. Lo que sea más alto que la pelota sigue contando."""
    zt, r = -0.25, 0.035
    bola = np.array([0.35, -0.45, zt + r])
    capa = np.vstack([_mesa(zt + dz) for dz in (0.0, 0.02, 0.04)])              # mesa de 4 cm de grueso (ruido)
    try:
        pl.poner_mapa(capa, 0.02, [(bola, r + 0.03)])
        plan = pl.planificar_pelota(Q0, _plano(bola), 2 * r, _plano((0.10, -0.50, zt + 0.002)))
        fb = next(f for f in plan.fases if f.nombre == "Bajar a la pelota")
        assert pl.valida(fb, fb.qs[-1]) is None
        # una caja de 15 cm justo al lado de la pelota sí cuenta
        caja = bola + np.stack(np.meshgrid(np.arange(0.06, 0.15, 0.02), np.arange(-0.08, 0.09, 0.02),
                                           np.arange(0.0, 0.16, 0.02)), -1).reshape(-1, 3)
        pl.poner_mapa(np.vstack([capa, caja, caja * [1, 1, 1] - [0.18, 0, 0]]), 0.02, [(bola, r + 0.03)])
        pl.zonas_apoyo([(bola, r + pl.zona_apoyo, r)])
        pl._choques = []
        with pl.con_mano("pelota_abierta"):
            assert pl.motivo(fb.qs[-1]) is not None
        assert "cm de alto" in pl._que_estorba(bola, r)               # y dice qué estorba
    finally:
        pl.poner_mapa(None, 0.02)


def test_avisa_si_la_pelota_esta_lejos(pl):
    lejos = np.array([0.0, -1.15, 0.0])
    assert "alcance" in pl._aviso_lejos(lejos).lower()
    assert pl._aviso_lejos(np.array([0.3, -0.45, -0.2])) == "" or "cámara" in pl._aviso_lejos(np.array([0.3, -0.45, -0.2]))


# ── pinza (13-nov) ───────────────────────────────────────────────────────────────────────────
def test_pinza_segun_el_tamano_de_la_pelota(pl):
    """Cuanto más pequeña, más cierra; el punto de agarre queda delante de la palma, entre las
    yemas; una pelota más grande que lo que abre la pinza no se intenta en pinza."""
    p4, p7 = pl.pinza(0.04), pl.pinza(0.067)
    assert p4 is not None and p7 is not None and p4["s"] > p7["s"] > 0.0
    assert p7["punto"][0] < -0.03 and p7["punto"][1] > OFFSET_PALMA_ORCA[1]     # delante de la palma, hacia los dedos
    assert pl.pinza(0.20) is None
    for k in (8, 9, 11, 12, 15, 16):                                          # aprieta: cierra más que al tocarla
        assert p7["apretar"][k] >= p7["cerrada"][k] >= p7["contacto"][k] - 1e-9


def test_plan_pelota_en_pinza_la_coge_entre_las_yemas(cadena, pl):
    from skill_shield.autonomo.choque_mano import yemas_mano
    plan = pl.planificar_pelota(Q0, _plano(BOLA), 0.067, _plano(DEJAR))
    assert pl.modo_pelota == "en pinza" and plan.pelota["pinza"]
    assert {f.mano for f in plan.fases if f.mano} >= {"pinza_abierta", "pinza_cerrada", "pinza_apretar"}
    fb = plan.fases[plan.pelota["i_bajar"]]
    Tw = cadena.fk_marcos(fb.qs[-1])[5]
    y = {k: Tw[:3, :3] @ v + Tw[:3, 3] for k, v in yemas_mano(
        config.RUTA_XML, tuple(sorted((k, round(v, 4)) for k, v in pl._posturas["pinza_contacto"].items())), 0.004).items()}
    r = 0.0335
    for dedo in ("pulgar", "indice"):                                         # las dos yemas tocan la pelota
        assert abs(np.linalg.norm(y[dedo] - np.array(BOLA)) - r) < 0.004
    assert np.dot(y["pulgar"] - BOLA, y["indice"] - BOLA) < 0                 # por lados opuestos
    ej = TA.Ejecutor(plan, {"abierta": {}}, 1.0, 2.0)                         # usa las posturas del plan
    assert "pinza_cerrada" in ej.posturas


# ── (17-nov) ir por arriba, apartarse para mirar y la mesa medida ─────────────────────────────
def _cajonera(zt=-0.25, X=(0.10, 0.60), Y=(-0.75, -0.30), z0=-0.95, res=0.02):
    """Lo que ve la cámara de una cajonera maciza: su tapa y sus lados (por dentro, nada)."""
    xs, ys = np.arange(X[0], X[1], res) + res / 2, np.arange(Y[0], Y[1], res) + res / 2
    zs = np.arange(z0, zt, res) + res / 2
    tapa = np.array([[x, y, zt - res / 2] for x in xs for y in ys])
    lados = [[x, y, z] for z in zs for x in xs for y in (ys[0], ys[-1])] + \
            [[x, y, z] for z in zs for y in ys for x in (xs[0], xs[-1])]
    return np.vstack([tapa, lados])


def _dentro_cajonera(P, zt=-0.25, X=(0.10, 0.60), Y=(-0.75, -0.30), h=0.005):
    return (P[:, 0] > X[0] + h) & (P[:, 0] < X[1] - h) & (P[:, 1] > Y[0] + h) & (P[:, 1] < Y[1] - h) & (P[:, 2] < zt - h)


@pytest.mark.parametrize("hueco", [0.0, 0.25])
def test_plan_de_la_pelota_no_mete_el_brazo_en_la_cajonera(cadena, hueco):
    """Caso del 17-nov: la mano empezaba cerca de la mesa y, al «Ir encima de la pelota», el antebrazo
    se metía en la cajonera (dentro de la zona de la pelota la mesa no contaba, y por dentro estaba
    hueca), más aún con el hueco que dejaba la mano en el mapa. Ahora: suelo también para el brazo,
    mesa maciza, y los tramos van por arriba."""
    pl = TA.Planificador(cadena, G)
    zt, r = -0.25, 0.035
    bola, dej = np.array([0.35, -0.45, zt + r]), np.array([0.20, -0.62, zt + 0.002])
    S = _cajonera()
    S = S[np.linalg.norm(S[:, :2] - bola[:2], axis=1) > hueco] if hueco else S
    rng = np.random.default_rng(1)
    hechos = 0
    while hechos < 4:
        q = Q0 + rng.normal(0, 0.5, 6)
        T = cadena.fk_marcos(q)[6]
        palma = T[:3, :3] @ OFFSET_PALMA_ORCA + T[:3, 3]
        if not (zt + 0.04 < palma[2] < zt + 0.25 and np.linalg.norm(palma[:2] - bola[:2]) < 0.35):
            continue
        pl.poner_mapa(S, 0.02, [(bola, r + 0.03), (dej, 0.15)])
        if pl.motivo(q) is not None or _dentro_cajonera(pl._puntos_robot(q)).any():
            continue
        hechos += 1
        # la mesa medida (plano de la profundidad, como la da pelotas.py): maciza hasta el suelo
        plan = pl.planificar_pelota(q, _plano(bola), 2 * r, _plano(dej), apoyo_z=zt,
                                    apoyo_poligono=[[0.10, -0.75], [0.60, -0.75], [0.60, -0.30], [0.10, -0.30]])
        for f in plan.fases:
            if f.qs is None:
                continue
            clave = f.mano_chk[0]
            mano = pl._nube_postura(clave) if clave else pl._mano_pts0
            for qq in f.qs:
                m = cadena.fk_marcos(qq)
                P = [mano @ m[5][:3, :3].T + m[5][:3, 3]] + [p @ m[k][:3, :3].T + m[k][:3, 3] for k, p, _ in pl._brazo_pts]
                assert not _dentro_cajonera(np.vstack(P)).any(), f"{f.nombre} se mete en la cajonera"
        assert pl.tramos_por_alto > 0
    pl.poner_mapa(None, 0.02)


def test_tramo_por_alto_sube_por_encima_de_lo_que_hay(cadena, pl):
    zt = -0.25
    pl.poner_mapa(_cajonera(zt), 0.02)
    pl._empezar(Q0)
    rng = np.random.default_rng(2)
    qs = []
    while len(qs) < 2:
        q = Q0 + rng.normal(0, 0.4, 6)
        p = pl._palma(q)
        if zt + 0.05 < p[2] < zt + 0.15 and pl.motivo(q) is None and not _dentro_cajonera(pl._puntos_robot(q)).any():
            if not qs or np.linalg.norm(p[:2] - pl._palma(qs[0])[:2]) > 0.15:
                qs.append(q)
    try:
        tramo = pl.tramo_por_alto(qs[0], qs[1])
    except TA.ErrorPlan:
        pytest.skip("estas dos posturas no tienen tramo por arriba")
    finally:
        pl.poner_mapa(None, 0.02)
    z = np.array([pl._palma(q)[2] for q in tramo])
    assert z.max() >= zt + pl.alto_margen - 0.011            # pasa por encima de la tapa + el margen
    assert np.allclose(tramo[0], qs[0]) and np.allclose(tramo[-1], qs[1])


def test_si_el_brazo_tapa_la_pelota_se_aparta_mira_y_vuelve_a_planificar():
    fr5 = FR5Falso()
    llamadas = []

    class PlanMirar(PlanFalso):
        def plan_mirar(self, q0, c):
            llamadas.append("mirar")
            if len(llamadas) > 1:
                raise AssertionError("solo se aparta una vez")
            qs = q0 + np.radians([0, 0, 0, 0, 0, 10]) * np.linspace(0, 1, 10)[:, None]
            p = TA.Plan(q0, [TA.Fase("Apartarse", qs), TA.Fase("Mirar la mesa", dur=0.2)], nombre="apartarse para mirar")
            p.replanificar = True
            return p

        def planificar_pelota(self, q0, T, d, Td, apoyo_z=None, **kw):
            llamadas.append(("pelota", apoyo_z))
            return TA.Plan(q0, _plano_pelota_falso().fases, nombre="coger, apretar y dejar la pelota")
    t = TA.Tarea(dict(G, AUTONOMO_TAREA_MOVER_MANO=False), ".", fr5, simulacro=False, fabrica_planificador=PlanMirar)
    herr = [dict(h, apoyo_z=-0.04) if h["id"] == 100 else h for h in HERR_PELOTA]
    ctx = lambda: dict(q_real=fr5.q, herramientas=herr, objetivo_id=100, mapa=None)
    t.orden({"tarea": dict(accion="planificar", id="m1")}, ctx)
    _esperar_plan(t, fr5)
    assert t.estado == "lista" and "me aparto" in t.texto and t.plan.nombre == "apartarse para mirar"
    t.orden({"tarea": dict(accion="ejecutar", id="m2")}, ctx)
    for _ in range(600):
        t.paso(1 / 60, fr5.q, True, ctx)
        if t.estado == "planificando":
            break
    assert t.estado == "planificando"                         # al acabar de mirar, planifica la tarea solo
    _esperar_plan(t, fr5)
    assert t.estado == "lista" and t.plan.nombre.endswith("pelota")   # y espera a «Ejecutar» (no arranca solo)
    assert llamadas == ["mirar", ("pelota", -0.04)]


def test_la_mesa_medida_pone_la_pelota_apoyada(pl):
    T = _plano((0.3, -0.5, -0.20))
    assert pl.apoyar(T, 0.07, -0.25)[2, 3] == pytest.approx(-0.215)     # 1,5 cm más abajo: apoyada en la mesa
    assert pl.apoyar(T, 0.07, -0.40)[2, 3] == pytest.approx(-0.20)      # no cuadra (15 cm): no se usa
    assert pl.apoyar(T, 0.07, None)[2, 3] == pytest.approx(-0.20)


def test_caso_real_17nov_el_pulgar_no_toca_la_mesa(cadena):
    """Plan real del 17-nov (diagnosticos/plan_ok_20261117_104644.npz): en la bajada en pinza el pulgar
    quedaba a ~1 cm de la mesa estimada y en el robot la tocaba. Ahora toda la mano le deja la holgura
    pedida y los tramos van por arriba."""
    from skill_shield.autonomo import calibracion as C
    d = np.load(os.path.join(RAIZ, "tests", "datos", "plan_pelota_pulgar_17nov.npz"))
    pl = TA.Planificador(cadena, G, C.cargar(os.path.join(RAIZ, "camara_robot.json"))["T_mundo_cam"])
    pl.poner_mapa(d["mapa"], float(d["res"]), [(e[:3], e[3]) for e in d["excluir"]])
    plan = pl.planificar_pelota(d["q"], d["T"][0], 0.07, d["T"][1])
    b = d["T"][0][:3, 3]
    z_mesa = b[2] - 0.035
    for f in plan.fases:
        if f.qs is None or not f.nombre.startswith(("Bajar a la pelota", "Subir con la pelota")):
            continue
        clave, _ = f.mano_chk
        for q in f.qs:
            Tb = cadena.fk_marcos(q)[5]
            P = pl._nube_postura(clave) @ Tb[:3, :3].T + Tb[:3, 3]
            sobre = np.linalg.norm(P[:, :2] - b[:2], axis=1) < 0.30
            assert P[sobre, 2].min() > z_mesa + pl.holgura_mesa_mano - 0.002, f.nombre
    assert pl.tramos_por_alto >= 1


def test_comprobacion_continua_ve_lo_que_hay_entre_dos_posturas(cadena):
    """(17-nov) Un cubo justo donde pasa la mano ENTRE dos posturas válidas: la comprobación por
    posturas sueltas no lo ve; la continua, sí."""
    pl = TA.Planificador(cadena, G)
    rng = np.random.default_rng(5)
    for _ in range(300):
        qa = Q0 + rng.normal(0, 0.3, 6)
        qb = qa + np.radians([14.0, 0, 0, 0, 0, 0])
        qm = 0.5 * (qa + qb)
        pl.poner_mapa(np.array([pl._palma(qm)]), 0.02)
        pl.min_grupo = 1
        if pl.motivo(qa) is None and pl.motivo(qb) is None and pl.motivo(qm) is not None:
            break
    else:
        pytest.skip("no encuentro un caso así")
    pl.continuo = False
    pl.comprobar_continuo(np.array([qa, qb]))                 # sin continua: no se entera
    pl.continuo = True
    with pytest.raises(TA.ErrorPlan, match="entre dos posturas"):
        pl.comprobar_continuo(np.array([qa, qb]))
    pl.poner_mapa(None, 0.02)


def test_lo_no_visto_cuenta_como_coste(cadena, pl):
    from skill_shield.autonomo.mapa import MapaOcupacion, empaquetar_visto, desempaquetar_visto
    m = MapaOcupacion(((-1.0, 0.6), (-1.6, 0.4), (-0.975, 0.9)), 0.02)
    v = desempaquetar_visto(empaquetar_visto(m))               # recién creado: nada visto
    assert v["n"] == tuple(m.n) and not v["conocido"].any()
    qs = Q0 + np.radians([10, 0, 0, 0, 0, 0]) * np.linspace(0, 1, 6)[:, None]
    pl.poner_mapa(None, 0.02)
    pl.poner_visto(v)
    assert pl.desconocido(qs) == pytest.approx(1.0)
    c_desc = pl.coste_tramo(qs)
    v["conocido"][:] = True
    assert pl.desconocido(qs) == 0.0 and pl.coste_tramo(qs) < c_desc
    pl.poner_visto(None)
    assert pl.desconocido(qs) == 0.0


def _en_mesa(pl, P):
    """Dentro de una mesa: losas (mundo simple, 18-nov) o discos de suelo (mundo completo)."""
    P = np.asarray(P, float)
    return bool(pl._mascara_losas(P, 0.0).any()) if pl.mundo_simple else pl._bajo_suelo(P, solo_discos=True)


def test_la_mesa_medida_es_suelo_en_toda_su_extension(pl):
    pl.poner_mapa(None, 0.02)
    pl.poner_mesa(-0.25, [[0.0, -0.8], [0.6, -0.8], [0.6, -0.4], [0.0, -0.4]])
    pl.zonas_apoyo(())
    assert _en_mesa(pl, [[0.30, -0.60, -0.27]])               # dentro y debajo: no
    assert not _en_mesa(pl, [[0.30, -0.60, -0.20]])           # encima: sí
    assert not _en_mesa(pl, [[0.80, -0.60, -0.27]])           # fuera de la mesa: sí
    pl.poner_mapa(None, 0.02)


def test_mundo_simple_quita_la_capa_de_la_mesa_pero_no_lo_que_hay_encima(cadena):
    """(18-nov) Con el mundo simple, lo que la cámara ve de la propia mesa (con su ruido de unos cm) no son
    obstáculos (la losa ya es la mesa); una caja de 10 cm encima, sí."""
    pl = TA.Planificador(cadena, dict(G, AUTONOMO_TAREA_MUNDO="simple"))
    pl.grupo_min = 1
    zt = -0.25
    x, y = np.meshgrid(np.arange(0.05, 0.55, 0.02), np.arange(-0.75, -0.35, 0.02))
    mesa = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, zt + 0.01)])
    ruido = mesa[::7] + [0.0, 0.0, 0.02]
    caja = np.array([[0.30, -0.55, zt + h] for h in np.arange(0.03, 0.11, 0.02)])
    pl.poner_mapa(np.vstack([mesa, ruido, caja]), 0.02)
    pl.poner_mesa(zt, [[0.0, -0.8], [0.6, -0.8], [0.6, -0.3], [0.0, -0.3]])
    pl.zonas_apoyo(())
    quedan = pl._arbol.data
    assert pl.quitados_mesa >= len(mesa)
    assert np.all(quedan[:, 2] > zt + 0.035)                  # solo lo de la caja por encima de la capa
    assert len(quedan) >= 3
    pl.poner_mapa(None, 0.02)


def test_las_mesas_del_mapa_son_suelo_y_no_estorban_si_el_robot_esta_debajo(cadena):
    """(17-nov) Las mesas que encuentra el mapa (planos) son suelo para mano y brazo; pero si el robot
    está AHORA debajo de una (mal medida, o un estante), esa no se usa."""
    from skill_shield.autonomo.mapa import MapaOcupacion, desempaquetar_visto, empaquetar_visto
    pl = TA.Planificador(cadena, G)
    m = MapaOcupacion(((-1.0, 0.6), (-1.6, 0.4), (-0.975, 0.9)), 0.02)
    v = desempaquetar_visto(empaquetar_visto(m))
    v["conocido"][:] = True
    mesa = np.array([[0.0, -0.8], [0.6, -0.8], [0.6, -0.4], [0.0, -0.4]])
    v["superficies"] = [(-0.25, mesa)]
    pl.poner_mapa(None, 0.02)
    pl.poner_visto(v)
    assert _en_mesa(pl, [[0.3, -0.6, -0.30]])
    # el robot, con la mano debajo de una «mesa» a z más alta que la palma: esa mesa no se usa
    p = pl._palma(Q0)
    estante = np.array([[p[0] - 0.2, p[1] - 0.2], [p[0] + 0.2, p[1] - 0.2], [p[0] + 0.2, p[1] + 0.2], [p[0] - 0.2, p[1] + 0.2]])
    v["superficies"] = [(-0.25, mesa), (float(p[2]) + 0.05, estante)]
    pl.poner_visto(v)
    pl._empezar(Q0)                                           # no da error: el estante se descarta
    assert len(pl._superficies) == 1 and pl._superficies[0][0] == pytest.approx(-0.25)
    pl.poner_visto(None)
    pl.poner_mapa(None, 0.02)


def test_correccion_fija_de_la_pelota_vista_desde_la_camara():
    """(18-nov) AUTONOMO_TAREA_PELOTA_CORRECCION_M = (derecha, lejos, arriba) en la imagen de la cámara."""
    t = TA.Tarea(dict(G, AUTONOMO_TAREA_PELOTA_CORRECCION_M=(-0.015, 0.02, 0.005), AUTONOMO_TAREA_PRECALENTAR=False),
                 ".", None, simulacro=True, fabrica_planificador=lambda: None)
    Tc = np.eye(4)
    Tc[:3, :3] = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])  # mira hacia -y (x de la imagen = -x del mundo)
    T = t._corregir_pelota(np.eye(4), Tc)
    assert np.allclose(T[:3, 3], (0.015, -0.02, 0.005))               # 1,5 cm a la izquierda de la imagen, 2 cm más lejos


# ══════════════════════ (23-nov) sin FR5, mano con par caída, pedal desactivado ══════════════════════
def test_sin_fr5_planifica_y_ejecuta_en_el_gemelo():
    """FR5 apagado (o AUTONOMO_CON_FR5 = False): la tarea sale de la pose del gemelo y se hace solo en
    el gemelo, aunque AUTONOMO_FR5_SIMULACRO sea False."""
    for fr5 in (None, FR5Falso()):
        if fr5 is not None:
            fr5.conectado = False                               # encendido nunca / apagado
        t = TA.Tarea(dict(G, AUTONOMO_TAREA_MOVER_MANO=False), ".", fr5, simulacro=False,
                     fabrica_planificador=PlanFalso)
        q_gem = np.radians([10.0, -20.0, 30.0, 0.0, 40.0, 0.0])
        ctx = lambda: dict(q_real=None, q_gemelo=q_gem, herramientas=HERR, objetivo_id=20, mapa=None)
        assert t.solo_gemelo()
        t.orden({"tarea": dict(accion="planificar", id="g-1")}, ctx)
        t0 = time.time()
        while t.estado == "planificando" and time.time() - t0 < 5:
            time.sleep(0.01)
            t.paso(0.0, None, False, ctx)
        assert t.estado == "lista" and np.allclose(t.plan.q0, q_gem) and t.solo_gemelo()
        t.orden({"tarea": dict(accion="ejecutar", id="g-2")}, ctx)
        assert t.estado == "pausada"
        q_vista = None
        for _ in range(3000):
            q_vista = t.paso(1 / 60, None, True, ctx)
            if t.estado == "terminada":
                break
        assert t.estado == "terminada" and q_vista is not None      # el gemelo enseña la tarea
        if fr5 is not None:
            assert fr5.hab == [] and fr5.pub == []                   # el FR5 real ni se toca


def test_plan_del_gemelo_no_se_vuelve_con_el_fr5_real():
    fr5 = FR5Falso()
    fr5.conectado = False
    t = TA.Tarea(dict(G, AUTONOMO_TAREA_MOVER_MANO=False), ".", fr5, simulacro=False, fabrica_planificador=PlanFalso)
    q_gem = np.zeros(6)
    ctx_g = lambda: dict(q_real=None, q_gemelo=q_gem, herramientas=HERR, objetivo_id=20, mapa=None)
    t.orden({"tarea": dict(accion="planificar", id="v-1")}, ctx_g)
    t0 = time.time()
    while t.estado == "planificando" and time.time() - t0 < 5:
        time.sleep(0.01)
        t.paso(0.0, None, False, ctx_g)
    assert t._q_inicio is not None and t._q_inicio_gemelo
    fr5.conectado = True                                        # ahora el FR5 está: «volver» con el real, no
    t.orden({"tarea": dict(accion="volver", id="v-2")},
            lambda: dict(q_real=fr5.q, q_gemelo=q_gem, herramientas=HERR, objetivo_id=20, mapa=None))
    assert t.estado == "error" and "gemelo" in t.texto


class _WrapperFalso:
    JOINT_A_CTRL = {"a": 0}
    _joints = ["a"]

    def __init__(self, n):
        self.n, self.conectado, self.desconectada = n, False, False

    def conectar(self):
        self.conectado = True
        return True

    def enviar_posiciones(self, arr):
        return self.conectado

    def desconectar(self):
        self.desconectada = True


def test_mano_activa_se_reconecta_sola_y_lo_dice():
    from skill_shield.autonomo.ejecucion import ManoActiva
    creados = []
    m = ManoActiva("/dev/falso", fabrica=lambda: creados.append(_WrapperFalso(len(creados))) or creados[-1],
                   reintento_s=0.2).iniciar()
    try:
        t0 = time.time()
        while not m.lista and time.time() - t0 < 2:
            time.sleep(0.01)
        assert m.lista and m.estado()["estado"] == "ok"
        creados[0].conectado = False                            # se cae (cable / alimentación)
        t0 = time.time()
        while m.perdidas == 0 and time.time() - t0 < 2:
            time.sleep(0.01)
        assert m.perdidas == 1 and creados[0].desconectada
        t0 = time.time()
        while not m.lista and time.time() - t0 < 3:
            time.sleep(0.01)
        assert m.lista and len(creados) == 2 and m.estado()["estado"] == "ok"
    finally:
        m.cerrar()
    assert not m.lista and creados[-1].desconectada


def test_mano_caida_a_mitad_aborta_la_tarea():
    t, fr5, mano, mon = _tarea(mover_mano=True)
    _orden(t, "planificar", 1, fr5)
    _esperar_plan(t, fr5)
    _orden(t, "ejecutar", 2, fr5)
    t0 = time.time()
    while not mano.lista and time.time() - t0 < 2:
        t.paso(1 / 60, fr5.q, True)
        time.sleep(0.01)
    for _ in range(20):
        t.paso(1 / 60, fr5.q, True)
    assert t.estado == "ejecutando" and fr5.hab == [True]
    mano.lista = False                                          # se cae la mano
    t.paso(1 / 60, fr5.q, True)
    assert t.estado == "error" and "OrcaHand" in t.texto and fr5.hab[-1] is False


def test_pedal_desactivado_permite_mover_mientras_la_web_vive():
    from skill_shield.autonomo.modo import Control
    c = Control()
    ahora = 1000.0
    c.actualizar(dict(pedal_obligatorio=False, pedal=False), ahora, ahora)
    assert c.permiso_tarea() and c.movimiento_permitido(True)
    c.actualizar(dict(pedal_obligatorio=True, pedal=False), ahora, ahora)
    assert not c.permiso_tarea()
    c.actualizar(dict(pedal_obligatorio=False, pedal=False), ahora - 5, ahora)    # web caída
    assert not c.permiso_tarea()
