"""
Pruebas de la tarea coger → usar → dejar (11-nov, paso 10), sin robot, sin mano y sin MuJoCo:

    python -m pytest tests_tarea.py -q

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                        # noqa: E402
from skill_shield.autonomo import tarea as TA                        # noqa: E402
from skill_shield.hardware.fr5_servo import ENVIANDO, ACERCANDO      # noqa: E402
from skill_shield.teleop_fr5.cinematica import OFFSET_PALMA_ORCA, cargar_cadena   # noqa: E402

G = {k: getattr(config, k) for k in dir(config) if k.isupper()}
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
    cam = cargar(os.path.join(os.path.dirname(os.path.abspath(__file__)), "camara_robot.json"))["T_mundo_cam"][:3, 3]
    m = [_marcador(cam - np.array(p), p) for p in pos]
    plan = pl.planificar(Q0, *m)
    assert np.allclose(plan.q_final, Q0) and max(pl.inclinaciones) > 0
    for f in plan.fases:
        if f.qs is not None:
            assert all(pl.motivo(q) is None for q in f.qs[::5])
