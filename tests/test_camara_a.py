"""
Pruebas del 19-nov (cámara A: OAK-D fija mirando la mesa de cerca), sin robot, sin cámaras y sin MuJoCo:

    python -m pytest tests/test_camara_a.py -q

  · camara_a.pelota_de_a: elige la pelota de la A que corresponde a la de la cabeza (cercana, estable, reciente).
  · Tarea: si la A ve la misma pelota, se planifica con su medida; si no cuadra, con la de cabeza.
  · Afinado: encima de la pelota se afina con la A (aunque la cabeza la vea mal o tapada).
  · modelo_colision: la cámara A, calibrada y activa, es un obstáculo más.
"""
import json
import os
import sys
import time

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import config                                                        # noqa: E402
from skill_shield.autonomo import tarea as TA                        # noqa: E402
from skill_shield.autonomo.camara_a import leer_canal, pelota_de_a   # noqa: E402
from skill_shield.hardware.fr5_servo import ACERCANDO, ENVIANDO      # noqa: E402
from skill_shield.teleop_fr5.cinematica import cargar_cadena         # noqa: E402

G = {k: getattr(config, k) for k in dir(config) if k.isupper()}
GA = dict(G, AUTONOMO_TAREA_MOVER_MANO=False, AUTONOMO_CAM_A=True, AUTONOMO_TAREA_PELOTA_CORRECCION_M=(0, 0, 0),
          AUTONOMO_CAM_A_PELOTA_CORRECCION_M=(0, 0, 0), AUTONOMO_CAM_A_OBLIGATORIA=False)
GA_OBLIGATORIA = dict(GA, AUTONOMO_CAM_A_OBLIGATORIA=True)       # (23-nov) la pelota solo se coge con la medida de la A
Q0 = np.radians([93.0, -17.0, 53.0, 43.0, 105.0, 25.0])
BOLA, DEJAR = np.array([0.35, -0.45, -0.15]), np.array([0.15, -0.45, -0.15])
T_A = np.eye(4)
T_A[:3, 3] = [0.35, -0.95, 0.15]                                     # la A, delante de la mesa y algo por encima


def _plano(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


def _pel(p, id_=100, estable=True, edad=0.0):
    return dict(id=id_, nombre="Pelota roja", tipo="pelota", estable=estable, diametro_m=0.067, edad_s=edad,
                T_mundo=_plano(p).tolist())


def _lista(bola):
    return [_pel(bola), dict(id=22, nombre="Punto de dejada", tipo="punto_dejada", estable=True,
                             T_mundo=_plano(DEJAR).tolist())]


@pytest.fixture(scope="module")
def cadena():
    return cargar_cadena(config.RUTA_XML, verbose=False)


# ── emparejar ────────────────────────────────────────────────────────────────────────────────
def test_pelota_de_a_elige_la_misma():
    c = np.array([0.3, -0.4, -0.15])
    lista = [_pel(c + [0.2, 0, 0], 101), _pel(c + [0.02, 0.01, 0], 102), _pel(c + [0.01, 0, 0], 103, estable=False)]
    assert pelota_de_a(lista, c, 0.06)["id"] == 102
    assert pelota_de_a(lista, c, 0.01) is None                       # la estable más cercana está a 2,2 cm
    assert pelota_de_a([_pel(c, edad=1.0)], c, 0.06, edad_max=0.5) is None     # vieja
    assert pelota_de_a([], c, 0.06) is None and pelota_de_a(lista, None, 0.06) is None
    # con el id de antes, esa (si sigue cerca), aunque haya otra más cerca
    lista2 = [_pel(c + [0.03, 0, 0], 7), _pel(c + [0.005, 0, 0], 8)]
    assert pelota_de_a(lista2, c, 0.06, id_pref=7)["id"] == 7


def test_leer_canal():
    class C:
        def __init__(self, b, t):
            self.b, self.t = b, t

        def leer(self):
            return self.b, self.t
    d = dict(lista=[_pel(BOLA)], T_mundo_cam=T_A.tolist())
    lista, T = leer_canal(C(json.dumps(d).encode(), time.time()))
    assert len(lista) == 1 and np.allclose(T, T_A)
    assert leer_canal(C(json.dumps(d).encode(), time.time() - 10)) == ([], None)
    assert leer_canal(None) == ([], None)


# ── la tarea con el planificador de verdad ───────────────────────────────────────────────────
class FR5Falso:
    def __init__(self):
        self.conectado, self.en_error = True, False
        self.q = Q0.copy()
        self._estado = ACERCANDO

    def publicar(self, q):
        self.q = np.array(q)

    def habilitar(self, si=True):
        self._estado = ENVIANDO if si else 1

    def estado(self):
        return dict(estado=self._estado)


def _planificar(cadena, ctx, g=GA, lista=True):
    fr5 = FR5Falso()
    t = TA.Tarea(g, ".", fr5, simulacro=False, fabrica_planificador=lambda: TA.Planificador(cadena, G))
    t.orden({"tarea": dict(accion="planificar", id="p1")}, lambda: dict(ctx, q_real=fr5.q))
    t0 = time.time()
    while t.estado == "planificando" and time.time() - t0 < 120:
        time.sleep(0.05)
        t.paso(0.0, fr5.q, False)
    if lista:
        assert t.estado == "lista", t.texto
    return t, fr5


def test_planifica_con_la_camara_a(cadena):
    da = np.array([0.02, -0.01, 0.0])                               # la A la ve 2,2 cm más allá
    t, _ = _planificar(cadena, dict(herramientas=_lista(BOLA), objetivo_id=100, mapa=None,
                                    herramientas_a=[_pel(BOLA + da, 105)], T_mundo_cam_a=T_A))
    assert t._fuente_plan == "a" and t.estado_web()["fuente_pelota"] == "a"
    assert np.allclose(t.plan.pelota["c"], BOLA + da)


def test_si_la_a_no_cuadra_usa_la_cabeza(cadena):
    t, _ = _planificar(cadena, dict(herramientas=_lista(BOLA), objetivo_id=100, mapa=None,
                                    herramientas_a=[_pel(BOLA + [0.12, 0, 0], 105)], T_mundo_cam_a=T_A))
    assert t._fuente_plan == "cabeza"
    assert np.allclose(t.plan.pelota["c"], BOLA)


def test_afina_con_la_camara_a(cadena, monkeypatch):
    """La cabeza tiene la pelota tapada por el brazo (o la ve mal, a 15 cm): se afina con la A."""
    monkeypatch.setattr(TA.Planificador, "fraccion_tapada",
                        lambda self, q, c, r, n=9, T_cam=None: 0.0 if T_cam is not None else 1.0)
    t, fr5 = _planificar(cadena, dict(herramientas=_lista(BOLA), objetivo_id=100, mapa=None))
    assert t._fuente_plan == "cabeza"                               # al planificar, sin la A
    i_b = [f.nombre for f in t.plan.fases].index("Bajar a la pelota")
    t.orden({"tarea": dict(accion="ejecutar", id="p2")}, lambda: dict(q_real=fr5.q))
    mover = np.array([0.015, -0.02, 0.0])
    ctx = lambda: dict(q_real=fr5.q, herramientas=_lista(BOLA + [0.15, 0, 0]), mapa=None,  # noqa: E731
                       herramientas_a=[_pel(BOLA + mover, 105)], T_mundo_cam_a=T_A)
    for _ in range(20000):
        t.paso(1 / 60, fr5.q, True, ctx)
        if t.estado == "error" or t.ejec.i > i_b:
            break
        if t.afinando:
            time.sleep(0.002)
    assert t.estado != "error", t.texto
    assert t.afinado_fuente == "a"
    assert np.allclose(t.plan.pelota["c"], BOLA + mover)
    assert t.afinado_cm == pytest.approx(np.linalg.norm(mover) * 100, abs=0.1)


# ── (23-nov) la cámara A obligatoria ─────────────────────────────────────────────────────────
def test_con_la_a_obligatoria_sin_la_a_no_se_coge(cadena):
    """Si la A no ve la pelota (o no cuadra con la cabeza), no se coge con la medida de la cabeza: error y no hay plan."""
    for ctx in (dict(herramientas=_lista(BOLA), objetivo_id=100, mapa=None),
                dict(herramientas=_lista(BOLA), objetivo_id=100, mapa=None,
                     herramientas_a=[_pel(BOLA + [0.12, 0, 0], 105)], T_mundo_cam_a=T_A)):
        t, _ = _planificar(cadena, ctx, g=GA_OBLIGATORIA, lista=False)
        assert t.estado == "error" and "cámara A" in t.texto and t.plan is None


def test_con_la_a_obligatoria_la_cabeza_no_recoloca_al_afinar(cadena, monkeypatch):
    """Planificada con la A; al afinar la A no la ve y la cabeza la ve 3 cm más allá: se baja a donde la midió la A."""
    monkeypatch.setattr(TA.Planificador, "fraccion_tapada", lambda self, q, c, r, n=9, T_cam=None: 0.0)
    da = np.array([0.01, -0.01, 0.0])
    t, fr5 = _planificar(cadena, dict(herramientas=_lista(BOLA), objetivo_id=100, mapa=None,
                                      herramientas_a=[_pel(BOLA + da, 105)], T_mundo_cam_a=T_A), g=GA_OBLIGATORIA)
    assert t._fuente_plan == "a"
    i_b = [f.nombre for f in t.plan.fases].index("Bajar a la pelota")
    t.orden({"tarea": dict(accion="ejecutar", id="p2")}, lambda: dict(q_real=fr5.q))
    ctx = lambda: dict(q_real=fr5.q, herramientas=_lista(BOLA + [0.03, 0, 0]), mapa=None,  # noqa: E731
                       herramientas_a=[], T_mundo_cam_a=T_A)
    for _ in range(20000):
        t.paso(1 / 60, fr5.q, True, ctx)
        if t.estado == "error" or t.ejec.i > i_b:
            break
        if t.afinando:
            time.sleep(0.002)
    assert t.estado != "error", t.texto
    assert t.afinado_fuente is None
    assert np.allclose(t.plan.pelota["c"], BOLA + da)


# ── la cámara A en el modelo de colisión ─────────────────────────────────────────────────────
def test_camara_a_es_obstaculo(cadena, tmp_path):
    from skill_shield.autonomo import calibracion as C
    from skill_shield.autonomo.mapa import modelo_colision
    ruta = str(tmp_path / "camara_a_robot.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(dict(T_mundo_cam=T_A.tolist(), T_brida_tablero=np.eye(4).tolist(),
                       intrinsecos=dict(K=np.eye(3).tolist())), f)
    assert np.allclose(C.cargar(ruta)["T_mundo_cam"], T_A)
    nombres = lambda m: [c.nombre for c in m.entorno.cajas]           # noqa: E731
    con = modelo_colision(cadena, dict(G, AUTONOMO_CAM_A=True, AUTONOMO_CAM_A_CALIB_ARCHIVO=ruta))
    sin = modelo_colision(cadena, dict(G, AUTONOMO_CAM_A=False, AUTONOMO_CAM_A_CALIB_ARCHIVO=ruta))
    assert "camara_a" in nombres(con) and "camara_a" not in nombres(sin)
