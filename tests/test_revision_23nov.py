"""
(23-nov) Revisión tras la prueba de la pelota en la caja:

  1. Tiempo de planificar: se apunta en qué se va (llegadas, caminos, rodeos, cuRobo) y cuRobo que no SIRVE (no falla,
     pero ningún tramo suyo vale) se aparta del plan tras unos intentos y se apaga tras varios planes así.
  2. Reintentos: si la pelota se ha movido desde el plan (al pulsar «Ejecutar») o se cae al llevarla (táctiles), se
     vuelve a planificar sola.
  3. Escena fija automática: si el plan no sale sin escena fija, la tarea pide escanearla y vuelve a planificar.

    python -m pytest tests/test_revision_23nov.py -q
"""
import os
import sys
import time
import types

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tests"))

import config                                                        # noqa: E402
from skill_shield.autonomo import tarea as TA                        # noqa: E402
from test_tarea import FR5Falso, PlanFalso, _plano, _plano_pelota_falso   # noqa: E402

G = {k: getattr(config, k) for k in dir(config) if k.isupper()}
G["AUTONOMO_CAM_A_OBLIGATORIA"] = False
G["AUTONOMO_TAREA_MOVER_MANO"] = False
G["AUTONOMO_TAREA_PLANIFICADOR"] = "propio"
G["AUTONOMO_TAREA_GUARDAR_PLANES"] = G["AUTONOMO_TAREA_GUARDAR_FALLOS"] = False
Q_CAJA = np.radians([149.5, -51.5, 68.0, -2.5, 96.5, 76.0])       # la postura de la prueba del 23-nov


def _herr(c=(0.3, -0.5, 0.0)):
    return [dict(id=100, nombre="Pelota roja", tipo="pelota", estable=True, diametro_m=0.065, T_mundo=_plano(c).tolist()),
            dict(id=22, nombre="Punto de dejada", tipo="punto_dejada", estable=True, T_mundo=np.eye(4).tolist())]


class EscenaFalsa:
    def __init__(self):
        self.cajas, self.voxeles, self.version = [], [], 0

    def actualizar(self):
        pass


def _tarea(tactil=None, falla_sin_escena=False, falla_siempre=False):
    fr5 = FR5Falso()
    llamadas = []
    escena = EscenaFalsa()

    class PlanPelota(PlanFalso):
        def __init__(self):
            self.escena, self.avisos = escena, []

        def poner_mapa(self, centros, res, excluir=()):
            return 0

        def planificar_pelota(self, q0, T, d, Td, **kw):
            llamadas.append(np.asarray(T, float)[:3, 3].copy())
            if falla_siempre or (falla_sin_escena and not escena.cajas):
                raise TA.ErrorPlan("bajar a la pelota: la mano tocaría la mesa (holgura 2 cm)")
            p = _plano_pelota_falso()
            p.fases[-1].vigilar_carga = True             # «Subir»: llevándola
            return TA.Plan(q0, p.fases, nombre="coger, apretar y dejar la pelota")
    t = TA.Tarea(dict(G), ".", fr5, simulacro=False, fabrica_planificador=PlanPelota, tactil=tactil)
    return t, fr5, llamadas, escena


def _ctx(fr5, c=(0.3, -0.5, 0.0)):
    return lambda: dict(q_real=fr5.q, herramientas=_herr(c), objetivo_id=100, mapa=None)


def _esperar(t, fr5, ctx, estados=("planificando",), pedal=False, n=500):
    for _ in range(n):
        if t.estado not in estados:
            return
        time.sleep(0.005)
        t.paso(1 / 60, fr5.q, pedal, ctx)


# ══════════════════════ 2. reintentos ══════════════════════
def test_si_la_pelota_se_ha_movido_al_ejecutar_vuelve_a_planificar_sola():
    t, fr5, llamadas, _ = _tarea()
    t.orden({"tarea": dict(accion="planificar", id="a1")}, _ctx(fr5))
    _esperar(t, fr5, _ctx(fr5))
    assert t.estado == "lista" and len(llamadas) == 1
    movida = _ctx(fr5, (0.34, -0.5, 0.0))                        # 4 cm
    t.orden({"tarea": dict(accion="ejecutar", id="a2")}, movida)
    assert t.estado == "planificando" and "se ha movido" in t.texto
    _esperar(t, fr5, movida)
    assert len(llamadas) == 2 and np.allclose(llamadas[-1], (0.34, -0.5, 0.0))
    assert t.estado == "pausada"                                 # sigue sola: solo falta el pedal


def test_con_el_ruido_de_la_camara_no_replanifica():
    t, fr5, llamadas, _ = _tarea()
    t.orden({"tarea": dict(accion="planificar", id="b1")}, _ctx(fr5))
    _esperar(t, fr5, _ctx(fr5))
    t.orden({"tarea": dict(accion="ejecutar", id="b2")}, _ctx(fr5, (0.305, -0.5, 0.0)))   # 5 mm
    assert t.estado == "pausada" and len(llamadas) == 1


def test_si_se_cae_al_llevarla_vuelve_a_planificar():
    estado = dict(fz=1.2)
    t, fr5, llamadas, _ = _tarea(tactil=lambda: dict(estado="ok", dedos=[dict(fz=estado["fz"])] * 5))
    ctx = _ctx(fr5)
    t.orden({"tarea": dict(accion="planificar", id="c1")}, ctx)
    _esperar(t, fr5, ctx)
    t.orden({"tarea": dict(accion="ejecutar", id="c2")}, ctx)
    replan = False
    for _ in range(3000):
        if t.ejec is not None and t.ejec.fase is not None and t.ejec.fase.nombre == "Subir":
            estado["fz"] = 0.0                                   # se le escurre
        t.paso(1 / 60, fr5.q, True, ctx)
        if t.estado == "planificando" or len(llamadas) > 1:
            replan = True
            break
        if t.estado in ("terminada", "error"):
            break
    assert replan, (t.estado, t.texto)
    assert t._reintentos == 1


def test_afinado_con_la_pelota_muy_movida_replanifica_en_vez_de_parar():
    t, fr5, llamadas, _ = _tarea()
    ctx = _ctx(fr5)
    t.orden({"tarea": dict(accion="planificar", id="d1")}, ctx)
    _esperar(t, fr5, ctx)
    t.orden({"tarea": dict(accion="ejecutar", id="d2")}, ctx)
    t._ctx_paso = ctx
    assert t._aplicar_afinado(("replanificar", "La pelota está a 15 cm de donde estaba al planificar")) is None
    assert t.estado == "planificando" and t._reintentos == 1


# ══════════════════════ 3. escena fija automática ══════════════════════
def test_sin_escena_fija_no_sale_la_escanea_y_vuelve_a_planificar():
    t, fr5, llamadas, escena = _tarea(falla_sin_escena=True)
    ctx = _ctx(fr5)
    t.orden({"tarea": dict(accion="planificar", id="e1")}, ctx)
    _esperar(t, fr5, ctx)
    assert t.estado == "escaneando", t.texto
    pe = t.peticion_escaneo
    assert pe and pe["id"] and any(np.linalg.norm(np.array(e[:3]) - (0.3, -0.5, 0.0)) < 1e-6 for e in pe["excluir"])
    for _ in range(5):                                           # aún escaneando
        t.paso(1 / 60, fr5.q, False, ctx)
    assert t.estado == "escaneando"
    escena.cajas, escena.version = [dict(nombre="caja")], 1      # la cámara ha guardado la escena
    t.paso(1 / 60, fr5.q, False, ctx)
    _esperar(t, fr5, ctx)
    assert t.estado == "lista" and len(llamadas) == 2


def test_si_ya_hay_escena_fija_y_no_sale_no_escanea_otra_vez():
    t, fr5, llamadas, escena = _tarea(falla_siempre=True)
    escena.cajas = [dict(nombre="caja")]
    ctx = _ctx(fr5)
    t.orden({"tarea": dict(accion="planificar", id="f1")}, ctx)
    _esperar(t, fr5, ctx)
    assert t.estado == "error" and t.peticion_escaneo is None and len(llamadas) == 1


def test_un_solo_escaneo_por_intento():
    t, fr5, llamadas, escena = _tarea(falla_siempre=True)
    ctx = _ctx(fr5)
    t.orden({"tarea": dict(accion="planificar", id="g1")}, ctx)
    _esperar(t, fr5, ctx)
    assert t.estado == "escaneando"
    escena.cajas, escena.version = [dict(nombre="caja")], 1
    t.paso(1 / 60, fr5.q, False, ctx)
    _esperar(t, fr5, ctx)
    assert t.estado == "error" and len(llamadas) == 2            # con escena tampoco: se dice el motivo y para


def test_la_camara_atiende_la_peticion_de_escaneo_una_vez():
    import json
    from skill_shield.autonomo.mapa import ConstructorMapa
    msg = json.dumps(dict(congelar=False, fijo_escanear_id="tarea-1-1", fijo_t=time.time(),
                          fijo_excluir=[[0.3, -0.5, 0.2, 0.08]])).encode()
    hechos = []
    falso = types.SimpleNamespace(canal_congelar=types.SimpleNamespace(leer=lambda: (msg, time.time())),
                                  _escaneo_hasta=None,
                                  escanear_fijo=lambda excluir=None: hechos.append(excluir))
    ConstructorMapa._peticion_tarea(falso)
    ConstructorMapa._peticion_tarea(falso)                       # el mismo id: no se repite
    assert hechos == [[[0.3, -0.5, 0.2, 0.08]]]
    viejo = json.dumps(dict(fijo_escanear_id="tarea-1-2", fijo_t=time.time() - 120)).encode()
    falso.canal_congelar = types.SimpleNamespace(leer=lambda: (viejo, time.time()))
    ConstructorMapa._peticion_tarea(falso)                       # una petición vieja (cámara reiniciada): no
    assert len(hechos) == 1


# ══════════════════════ 1. cuRobo que no sirve ══════════════════════
class CuroboInutil:
    def __init__(self):
        self.llamadas = 0

    def actualizar_mundo(self, *a, **k):
        pass

    def plan(self, q0, q1):
        self.llamadas += 1
        time.sleep(0.01)
        return None


@pytest.fixture(scope="module")
def pl():
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    p = TA.Planificador(cargar_cadena(G["RUTA_XML"], verbose=False), G)
    p.poner_mapa(None, 0.02)
    return p


def test_curobo_que_no_resuelve_nada_se_aparta_y_se_apaga(pl):
    cu = CuroboInutil()
    pl.curobo = cu
    q1 = Q_CAJA + np.radians([10, 0, 0, 0, 0, 0])
    for plan in range(3):
        pl._empezar(Q_CAJA)
        for _ in range(6):
            pl.conectar(Q_CAJA, q1, nombre="ir")             # el planificador propio lo resuelve igual
        r = pl.resumen_tiempos()
        if plan < 2:
            assert r["curobo_apartado"] and r["curobo_intentos"] == pl.curobo_fallos_max_plan
            assert r["tiempos"].get("curobo", 0) > 0
    assert pl.curobo is None and cu.llamadas == 2 * pl.curobo_fallos_max_plan     # al 3.º plan, ni se intenta
    assert any("cuRobo apagado" in a for a in pl.avisos)
    pl._curobo_pausado, pl.planes_curobo_inutiles, pl._plan_empezado = None, 0, False
