"""Pruebas del agarre asistido del modo GRABACION (16-nov), sin robot ni cámaras:

    python -m pytest tests/test_agarre_asistido.py -q

  · Planificador.planificar_previa (cadena y mallas reales): lleva la mano encima de la pelota,
    con la bajada comprobada, y se para ahí.
  · TeleopRelativa: embrague, escala de desplazamiento y giro, giro bloqueado, velocidad máxima,
    saltos de la IK y frenado contra obstáculos.
  · AgarreAsistido: plan → llevar (solo con pedal) → afinar → listo → pisar = grabar → soltar =
    guardar → relativo sin grabar; cancelar por desvío o si el robot se movió al planificar.
  · El objetivo elegido ([O]) y el fijado para la grabación.
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
from skill_shield.ia.agarre_asistido import AgarreAsistido, TeleopRelativa, escalar_giro   # noqa: E402
from skill_shield.ia.fuentes import FuentesGrabacion                 # noqa: E402
from skill_shield.teleop_fr5.cinematica import (OFFSET_MUNECA_ORCA, OFFSET_PALMA_ORCA, cargar_cadena,  # noqa: E402
                                                rot_eje)

G = {k: getattr(config, k) for k in dir(config) if k.isupper()}
CADENA = cargar_cadena(config.RUTA_XML, verbose=False)
Q0 = np.radians([93.0, -17.0, 53.0, 43.0, 105.0, 25.0])          # la de tests/test_tarea.py
BOLA = (0.35, -0.45, -0.15)


def _muneca(q):
    T = CADENA.fk_marcos(np.asarray(q, float))[6]
    return T[:3, :3] @ OFFSET_MUNECA_ORCA + T[:3, 3], T[:3, :3]


def _humano(p, R=np.eye(3)):
    return dict(p_muneca=np.asarray(p, float), R_mano=np.asarray(R, float))


# ══════════════════════ planificador: solo hasta la postura de agarre ══════════════════════
def test_planificar_previa_pelota_se_para_encima():
    pl = TA.Planificador(CADENA, G)
    T = np.eye(4)
    T[:3, 3] = BOLA
    plan = pl.planificar_previa(Q0, dict(tipo="pelota", T_mundo=T, diametro_m=0.067, nombre="Pelota"))
    assert [f.nombre for f in plan.fases] == ["Abrir la mano", "Ir encima de la pelota"]
    q_fin = plan.fases[-1].qs[-1]
    assert np.allclose(plan.q_final, q_fin)
    # la bajada comprobada empieza donde acaba el plan y deja la mano en la pelota
    assert np.allclose(plan.pelota["bajada"][0], q_fin, atol=1e-9)
    Tm = CADENA.fk_marcos(q_fin)[6]
    palma = Tm[:3, :3] @ plan.pelota["punto"] + Tm[:3, 3]
    assert 0.03 < np.linalg.norm(palma - np.array(BOLA)) < 0.35          # encima, sin tocarla
    for f in plan.fases:
        if f.qs is not None:
            assert all(pl.valida(f, q) is None for q in f.qs[::4])


# ══════════════════════ modo relativo (embrague) ══════════════════════
def test_escalar_giro():
    R = rot_eje([0.3, -0.5, 0.8], 0.9)
    Rm = escalar_giro(R, 0.5)
    assert np.allclose(Rm @ Rm, R, atol=1e-9)
    assert np.allclose(escalar_giro(np.eye(3), 0.5), np.eye(3))
    assert np.allclose(escalar_giro(R, 1.0), R, atol=1e-9)


def test_relativo_copia_desplazamientos_escalados():
    rel = TeleopRelativa(CADENA, escala_pos=0.5, escala_giro=0.5, v_max=100.0, salto_max=1.0)
    assert rel.paso(_humano([0, 0, 0]))[0] is None                       # sin embrague, nada
    p_r0, R_r0 = _muneca(Q0)
    h0 = np.array([0.4, 0.1, 1.2])                                         # tu mano: donde sea
    rel.enganchar(_humano(h0), Q0)
    q, info = rel.paso(_humano(h0 + [0.04, -0.02, 0.06]), dt=0.03)
    assert info["estado"] == "ok"
    p_r, R_r = _muneca(q)
    assert np.allclose(p_r - p_r0, [0.02, -0.01, 0.03], atol=1e-6)       # la mitad
    assert np.allclose(R_r, R_r0, atol=1e-6)
    # giro de tu mano 0,4 rad → el robot gira 0,2 rad (en el mundo), con la muñeca en su sitio
    Rh = rot_eje([0, 0, 1], 0.4)
    q, _ = rel.paso(_humano(h0, Rh), dt=0.03)
    p_r, R_r = _muneca(q)
    assert np.allclose(R_r, rot_eje([0, 0, 1], 0.2) @ R_r0, atol=1e-6) and np.allclose(p_r, p_r0, atol=1e-6)
    # giro bloqueado: solo cuenta la traslación
    q, _ = rel.paso(_humano(h0 + [0.02, 0, 0], Rh), dt=0.03, bloquear_giro=True)
    p_r, R_r = _muneca(q)
    assert np.allclose(R_r, R_r0, atol=1e-6) and np.allclose(p_r - p_r0, [0.01, 0, 0], atol=1e-6)
    # sin medida de tu mano: se queda donde está
    q2, info = rel.paso(None)
    assert np.allclose(q2, q) and info["estado"] == "sin_medida"
    rel.soltar()
    assert not rel.enganchado


def test_relativo_velocidad_salto_y_alcance():
    rel = TeleopRelativa(CADENA, escala_pos=1.0, v_max=0.5, salto_max=0.3)
    h0 = np.zeros(3)
    rel.enganchar(_humano(h0), Q0)
    q, _ = rel.paso(_humano(h0 + [0.15, 0, 0]), dt=0.02)
    assert np.max(np.abs(q - Q0)) <= 0.5 * 0.02 + 1e-9                    # velocidad limitada
    q, info = rel.paso(_humano(h0 + [5.0, 0, 0]), dt=0.02)                # fuera de alcance
    assert info["estado"] == "fuera_de_alcance"
    rel.enganchar(_humano(h0), Q0)
    q, info = rel.paso(_humano(h0, rot_eje([1, 0, 0], 2.5)), dt=0.02)     # giro enorme: la IK salta
    assert info["estado"] in ("salto", "fuera_de_alcance") and np.allclose(q, Q0)


class _ColFalsa:
    """Choca si la muñeca baja de z_min."""
    def __init__(self, z_min):
        self.z_min = z_min

    def colisiones(self, q):
        return [("mano", "mapa", 0.0)] if _muneca(q)[0][2] < self.z_min else []

    def libre(self, q):
        return not self.colisiones(q)


def test_relativo_frena_contra_obstaculos():
    rel = TeleopRelativa(CADENA, escala_pos=1.0, v_max=100.0, salto_max=1.0)
    p0, _ = _muneca(Q0)
    col = _ColFalsa(p0[2] - 0.03)
    rel.enganchar(_humano(np.zeros(3)), Q0)
    q, info = rel.paso(_humano([0, 0, -0.08]), dt=0.03, col=col)
    assert info["estado"] == "bloqueado" and "mapa" in info["motivo"]
    z = _muneca(q)[0][2]
    assert col.libre(q) and z < p0[2] - 0.02                              # lo más cerca posible


# ══════════════════════ la secuencia completa ══════════════════════
class _PlanificadorFalso:
    def __init__(self):
        self.mapas = []

    def poner_mapa(self, centros, res, excluir=()):
        self.mapas.append((centros, res, list(excluir)))

    def planificar_previa(self, q0, det):
        q0 = np.asarray(q0, float)
        qs = q0 + np.radians([10, -5, 5, 0, 0, 0]) * np.linspace(0, 1, 12)[:, None]
        return TA.Plan(q0, [TA.Fase("Abrir la mano", mano="pelota_abierta"), TA.Fase("Ir encima", qs)],
                       nombre="llevar la mano a la pelota (falso)")


def _agarre(**cfg):
    pl = _PlanificadorFalso()
    g = dict(G, GRABACION_AFINAR_S=0.0, GRABACION_V_MAX=50.0, AUTONOMO_TAREA_VEL_DEG_S=60.0,
             AUTONOMO_TAREA_ACEL_DEG_S2=400.0, **cfg)
    return AgarreAsistido(g, RAIZ, CADENA, fabrica_planificador=lambda: pl), pl


def _det():
    T = np.eye(4)
    T[:3, 3] = BOLA
    return dict(id=3, nombre="Pelota de tenis", tipo="pelota", T_mundo=T, diametro_m=0.067)


def _esperar_plan(ag):
    ag._hilo.join(5.0)
    return ag.paso(0.03)


def test_secuencia_llevar_listo_grabar_guardar():
    ag, pl = _agarre()
    assert not ag.pedir_plan(Q0, None) and "objeto" in ag.aviso and ag.estado == "libre"
    assert ag.pedir_plan(Q0, _det(), mapa=(0.02, np.zeros((5, 3))))
    assert ag.controla_brazo and ag.estado == "planificando"
    q = _esperar_plan(ag)
    assert ag.estado == "llevando" and np.allclose(q, Q0)
    assert pl.mapas[-1][1] == 0.02 and np.allclose(pl.mapas[-1][2][0][0], BOLA)   # la pelota no es obstáculo
    # sin pedal no avanza (y el FR5 no se arma)
    for _ in range(10):
        q = ag.paso(0.05, q_real=Q0)
    assert np.allclose(q, Q0) and not ag.quiere_mover
    assert ag.pedal(True) is None and ag.quiere_mover
    for _ in range(200):
        q = ag.paso(0.05, q_real=q)
        if ag.estado != "llevando":
            break
    q_fin = Q0 + np.radians([10, -5, 5, 0, 0, 0])
    assert np.allclose(q, q_fin)
    ag.paso(0.05)
    assert ag.estado == "listo" and ag.necesita_humano
    assert ag.pedal(False) is None                                         # soltar en «listo»: nada
    # pisar en LISTO = empieza a grabarse; tu mano manda en relativo
    assert ag.pedal(True, q_robot=q_fin, obj_humano=None) is None and ag.estado == "listo"   # sin tu mano no
    h0 = np.array([0.3, 0.2, 1.1])
    assert ag.pedal(True, q_robot=q_fin, obj_humano=_humano(h0)) == "grabar"
    assert ag.estado == "agarrando" and ag.quiere_mover
    q = ag.paso(0.03, obj_humano=_humano(h0 + [0, 0, -0.04]))
    assert np.allclose(_muneca(q)[0] - _muneca(q_fin)[0], [0, 0, -0.02], atol=1e-4)
    # soltar = se para y se guarda
    assert ag.pedal(False) == "parar_grabacion" and ag.estado == "relativo" and ag.n_grabados == 1
    assert not ag.quiere_mover and np.allclose(ag.paso(0.03, obj_humano=_humano(h0)), q)
    # volver a pisar: relativo SIN grabar
    assert ag.pedal(True, q_robot=q, obj_humano=_humano(h0)) is None and ag.quiere_mover
    assert ag.pedal(False) is None and ag.n_grabados == 1
    assert "RELATIVO" in ag.texto() and ag.texto().isascii()
    ag.cancelar()
    assert not ag.controla_brazo and ag.paso(0.03) is None


def test_se_cancela_si_el_robot_no_sigue_o_se_movio():
    ag, _ = _agarre()
    ag.pedir_plan(Q0, _det())
    q = _esperar_plan(ag)
    ag.pedal(True)
    q = ag.paso(0.05, q_real=Q0)
    ag._t_desvio = time.monotonic() - 1.0
    assert ag.paso(0.05, q_real=Q0 + math.radians(20)) is None
    assert ag.estado == "libre" and "desv" in ag.aviso
    # el robot se movió mientras se planificaba → no vale el plan
    ag.pedir_plan(Q0, _det())
    ag._hilo.join(5.0)
    assert ag.paso(0.03, q_real=Q0 + math.radians(10)) is None
    assert ag.estado == "libre" and "movido" in ag.aviso
    # [L] durante el plan = cancelar: un resultado que llegue tarde ya no vale
    ag.pedir_plan(Q0, _det())
    ag.cancelar("cancelado")
    ag._hilo.join(5.0)
    assert ag.paso(0.03) is None and ag.estado == "libre"


def test_error_del_planificador_se_ve():
    class _Malo(_PlanificadorFalso):
        def planificar_previa(self, q0, det):
            raise TA.ErrorPlan("no llego a la pelota")
    ag = AgarreAsistido(dict(G), RAIZ, CADENA, fabrica_planificador=_Malo)
    ag.pedir_plan(Q0, _det())
    ag._hilo.join(5.0)
    assert ag.paso(0.03) is None and ag.estado == "libre" and "no llego a la pelota" in ag.aviso


# ══════════════════════ objetivo elegido y fijado ══════════════════════
class _CabezaFalsa:
    def __init__(self, lista):
        self.lista = lista

    def objetos(self):
        return self.lista

    def estado(self):
        return "ok"

    def publicar_pose(self, q, real, caps=None):
        pass

    def poner_objetivo(self, id_):
        pass

    def mapa(self):
        return None


def _lista(palma):
    T1, T2, T3 = np.eye(4), np.eye(4), np.eye(4)
    T1[:3, 3], T2[:3, 3], T3[:3, 3] = palma + [0, 0, -0.1], palma + [0.3, 0, -0.1], palma + [0.1, 0, -0.1]
    return [dict(id=3, nombre="Pelota cerca", tipo="pelota", T_mundo=T1.tolist(), diametro_m=0.067),
            dict(id=20, nombre="Destornillador", tipo="herramienta", T_mundo=T2.tolist()),
            dict(id=40, nombre="Punto de uso", tipo="punto_uso", T_mundo=T3.tolist())]


def test_objetivo_elegido_con_O_y_fijado_al_grabar():
    from skill_shield.ia.cabeza import MundoGrabacion
    palma = CADENA.fk(Q0)["palma"]
    lista = _lista(palma)
    fuentes = FuentesGrabacion([f"j{i}" for i in range(6)], [f"m{i}" for i in range(17)], cadena=CADENA,
                               objetos=lambda: lista, radio_objeto=0.5)
    mundo = MundoGrabacion(_CabezaFalsa(lista), fuentes)
    mundo.paso(None, Q0)
    assert mundo.objetivo["id"] == 3                                      # el más cercano
    assert mundo.siguiente_objetivo() == "Pelota cerca"
    assert mundo.siguiente_objetivo() == "Destornillador"                 # el punto de uso no se coge
    mundo.paso(None, Q0)
    assert mundo.objetivo["id"] == 20 and "elegido" in mundo.texto()
    assert mundo.objetivo_para_plan()["id"] == 20
    assert mundo.siguiente_objetivo() is None                             # vuelta a automático
    mundo.paso(None, Q0)
    assert mundo.objetivo["id"] == 3
    # el objetivo del agarre se fija ANTES de empezar a grabar: no la 1ª detección cercana
    fuentes.fijar_objetivo(lista[1])
    fuentes.iniciar_episodio()
    assert fuentes.objetivo_clave() == "herramienta:20"
    fila = fuentes.muestra(Q0)
    assert fila[fuentes.columnas().index("obj:id")] == 20.0
    fuentes.fin_episodio()
    fuentes.fijar_objetivo(None)
    fuentes.iniciar_episodio()
    fuentes.muestra(Q0)
    assert fuentes.objetivo_clave() == "pelota:3"
    fuentes.fin_episodio()
