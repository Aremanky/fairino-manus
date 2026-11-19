"""
(18-nov) Registro de cada intento del modo autónomo (diagnosticos/intentos.jsonl) y su resumen
(diagnosticos/resumen_intentos.py), con el FR5 y los planes falsos de test_tarea.py.

    python -m pytest tests/test_registro_intentos.py -q
"""
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tests"))
sys.path.insert(0, os.path.join(RAIZ, "diagnosticos"))

import resumen_intentos as RI                                                # noqa: E402
import test_tarea as TT                                                      # noqa: E402
from skill_shield.autonomo import tarea as TA                                # noqa: E402


@pytest.fixture
def registro(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_SHIELD_REGISTRO_INTENTOS", "1")
    return tmp_path / "intentos.jsonl"


def _con(t, ruta):
    t.cfg["AUTONOMO_TAREA_REGISTRO"] = str(ruta)
    return t


def test_un_agarre_bueno_queda_apuntado(registro):
    t, fr5, _ = TT._tarea_pelota(lambda: dict(estado="ok", dedos=[dict(fz=1.0)] * 5))
    TT._ejecutar_pelota(_con(t, registro), fr5)
    assert t.estado == "terminada"
    I = RI.leer(str(registro))
    assert len(I) == 1
    ev = list(I.values())[0]
    assert [e["evento"] for e in ev] == ["plan", "agarre", "fin"]
    fin = ev[-1]
    assert fin["ok"] and fin["objetivo"] == "Pelota roja" and np.allclose(fin["centro_mundo"], (0.3, -0.5, 0.0))
    assert RI.clasificar(ev) == "éxito"
    assert "ÉXITO: 1/1" in "\n".join(RI.resumir(I))


def test_si_los_tactiles_no_la_notan_queda_como_no_la_coge(registro):
    t, fr5, _ = TT._tarea_pelota(lambda: dict(estado="ok", dedos=[dict(fz=0.0)] * 5))
    TT._ejecutar_pelota(_con(t, registro), fr5)
    assert t.estado == "error"
    ev = list(RI.leer(str(registro)).values())[0]
    assert any(e["evento"] == "agarre" and e["ok"] is False for e in ev)
    assert RI.clasificar(ev) == "no la coge"


def test_un_plan_fallido_queda_con_su_motivo(registro):
    class SinPlan(TT.PlanFalso):
        def planificar_pelota(self, *a, **k):
            raise TA.ErrorPlan("no encuentro cómo llegar a la pelota · lo que estorba (a unos 3 cm de la pelota)")
    t = TA.Tarea(dict(TT.G, AUTONOMO_TAREA_MOVER_MANO=False), ".", TT.FR5Falso(), simulacro=False,
                 fabrica_planificador=SinPlan)
    fr5 = t.fr5
    t.orden({"tarea": dict(accion="planificar", id="p1")},
            lambda: dict(q_real=fr5.q, herramientas=TT.HERR_PELOTA, objetivo_id=100, mapa=None))
    TT._esperar_plan(_con(t, registro), fr5)
    assert t.estado == "error"
    I = RI.leer(str(registro))
    ev = list(I.values())[0]
    assert ev[0]["evento"] == "plan" and not ev[0]["ok"] and "estorba" in ev[0]["motivo"]
    assert RI.clasificar(ev) == "no planifica"
    txt = "\n".join(RI.resumir(I))
    assert "ÉXITO: 0/1" in txt and "1× no encuentro cómo llegar" in txt


def test_sin_permiso_no_escribe(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_SHIELD_REGISTRO_INTENTOS", "0")
    ruta = tmp_path / "intentos.jsonl"
    t, fr5, _ = TT._tarea_pelota(lambda: dict(estado="ok", dedos=[dict(fz=1.0)] * 5))
    TT._ejecutar_pelota(_con(t, ruta), fr5)
    assert t.estado == "terminada" and not ruta.exists()


def test_intervalo_de_wilson():
    lo, hi = RI.wilson(3, 4)
    assert 0.25 < lo < 0.35 and 0.9 < hi < 0.97                 # 3 de 4: puede ser casi cualquier cosa
    lo, hi = RI.wilson(18, 20)
    assert lo > 0.65
