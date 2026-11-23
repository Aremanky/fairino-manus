"""(23-nov) EstimadorCarga: con pares sintéticos del propio modelo debe sacar la masa de la brida."""
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from skill_shield.hardware.carga_fr5 import EstimadorCarga  # noqa: E402

XML = os.path.join(RAIZ, "fairino_description", "mjcf", "fairino_mjcf.xml")
Q = np.radians([149.5, -51.5, 68.0, -2.5, 96.5, 76.0])


def _par(est, q, m, c_brida=(0.0, 0.03, 0.13), signo=-1.0, con_brazo=True):
    A, k = est.sistema(q)
    c = est.cadena.fk_marcos(q)[5][:3, :3] @ np.asarray(c_brida)
    return signo * (A @ np.r_[m, m * c] + (k if con_brazo else 0.0))


def test_masa_en_posturas_al_azar():
    est = EstimadorCarga(XML)
    rng = np.random.default_rng(0)
    lim = est.cadena.limites
    for _ in range(50):
        q = rng.uniform(lim[:, 0] * 0.8, lim[:, 1] * 0.8)
        for m in (1.3, 2.0):
            x, _ = est.resolver(q, _par(est, q, m), -1.0)
            assert abs(x[0] - m) < 0.08


def test_absoluto_signo_frenos_y_movimiento():
    est = EstimadorCarga(XML)
    par = _par(est, Q, 1.8, signo=+1.0)                 # signo al revés del habitual: se detecta solo
    for i in range(30):
        r = est.actualizar(Q, par, ahora=i * 0.1, habilitado=True)
    assert r["estable"] and est.modo == "absoluto" and est.signo == 1.0 and abs(r["kg"] - 1.8) < 0.08
    r = est.actualizar(Q, par, ahora=3.1, habilitado=False)   # frenos: se queda el último valor
    assert r["frenado"] and abs(r["kg"] - 1.8) < 0.08
    r = est.actualizar(Q + 0.05, par, ahora=3.2)        # ~29 °/s
    assert r["moviendo"] and r["kg"] is not None
    assert est.actualizar(None, None, ahora=3.3)["texto"] == "sin lectura del FR5"


def test_neto_controladora_resta_gravedad():
    """Par ≈ 0 con 1,3 kg puestos en la controladora; luego coge 0,5 kg más."""
    est = EstimadorCarga(XML)
    rng = np.random.default_rng(1)
    for i in range(20):
        r = est.actualizar(Q, rng.normal(0, 0.05, 6), ahora=i * 0.1, habilitado=True, carga_ctrl=1.3)
    assert est.modo == "neto" and abs(r["kg"] - 1.3) < 0.1
    extra = _par(est, Q, 0.5, con_brazo=False)          # el par neto que hace lo cogido
    for i in range(20, 60):
        r = est.actualizar(Q, extra, ahora=i * 0.1, habilitado=True, carga_ctrl=1.3)
    assert abs(r["kg"] - 1.8) < 0.1


def test_par_cero_sin_saber_si_esta_habilitado():
    est = EstimadorCarga(XML)
    for i in range(20):
        r = est.actualizar(Q, np.zeros(6), ahora=i * 0.1, habilitado=None, carga_ctrl=1.3)
    assert r["kg"] is None and "deshabilitado" in r["texto"]
