"""
(18-nov, tarde) La pelota NO pone cubos de colisión, ni en el mapa ni en el planificador, aunque su
centro estimado se desvíe a lo largo del rayo de la cámara tanto como el ruido del estéreo a esa
distancia (AUTONOMO_TAREA_RUIDO_PROF_K · d²); y lo que hay al lado de la pelota SIGUE contando.

Caso real (plan_fallido_20261118_122557): «bajar a la pelota: la mano toca un obstáculo del mapa · lo que
estorba (el mapa lo ve a unos 3 cm de la pelota, de unos 8 cm de alto)»: a 3 cm del centro en planta y
hasta 1 cm por encima de su parte de arriba, es decir, la PROPIA pelota (7 cm). Las exclusiones eran una
esfera/caja de r + 1 cm alrededor del centro estimado, más pequeña que donde caen de verdad sus cubos:
un cubo de 2 cm que toca la pelota tiene el centro hasta 1,7 cm fuera de ella, la TSDF engorda cada
superficie hacia la cámara y el centro se desvía a lo largo del rayo con el ruido de la profundidad.

Además, el mapa ya no RELLENA nada: lo ocupado es solo lo que mide la cámara (antes las mesas vistas como
plano se rellenaban en toda su envolvente y se borraba lo que hubiera en los ~6 cm de encima).

    python -m pytest tests/test_pelota_sin_cubos.py -q
"""
import math
import os
import sys
import time

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tests"))

import config                                                                 # noqa: E402
import test_mapa_tsdf as TM                                                   # noqa: E402
from skill_shield.autonomo.mapa import ConstructorMapa, MapaOcupacion, OCUPADO   # noqa: E402
from skill_shield.autonomo.pelotas import DetectorPelotas, _Pista             # noqa: E402

G = {k: getattr(config, k) for k in dir(config) if k.isupper()}
R = 0.035
RES = 0.02
Q_CUBO = 0.5 * math.sqrt(3) * RES                 # un cubo que toca la pelota: su centro, hasta aquí fuera
C_PELOTA = np.array([0.35, -0.95, TM.Z_TAPA + R])
# una caja al LADO de la pelota (en x, de lado para la cámara), a 3,5 cm de su superficie: tiene que seguir
CAJA_LADO = ((C_PELOTA[0] + R + 0.035, C_PELOTA[0] + R + 0.075), (C_PELOTA[1] - 0.02, C_PELOTA[1] + 0.02),
             (TM.Z_TAPA, TM.Z_TAPA + 0.06))


def _escena(T, con_caja=True):
    """Profundidad sin ruido (m) de la cajonera + la pelota (+ la caja de al lado)."""
    dw, dc = TM._rayos(T)
    o = T[:3, 3]
    oc = o - C_PELOTA
    b = dw @ oc
    a = np.sum(dw ** 2, -1)
    disc = b ** 2 - a * (oc @ oc - R * R)
    t_b = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
    t = np.minimum(TM._caja(o, dw, TM.CAJONERA), t_b)
    if con_caja:
        t = np.minimum(t, TM._caja(o, dw, CAJA_LADO))
    return np.where(np.isfinite(t), t * dc[:, 2], 0.0).reshape(TM.H, TM.W_)


def _ruido(z, rng):
    """Ruido k·z² y píxeles «voladores» en los bordes (mezcla de lo de delante y lo de detrás)."""
    zz = z + 0.4 * rng.normal(0, 1, z.shape) * 0.010 * z * z
    dz = np.abs(np.diff(zz, axis=1, prepend=zz[:, :1])) > 0.05
    borde = np.zeros_like(dz)
    for k in range(-2, 3):
        borde |= np.roll(dz, k, axis=1) | np.roll(dz, k, axis=0)
    mez = rng.random(zz.shape)
    zz = np.where(borde & (rng.random(zz.shape) < 0.6),
                  np.roll(zz, 3, axis=1) * mez + np.roll(zz, -3, axis=1) * (1 - mez), zz)
    return (np.clip(zz, 0, 10) * 1000).astype(np.uint16)


def _detector(T, desvio):
    """Detector con la pelota vista y su centro desviado 'desvio' m a lo largo del rayo de la cámara."""
    cc = (np.linalg.inv(T) @ np.r_[C_PELOTA, 1.0])[:3]
    u = cc[0] / cc[2] * TM.K[0, 0] + TM.K[0, 2]
    v = cc[1] / cc[2] * TM.K[1, 1] + TM.K[1, 2]
    rp = R * TM.K[0, 0] / cc[2]
    p = cc * (1.0 + desvio / np.linalg.norm(cc))
    det = DetectorPelotas(G, RAIZ)
    det.T_mundo_cam = T
    e = dict(nombre="Pelota roja", color_gemelo=(1, 0, 0))
    det.ultimas = [(e, p, 2 * R, (u, v, rp))]
    det.pistas[100] = _Pista(100, e, p, 2 * R, time.time())
    return det, (T @ np.r_[p, 1.0])[:3]


def _mapa_como_el_modo_autonomo(T, det, n=15, exclusiones=True):
    m = MapaOcupacion(TM.LIM, RES, z_suelo=-0.975, metodo="tsdf", motor="numpy",
                      dist_max=G["AUTONOMO_MAPA_DIST_MAX"], k_trunc=G["AUTONOMO_MAPA_TSDF_K"],
                      min_grupo=G["AUTONOMO_MAPA_MIN_GRUPO"], superficies=G["AUTONOMO_MAPA_MESAS"])
    if exclusiones:                                  # lo mismo que camara.py → ConstructorMapa.paso
        m.excluir = ConstructorMapa.zonas([det.caja_mundo(100)] + det.cajas_mundo())
    z = _escena(T)
    rng = np.random.default_rng(0)
    for i in range(n):
        prof = _ruido(z, rng)
        if exclusiones:
            prof = ConstructorMapa.quitar_pixeles(prof, det.pixeles_halo(), TM.K, T)
        m.actualizar(prof, TM.K, T, ahora=1000.0 + i)
    return m._centros[(m.estado(1000.0 + n) == OCUPADO).reshape(-1)]


def _de_la_pelota(O):
    """Cubos que se solapan con la pelota de verdad (por encima de su base: la mesa de debajo es mesa)."""
    O = np.asarray(O, float).reshape(-1, 3)
    return O[(np.linalg.norm(O - C_PELOTA, axis=1) < R + Q_CUBO) & (O[:, 2] > C_PELOTA[2] - R + 0.5 * RES)]


def _de_la_caja(O):
    (x0, x1), (y0, y1), (z0, z1) = CAJA_LADO
    O = np.asarray(O, float).reshape(-1, 3)
    return O[(O[:, 0] > x0 - 0.01) & (O[:, 0] < x1 + 0.01) & (O[:, 1] > y0 - 0.01) & (O[:, 1] < y1 + 0.01)
             & (O[:, 2] > z0 + 0.01) & (O[:, 2] < z1 + 0.01)]


def _ruido_a(T):
    return G["AUTONOMO_TAREA_RUIDO_PROF_K"] * float(np.sum((C_PELOTA - T[:3, 3]) ** 2))


@pytest.mark.parametrize("signo", [+1.0, -1.0])
def test_el_mapa_no_pone_cubos_de_la_pelota(signo):
    T = TM._T()
    desvio = signo * _ruido_a(T)                    # lo que dice el proyecto que se equivoca el estéreo ahí
    det, _ = _detector(T, desvio)
    O = _mapa_como_el_modo_autonomo(T, det)
    pel = _de_la_pelota(O)
    assert not len(pel), f"desvío {100 * desvio:+.1f} cm: {len(pel)} cubos de la pelota en el mapa: " \
        f"{np.round(pel - C_PELOTA, 3).tolist()}"
    assert len(_de_la_caja(O)) >= 3, "la caja de al lado de la pelota ha desaparecido del mapa"


@pytest.fixture(scope="module")
def pl():
    from skill_shield.autonomo import tarea as TA
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    return TA.Planificador(cargar_cadena(config.RUTA_XML, verbose=False), G, T_mundo_cam=TM._T())


@pytest.mark.parametrize("signo", [+1.0, -1.0])
def test_el_planificador_quita_toda_la_pelota_y_deja_lo_de_al_lado(pl, signo):
    """Aunque al mapa se le cuelen cubos de la pelota (aquí, el peor caso: el mapa sin ninguna exclusión),
    lo que Tarea manda quitar al planificador se lleva la pelota entera y deja la caja de al lado."""
    from skill_shield.autonomo.tarea import zona_excluida
    T = TM._T()
    det, c_est = _detector(T, signo * _ruido_a(T))
    O = _mapa_como_el_modo_autonomo(T, det, exclusiones=False)
    assert len(_de_la_pelota(O)) > 0                # sin exclusiones, la pelota está en el mapa
    h = dict(tipo="pelota", T_mundo=np.r_[np.c_[np.eye(3), c_est], [[0, 0, 0, 1]]].tolist(), diametro_m=2 * R)
    try:
        pl.poner_mapa(O, RES, zona_excluida(h, T, G))
        pel = _de_la_pelota(pl._C)
        assert not len(pel), f"quedan {len(pel)} cubos de la pelota: {np.round(pel - C_PELOTA, 3).tolist()}"
        assert len(_de_la_caja(pl._C)) >= 3, "la caja de al lado de la pelota ya no cuenta"
    finally:
        pl.poner_mapa(None, RES)


@pytest.mark.parametrize("signo", [+1.0, -1.0])
def test_con_sus_cubos_desviados_se_puede_coger_la_pelota(pl, signo):
    """El caso real, en el planificador: pelota sobre una mesa medida (apoyo_z, como la da pelotas.py), con sus
    cubos desplazados a lo largo del rayo lo que dice el ruido del estéreo. Con r + 1 cm salía «bajar a la pelota:
    la mano toca un obstáculo del mapa · lo que estorba (a unos 2 cm de la pelota, de unos 9 cm de alto)»."""
    from skill_shield.autonomo.tarea import zona_excluida
    T = TM._T()
    zt = -0.25
    c = np.array([0.35, -0.45, zt + R])
    u = (T[:3, 3] - c) / np.linalg.norm(T[:3, 3] - c)
    real = c + u * signo * G["AUTONOMO_TAREA_RUIDO_PROF_K"] * float(np.sum((T[:3, 3] - c) ** 2))
    g_ = np.arange(-0.3, 0.3001, RES)
    X, Y, Z = np.meshgrid(g_, g_, np.arange(-0.04, 0.12, RES))
    P = (np.floor((np.stack([X, Y, Z], -1).reshape(-1, 3) + [c[0], c[1], zt]) / RES) + 0.5) * RES
    mesa = P[np.abs(P[:, 2] - (zt - 0.5 * RES)) < 1e-6]
    bola = P[(np.linalg.norm(P - real, axis=1) <= R + 0.5 * RES) & (P[:, 2] > zt + 0.005)]
    dej = np.array([0.20, -0.62, zt + 0.002])
    h = dict(tipo="pelota", T_mundo=np.r_[np.c_[np.eye(3), c], [[0, 0, 0, 1]]], diametro_m=2 * R)
    Tp, Td = np.eye(4), np.eye(4)
    Tp[:3, 3], Td[:3, 3] = c, dej
    try:
        pl.poner_mapa(np.vstack([mesa, bola]), RES, zona_excluida(h, T, G) + [(dej, 0.09)])
        assert not (np.linalg.norm(pl._C - real, axis=1) <= R + 0.5 * RES).any()
        plan = pl.planificar_pelota(np.radians([93.0, -17.0, 53.0, 43.0, 105.0, 25.0]), Tp, 2 * R, Td, apoyo_z=zt,
                                    apoyo_poligono=[[c[0] - 0.3, c[1] - 0.3], [c[0] + 0.3, c[1] - 0.3],
                                                    [c[0] + 0.3, c[1] + 0.3], [c[0] - 0.3, c[1] + 0.3]])
        assert "Bajar a la pelota" in [f.nombre for f in plan.fases]
    finally:
        pl.poner_mapa(None, RES)


def test_el_mapa_no_rellena_las_mesas():
    """Un hueco en la tapa (lo que la cámara NO ve) no se rellena, y lo bajo que hay encima de la tapa
    (aquí, 4 cm) sigue en el mapa: antes la tapa vista como plano se rellenaba en toda su envolvente y se
    borraba lo que hubiera hasta ~6 cm por encima."""
    T = TM._T()
    dw, dc = TM._rayos(T)
    o = T[:3, 3]
    bajo = ((0.20, 0.26), (-1.05, -0.99), (TM.Z_TAPA, TM.Z_TAPA + 0.04))       # algo de 4 cm sobre la tapa
    t = np.minimum(TM._caja(o, dw, TM.CAJONERA), TM._caja(o, dw, bajo))
    z = np.where(np.isfinite(t), t * dc[:, 2], 0.0)
    P = o + dw * np.where(np.isfinite(t), t, 0.0)[:, None]
    hueco = (np.abs(P[:, 0] - 0.45) < 0.05) & (np.abs(P[:, 1] + 0.95) < 0.05) & (np.abs(P[:, 2] - TM.Z_TAPA) < 0.01)
    z = np.where(hueco, 0.0, z).reshape(TM.H, TM.W_)                          # la cámara no mide ahí
    m = MapaOcupacion(TM.LIM, RES, z_suelo=-0.975, metodo="tsdf", motor="numpy", dist_max=1.7, superficies=True)
    rng = np.random.default_rng(0)
    for i in range(10):
        m.actualizar((np.clip(z + rng.normal(0, 0.002, z.shape), 0, 10) * 1000).astype(np.uint16), TM.K, T,
                     ahora=1000.0 + i)
    assert m.superficies(), "la tapa se sigue detectando como plano (la usa el planificador)"
    O = m._centros[(m.estado(1010.0) == OCUPADO).reshape(-1)]
    en_hueco = (np.abs(O[:, 0] - 0.45) < 0.03) & (np.abs(O[:, 1] + 0.95) < 0.03) & (np.abs(O[:, 2] - TM.Z_TAPA) < 0.03)
    assert not en_hueco.any(), f"{int(en_hueco.sum())} cubos inventados en el hueco de la tapa"
    sobre = (O[:, 0] > 0.19) & (O[:, 0] < 0.27) & (O[:, 1] > -1.06) & (O[:, 1] < -0.98) & (O[:, 2] > TM.Z_TAPA + 0.02)
    assert sobre.sum() >= 2, "lo de 4 cm encima de la tapa ha desaparecido del mapa"
