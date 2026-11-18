"""
Pruebas de la detección de pelotas (11-nov), sin cámara: imágenes y profundidad sintéticas.

    python -m pytest tests/test_pelotas.py -q
"""
import os
import sys

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del repo (tests/..)
sys.path.insert(0, RAIZ)

import config                                                          # noqa: E402
from skill_shield.autonomo.pelotas import (DetectorPelotas, candidatas, catalogo,   # noqa: E402
                                           rango_sugerido)

K = np.array([[470.0, 0, 320], [0, 470.0, 240], [0, 0, 1]])
HSV = {"Pelota amarilla": (24, 200, 200), "Pelota de tenis": (33, 160, 200), "Pelota verde": (48, 150, 180),
       "Pelota roja": (178, 190, 160)}


def _escena(pelotas, fondo_z=2.0):
    """pelotas: [(nombre, centro 3D en la cámara, diámetro)] → (imagen BGR, profundidad mm)."""
    img = np.full((480, 640, 3), 120, np.uint8)
    prof = np.full((480, 640), fondo_z * 1000, np.float64)
    vv, uu = np.mgrid[0:480, 0:640]
    ray = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    for nombre, c, d in sorted(pelotas, key=lambda p: -p[1][2]):
        c, r = np.asarray(c, float), d / 2
        # intersección rayo-esfera: t² |ray|² − 2 t ray·c + |c|² − r² = 0
        a = np.sum(ray * ray, -1)
        b = ray @ c
        disc = b * b - a * (c @ c - r * r)
        dentro = disc > 0
        t = (b - np.sqrt(np.where(dentro, disc, 0))) / a
        prof[dentro] = np.minimum(prof[dentro], t[dentro] * 1000)
        h, s, v = HSV[nombre]
        bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
        img[dentro] = bgr
    return img, prof.astype(np.uint16)


def test_encuentra_cada_pelota_en_su_sitio():
    reales = [("Pelota amarilla", (0.15, 0.10, 0.90), 0.09), ("Pelota de tenis", (-0.05, 0.12, 0.85), 0.067),
              ("Pelota verde", (0.02, 0.05, 1.00), 0.045), ("Pelota roja", (-0.20, 0.08, 0.95), 0.08)]
    img, prof = _escena(reales)
    hallado = {e["nombre"]: (c, d) for e, c, d, _ in candidatas(img, prof, K, catalogo(config))}
    assert set(hallado) == {n for n, _, _ in reales}
    for n, c, d in reales:
        assert np.linalg.norm(hallado[n][0] - np.array(c)) < 0.015, n          # centro a menos de 1,5 cm
        assert abs(hallado[n][1] - d) < 0.25 * d, n


def test_mismo_color_pero_otro_tamano_no_es_pelota_de_tenis():
    img, prof = _escena([("Pelota de tenis", (0.0, 0.0, 0.9), 0.20)])      # «tenis» de 20 cm: no
    assert not [1 for e, *_ in candidatas(img, prof, K, catalogo(config)) if e["nombre"] == "Pelota de tenis"]


def test_sin_profundidad_no_inventa():
    img, _ = _escena([("Pelota roja", (0.0, 0.0, 0.9), 0.08)])
    assert candidatas(img, np.zeros((480, 640), np.uint16), K, catalogo(config)) == []


def test_seguimiento_estable_y_en_el_mundo():
    det = DetectorPelotas(config, RAIZ)
    img, prof = _escena([("Pelota roja", (0.0, 0.0, 0.9), 0.08), ("Pelota verde", (0.2, 0.0, 0.9), 0.045)])
    for i in range(6):
        det.paso(img, K, prof, ahora=100 + i * 0.1)
    lista = det.lista(ahora=100.6)
    assert len(lista) == 2 and all(p["estable"] and p["tipo"] == "pelota" for p in lista)
    assert len({p["id"] for p in lista}) == 2
    roja = next(p for p in lista if p["nombre"] == "Pelota roja")
    T = np.array(roja["T_cam"])
    assert np.allclose(T[:3, 3], [0, 0, 0.9], atol=0.015)
    assert T[:3, 2] @ np.array([0, 0, -1.0]) > 0.99                     # z hacia la cámara
    if det.T_mundo_cam is not None:
        assert np.allclose(np.array(roja["T_mundo"]), det.T_mundo_cam @ T, atol=1e-4)
    assert det.lista(ahora=110) == []                                     # olvidadas


def test_rango_sugerido_del_rojo_da_la_vuelta():
    parche = np.zeros((9, 9, 3), np.uint8)
    parche[..., 0], parche[..., 1], parche[..., 2] = 177, 200, 160
    r = rango_sugerido(parche)
    assert len(r) == 2 and r[0][0][0] == 0 and r[1][1][0] == 179


def test_altura_de_la_mesa_con_un_plano_alrededor_de_la_pelota():
    """(17-nov) La mesa medida con la profundidad de alrededor de la pelota (RANSAC), aunque el centro
    de la pelota venga 2 cm mal en altura y la profundidad tenga ruido."""
    from skill_shield.autonomo.pelotas import apoyo_bajo
    K = np.array([[450.0, 0, 320], [0, 450.0, 240], [0, 0, 1]])
    cam = np.array([0.0, 0.0, 0.5])
    z = np.array([0.0, -0.8, -0.75]); z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    T = np.eye(4); T[:3, :3] = np.column_stack([x, np.cross(z, x), z]); T[:3, 3] = cam
    zt, r = -0.25, 0.035
    c = np.array([0.0, -0.75, zt + r])
    v, u = np.mgrid[0:480, 0:640]
    rays = np.stack([(u - 320) / 450.0, (v - 240) / 450.0, np.ones_like(u, float)], -1) @ T[:3, :3].T
    t_mesa = (zt - cam[2]) / rays[..., 2]
    oc = cam - c
    b = rays @ oc; a = np.sum(rays ** 2, -1); disc = b ** 2 - a * (oc @ oc - r * r)
    t_bola = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
    t = np.minimum(np.where(t_mesa > 0, t_mesa, np.inf), t_bola)
    prof = (t * (rays @ T[:3, 2]))                                   # z en la cámara
    prof = prof + np.random.default_rng(0).normal(0, 0.003, prof.shape)
    prof_mm = np.where(np.isfinite(prof), prof * 1000, 0).astype(np.uint16)
    cc = np.linalg.inv(T) @ np.r_[c, 1.0]
    uu, vv = cc[0] / cc[2] * 450 + 320, cc[1] / cc[2] * 450 + 240
    r_px = r * 450 / cc[2]
    mal = cc[:3] + np.linalg.inv(T)[:3, :3] @ [0, 0, 0.02]           # el centro estimado, 2 cm alto
    zm = apoyo_bajo(prof_mm, K, T, uu, vv, r_px, mal, 2 * r)
    assert zm is not None and abs(zm - zt) < 0.004
    assert apoyo_bajo(np.zeros_like(prof_mm), K, T, uu, vv, r_px, mal, 2 * r) is None   # sin profundidad: nada
    # (17-nov) y hasta dónde llega: la envolvente de la mesa vista alrededor contiene la pelota y es amplia
    zm2, poli = apoyo_bajo(prof_mm, K, T, uu, vv, r_px, mal, 2 * r, mesa_m=0.35)
    from scipy.spatial import Delaunay
    assert abs(zm2 - zt) < 0.004 and poli is not None and Delaunay(np.array(poli)).find_simplex(c[:2]) >= 0
    P = np.array(poli)
    assert (P[:, 0].max() - P[:, 0].min()) > 0.4 and (P[:, 1].max() - P[:, 1].min()) > 0.3


def test_el_aro_rojo_de_una_articulacion_no_es_una_pelota():
    """(18-nov) La tapa de las articulaciones del FR5: disco blanco con un ANILLO rojo. Su contorno exterior es
    redondo, pero está hueco: no es una pelota (salía como «Pelota roja» flotando junto al brazo)."""
    img = np.full((480, 640, 3), 120, np.uint8)
    prof = np.full((480, 640), 600, np.uint16)
    rojo = cv2.cvtColor(np.uint8([[[178, 190, 160]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    cv2.circle(img, (320, 240), 40, rojo, 7)                                  # el aro
    cv2.circle(img, (320, 240), 33, (235, 235, 235), -1)                      # la tapa blanca
    assert not [1 for e, *_ in candidatas(img, prof, K, catalogo(config)) if e["nombre"] == "Pelota roja"]
    # una pelota con un brillo blanco en medio sí sigue contando
    img2, prof2 = _escena([("Pelota roja", (0.0, 0.0, 0.9), 0.08)])
    cv2.circle(img2, (325, 235), 4, (255, 255, 255), -1)
    assert [1 for e, *_ in candidatas(img2, prof2, K, catalogo(config)) if e["nombre"] == "Pelota roja"]
