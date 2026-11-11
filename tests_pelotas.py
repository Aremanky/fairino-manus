"""
Pruebas de la detección de pelotas (11-nov), sin cámara: imágenes y profundidad sintéticas.

    python -m pytest tests_pelotas.py -q
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    det = DetectorPelotas(config, os.path.dirname(os.path.abspath(__file__)))
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
