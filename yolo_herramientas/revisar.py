#!/usr/bin/env python3
"""
Pinta unas cuantas imágenes con sus máscaras para comprobar a ojo que una conversión está bien
(herramienta en color, MANGO en verde con borde blanco). Se guardan en <carpeta>/revision/.

    python revisar.py <DATOS>/convertidos/handal            # 24 al azar
    python revisar.py <DATOS>/yolo/herramientas_v1 --n 48
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np

from comun import cargar_config

COLORES = [(60, 200, 60), (255, 120, 0), (0, 140, 255), (200, 0, 200), (0, 220, 255), (255, 0, 120),
           (120, 120, 255), (0, 255, 160), (180, 180, 0), (255, 200, 120), (90, 60, 200), (60, 160, 255)]


def pintar(img, lab, nombres):
    h, w = img.shape[:2]
    capa = img.copy()
    lineas = []
    with open(lab) as f:
        for linea in f:
            v = linea.split()
            if len(v) >= 7:
                lineas.append((int(v[0]), np.array(v[1:], float).reshape(-1, 2) * [w, h]))
    lineas.sort(key=lambda x: x[0] == 0)               # el mango, encima
    for c, p in lineas:
        pts = np.round(p).astype(np.int32)
        cv2.fillPoly(capa, [pts], COLORES[c % len(COLORES)])
        cv2.polylines(img, [pts], True, (255, 255, 255) if c == 0 else COLORES[c % len(COLORES)], 2)
        x, y = pts.min(0) if c else (pts.min(0)[0], pts.max(0)[1] + 16)   # el mango, debajo
        cv2.putText(img, nombres[c] if c < len(nombres) else str(c), (int(x), max(12, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return cv2.addWeighted(capa, 0.45, img, 0.55, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--split", default="*")
    a = ap.parse_args()
    nombres = cargar_config()["clases"]
    imgs = sorted(glob.glob(os.path.join(a.carpeta, "images", a.split, "*")))
    if not imgs:
        raise SystemExit(f"✗ No hay imágenes en {a.carpeta}/images/{a.split}/")
    random.seed(0)
    sel = random.sample(imgs, min(a.n, len(imgs)))
    salida = os.path.join(a.carpeta, "revision")
    os.makedirs(salida, exist_ok=True)
    teselas = []
    for ruta in sel:
        lab = ruta.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        lab = os.path.splitext(lab)[0] + ".txt"
        img = cv2.imread(ruta)
        if img is None or not os.path.exists(lab):
            continue
        t = pintar(img, lab, nombres)
        f = 400 / max(t.shape[:2])
        t = cv2.resize(t, (int(t.shape[1] * f), int(t.shape[0] * f)))
        t = cv2.copyMakeBorder(t, 0, 400 - t.shape[0], 0, 400 - t.shape[1], cv2.BORDER_CONSTANT)
        teselas.append(t)
    for i in range(0, len(teselas), 12):
        grupo = teselas[i:i + 12]
        grupo += [np.zeros_like(teselas[0])] * (12 - len(grupo))
        mosaico = np.vstack([np.hstack(grupo[j:j + 4]) for j in range(0, 12, 4)])
        cv2.imwrite(os.path.join(salida, f"revision_{i // 12:02d}.jpg"), mosaico)
    print(f"✅ {len(teselas)} imágenes pintadas en {salida}/")


if __name__ == "__main__":
    main()
