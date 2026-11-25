#!/usr/bin/env python3
"""
Cualquier dataset en formato YOLO-seg (Roboflow, una exportación de CVAT «Ultralytics YOLO Segmentation»,
vuestras fotos ya etiquetadas…) → nuestras clases.

Lee los nombres de clase de su data.yaml y los traduce con el bloque «nombres» de clases.yaml (lo que no
esté ahí se ignora y se dice). Sirve con las dos disposiciones habituales:
    train/images + train/labels (+ valid/, test/)       ← Roboflow
    images/train + labels/train (+ val)                 ← Ultralytics / CVAT

    python convertir_yolo_seg.py <carpeta> --fuente propias_sala
    python convertir_yolo_seg.py <carpeta> --fuente propias_sala --repartir     # train/val por nuestra cuenta
"""
import argparse
import glob
import hashlib
import os
import re
import sys
from collections import Counter

import yaml

from comun import Escritor, cargar_config, particion, ruta_datos, traductor

EXT_IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def nombres_de(carpeta):
    cands = glob.glob(os.path.join(carpeta, "*.yaml")) + glob.glob(os.path.join(carpeta, "*.yml"))
    for c in cands:
        with open(c, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        n = d.get("names")
        if isinstance(n, dict):
            return {int(k): str(v) for k, v in n.items()}
        if isinstance(n, list):
            return dict(enumerate(map(str, n)))
    sys.exit(f"✗ No encuentro un data.yaml con «names» en {carpeta}")


def imagen_de(ruta_txt):
    base = os.path.splitext(ruta_txt)[0]
    for a, b in (("/labels/", "/images/"), ("labels", "images")):
        cand = base.replace(a, b, 1)
        for ext in EXT_IMG + tuple(e.upper() for e in EXT_IMG):
            if os.path.exists(cand + ext):
                return cand + ext
    return None


def split_de(ruta_rel):
    partes = re.split(r"[\\/]", ruta_rel.lower())
    return "val" if any(p in ("valid", "val", "test", "validation") for p in partes) else "train"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta")
    ap.add_argument("--fuente", required=True, help="nombre corto para esta fuente (p. ej. propias_sala)")
    ap.add_argument("--repartir", action="store_true", help="ignorar sus splits y repartir train/val por imagen")
    ap.add_argument("--frac-val", type=float, default=0.15)
    ap.add_argument("--max-lado", type=int, default=1280)
    ap.add_argument("--max-fondo", type=float, default=0.1,
                    help="imágenes sin ninguna herramienta nuestra que se guardan, como fracción de las que sí "
                         "tienen (0,1 = un 10 %%; en los datasets de taller sobran fotos de oficina, EPI…)")
    a = ap.parse_args()

    a.carpeta = ruta_datos(a.carpeta)
    cfg = cargar_config()
    traducir = traductor(cfg)
    nombres = nombres_de(a.carpeta)
    traduccion, ignoradas = {}, []
    for i, n in nombres.items():
        c = traducir(n)
        if c is not None:
            traduccion[i] = c
        else:
            ignoradas.append(n)
    print("Clases → " + " · ".join(f"{nombres[i]}→{cfg['clases'][c]}" for i, c in traduccion.items()))
    if ignoradas:
        print("⚠ Se ignoran (añádelas a «nombres» en clases.yaml si las quieres): " + ", ".join(ignoradas))

    esc = Escritor(a.fuente, cfg, max_lado=a.max_lado, limpiar=True)
    st = Counter()
    pendientes, fondos = [], []
    for txt in sorted(glob.glob(os.path.join(a.carpeta, "**", "*.txt"), recursive=True)):
        rel = os.path.relpath(txt, a.carpeta)
        if "labels" not in rel.replace("\\", "/").split("/"):
            continue
        img = imagen_de(txt)
        if img is None:
            st["etiquetas sin imagen"] += 1
            continue
        inst = []
        with open(txt) as f:
            for linea in f:
                v = linea.split()
                if not v:
                    continue
                c = int(float(v[0]))
                if c not in traduccion:
                    st["instancias de clases ignoradas"] += 1
                    continue
                if len(v) == 5:                        # es una caja, no un polígono
                    st["cajas (sin máscara) que se saltan"] += 1
                    continue
                p = [float(x) for x in v[1:]]
                if len(p) < 6 or len(p) % 2:
                    st["líneas mal formadas"] += 1
                    continue
                inst.append((traduccion[c], p))
        split = particion(os.path.basename(img), a.frac_val) if a.repartir else split_de(rel)
        nombre = a.fuente + "_" + re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(rel)[0]).strip("_")
        (pendientes if inst else fondos).append((nombre, img, inst, split))
    n_fondo = min(len(fondos), int(a.max_fondo * len(pendientes)))
    if len(fondos) > n_fondo:
        st["imágenes sin herramientas descartadas"] = len(fondos) - n_fondo
    fondos.sort(key=lambda x: hashlib.md5(x[0].encode()).hexdigest())     # al azar, pero siempre las mismas
    for nombre, img, inst, split in pendientes + fondos[:n_fondo]:
        if esc.guardar(nombre, img, inst, split):
            st["imágenes" if inst else "imágenes de fondo"] += 1
    if st["cajas (sin máscara) que se saltan"]:
        print("⚠ Este dataset tiene cajas, no máscaras: pásalo por cajas_a_mascaras.py")
    print(" · ".join(f"{k}: {v}" for k, v in st.items()))
    esc.cerrar(dict(st))


if __name__ == "__main__":
    main()
