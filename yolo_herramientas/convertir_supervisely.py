#!/usr/bin/env python3
"""
Dataset en formato Supervisely (así se bajan los de Dataset Ninja: METU-ALET, herramientas de obra…) →
dataset YOLO con los nombres de clase ORIGINALES. Las cajas quedan como cajas («c xc yc w h») y los
polígonos y bitmaps, como polígonos. Después:
    python cajas_a_mascaras.py <salida> <salida>_seg          # SAM 2 saca la máscara de cada caja
    python convertir_yolo_seg.py <salida>_seg --fuente alet   # y a nuestras clases

    python convertir_supervisely.py $HERRAMIENTAS_DATOS/ninja/metu-alet $HERRAMIENTAS_DATOS/yolo_crudo/alet

Supervisely: <proyecto>/<dataset>/img/<imagen> y <proyecto>/<dataset>/ann/<imagen>.json, con
{"size": {"height", "width"}, "objects": [{"classTitle", "geometryType", "points": {"exterior": …}, "bitmap"}]}.
"""
import argparse
import base64
import glob
import json
import os
import re
import shutil
import zlib
from collections import Counter

import cv2
import numpy as np
import yaml

from comun import mascara_a_poligono, particion


def bitmap_a_mascara(obj, h, w):
    """Bitmap de Supervisely: PNG comprimido con zlib y en base64, colocado en «origin»."""
    datos = zlib.decompress(base64.b64decode(obj["bitmap"]["data"]))
    png = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_UNCHANGED)
    trozo = (png[:, :, 3] if png.ndim == 3 and png.shape[2] == 4 else (png if png.ndim == 2 else png.max(2))) > 0
    x0, y0 = obj["bitmap"]["origin"]
    m = np.zeros((h, w), np.uint8)
    hh, ww = min(trozo.shape[0], h - y0), min(trozo.shape[1], w - x0)
    m[y0:y0 + hh, x0:x0 + ww] = trozo[:hh, :ww]
    return m


def enlazar(o, d):
    try:
        os.link(o, d)
    except OSError:
        shutil.copyfile(o, d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("salida")
    ap.add_argument("--frac-val", type=float, default=0.1)
    a = ap.parse_args()

    anns = sorted(glob.glob(os.path.join(a.entrada, "**", "ann", "*.json"), recursive=True))
    if not anns:
        raise SystemExit(f"✗ No hay carpetas ann/ con .json en {a.entrada} (¿está descomprimido?)")
    nombres = []
    meta = glob.glob(os.path.join(a.entrada, "**", "meta.json"), recursive=True)
    if meta:
        with open(meta[0], encoding="utf-8") as f:
            nombres = [c["title"] for c in json.load(f).get("classes", [])]
    idx = {n: i for i, n in enumerate(nombres)}
    shutil.rmtree(a.salida, ignore_errors=True)
    for s in ("train", "val"):
        os.makedirs(os.path.join(a.salida, "images", s))
        os.makedirs(os.path.join(a.salida, "labels", s))
    st = Counter()
    for ra in anns:
        dir_ds = os.path.dirname(os.path.dirname(ra))
        nombre_img = os.path.basename(ra)[:-5]                     # «foto.jpg.json» → «foto.jpg»
        img = os.path.join(dir_ds, "img", nombre_img)
        if not os.path.exists(img):
            st["anotaciones sin imagen"] += 1
            continue
        with open(ra, encoding="utf-8") as f:
            d = json.load(f)
        h, w = d["size"]["height"], d["size"]["width"]
        lineas = []
        for o in d.get("objects", []):
            t = o.get("classTitle")
            if t not in idx:
                idx[t] = len(nombres)
                nombres.append(t)
            c, g = idx[t], o.get("geometryType")
            ext = (o.get("points") or {}).get("exterior") or []
            if g == "rectangle" and len(ext) == 2:
                (x1, y1), (x2, y2) = ext
                x1, x2 = sorted((max(0, x1), min(w, x2)))
                y1, y2 = sorted((max(0, y1), min(h, y2)))
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    lineas.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                    st["cajas"] += 1
            elif g == "polygon" and len(ext) >= 3:
                p = np.asarray(ext, np.float64)
                lineas.append(f"{c} " + " ".join(f"{v:.6f}" for v in (p / [w, h]).clip(0, 1).reshape(-1)))
                st["polígonos"] += 1
            elif g == "bitmap" and o.get("bitmap"):
                p = mascara_a_poligono(bitmap_a_mascara(o, h, w))
                if p is not None:
                    lineas.append(f"{c} " + " ".join(f"{v:.6f}" for v in p))
                    st["bitmaps"] += 1
            else:
                st[f"geometría «{g}» que no se usa"] += 1
        ds = os.path.basename(dir_ds).lower()
        split = "val" if re.search(r"val|test", ds) else particion(nombre_img, a.frac_val)
        base = re.sub(r"[^A-Za-z0-9]+", "_", f"{ds}_{os.path.splitext(nombre_img)[0]}").strip("_")
        ext_img = os.path.splitext(nombre_img)[1].lower() or ".jpg"
        enlazar(img, os.path.join(a.salida, "images", split, base + ext_img))
        with open(os.path.join(a.salida, "labels", split, base + ".txt"), "w") as f:
            f.write("\n".join(lineas) + ("\n" if lineas else ""))
        st[f"imágenes {split}"] += 1
    with open(os.path.join(a.salida, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(path=os.path.abspath(a.salida), train="images/train", val="images/val",
                            names=dict(enumerate(nombres))), f, allow_unicode=True, sort_keys=False)
    print(" · ".join(f"{k}: {v}" for k, v in st.items()))
    print(f"Clases ({len(nombres)}): " + ", ".join(nombres))
    print(f"✅ {a.salida}")


if __name__ == "__main__":
    main()
