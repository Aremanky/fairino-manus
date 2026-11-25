#!/usr/bin/env python3
"""
Dataset de DETECCIÓN (cajas, formato YOLO: «clase xc yc w h») → dataset de SEGMENTACIÓN con SAM 2:
cada caja se le pasa a SAM como pista y su máscara pasa a ser el polígono. Sirve para METU-ALET, el de
herramientas de obra o cualquier proyecto de Roboflow que solo tenga cajas (exportado en «YOLOv8»).

    python cajas_a_mascaras.py <carpeta_con_cajas> <carpeta_salida>
    python convertir_yolo_seg.py <carpeta_salida> --fuente alet        # y luego, a nuestras clases

La salida conserva la estructura y el data.yaml de la entrada; lo que ya era polígono se deja igual. Las máscaras de SAM con cajas suelen ser
buenas, pero REVÍSALAS (revisar.py) antes de entrenar con ellas. Si se corta, al relanzarlo sigue
por donde iba. Estas fuentes no traen el mango:
completadlo con preetiquetar.py --completar-mango.
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np

from comun import cargar_config, mascara_a_poligono, ruta_datos, traductor

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def clases_utiles(entrada):
    """Ids del data.yaml de la entrada cuyas clases traduce clases.yaml (bloque «nombres»); las demás
    (EPI, oficina, jardín…) no se pasan por SAM: no se van a usar y es lo que más tiempo lleva.
    None si no hay data.yaml que leer (entonces se pasan todas)."""
    import yaml
    ys = glob.glob(os.path.join(entrada, "*.y*ml"))
    if not ys:
        return None
    with open(ys[0], encoding="utf-8") as f:
        nombres = (yaml.safe_load(f) or {}).get("names")
    if isinstance(nombres, list):
        nombres = dict(enumerate(nombres))
    if not nombres:
        return None
    traducir = traductor(cargar_config())
    utiles = {int(i) for i, n in nombres.items() if traducir(n) is not None}
    print(f"   SAM solo para {len(utiles)} de {len(nombres)} clases: "
          + ", ".join(str(nombres[i]) for i in sorted(utiles)))
    return utiles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("salida")
    ap.add_argument("--sam", default="sam2.1_b.pt", help="modelo SAM de Ultralytics (se baja solo)")
    ap.add_argument("--todas", action="store_true",
                    help="pasar por SAM todas las cajas (por defecto, solo las de clases que tiene clases.yaml)")
    a = ap.parse_args()
    a.entrada, a.salida = ruta_datos(a.entrada), ruta_datos(a.salida)
    utiles = None if a.todas else clases_utiles(a.entrada)
    from ultralytics import SAM
    sam = SAM(a.sam)

    for y in glob.glob(os.path.join(a.entrada, "*.y*ml")):
        os.makedirs(a.salida, exist_ok=True)
        shutil.copyfile(y, os.path.join(a.salida, os.path.basename(y)))
    n = 0
    txts = sorted(set(glob.glob(os.path.join(a.entrada, "**", "labels", "**", "*.txt"), recursive=True)))
    for txt in txts:
        base = os.path.splitext(txt.replace(os.sep + "labels" + os.sep, os.sep + "images" + os.sep, 1))[0]
        img = next((base + e for e in EXT + tuple(x.upper() for x in EXT) if os.path.exists(base + e)), None)
        if img is None:
            continue
        rel_lab = os.path.relpath(txt, a.entrada)
        rel_img = os.path.relpath(img, a.entrada)
        os.makedirs(os.path.dirname(os.path.join(a.salida, rel_lab)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(a.salida, rel_img)), exist_ok=True)
        if os.path.exists(os.path.join(a.salida, rel_lab)):      # ya hecha (se puede cortar y seguir)
            n += 1
            continue
        try:
            os.link(img, os.path.join(a.salida, rel_img))
        except OSError:
            shutil.copyfile(img, os.path.join(a.salida, rel_img))
        im = cv2.imread(img)
        h, w = im.shape[:2]
        clases, cajas, lineas = [], [], []
        with open(txt) as f:
            for l in f:
                v = l.split()
                if len(v) >= 7:                        # ya es un polígono: se deja tal cual
                    lineas.append(l.strip())
                elif len(v) == 5:
                    c, xc, yc, bw, bh = int(float(v[0])), *map(float, v[1:])
                    if utiles is not None and c not in utiles:     # no es de obra: no se aprende
                        continue
                    clases.append(c)
                    cajas.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
        if cajas:
            r = sam(img, bboxes=cajas, verbose=False)[0]
            mascaras = r.masks.data.cpu().numpy() if r.masks is not None else []
            for c, m in zip(clases, mascaras):
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                p = mascara_a_poligono(m)
                if p is not None:
                    lineas.append(f"{c} " + " ".join(f"{v:.6f}" for v in p))
        with open(os.path.join(a.salida, rel_lab), "w") as f:
            f.write("\n".join(lineas) + ("\n" if lineas else ""))
        n += 1
        if n % 200 == 0:
            print(f"   {n}/{len(txts)} imágenes", flush=True)
    print(f"✅ {n} imágenes con máscaras de SAM en {a.salida}")


if __name__ == "__main__":
    main()
