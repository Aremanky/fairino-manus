#!/usr/bin/env python3
"""
Usa un modelo ya entrenado para ahorrar trabajo de etiquetado. Dos usos:

1) Preetiquetar vuestras fotos para corregirlas en CVAT (mucho más rápido que dibujar desde cero):
    python preetiquetar.py --modelo runs/segment/herramientas_v1/weights/best.pt \
                           --imagenes <DATOS>/propias/sala_martillos/images
   → <DATOS>/propias/sala_martillos/cvat_preetiquetas.zip (formato «Ultralytics YOLO Segmentation»):
     en CVAT, la tarea con esas fotos → Actions → Upload annotations → ese formato → el zip.

2) Poner el MANGO a una fuente que solo trae la herramienta (Roboflow, cajas pasadas por SAM…):
    python preetiquetar.py --modelo …/best.pt --completar-mango rf_tools_segmentation_0jsgu
   → nueva fuente <fuente>_mango en convertidos/: sus etiquetas + los mangos que el modelo ve DENTRO de
     una herramienta ya etiquetada (con confianza ≥ --conf). Revisadla con revisar.py antes de usarla.
"""
import argparse
import glob
import os
import shutil
import zipfile

import cv2
import numpy as np
import yaml

from comun import cargar_config, dir_datos, poligono_a_mascara


def predecir(modelo, ruta, conf, imgsz):
    r = modelo.predict(ruta, conf=conf, imgsz=imgsz, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []
    return [(int(c), float(s), np.asarray(p, np.float64).reshape(-1).tolist())
            for c, s, p in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.masks.xyn) if len(p) >= 3]


def para_cvat(modelo, carpeta_img, conf, imgsz, nombres):
    base = os.path.dirname(os.path.abspath(carpeta_img.rstrip("/")))
    tmp = os.path.join(base, "cvat_preetiquetas")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "labels", "train"))
    imgs = sorted(f for f in glob.glob(os.path.join(carpeta_img, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    with open(os.path.join(tmp, "train.txt"), "w") as lista:
        for i, ruta in enumerate(imgs):
            nombre = os.path.basename(ruta)
            with open(os.path.join(tmp, "labels", "train", os.path.splitext(nombre)[0] + ".txt"), "w") as f:
                for c, _, p in predecir(modelo, ruta, conf, imgsz):
                    f.write(f"{c} " + " ".join(f"{v:.6f}" for v in p) + "\n")
            lista.write(f"data/images/train/{nombre}\n")
            if i % 50 == 0:
                print(f"   {i}/{len(imgs)}", flush=True)
    with open(os.path.join(tmp, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(path=".", train="train.txt", names=dict(enumerate(nombres))), f, allow_unicode=True)
    zip_ruta = tmp + ".zip"
    with zipfile.ZipFile(zip_ruta, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, fs in os.walk(tmp):
            for fn in fs:
                z.write(os.path.join(d, fn), os.path.relpath(os.path.join(d, fn), tmp))
    print(f"✅ {len(imgs)} imágenes preetiquetadas → {zip_ruta}")


def completar_mango(modelo, fuente, conf, imgsz, dentro_min):
    orig = os.path.join(dir_datos(), "convertidos", fuente)
    dest = orig + "_mango"
    shutil.rmtree(dest, ignore_errors=True)
    n_img = n_mangos = 0
    for split in ("train", "val"):
        os.makedirs(os.path.join(dest, "images", split))
        os.makedirs(os.path.join(dest, "labels", split))
        for img in sorted(glob.glob(os.path.join(orig, "images", split, "*"))):
            stem = os.path.splitext(os.path.basename(img))[0]
            with open(os.path.join(orig, "labels", split, stem + ".txt")) as f:
                lineas = [l.strip() for l in f if l.strip()]
            im = cv2.imread(img)
            h, w = im.shape[:2]
            herramientas = [poligono_a_mascara([float(x) for x in l.split()[1:]], h, w) for l in lineas
                            if int(l.split()[0]) != 0]
            nuevas = []
            if herramientas and not any(int(l.split()[0]) == 0 for l in lineas):
                for c, s, p in predecir(modelo, img, conf, imgsz):
                    if c != 0:
                        continue
                    m = poligono_a_mascara(p, h, w).astype(bool)
                    if m.sum() and max((m & (t > 0)).sum() / m.sum() for t in herramientas) >= dentro_min:
                        nuevas.append("0 " + " ".join(f"{v:.6f}" for v in p))
            try:
                os.link(img, os.path.join(dest, "images", split, os.path.basename(img)))
            except OSError:
                shutil.copyfile(img, os.path.join(dest, "images", split, os.path.basename(img)))
            with open(os.path.join(dest, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lineas + nuevas) + "\n")
            n_img += 1
            n_mangos += len(nuevas)
    print(f"✅ {n_img} imágenes, {n_mangos} mangos añadidos → {dest}\n   Revísalo: python revisar.py {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--imagenes", help="carpeta de fotos sin etiquetar → zip para CVAT")
    g.add_argument("--completar-mango", metavar="FUENTE", help="fuente de convertidos/ sin mango")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--dentro", type=float, default=0.8, help="fracción del mango que tiene que caer en la herramienta")
    a = ap.parse_args()
    from ultralytics import YOLO
    modelo = YOLO(a.modelo)
    if a.imagenes:
        para_cvat(modelo, a.imagenes, a.conf, a.imgsz, cargar_config()["clases"])
    else:
        completar_mango(modelo, a.completar_mango, max(a.conf, 0.5), a.imgsz, a.dentro)


if __name__ == "__main__":
    main()
