#!/usr/bin/env python3
"""
PACO-LVIS → YOLO-seg con nuestras clases (herramienta + mango).

PACO trae, en formato COCO, la máscara de cada objeto y la de cada una de sus partes
(«hammer:handle», «pliers:jaw»…). Aquí: cada herramienta de clases.yaml → su máscara con su
clase, y la unión de sus partes de agarre (paco.partes_mango) → una instancia «mango».

    python convertir_paco.py --anotaciones <DATOS>/paco/annotations --imagenes <DATOS>/coco
    python convertir_paco.py ... --listar        # solo cuenta lo que hay de cada herramienta

Las imágenes son las de COCO 2017: solo hacen falta las que tienen herramientas (unos pocos miles) y
se bajan solas, una a una, a <DATOS>/coco/{train2017,val2017}/. PACO train y test → nuestro train;
PACO val → nuestro val.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from comun import Escritor, cargar_config, dir_datos, id_clase, mascara_a_poligono

ARCHIVOS = {"train": "paco_lvis_v1_train.json", "test": "paco_lvis_v1_test.json", "val": "paco_lvis_v1_val.json"}
A_NUESTRO = {"train": "train", "test": "train", "val": "val"}


def decodificar(seg, h, w):
    from pycocotools import mask as mu
    if isinstance(seg, list):                          # polígonos
        rle = mu.merge(mu.frPyObjects(seg, h, w))
    elif isinstance(seg.get("counts"), list):          # RLE sin comprimir
        rle = mu.frPyObjects(seg, h, w)
    else:                                              # RLE comprimido
        rle = dict(seg)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode()
    return mu.decode(rle)


def ruta_imagen(raiz, file_name):
    """PACO guarda file_name al estilo de LVIS/COCO (a veces «COCO_val2014_000000001268.jpg»):
    se busca tal cual y, si no, por su número en train2017/ y val2017/."""
    cand = [os.path.join(raiz, file_name)]
    m = re.search(r"(\d{12})", os.path.basename(file_name))
    if m:
        cand += [os.path.join(raiz, s, m.group(1) + ".jpg") for s in ("train2017", "val2017")]
    for c in cand:
        if os.path.exists(c):
            return c
    return None


def bajar_las_que_falten(raiz, file_names, hilos=16):
    """Solo hacen falta unos pocos miles de imágenes de COCO (las que tienen herramientas): en vez de bajar
    los 19 GB de train2017 + val2017, se bajan una a una de images.cocodataset.org."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    def una(fn):
        if ruta_imagen(raiz, fn):
            return True
        m = re.search(r"(\d{12})", os.path.basename(fn))
        if not m:
            return False
        for s in ("train2017", "val2017"):
            destino = os.path.join(raiz, s, m.group(1) + ".jpg")
            try:
                with urllib.request.urlopen(f"http://images.cocodataset.org/{s}/{m.group(1)}.jpg", timeout=30) as r:
                    datos = r.read()
                os.makedirs(os.path.dirname(destino), exist_ok=True)
                with open(destino + ".tmp", "wb") as f:
                    f.write(datos)
                os.replace(destino + ".tmp", destino)
                return True
            except Exception:
                continue
        return False

    faltan = [fn for fn in file_names if not ruta_imagen(raiz, fn)]
    if not faltan:
        return
    print(f"⬇ Bajando {len(faltan)} imágenes de COCO a {raiz} …", flush=True)
    with ThreadPoolExecutor(hilos) as ex:
        ok = sum(ex.map(una, faltan))
    print(f"   {ok}/{len(faltan)} bajadas", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anotaciones", default=os.path.join(dir_datos(), "paco", "annotations"))
    ap.add_argument("--imagenes", default=os.path.join(dir_datos(), "coco"), help="carpeta con train2017/ y val2017/")
    ap.add_argument("--splits", nargs="+", default=["train", "test", "val"], choices=list(ARCHIVOS))
    ap.add_argument("--listar", action="store_true", help="solo contar herramientas y partes, sin convertir")
    ap.add_argument("--no-bajar", action="store_true",
                    help="no bajar de internet las imágenes de COCO que falten (si ya tenéis train2017/ y val2017/)")
    ap.add_argument("--max-lado", type=int, default=1280)
    a = ap.parse_args()

    cfg = cargar_config()
    obj_map = {k: id_clase(cfg, v) for k, v in cfg["paco"]["objetos"].items()}
    partes_mango = set(cfg["paco"]["partes_mango"])
    id_mango = id_clase(cfg, "mango")
    esc = None if a.listar else Escritor("paco", cfg, max_lado=a.max_lado, limpiar=True)
    faltan, cuenta = 0, Counter()

    for split in a.splits:
        ruta = os.path.join(a.anotaciones, ARCHIVOS[split])
        if not os.path.exists(ruta):
            sys.exit(f"✗ No está {ruta} (¿has hecho ./descargar.sh paco?)")
        print(f"📖 Leyendo {ruta} (tarda un poco)…", flush=True)
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        cats = {c["id"]: c["name"] for c in d["categories"]}
        imgs = {im["id"]: im for im in d["images"]}
        objetos = defaultdict(list)                     # image_id → [ann de herramienta]
        partes = defaultdict(list)                      # obj_ann_id → [ann de parte de agarre]
        for an in d["annotations"]:
            nombre = cats.get(an["category_id"], "")
            if ":" in nombre:
                obj, parte = nombre.split(":", 1)
                if obj in obj_map and parte in partes_mango:
                    partes[an.get("obj_ann_id")].append(an)
                if obj in obj_map:
                    cuenta[f"{split} · parte {nombre}"] += 1
            elif nombre in obj_map:
                objetos[an["image_id"]].append((nombre, an))
                cuenta[f"{split} · {nombre}"] += 1
        del d
        if a.listar:
            continue
        print(f"   {len(objetos)} imágenes con herramientas en {split}", flush=True)
        if not a.no_bajar:
            bajar_las_que_falten(a.imagenes, [imgs[i]["file_name"] for i in objetos])
        for n, (img_id, anns) in enumerate(sorted(objetos.items())):
            im = imgs[img_id]
            ruta_img = ruta_imagen(a.imagenes, im["file_name"])
            if ruta_img is None:
                faltan += 1
                continue
            h, w = im["height"], im["width"]
            inst = []
            for nombre, an in anns:
                p = mascara_a_poligono(decodificar(an["segmentation"], h, w))
                if p is None:
                    continue
                inst.append((obj_map[nombre], p))
                ms = [decodificar(pa["segmentation"], h, w) for pa in partes.get(an["id"], [])]
                if ms:
                    pm = mascara_a_poligono(np.any(np.stack(ms), axis=0))
                    if pm is not None:
                        inst.append((id_mango, pm))
            if inst:
                esc.guardar(f"paco_{img_id}", ruta_img, inst, A_NUESTRO[split])
            if n % 500 == 0:
                print(f"   {split}: {n}/{len(objetos)}", flush=True)

    print("\nHerramientas y partes de agarre en PACO:")
    for k, v in sorted(cuenta.items()):
        print(f"   {k:45s} {v}")
    if esc is not None:
        if faltan:
            print(f"⚠ {faltan} imágenes no estaban en {a.imagenes} (¿falta train2017/ o val2017/?)")
        esc.cerrar(dict(imagenes_que_faltan=faltan))


if __name__ == "__main__":
    main()
