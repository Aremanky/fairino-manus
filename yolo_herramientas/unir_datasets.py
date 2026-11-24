#!/usr/bin/env python3
"""
Junta las fuentes ya convertidas (<DATOS>/convertidos/*) en un dataset de YOLO listo para entrenar:
    <DATOS>/yolo/<nombre>/images/{train,val}, labels/{train,val} y <nombre>.yaml

    python unir_datasets.py --nombre herramientas_v1                       # todas las fuentes CON mango
    python unir_datasets.py --nombre herramientas_v1 --repetir propias_sala=5   # vuestras fotos pesan x5
    python unir_datasets.py --nombre v2 --fuentes paco handal propias_sala rf_tools_segmentation_0jsgu \
                            --incluir-sin-mango

Por qué se dejan fuera por defecto las fuentes SIN mango: en sus imágenes hay mangos que nadie ha marcado y
YOLO aprendería que eso es «fondo». Úsalas después de preetiquetarles el mango (preetiquetar.py).
Los archivos se enlazan (no se copian): no ocupa más disco.
"""
import argparse
import glob
import json
import os
import shutil
from collections import Counter

import yaml

from comun import cargar_config, dir_datos


def enlazar(origen, destino):
    if os.path.exists(destino):
        os.remove(destino)
    try:
        os.link(origen, destino)
    except OSError:
        shutil.copyfile(origen, destino)


def contar(dir_fuente, n_clases):
    c = Counter()
    for txt in glob.glob(os.path.join(dir_fuente, "labels", "*", "*.txt")):
        with open(txt) as f:
            for linea in f:
                if linea.strip():
                    c[int(linea.split()[0])] += 1
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", default="herramientas_v1")
    ap.add_argument("--fuentes", nargs="*", default=None, help="por defecto, todas las de convertidos/")
    ap.add_argument("--repetir", nargs="*", default=[], metavar="FUENTE=N",
                    help="repetir N veces las imágenes de TRAIN de una fuente (las vuestras pesan más)")
    ap.add_argument("--incluir-sin-mango", action="store_true")
    a = ap.parse_args()

    cfg = cargar_config()
    nombres = cfg["clases"]
    base = os.path.join(dir_datos(), "convertidos")
    fuentes = a.fuentes or sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    repetir = {k: int(v) for k, v in (x.split("=") for x in a.repetir)}
    destino = os.path.join(dir_datos(), "yolo", a.nombre)
    if os.path.isdir(destino):
        shutil.rmtree(destino)
    for s in ("train", "val"):
        os.makedirs(os.path.join(destino, "images", s))
        os.makedirs(os.path.join(destino, "labels", s))

    total = {"train": Counter(), "val": Counter()}
    imgs = Counter()
    for fu in fuentes:
        d = os.path.join(base, fu)
        if not os.path.isdir(d):
            print(f"⚠ {fu}: no está en {base}, se salta")
            continue
        c = contar(d, len(nombres))
        herramientas = sum(v for k, v in c.items() if k != 0)
        if herramientas and not c[0] and not a.incluir_sin_mango:
            print(f"⚠ {fu}: tiene herramientas pero NINGÚN mango → fuera (ver --incluir-sin-mango)")
            continue
        n_rep = repetir.get(fu, 1)
        for s in ("train", "val"):
            for img in sorted(glob.glob(os.path.join(d, "images", s, "*"))):
                stem = os.path.splitext(os.path.basename(img))[0]
                lab = os.path.join(d, "labels", s, stem + ".txt")
                if not os.path.exists(lab):
                    continue
                with open(lab) as f:
                    clases = [int(l.split()[0]) for l in f if l.strip()]
                for r in range(n_rep if s == "train" else 1):
                    suf = f"_r{r}" if r else ""
                    enlazar(img, os.path.join(destino, "images", s, stem + suf + os.path.splitext(img)[1]))
                    enlazar(lab, os.path.join(destino, "labels", s, stem + suf + ".txt"))
                    total[s].update(clases)
                    imgs[s] += 1
        print(f"✓ {fu}" + (f" (x{n_rep} en train)" if n_rep > 1 else ""))

    ruta_yaml = os.path.join(destino, a.nombre + ".yaml")
    with open(ruta_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(path=destino, train="images/train", val="images/val",
                            names={i: n for i, n in enumerate(nombres)}), f, allow_unicode=True, sort_keys=False)
    with open(os.path.join(destino, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fuentes=fuentes, imagenes=imgs,
                       instancias={s: {nombres[k]: v for k, v in sorted(total[s].items())} for s in total}),
                  f, indent=2, ensure_ascii=False)

    print(f"\n{'clase':16s} {'train':>8s} {'val':>8s}")
    for i, n in enumerate(nombres):
        print(f"{n:16s} {total['train'][i]:8d} {total['val'][i]:8d}" + ("   ← sin ejemplos" if not total['train'][i] else ""))
    print(f"{'IMÁGENES':16s} {imgs['train']:8d} {imgs['val']:8d}")
    print(f"\n✅ Dataset listo: {ruta_yaml}")


if __name__ == "__main__":
    main()
