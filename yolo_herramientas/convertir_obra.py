#!/usr/bin/env python3
"""
«Image Dataset for Object Detection of Small Size Construction Tools» (Zenodo 6530106, CC BY 4.0):
25 084 fotos de 12 herramientas de obra de interior (bucket, cutter, drill, grinder, hammer, knife, saw,
shovel, spanner, tacker, trowel, wrench), la cuarta parte en obras reales, con cajas.

Lee los DATA*.zip de $HERRAMIENTAS_DATOS/obra/ SIN descomprimirlos (pesan ~110 GB) y deja en
$HERRAMIENTAS_DATOS/yolo_crudo/obra/ un dataset YOLO de cajas con las imágenes reducidas a 1280 px y un
data.yaml con sus nombres originales. Después van por SAM 2 y a nuestras clases (publicos.sh convertir).

    python convertir_obra.py --inspeccionar     # PRIMERO: cómo vienen los zip y un mosaico con los nombres
    python convertir_obra.py                    # convierte los zip que estén completos
    python convertir_obra.py --base1            # si --inspeccionar dice que los ids empiezan en 1

Los zip que aún se están bajando se saltan; al relanzarlo, lo ya convertido no se repite.
Cita: Lee, Jeon y Shin (2023), KSCE Journal of Civil Engineering, doi:10.1007/s12205-022-1011-7.
"""
import argparse
import glob
import os
import random
import re
import zipfile
from collections import Counter, defaultdict

import cv2
import numpy as np
import yaml

from comun import cargar_config, dir_datos, particion

EXT_IMG = (".jpg", ".jpeg", ".png", ".bmp")


def zips_completos(carpeta):
    ok, faltan = [], []
    for z in sorted(glob.glob(os.path.join(carpeta, "DATA*.zip"))):
        (ok if zipfile.is_zipfile(z) else faltan).append(z)
    for z in faltan:
        print(f"   · {os.path.basename(z)} no está completo (¿se sigue bajando?): se salta")
    return ok


def emparejar(nombres):
    """[(etiqueta.txt, imagen)] dentro de un zip: misma base de nombre; si hay varias imágenes con la
    misma base, la que comparte más ruta con la etiqueta."""
    imgs = defaultdict(list)
    for n in nombres:
        if n.lower().endswith(EXT_IMG):
            imgs[os.path.splitext(os.path.basename(n))[0]].append(n)
    pares = []
    for n in nombres:
        if not n.lower().endswith(".txt"):
            continue
        cand = imgs.get(os.path.splitext(os.path.basename(n))[0])
        if cand:
            pares.append((n, max(cand, key=lambda i: len(os.path.commonprefix([i, n])))))
    return pares


def leer_cajas(texto):
    """Líneas «clase xc yc w h» (con o sin comas/paréntesis) → [(clase como texto, xc, yc, w, h)]."""
    cajas = []
    for linea in texto.splitlines():
        v = [t for t in re.split(r"[,\s()\[\]]+", linea.strip()) if t]
        if len(v) != 5:
            continue
        try:
            cajas.append((v[0], *map(float, v[1:])))
        except ValueError:
            continue
    return cajas


def nombre_clase(token, origen, base1):
    if re.fullmatch(r"-?\d+(\.0+)?", token):
        i = int(float(token)) - (1 if base1 else 0)
        return origen[i] if 0 <= i < len(origen) else None
    return token.strip().lower()


def leer_imagen(z, nombre):
    datos = np.frombuffer(z.read(nombre), np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def inspeccionar(zips, origen, base1, salida):
    ids, muestras = Counter(), []
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            nombres = z.namelist()
            ext = Counter(os.path.splitext(n)[1].lower() for n in nombres if not n.endswith("/"))
            carpetas = Counter(os.path.dirname(n).split("/")[0] for n in nombres)
            pares = emparejar(nombres)
            print(f"\n📦 {os.path.basename(zp)}: {len(nombres)} entradas · extensiones {dict(ext)}")
            print(f"   carpetas de primer nivel: {dict(carpetas.most_common(8))}")
            print(f"   {len(pares)} parejas etiqueta-imagen")
            for t, i in pares[:2]:
                print(f"   ej.: {t}  ↔  {i}")
                for linea in z.read(t).decode("utf-8", "replace").splitlines()[:3]:
                    print(f"        «{linea}»")
            aqui = Counter()
            for t, _ in pares:
                aqui.update(c[0] for c in leer_cajas(z.read(t).decode("utf-8", "replace")))
            ids.update(aqui)
            print("   cajas por clase en este zip: " + " · ".join(
                f"{nombre_clase(k, origen, base1)} {n}" for k, n in sorted(aqui.items(), key=lambda x: -x[1])))
            muestras += [(zp, t, i) for t, i in random.Random(0).sample(pares, min(12, len(pares)))]
    print("\nClases vistas en las etiquetas (id o nombre: cajas):")
    for k, n in sorted(ids.items(), key=lambda x: (len(x[0]), x[0])):
        print(f"   {k:>8s}: {n:6d}  → {nombre_clase(k, origen, base1)}")
    numeros = sorted(int(float(k)) for k in ids if re.fullmatch(r"-?\d+(\.0+)?", k))
    if numeros:
        if numeros[0] >= 1 and numeros[-1] == len(origen) and not base1:
            print(f"⚠ Los ids van de 1 a {len(origen)}: parece que empiezan en 1 → usa --base1")
        elif numeros[-1] >= len(origen) + (1 if base1 else 0):
            print(f"⚠ Hay ids mayores que las {len(origen)} clases de «obra.origen» en clases.yaml")
    # mosaico con los nombres que salen de «origen»: si una caja dice «drill» y es un martillo, el orden está mal
    tiles = []
    for zp, t, i in random.Random(1).sample(muestras, min(12, len(muestras))):
        with zipfile.ZipFile(zp) as z:
            im = leer_imagen(z, i)
            if im is None:
                continue
            h, w = im.shape[:2]
            for c, xc, yc, bw, bh in leer_cajas(z.read(t).decode("utf-8", "replace")):
                x0, y0 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
                x1, y1 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
                grosor = max(2, w // 300)
                cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), grosor)
                cv2.putText(im, str(nombre_clase(c, origen, base1)), (x0, max(y0 - 8, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, w / 900, (0, 255, 0), grosor)
        tiles.append(cv2.resize(im, (480, 360)))
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        mosaico = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
        cv2.imwrite(salida, mosaico)
        print(f"\n👀 Mira {salida}: el nombre de cada caja tiene que ser la herramienta que encierra.\n"
              f"   Si no cuadra, el orden de «obra.origen» en clases.yaml no es el suyo (o hay que usar --base1).")


def convertir(zips, origen, base1, salida, max_lado, maximo):
    for s in ("train", "val"):
        os.makedirs(os.path.join(salida, "images", s), exist_ok=True)
        os.makedirs(os.path.join(salida, "labels", s), exist_ok=True)
    with open(os.path.join(salida, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(names={i: n for i, n in enumerate(origen)}), f, allow_unicode=True, sort_keys=False)
    idx = {n: i for i, n in enumerate(origen)}
    st = Counter()
    for zp in zips:
        parte = re.sub(r"\W+", "", os.path.splitext(os.path.basename(zp))[0]).lower()
        with zipfile.ZipFile(zp) as z:
            pares = emparejar(z.namelist())
            print(f"📦 {os.path.basename(zp)}: {len(pares)} imágenes", flush=True)
            for k, (t, i) in enumerate(pares):
                if maximo and st["imágenes"] >= maximo:
                    break
                base = os.path.splitext(i)[0]
                nombre = "obra_" + parte + "_" + re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")
                split = particion(nombre)
                lab = os.path.join(salida, "labels", split, nombre + ".txt")
                if os.path.exists(lab):                 # ya convertida
                    st["imágenes"] += 1
                    continue
                lineas = []
                for c, xc, yc, bw, bh in leer_cajas(z.read(t).decode("utf-8", "replace")):
                    n = nombre_clase(c, origen, base1)
                    if n not in idx:
                        st["cajas de clase desconocida"] += 1
                        continue
                    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                        st["cajas fuera de la imagen"] += 1
                        continue
                    lineas.append(f"{idx[n]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                im = leer_imagen(z, i)
                if im is None:
                    st["imágenes ilegibles"] += 1
                    continue
                h, w = im.shape[:2]
                if max(h, w) > max_lado:
                    f = max_lado / max(h, w)
                    im = cv2.resize(im, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(salida, "images", split, nombre + ".jpg"), im,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(lab, "w") as f:               # la etiqueta, al final: si se corta, se rehace
                    f.write("\n".join(lineas) + ("\n" if lineas else ""))
                st["imágenes"] += 1
                st["cajas"] += len(lineas)
                if (k + 1) % 500 == 0:
                    print(f"   {k + 1}/{len(pares)}", flush=True)
    print(" · ".join(f"{k}: {v}" for k, v in st.items()))
    print(f"✅ {salida}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default=os.path.join(dir_datos(), "obra"))
    ap.add_argument("--salida", default=os.path.join(dir_datos(), "yolo_crudo", "obra"))
    ap.add_argument("--inspeccionar", action="store_true")
    ap.add_argument("--base1", action="store_true", help="los ids de clase de sus etiquetas empiezan en 1")
    ap.add_argument("--max-lado", type=int, default=1280)
    ap.add_argument("--max", type=int, default=0, help="como mucho N imágenes (para probar)")
    a = ap.parse_args()
    origen = [str(n).lower() for n in cargar_config()["obra"]["origen"]]
    zips = zips_completos(a.entrada)
    if not zips:
        raise SystemExit(f"✗ No hay ningún DATA*.zip completo en {a.entrada} (./yolo_herramientas/descargar.sh obra)")
    if a.inspeccionar:
        inspeccionar(zips, origen, a.base1, os.path.join(a.entrada, "revision_obra.jpg"))
    else:
        convertir(zips, origen, a.base1, a.salida, a.max_lado, a.max)


if __name__ == "__main__":
    main()
