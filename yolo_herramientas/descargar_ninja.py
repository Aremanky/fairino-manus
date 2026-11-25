#!/usr/bin/env python3
"""
Dataset Ninja (formato Supervisely): baja y descomprime los datasets de clases.yaml → dataset_ninja en
$HERRAMIENTAS_DATOS/ninja/<nombre>/ (METU-ALET → ninja/metu-alet/).

    python descargar_ninja.py          # (lo hace descargar.sh ninja)

Primero mira si ya hay un .tar/.zip bueno en la carpeta (por ejemplo, bajado a mano desde la web); si no,
lo intenta con dataset-tools. Si la descarga sale vacía o rota la borra y explica cómo bajarlo a mano,
y sigue con el siguiente: un dataset que falle no para a los demás.
"""
import glob
import os
import re
import shutil
import sys
import tarfile
import zipfile

from comun import cargar_config, dir_datos

# Página de cada dataset en Dataset Ninja (botón «Download» → formato Supervisely)
PAGINAS = {"METU-ALET": "https://datasetninja.com/alet"}
MIN_BYTES = 1 << 20     # un archivo de menos de 1 MB no es un dataset: es un error del servidor


def carpeta(destino, nombre):
    return os.path.join(destino, re.sub(r"[^a-z0-9]+", "-", nombre.lower()).strip("-"))


def anotaciones(d):
    return glob.glob(os.path.join(d, "**", "ann", "*.json"), recursive=True)


def archivos(d):
    return [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
            if os.path.isfile(p) and re.search(r"\.(tar|tar\.gz|tgz|zip)$", p, re.I)]


def es_valido(p):
    if os.path.getsize(p) < MIN_BYTES:
        return False
    try:
        return zipfile.is_zipfile(p) if p.lower().endswith(".zip") else tarfile.is_tarfile(p)
    except OSError:
        return False


def descomprimir(p):
    """True si se descomprime entero. Un .tar cortado (descarga incompleta) solo falla al llegar al final:
    entonces se aparta como .incompleto y se borra lo que se hubiera sacado, para no dejarlo a medias."""
    dst = os.path.dirname(p)
    antes = set(os.listdir(dst))
    try:
        if p.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                z.extractall(dst)
        else:
            with tarfile.open(p) as tf:
                try:
                    tf.extractall(dst, filter="data")
                except TypeError:           # Python < 3.12
                    tf.extractall(dst)
    except (tarfile.TarError, zipfile.BadZipFile, EOFError, OSError) as e:
        for n in set(os.listdir(dst)) - antes:
            r = os.path.join(dst, n)
            shutil.rmtree(r) if os.path.isdir(r) else os.remove(r)
        os.replace(p, p + ".incompleto")
        print(f"   ❌ {os.path.basename(p)} está cortado ({e}): la descarga no terminó. Se aparta como "
              f"{os.path.basename(p)}.incompleto")
        return False
    os.remove(p)
    return True


def buenos(d):
    """Archivos que valen; borra los vacíos o rotos (de una descarga fallida)."""
    ok = []
    for p in archivos(d):
        if es_valido(p):
            ok.append(p)
        else:
            print(f"   🗑 {os.path.basename(p)} está vacío o roto ({os.path.getsize(p)} B): se borra")
            with open(p, "rb") as f:
                cabeza = f.read(400)
            texto = cabeza.decode("utf-8", "replace")
            if cabeza and sum(ch.isprintable() or ch in "\n\r\t" for ch in texto) > 0.9 * len(texto):
                print("      en vez del dataset, el servidor ha mandado esto:\n      "
                      + " ".join(texto.split())[:300])
            os.remove(p)
    return ok


def bajar_con_dataset_tools(nombre, d):
    try:
        import dataset_tools as dtools
    except ImportError:
        print("   (dataset-tools no está instalado)")
        return
    try:
        dtools.download(dataset=nombre, dst_dir=d)
    except Exception as e:      # noqa: BLE001 — cualquier fallo de la librería se trata igual
        print(f"   dataset-tools ha fallado: {type(e).__name__}: {e}")


def main():
    destino = os.path.join(dir_datos(), "ninja")
    fallan = []
    for nombre in cargar_config().get("dataset_ninja") or []:
        d = carpeta(destino, nombre)
        os.makedirs(d, exist_ok=True)
        hecho = os.path.join(d, ".descomprimido")
        if os.path.exists(hecho):
            print(f"✅ {nombre} ya está en {d}")
            continue
        print(f"⬇ {nombre} → {d}")
        ok = buenos(d)
        if not ok:
            bajar_con_dataset_tools(nombre, d)
            ok = buenos(d)
        if not ok:
            pagina = PAGINAS.get(nombre, f"https://datasetninja.com/ (busca «{nombre}»)")
            print(f"❌ {nombre}: la descarga automática no ha traído nada. Bájalo a mano:\n"
                  f"     1. Abre {pagina} → botón «Download» → formato Supervisely (el completo, no el «sample»).\n"
                  f"     2. Deja el .tar tal cual en  {d}/\n"
                  f"     3. Vuelve a lanzar  ./yolo_herramientas/descargar.sh ninja  (lo comprueba y lo descomprime)")
            fallan.append(nombre)
            continue
        enteros = True
        for p in ok:
            print(f"   descomprimiendo {os.path.basename(p)}…")
            enteros = descomprimir(p) and enteros
        if not enteros:
            fallan.append(nombre)
            continue
        open(hecho, "w").close()
        n = len(anotaciones(d))
        print(f"   {n} anotaciones" if n else f"⚠ {nombre}: descomprimido, pero sin carpetas ann/ con .json "
                                            f"(¿no es formato Supervisely?)")
    if fallan:
        print("⚠ Sin bajar: " + ", ".join(fallan) + " (el resto del flujo funciona igual, solo con menos datos)")
        sys.exit(1)


if __name__ == "__main__":
    main()
