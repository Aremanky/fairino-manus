#!/usr/bin/env python3
"""
HANDAL (NVIDIA, CC BY-NC-SA 4.0: solo investigación) desde su carpeta de Google Drive, y SOLO las
categorías de herramientas (martillos, alicates, destornilladores, llaves, carracas y taladros): las de
cocina no se bajan. Es la fuente principal del mango (trae la máscara del mango en cada imagen).

    python descargar_handal.py --listar          # qué hay en la carpeta y qué se bajaría
    python descargar_handal.py                   # baja, descomprime y borra cada zip
    python descargar_handal.py --conservar-zips  # igual, pero sin borrar los zip

Cada zip se descomprime sin las carpetas depth/ (no se usan y ocupan mucho) y después se borra: lo que
queda es lo que lee convertir_handal.py. Se puede cortar: al relanzarlo, lo terminado no se repite y la
descarga a medias sigue por donde iba.

Si Google corta («demasiadas descargas de este archivo»), relánzalo más tarde o bájalo a mano desde el
navegador y deja el zip en $HERRAMIENTAS_DATOS/handal/zips/: al relanzar se descomprime igual.
"""
import argparse
import fcntl
import glob
import os
import re
import shutil
import sys
import zipfile

from comun import dir_datos

URL = "https://drive.google.com/drive/folders/1ANTZAXDbZbszyxa7ZO9_DcFHe8tbdCIN"
HERRAMIENTA = re.compile(r"hammer|plier|screwdriver|wrench|ratchet|drill", re.I)
EXTRA = re.compile(r"model|mesh|handle|afford", re.I)          # por si los modelos van en un zip aparte
SIN_DEPTH = re.compile(r"(^|/)depth(/|$)")
MIN_LIBRE_GB = float(os.environ.get("HANDAL_MIN_LIBRE_GB", 40))    # no empezar un zip con menos sitio


def libre_gb(d):
    return shutil.disk_usage(d).free / 1e9


def listar():
    import gdown
    try:
        return gdown.download_folder(url=URL, skip_download=True, quiet=True) or []
    except Exception as e:      # noqa: BLE001 — cualquier fallo de gdown/Drive
        print(f"✗ No se ha podido leer la carpeta de Drive ({str(e)[:300]})")
        return None


def interesa(nombre):
    return nombre.lower().endswith(".zip") and bool(HERRAMIENTA.search(nombre) or EXTRA.search(nombre))


def descomprimir(z, destino):
    """Descomprime sin depth/. True si ha ido bien."""
    try:
        with zipfile.ZipFile(z) as zf:
            miembros = [m for m in zf.infolist() if not SIN_DEPTH.search(m.filename)]
            print(f"   📦 descomprimiendo {os.path.basename(z)}: {len(miembros)} archivos "
                  f"(sin depth/: {len(zf.infolist()) - len(miembros)} menos)", flush=True)
            for i, m in enumerate(miembros, 1):
                zf.extract(m, destino)
                if i % 20000 == 0:
                    print(f"      {i}/{len(miembros)}", flush=True)
        return True
    except (zipfile.BadZipFile, EOFError, OSError) as e:
        print(f"   ❌ {os.path.basename(z)} está cortado o roto ({e}): bórralo y relanza")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--conservar-zips", action="store_true")
    a = ap.parse_args()
    d = os.path.join(dir_datos(), "handal")
    zips = os.path.join(d, "zips")
    os.makedirs(zips, exist_ok=True)
    if not a.listar:                    # que no lo bajen dos terminales a la vez
        cerrojo = open(os.path.join(d, ".descargando"), "w")
        try:
            fcntl.flock(cerrojo, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("⏳ HANDAL ya se está bajando en otra terminal: aquí se salta")
            return

    archivos = listar()
    if a.listar:
        if archivos:
            print(f"Carpeta de HANDAL ({URL}):")
            for f in archivos:
                n = os.path.basename(f.path)
                print(f"   {'✓ se baja ' if interesa(n) else '· no      '} {f.path}")
        return

    def hecho(n):
        return os.path.join(d, f".{n}.ok")

    def ya(n):      # (.descomprimido: la marca que dejaba la versión anterior de descargar.sh)
        return os.path.exists(hecho(n)) or os.path.exists(os.path.join(d, f".{n}.descomprimido"))

    fallan = []
    if archivos is None:
        print(f"   Bájalos a mano desde {URL} (solo los de herramientas) y déjalos en {zips}/")
        archivos = []
    pendientes = [f for f in archivos if interesa(os.path.basename(f.path)) and not ya(os.path.basename(f.path))]
    if archivos:
        print(f"HANDAL: {len([f for f in archivos if interesa(os.path.basename(f.path))])} zip de herramientas "
              f"en Drive, {len(pendientes)} por bajar · libre: {libre_gb(d):.0f} GB")
    import gdown
    for f in pendientes:
        n = os.path.basename(f.path)
        z = os.path.join(zips, n)
        if not (os.path.exists(z) and zipfile.is_zipfile(z)):
            if libre_gb(d) < MIN_LIBRE_GB:
                print(f"✗ Quedan menos de {MIN_LIBRE_GB} GB libres: paro antes de bajar {n}")
                fallan.append(n)
                break
            print(f"\n⬇ {n}", flush=True)
            try:
                r = gdown.download(id=f.id, output=z, quiet=False, resume=True)
            except Exception as e:  # noqa: BLE001
                r = None
                print(f"   ✗ {str(e)[:400]}")
            if not r or not os.path.exists(z) or not zipfile.is_zipfile(z):
                print(f"   ✗ {n} no se ha podido bajar entero (si Google dice «demasiadas descargas», prueba "
                      f"más tarde o bájalo a mano a {zips}/)")
                fallan.append(n)
                continue
    # descomprimir lo que haya en zips/ (lo bajado ahora y lo que se haya dejado a mano)
    for z in sorted(glob.glob(os.path.join(zips, "*.zip"))):
        n = os.path.basename(z)
        if ya(n):
            continue
        if not zipfile.is_zipfile(z):
            print(f"   · {n} no está completo (¿se sigue bajando?): se salta")
            continue
        if descomprimir(z, d):
            open(hecho(n), "w").close()
            if not a.conservar_zips:
                os.remove(z)
        else:
            fallan.append(n)
    listos = len(glob.glob(os.path.join(d, ".*.ok")) + glob.glob(os.path.join(d, ".*.descomprimido")))
    print(f"\nHANDAL: {listos} categorías listas en {d} · libre: {libre_gb(d):.0f} GB")
    if fallan:
        print("⚠ Pendientes: " + ", ".join(fallan) + " (relanza para seguir)")
        sys.exit(1)
    print("Siguiente: python yolo_herramientas/convertir_handal.py --inspeccionar")


if __name__ == "__main__":
    main()
