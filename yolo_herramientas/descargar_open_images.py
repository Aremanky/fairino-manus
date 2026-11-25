#!/usr/bin/env python3
"""
Open Images V7 (Google): baja SOLO las imágenes con las herramientas de clases.yaml → open_images, con sus
cajas, mediante FiftyOne, y las deja en formato YOLO (cajas) en $HERRAMIENTAS_DATOS/yolo_crudo/open_images/.

    pip install fiftyone            # (lo hace descargar.sh open_images)
    python descargar_open_images.py
    python descargar_open_images.py --max 2000        # menos imágenes por split, para probar

Después: cajas_a_mascaras.py (SAM 2) y convertir_yolo_seg.py, o todo seguido con publicos.sh.

La primera vez baja los índices de Open Images (unos 3 GB: la lista de imágenes y la de cajas de train), que
con una conexión lenta tarda bastante. Se puede cortar con Ctrl+C: al relanzarlo sigue por donde iba.
"""
import argparse
import glob
import json
import os
import time

from comun import cargar_config, dir_datos


def _descarga_reanudable(original):
    """Sustituye a eta.core.web.download_file (lo que usa FiftyOne para los índices de Open Images).

    El original escribe directamente en el archivo final: si se corta, queda un CSV a medias que FiftyOne
    da por bueno en el siguiente intento. Este baja a <archivo>.part, sigue por donde iba (Range) y solo
    al terminar lo renombra al nombre final."""
    def descargar(url, path=None, chunk_size=None, quiet=False, **kw):
        if path is None:
            return original(url, path=path, chunk_size=chunk_size, quiet=quiet, **kw)
        import requests
        parte = path + ".part"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        hecho = os.path.getsize(parte) if os.path.exists(parte) else 0
        cab = {"Range": f"bytes={hecho}-"} if hecho else {}
        with requests.get(url, headers=cab, stream=True, timeout=60) as r:
            if hecho and r.status_code == 416:          # el .part ya estaba completo
                os.replace(parte, path)
                return
            r.raise_for_status()
            if hecho and r.status_code != 206:          # el servidor no deja seguir: desde cero
                hecho = 0
            total = hecho + int(r.headers.get("Content-Length") or 0)
            nombre = os.path.basename(path)
            if hecho:
                print(f"   ↻ {nombre}: sigo desde {hecho / 1e6:.0f} MB")
            t0, b0, ultimo = time.time(), hecho, 0.0
            with open(parte, "ab" if hecho else "wb") as f:
                for trozo in r.iter_content(chunk_size=1 << 20):
                    f.write(trozo)
                    hecho += len(trozo)
                    ahora = time.time()
                    if ahora - ultimo > 2 and total:
                        v = (hecho - b0) / max(ahora - t0, 1e-6)
                        resto = (total - hecho) / v / 60 if v else 0
                        print(f"\r   {nombre}: {hecho / 1e6:,.0f}/{total / 1e6:,.0f} MB · "
                              f"{v / 1e6:.1f} MB/s · faltan ~{resto:.0f} min   ", end="", flush=True)
                        ultimo = ahora
            if total:
                print()
        if total and hecho != total:
            raise IOError(f"{nombre}: incompleto ({hecho} de {total} B); relanza y seguirá por donde iba")
        os.replace(parte, path)
    return descargar


def _rescatar_cortados(raices):
    """Un CSV que no acaba en salto de línea se quedó a medias en una descarga cortada (antes de este
    arreglo): se renombra a .part para que la descarga siga desde ahí en vez de usarlo roto."""
    for raiz in raices:
        for p in glob.glob(os.path.join(raiz, "*", "metadata", "*.csv")) + \
                 glob.glob(os.path.join(raiz, "*", "labels", "*.csv")):
            try:
                with open(p, "rb") as f:
                    f.seek(-1, os.SEEK_END)
                    completo = f.read(1) == b"\n"
            except OSError:                             # vacío
                completo = False
            if not completo:
                print(f"   ✂ {p} se quedó a medias: se retomará la descarga")
                os.replace(p, p + ".part")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=None, help="imágenes como mucho por split (por defecto, clases.yaml)")
    ap.add_argument("--salida", default=os.path.join(dir_datos(), "yolo_crudo", "open_images"))
    ap.add_argument("--rehacer", action="store_true", help="exportar otra vez aunque ya esté hecho")
    a = ap.parse_args()
    cfg = cargar_config()["open_images"]
    maximo = a.max or cfg.get("max_por_split")
    pedido = dict(clases=list(cfg["clases"]), max=maximo)
    hecho = os.path.join(a.salida, ".completo.json")
    try:
        with open(hecho, encoding="utf-8") as f:
            ya = json.load(f) == pedido
    except (OSError, ValueError):
        ya = False
    if ya and not a.rehacer:
        print(f"✅ Open Images ya está en {a.salida} (--rehacer para repetirlo)")
        return

    import eta.core.web as etaw
    import fiftyone as fo
    import fiftyone.utils.openimages as fouo
    import fiftyone.zoo as foz

    etaw.download_file = _descarga_reanudable(etaw.download_file)
    # FiftyOne guarda lo descargado en su carpeta del zoo (~/fiftyone salvo que se cambie
    # fo.config.dataset_zoo_dir); lo que ya está ahí no se vuelve a bajar.
    _rescatar_cortados({os.path.join(fo.config.dataset_zoo_dir, "open-images-v7")})

    disponibles = set(fouo.get_classes())
    clases = [c for c in cfg["clases"] if c in disponibles]
    faltan = [c for c in cfg["clases"] if c not in disponibles]
    if faltan:
        print("⚠ No existen en Open Images y se saltan: " + ", ".join(faltan))
    print("Clases: " + ", ".join(clases))

    for split, nuestro in (("train", "train"), ("validation", "val")):
        # Sin «dataset_dir» (esta versión de FiftyOne lo pasa dos veces y falla) y sin «overwrite»
        # (borra la carpeta descargada y lo baja todo otra vez): solo se rehace la base de FiftyOne.
        nombre = f"herramientas_oi_{split}"
        print(f"   {split}: preparando… (la primera vez lee un índice de 2 GB y puede estar varios minutos "
              f"sin escribir nada: no lo cortes)", flush=True)
        if fo.dataset_exists(nombre):
            fo.delete_dataset(nombre)
        ds = foz.load_zoo_dataset(
            "open-images-v7", split=split, label_types=["detections"], classes=clases, only_matching=True,
            max_samples=maximo, shuffle=True, seed=0, dataset_name=nombre)
        esquema = ds.get_field_schema()
        campo = next((c for c in ("ground_truth", "detections") if c in esquema), None)
        if campo is None:
            campo = next(c for c, t in esquema.items() if "Detections" in str(t) and c != "segmentations")
        print(f"   {split}: {len(ds)} imágenes · cajas en el campo «{campo}»")
        ds.export(export_dir=a.salida, dataset_type=fo.types.YOLOv5Dataset, label_field=campo,
                  split=nuestro, classes=clases)
    with open(hecho, "w", encoding="utf-8") as f:
        json.dump(pedido, f)
    print(f"✅ {a.salida}")


if __name__ == "__main__":
    main()
