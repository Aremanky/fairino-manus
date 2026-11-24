#!/usr/bin/env python3
"""
Open Images V7 (Google): baja SOLO las imágenes con las herramientas de clases.yaml → open_images, con sus
cajas, mediante FiftyOne, y las deja en formato YOLO (cajas) en $HERRAMIENTAS_DATOS/yolo_crudo/open_images/.

    pip install fiftyone            # (lo hace descargar.sh open_images)
    python descargar_open_images.py
    python descargar_open_images.py --max 2000        # menos imágenes por split, para probar

Después: cajas_a_mascaras.py (SAM 2) y convertir_yolo_seg.py, o todo seguido con publicos.sh.
"""
import argparse
import os

from comun import cargar_config, dir_datos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=None, help="imágenes como mucho por split (por defecto, clases.yaml)")
    ap.add_argument("--salida", default=os.path.join(dir_datos(), "yolo_crudo", "open_images"))
    a = ap.parse_args()

    import fiftyone as fo
    import fiftyone.utils.openimages as fouo
    import fiftyone.zoo as foz

    cfg = cargar_config()["open_images"]
    disponibles = set(fouo.get_classes())
    clases = [c for c in cfg["clases"] if c in disponibles]
    faltan = [c for c in cfg["clases"] if c not in disponibles]
    if faltan:
        print("⚠ No existen en Open Images y se saltan: " + ", ".join(faltan))
    print("Clases: " + ", ".join(clases))
    maximo = a.max or cfg.get("max_por_split")

    for split, nuestro in (("train", "train"), ("validation", "val")):
        ds = foz.load_zoo_dataset(
            "open-images-v7", split=split, label_types=["detections"], classes=clases, only_matching=True,
            max_samples=maximo, shuffle=True, seed=0,
            dataset_dir=os.path.join(dir_datos(), "fiftyone", "open-images-v7"),
            dataset_name=f"herramientas_oi_{split}", overwrite=True)
        esquema = ds.get_field_schema()
        campo = next((c for c in ("ground_truth", "detections") if c in esquema), None)
        if campo is None:
            campo = next(c for c, t in esquema.items() if "Detections" in str(t) and c != "segmentations")
        print(f"   {split}: {len(ds)} imágenes · cajas en el campo «{campo}»")
        ds.export(export_dir=a.salida, dataset_type=fo.types.YOLOv5Dataset, label_field=campo,
                  split=nuestro, classes=clases)
    print(f"✅ {a.salida}")


if __name__ == "__main__":
    main()
