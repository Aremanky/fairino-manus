#!/usr/bin/env python3
"""
Baja los proyectos de segmentación de Roboflow Universe de la lista «roboflow» de clases.yaml, en formato
YOLO-seg, a <DATOS>/roboflow/<proyecto>/ y los convierte a nuestras clases (convertir_yolo_seg.py).

Hace falta una clave gratuita de Roboflow (https://app.roboflow.com → Settings → API Keys):
    export ROBOFLOW_API_KEY=xxxxxxxx
    python descargar_roboflow.py
"""
import os
import subprocess
import sys

from comun import AQUI, cargar_config, dir_datos


def main():
    clave = os.environ.get("ROBOFLOW_API_KEY")
    if not clave:
        sys.exit("✗ Falta la clave: export ROBOFLOW_API_KEY=… (app.roboflow.com → Settings → API Keys)")
    from roboflow import Roboflow
    rf = Roboflow(api_key=clave)
    cfg = cargar_config()
    for p in cfg.get("roboflow") or []:
        destino = os.path.join(dir_datos(), "roboflow", p["proyecto"])
        print(f"\n⬇ {p['workspace']}/{p['proyecto']} v{p['version']} → {destino}")
        try:
            proyecto = rf.workspace(p["workspace"]).project(p["proyecto"])
            proyecto.version(int(p["version"])).download("yolov8", location=destino, overwrite=True)
        except Exception as e:
            print(f"   ✗ No se ha podido bajar ({e}). Mira en la web del proyecto qué versiones tiene y "
                  f"corrige «version» en clases.yaml.")
            continue
        fuente = "rf_" + p["proyecto"].replace("-", "_")
        subprocess.run([sys.executable, os.path.join(AQUI, "convertir_yolo_seg.py"), destino, "--fuente", fuente],
                       check=False)


if __name__ == "__main__":
    main()
