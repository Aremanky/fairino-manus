#!/usr/bin/env python3
"""
Baja los proyectos de Roboflow Universe de la lista «roboflow» de clases.yaml en formato YOLO a
$HERRAMIENTAS_DATOS/yolo_crudo/rf_<proyecto>/ y los convierte a nuestras clases:
    tipo: seg   → convertir_yolo_seg.py directamente
    tipo: cajas → cajas_a_mascaras.py (SAM 2) y después convertir_yolo_seg.py

Hace falta una clave gratuita de Roboflow (https://app.roboflow.com → Settings → API Keys):
    export ROBOFLOW_API_KEY=xxxxxxxx
    python descargar_roboflow.py                 # todos
    python descargar_roboflow.py --sin-convertir # solo bajar
"""
import argparse
import os
import subprocess
import sys

from comun import AQUI, cargar_config, dir_datos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sin-convertir", action="store_true")
    a = ap.parse_args()
    clave = os.environ.get("ROBOFLOW_API_KEY")
    if not clave:
        sys.exit("✗ Falta la clave: export ROBOFLOW_API_KEY=… (app.roboflow.com → Settings → API Keys)")
    from roboflow import Roboflow
    rf = Roboflow(api_key=clave)
    for p in cargar_config().get("roboflow") or []:
        fuente = "rf_" + p["proyecto"].replace("-", "_")
        crudo = os.path.join(dir_datos(), "yolo_crudo", fuente)
        print(f"\n⬇ {p['workspace']}/{p['proyecto']} v{p['version']} ({p.get('tipo', 'seg')}) → {crudo}")
        if not os.path.isdir(crudo):
            try:
                rf.workspace(p["workspace"]).project(p["proyecto"]).version(int(p["version"])).download(
                    "yolov8", location=crudo, overwrite=True)
            except Exception as e:
                print(f"   ✗ No se ha podido bajar ({e}). Mira en la web del proyecto qué versiones tiene y "
                      f"corrige «version» en clases.yaml.")
                continue
        if a.sin_convertir:
            continue
        origen = crudo
        if p.get("tipo") == "cajas":
            origen = crudo + "_seg"
            subprocess.run([sys.executable, os.path.join(AQUI, "cajas_a_mascaras.py"), crudo, origen], check=False)
        subprocess.run([sys.executable, os.path.join(AQUI, "convertir_yolo_seg.py"), origen, "--fuente", fuente],
                       check=False)


if __name__ == "__main__":
    main()
