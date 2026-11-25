#!/usr/bin/env python3
"""
Baja los proyectos de Roboflow Universe de la lista «roboflow» de clases.yaml en formato YOLO a
$HERRAMIENTAS_DATOS/yolo_crudo/rf_<proyecto>/ y los convierte a nuestras clases: las cajas pasan por
SAM 2 (cajas_a_mascaras.py, que deja los polígonos como vienen) y después convertir_yolo_seg.py.

Hace falta la clave PRIVADA de Roboflow (https://app.roboflow.com → Settings → API Keys → «Private API
Key»; la que empieza por «rf_» es la publicable y no sirve para descargar):
    export ROBOFLOW_API_KEY=xxxxxxxx
    python descargar_roboflow.py                 # bajar lo que falte y convertir
    python descargar_roboflow.py --sin-convertir # solo bajar

Sin «version» en clases.yaml se baja la última versión del proyecto. Un proyecto que no exista o no se
pueda bajar se salta con un aviso y no para a los demás.
"""
import argparse
import json
import os
import subprocess
import sys

from comun import AQUI, cargar_config, dir_datos

API = "https://api.roboflow.com"


class ClaveMala(Exception):
    pass


def versiones(workspace, proyecto, clave):
    """Números de versión publicados del proyecto (lista vacía si no tiene); None si no existe o es privado."""
    import requests
    r = requests.get(f"{API}/{workspace}/{proyecto}", params={"api_key": clave}, timeout=60)
    if r.status_code == 401:
        raise ClaveMala(r.text[:300])
    if r.status_code != 200:
        return None
    vs = []
    for v in r.json().get("versions") or []:
        try:
            vs.append(int(str(v.get("id", "")).rstrip("/").rsplit("/", 1)[-1]))
        except ValueError:
            pass
    return sorted(vs)


def ya_bajado(crudo):
    """Versión ya bajada (de su .completo.json), 0 si se bajó con la versión anterior de este script
    (hay data.yaml y etiquetas) y None si no está o quedó a medias."""
    try:
        with open(os.path.join(crudo, ".completo.json"), encoding="utf-8") as f:
            return json.load(f)["version"]
    except (OSError, ValueError, KeyError):
        pass
    hay_yaml = os.path.exists(os.path.join(crudo, "data.yaml"))
    hay_txt = any(n.endswith(".txt") for _, _, fs in os.walk(crudo) for n in fs)
    return 0 if hay_yaml and hay_txt else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sin-convertir", action="store_true")
    a = ap.parse_args()
    clave = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not clave:
        sys.exit("✗ Falta la clave: export ROBOFLOW_API_KEY=… (app.roboflow.com → Settings → API Keys → "
                 "Private API Key)")
    publicable = clave.startswith("rf_")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=clave)
    except Exception as e:      # noqa: BLE001 — la librería lanza RuntimeError con el JSON del servidor
        sys.exit(f"✗ Roboflow no acepta la clave ({str(e)[:200]}). "
                 + ("Esa parece la clave publicable (empieza por «rf_»), que no sirve para descargar. "
                    if publicable else "")
                 + "Copia la «Private API Key» de app.roboflow.com → Settings → API Keys")

    fallan = []
    for p in cargar_config().get("roboflow") or []:
        ws, pr = p["workspace"], p["proyecto"]
        fuente = "rf_" + pr.replace("-", "_")
        crudo = os.path.join(dir_datos(), "yolo_crudo", fuente)
        pedida = p.get("version")
        bajada = ya_bajado(crudo)
        if bajada is not None and (pedida is None or bajada in (0, int(pedida))):
            print(f"\n✅ {ws}/{pr} ya está en {crudo}")
        else:
            try:
                vs = versiones(ws, pr, clave)
            except ClaveMala as e:
                sys.exit(f"✗ Roboflow rechaza la clave ({e}). Usa la «Private API Key».")
            except Exception as e:  # noqa: BLE001 — red
                print(f"\n✗ {ws}/{pr}: no se ha podido consultar ({e}); se salta")
                fallan.append(f"{ws}/{pr}")
                continue
            if vs is None:
                print(f"\n✗ {ws}/{pr}: el proyecto no existe o no es público; quítalo de clases.yaml")
                fallan.append(f"{ws}/{pr}")
                continue
            if not vs:
                print(f"\n✗ {ws}/{pr}: no tiene ninguna versión publicada; se salta")
                fallan.append(f"{ws}/{pr}")
                continue
            if pedida is not None and int(pedida) not in vs:
                print(f"\n⚠ {ws}/{pr}: no tiene la versión {pedida} (tiene {vs}); se baja la {vs[-1]}")
            v = int(pedida) if pedida is not None and int(pedida) in vs else vs[-1]
            print(f"\n⬇ {ws}/{pr} v{v} → {crudo}")
            try:
                rf.workspace(ws).project(pr).version(v).download("yolov8", location=crudo, overwrite=True)
            except Exception as e:  # noqa: BLE001
                print(f"   ✗ No se ha podido bajar ({str(e)[:300]})")
                fallan.append(f"{ws}/{pr}")
                continue
            with open(os.path.join(crudo, ".completo.json"), "w", encoding="utf-8") as f:
                json.dump(dict(workspace=ws, proyecto=pr, version=v), f)
        if a.sin_convertir:
            continue
        # cajas → máscaras con SAM (los polígonos se dejan igual) y a nuestras clases
        subprocess.run([sys.executable, os.path.join(AQUI, "cajas_a_mascaras.py"), crudo, crudo + "_seg"],
                       check=False)
        subprocess.run([sys.executable, os.path.join(AQUI, "convertir_yolo_seg.py"), crudo + "_seg",
                        "--fuente", fuente], check=False)
    if fallan:
        print("\n⚠ Sin bajar: " + ", ".join(fallan) + " (los demás sí; el resto del flujo funciona igual)")


if __name__ == "__main__":
    main()
