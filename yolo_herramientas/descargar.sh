#!/usr/bin/env bash
# Descarga de los datasets públicos a $HERRAMIENTAS_DATOS (por defecto ~/datasets_herramientas).
#
#   ./yolo_herramientas/descargar.sh paco        # anotaciones de PACO-LVIS (las imágenes las baja convertir_paco.py)
#   ./yolo_herramientas/descargar.sh handal      # HANDAL desde Google Drive (MUCHOS GB: ver abajo)
#   ./yolo_herramientas/descargar.sh roboflow    # proyectos de Roboflow de clases.yaml (hace falta ROBOFLOW_API_KEY)
#   ./yolo_herramientas/descargar.sh ninja       # Dataset Ninja de clases.yaml (METU-ALET: 49 herramientas, ~2,7 GB)
#   ./yolo_herramientas/descargar.sh open_images # Open Images V7: solo las imágenes con herramientas (FiftyOne)
#   ./yolo_herramientas/descargar.sh coco        # (opcional) COCO train2017+val2017 enteros, 19 GB
set -uo pipefail
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$RAIZ/venv_yolo/bin/python"
DATOS="${HERRAMIENTAS_DATOS:-$HOME/datasets_herramientas}"
mkdir -p "$DATOS"
echo "📂 Datos en $DATOS  (libre: $(df -h "$DATOS" | awk 'NR==2{print $4}'))"

paco() {
    local d="$DATOS/paco/annotations"
    mkdir -p "$d" && cd "$d" || exit 1
    if ls paco_lvis_v1_*.json >/dev/null 2>&1; then echo "✅ PACO ya está en $d"; return; fi
    wget -c https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip || exit 1
    echo "02ac4edb22c251e07853e6231d69aec3fad0a180f03de2f8c880650322debc80  paco_lvis_v1.zip" | sha256sum -c - \
        || { echo "❌ el zip no coincide con el de PACO (descarga corrupta: bórralo y repite)"; exit 1; }
    unzip -o -q paco_lvis_v1.zip
    find . -mindepth 2 -name 'paco_lvis_v1_*.json' -exec mv {} . \;
    ls -lh paco_lvis_v1_*.json && echo "✅ PACO listo"
}

handal() {
    local d="$DATOS/handal"
    local url="https://drive.google.com/drive/folders/1ANTZAXDbZbszyxa7ZO9_DcFHe8tbdCIN"
    mkdir -p "$d/zips"
    cat <<EOF
HANDAL está en Google Drive ($url), un zip por categoría, y entero pesa MUCHO.
Solo hacen falta las categorías de herramientas: hammer, pliers (las 3), power drill, ratchet,
screwdriver y wrench (las 2). Lo más fiable: abrir el enlace en el navegador, bajar esos zips y
dejarlos en  $d/zips/  ; después, volver a lanzar  ./yolo_herramientas/descargar.sh handal  para
descomprimirlos.
EOF
    if ! ls "$d"/zips/*.zip >/dev/null 2>&1; then
        read -r -p "¿Intento bajar la carpeta ENTERA con gdown (todas las categorías, incluidas las de cocina)? [s/N] " r
        if [[ "$r" =~ ^[sS]$ ]]; then
            "$PY" -m gdown --folder "$url" -O "$d/zips" --remaining-ok || echo "⚠️  gdown ha fallado (límite de Google): bájalos a mano"
        fi
    fi
    for z in "$d"/zips/*.zip; do
        [ -f "$z" ] || continue
        local hecho="$d/.$(basename "$z").descomprimido"
        [ -f "$hecho" ] && continue
        echo "📦 $(basename "$z")"
        unzip -o -q "$z" -d "$d" && touch "$hecho"
    done
    echo "Ahora:  $PY yolo_herramientas/convertir_handal.py --inspeccionar"
}

roboflow() {
    [ -n "${ROBOFLOW_API_KEY:-}" ] || { echo "❌ export ROBOFLOW_API_KEY=… (app.roboflow.com → Settings → API Keys)"; exit 1; }
    cd "$RAIZ/yolo_herramientas" && "$PY" descargar_roboflow.py
}

ninja() {
    "$PY" -m pip install -q --upgrade dataset-tools || { echo "❌ no se pudo instalar dataset-tools"; exit 1; }
    "$PY" - "$DATOS/ninja" "$RAIZ/yolo_herramientas" <<'EOF2'
import glob, os, re, sys, tarfile
destino, yh = sys.argv[1], sys.argv[2]
sys.path.insert(0, yh)
from comun import cargar_config
import dataset_tools as dtools
for nombre in cargar_config().get("dataset_ninja") or []:
    d = os.path.join(destino, re.sub(r"[^a-z0-9]+", "-", nombre.lower()).strip("-"))
    if glob.glob(os.path.join(d, "**", "ann", "*.json"), recursive=True):
        print(f"✅ {nombre} ya está en {d}")
        continue
    print(f"⬇ {nombre} → {d}")
    os.makedirs(d, exist_ok=True)
    dtools.download(dataset=nombre, dst_dir=d)
    for t in glob.glob(os.path.join(d, "**", "*.tar"), recursive=True):
        with tarfile.open(t) as tf:
            try:
                tf.extractall(os.path.dirname(t), filter="data")
            except TypeError:
                tf.extractall(os.path.dirname(t))
        os.remove(t)
    print(f"   {len(glob.glob(os.path.join(d, '**', 'ann', '*.json'), recursive=True))} anotaciones")
EOF2
}

open_images() {
    "$PY" -c "import fiftyone" 2>/dev/null || "$PY" -m pip install -q fiftyone || { echo "❌ no se pudo instalar fiftyone"; exit 1; }
    cd "$RAIZ/yolo_herramientas" && "$PY" descargar_open_images.py
}

coco() {
    mkdir -p "$DATOS/coco" && cd "$DATOS/coco" || exit 1
    for s in train2017 val2017; do
        [ -d "$s" ] && [ "$(ls "$s" | wc -l)" -gt 4000 ] && { echo "✅ $s ya está"; continue; }
        wget -c "http://images.cocodataset.org/zips/$s.zip" && unzip -q -o "$s.zip"
    done
}

case "${1:-}" in
    paco) paco ;;
    handal) handal ;;
    roboflow) roboflow ;;
    ninja) ninja ;;
    open_images) open_images ;;
    coco) coco ;;
    *) sed -n '2,9p' "$0" | sed 's/^# \?//'; exit 2 ;;
esac
