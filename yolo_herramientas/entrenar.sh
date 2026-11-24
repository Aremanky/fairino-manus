#!/usr/bin/env bash
# Entrena el YOLO de segmentación (herramientas + mango) en la RTX 5080.
#
#   ./yolo_herramientas/entrenar.sh herramientas_v1                 # yolo26s-seg, 150 épocas, 960 px
#   MODELO=yolo26m-seg.pt EPOCAS=200 ./yolo_herramientas/entrenar.sh herramientas_v2
#   BATCH=8 ./yolo_herramientas/entrenar.sh herramientas_v1         # si se queda sin memoria de la GPU
#
# Resultado: runs/segment/<nombre>/weights/best.pt (+ gráficas y matriz de confusión en la misma carpeta).
# Se puede cortar con Ctrl+C y seguir luego con:  yolo segment train resume model=runs/segment/<nombre>/weights/last.pt
set -euo pipefail
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"
NOMBRE="${1:?di el nombre del dataset (el de unir_datasets.py --nombre)}"
DATOS="${HERRAMIENTAS_DATOS:-$HOME/datasets_herramientas}"
YAML="$DATOS/yolo/$NOMBRE/$NOMBRE.yaml"
[ -f "$YAML" ] || { echo "❌ No está $YAML: primero  python yolo_herramientas/unir_datasets.py --nombre $NOMBRE"; exit 1; }
source venv_yolo/bin/activate

# imgsz 960: los mangos son finos y a ~1 m ocupan pocos píxeles; 640 se queda corto.
# degrees/flipud: la herramienta puede estar en cualquier orientación sobre la mesa.
# overlap_mask=False: IMPRESCINDIBLE. El mango está DENTRO de la herramienta; con True (lo de serie) YOLO junta
# todas las máscaras en una y la herramienta se quedaría con un agujero donde está el mango.
yolo segment train \
    model="${MODELO:-yolo26s-seg.pt}" \
    data="$YAML" \
    epochs="${EPOCAS:-150}" \
    imgsz="${IMGSZ:-960}" \
    batch="${BATCH:-16}" \
    device=0 \
    workers=8 \
    patience=40 \
    cos_lr=True \
    close_mosaic=15 \
    degrees=15 \
    flipud=0.3 \
    overlap_mask=False \
    project=runs/segment \
    name="$NOMBRE" \
    exist_ok=True \
    plots=True

echo
echo "✅ Modelo: runs/segment/$NOMBRE/weights/best.pt"
echo "   Evaluar:  yolo segment val model=runs/segment/$NOMBRE/weights/best.pt data=$YAML imgsz=${IMGSZ:-960}"
