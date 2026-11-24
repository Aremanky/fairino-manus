#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════════════════════
#  Entrenamiento SOLO con datasets públicos (sin fotos propias), en dos fases:
#
#   FASE 1 · «mango»: PACO + HANDAL, las únicas fuentes con el mango marcado → modelo que ya sabe qué es
#            un mango.
#   COMPLETAR: ese modelo pone el mango a las fuentes que solo traen la herramienta (METU-ALET, Open
#            Images, Roboflow…), dentro de cada herramienta etiquetada.
#   FASE 2 · «amplio»: todas las fuentes juntas → el modelo final con las 37 herramientas de clases.yaml.
#
#   ./yolo_herramientas/publicos.sh descargar    # PACO, METU-ALET, Open Images y Roboflow (HANDAL, a mano)
#   ./yolo_herramientas/publicos.sh convertir    # todo a nuestras clases (las cajas, con SAM 2)
#   ./yolo_herramientas/publicos.sh fase1        # une PACO + HANDAL y entrena mango_v1
#   ./yolo_herramientas/publicos.sh completar    # mango_v1 pone el mango a las demás fuentes
#   ./yolo_herramientas/publicos.sh fase2        # une todo y entrena herramientas_amplio
#   ./yolo_herramientas/publicos.sh todo         # convertir → fase1 → completar → fase2 (tras descargar)
#
#  Cada paso se puede repetir: lo ya bajado o convertido con SAM no se rehace.
#  Variables: EPOCAS1 (80), EPOCAS2 (150), MODELO1 (yolo26s-seg.pt), MODELO2 (el best.pt de la fase 1),
#             MAX_HANDAL (15000 imágenes de HANDAL en la fase 2), ROBOFLOW_API_KEY.
# ═══════════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"
YH="$RAIZ/yolo_herramientas"
PY="$RAIZ/venv_yolo/bin/python"
export HERRAMIENTAS_DATOS="${HERRAMIENTAS_DATOS:-$HOME/datasets_herramientas}"
D="$HERRAMIENTAS_DATOS"
CRUDO="$D/yolo_crudo"
MANGO_PT="runs/segment/mango_v1/weights/best.pt"

titulo() { echo; echo "━━━━━━━━ $* ━━━━━━━━"; }

descargar() {
    titulo "PACO"
    "$YH/descargar.sh" paco
    titulo "Dataset Ninja (METU-ALET…)"
    "$YH/descargar.sh" ninja
    titulo "Open Images"
    "$YH/descargar.sh" open_images
    titulo "Roboflow"
    if [ -n "${ROBOFLOW_API_KEY:-}" ]; then "$PY" "$YH/descargar_roboflow.py" --sin-convertir
    else echo "⚠️  sin ROBOFLOW_API_KEY: se salta Roboflow"; fi
    titulo "HANDAL"
    ls "$D"/handal/zips/*.zip >/dev/null 2>&1 || echo "⚠️  HANDAL se baja a mano: ./yolo_herramientas/descargar.sh handal (explica cómo)"
    "$YH/descargar.sh" handal </dev/null || true
}

cajas_y_convertir() {   # $1 carpeta YOLO cruda (cajas)   $2 fuente
    [ -d "$1" ] || { echo "· $2: no está ($1), se salta"; return; }
    "$PY" "$YH/cajas_a_mascaras.py" "$1" "$1_seg" && \
        "$PY" "$YH/convertir_yolo_seg.py" "$1_seg" --fuente "$2"
}

convertir() {
    titulo "PACO"
    "$PY" "$YH/convertir_paco.py"
    titulo "HANDAL"
    if [ -d "$D/handal" ] && [ -n "$(find "$D/handal" -name scene_gt.json -print -quit)" ]; then
        "$PY" "$YH/convertir_handal.py" --inspeccionar | tail -3
        "$PY" "$YH/convertir_handal.py" --cada 10
    else
        echo "⚠️  HANDAL no está descomprimido: sin él casi no habrá mangos para la fase 1"
    fi
    titulo "Dataset Ninja → cajas → SAM 2"
    for d in "$D"/ninja/*/; do
        [ -d "$d" ] || continue
        n="$(basename "$d" | tr 'A-Z-' 'a-z_')"
        "$PY" "$YH/convertir_supervisely.py" "$d" "$CRUDO/$n" && cajas_y_convertir "$CRUDO/$n" "$n"
    done
    titulo "Open Images → SAM 2"
    cajas_y_convertir "$CRUDO/open_images" open_images
    titulo "Roboflow"
    if [ -n "${ROBOFLOW_API_KEY:-}" ]; then "$PY" "$YH/descargar_roboflow.py"; else echo "· sin ROBOFLOW_API_KEY: se salta"; fi
    titulo "Revisión"
    for f in "$D"/convertidos/*/; do "$PY" "$YH/revisar.py" "$f" --n 24 | tail -1; done
    echo "👀 Mira los mosaicos de $D/convertidos/*/revision/ antes de entrenar"
}

fase1() {
    titulo "FASE 1: unir PACO + HANDAL"
    "$PY" "$YH/unir_datasets.py" --nombre mango_v1 --fuentes paco handal || exit 1
    titulo "FASE 1: entrenar mango_v1"
    MODELO="${MODELO1:-yolo26s-seg.pt}" EPOCAS="${EPOCAS1:-80}" "$YH/entrenar.sh" mango_v1
}

completar() {
    [ -f "$MANGO_PT" ] || { echo "❌ No está $MANGO_PT: primero la fase 1"; exit 1; }
    titulo "COMPLETAR el mango con mango_v1"
    for f in "$D"/convertidos/*/; do
        n="$(basename "$f")"
        case "$n" in paco|handal|handal_*|*_mango) continue ;; esac
        if grep -qs '"mango"' "$f/info.json"; then echo "· $n ya tiene mangos"; continue; fi
        "$PY" "$YH/preetiquetar.py" --modelo "$MANGO_PT" --completar-mango "$n"
        "$PY" "$YH/revisar.py" "$D/convertidos/${n}_mango" --n 24 | tail -1
    done
}

fase2() {
    titulo "FASE 2: unir todo"
    "$PY" "$YH/unir_datasets.py" --nombre herramientas_amplio --incluir-sin-mango \
        --max handal="${MAX_HANDAL:-15000}" || exit 1
    titulo "FASE 2: entrenar herramientas_amplio"
    local m="${MODELO2:-$MANGO_PT}"
    [ -f "$m" ] || m="yolo26m-seg.pt"
    MODELO="$m" EPOCAS="${EPOCAS2:-150}" "$YH/entrenar.sh" herramientas_amplio
}

case "${1:-}" in descargar|convertir|fase1|completar|fase2|todo)
    [ -x "$PY" ] || { echo "❌ Falta venv_yolo: ./yolo_herramientas/instalar.sh"; exit 1; } ;;
esac
case "${1:-}" in
    descargar) descargar ;;
    convertir) convertir ;;
    fase1) fase1 ;;
    completar) completar ;;
    fase2) fase2 ;;
    todo) convertir; fase1; completar; fase2 ;;
    *) sed -n '3,20p' "$YH/publicos.sh" | sed 's/^# \?//'; exit 2 ;;
esac
