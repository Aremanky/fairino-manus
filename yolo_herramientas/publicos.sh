#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════════════════════
#  Entrenamiento SOLO con datasets públicos (sin fotos propias) de HERRAMIENTAS DE OBRA E INSTALACIONES
#  (las 25 de clases.yaml + el mango), en dos fases:
#
#   FASE 1 · «mango»: PACO + HANDAL, las únicas fuentes con el mango marcado → modelo que ya sabe qué es
#            un mango.
#   COMPLETAR: ese modelo pone el mango a las fuentes que solo traen la herramienta (el de obra, Open
#            Images, Roboflow…), dentro de cada herramienta etiquetada.
#   FASE 2 · «obra»: todas las fuentes juntas → el modelo final (herramientas_obra).
#
#   ./yolo_herramientas/publicos.sh descargar    # PACO, Open Images, Roboflow, HANDAL y el de obra
#                                                # (lo que falte; HANDAL y obra son los largos)
#   ./yolo_herramientas/publicos.sh resumen      # qué está ya bajado y qué falta
#   ./yolo_herramientas/publicos.sh convertir    # todo a nuestras clases (las cajas, con SAM 2)
#   ./yolo_herramientas/publicos.sh fase1        # une PACO + HANDAL y entrena mango_v1
#   ./yolo_herramientas/publicos.sh completar    # mango_v1 pone el mango a las demás fuentes
#   ./yolo_herramientas/publicos.sh fase2        # une todo y entrena herramientas_obra
#   ./yolo_herramientas/publicos.sh todo         # convertir → fase1 → completar → fase2 (tras descargar)
#
#  Cada paso se puede repetir: lo ya bajado o convertido con SAM no se rehace.
#  Variables: EPOCAS1 (80), EPOCAS2 (150), MODELO1 (yolo26s-seg.pt), MODELO2 (el best.pt de la fase 1),
#             MAX_HANDAL (15000 imágenes de HANDAL en la fase 2), ROBOFLOW_API_KEY,
#             OBRA_PARTES ("1 2 3 4": zip del dataset de obra), OBRA_BASE1 (=1 si sus ids empiezan en 1).
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

descargar() {   # lo pequeño primero; HANDAL y el de obra, al final (son los largos)
    titulo "PACO"
    "$YH/descargar.sh" paco
    if [ -n "$(ninja_lista)" ]; then
        titulo "Dataset Ninja"
        "$YH/descargar.sh" ninja
    fi
    titulo "Open Images"
    "$YH/descargar.sh" open_images
    titulo "Roboflow"
    if [ -n "${ROBOFLOW_API_KEY:-}" ]; then "$PY" "$YH/descargar_roboflow.py" --sin-convertir
    else echo "⚠️  sin ROBOFLOW_API_KEY: se salta Roboflow"; fi
    titulo "HANDAL (Google Drive: solo herramientas; la fuente principal del mango)"
    "$YH/descargar.sh" handal
    titulo "Small Size Construction Tools (Zenodo, ~110 GB)"
    "$YH/descargar.sh" obra
    resumen
}

ninja_lista() {
    "$PY" -c "import sys; sys.path.insert(0, '$YH')
from comun import cargar_config; print(*(cargar_config().get('dataset_ninja') or []))"
}

resumen() {   # qué hay ya bajado; relanzar «descargar» solo baja lo que falte
    titulo "RESUMEN ($D)"
    local s n
    ls "$D"/paco/annotations/paco_lvis_v1_*.json >/dev/null 2>&1 && s="✅" || s="❌"; echo "$s PACO"
    [ -f "$CRUDO/open_images/.completo.json" ] && s="✅" || s="❌"; echo "$s Open Images"
    n=$(for x in "$CRUDO"/rf_*/; do case "$x" in *_seg/) continue ;; esac; [ -f "$x/data.yaml" ] && echo "$x"; done | wc -l)
    [ "$n" -gt 0 ] && s="✅" || s="·"; echo "$s Roboflow: $n proyectos (hace falta ROBOFLOW_API_KEY)"
    n=$(ls -a "$D"/handal/ 2>/dev/null | grep -cE '^\..*\.(ok|descomprimido)$')
    [ "$n" -gt 0 ] && s="✅" || s="❌"; echo "$s HANDAL: $n categorías descomprimidas"
    local n_ok n_tot
    n_ok=$(ls "$D"/obra/DATA*.zip.ok 2>/dev/null | wc -l)
    n_tot=$(echo ${OBRA_PARTES:-$("$PY" -c "import sys; sys.path.insert(0, '$YH')
from comun import cargar_config; print(*cargar_config()['obra']['partes'])")} | wc -w)
    [ "$n_ok" -ge "$n_tot" ] && s="✅" || s="⏳"
    echo "$s Small Size Construction Tools: $n_ok de $n_tot zip"
    for n in $(ninja_lista); do
        local c="$D/ninja/$(echo "$n" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]\+/-/g; s/^-//; s/-$//')"
        [ -f "$c/.descomprimido" ] && s="✅" || s="❌"; echo "$s $n (Dataset Ninja)"
    done
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
        [ -f "$d/.descomprimido" ] || continue          # solo los bajados enteros
        n="$(basename "$d" | tr 'A-Z-' 'a-z_')"
        "$PY" "$YH/convertir_supervisely.py" "$d" "$CRUDO/$n" && cajas_y_convertir "$CRUDO/$n" "$n"
    done
    titulo "Small Size Construction Tools → SAM 2"
    # los zip, una vez convertidos, se pueden borrar (dejando los .ok): lo convertido queda en yolo_crudo/obra
    if ls "$D"/obra/DATA*.zip >/dev/null 2>&1; then
        "$PY" "$YH/convertir_obra.py" ${OBRA_BASE1:+--base1} || echo "⚠️  convertir_obra.py ha fallado"
    fi
    if [ -d "$CRUDO/obra" ]; then cajas_y_convertir "$CRUDO/obra" obra
    else echo "· no está bajado ($D/obra): se salta"; fi
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
        if "$PY" -c "import json,sys; i=json.load(open(sys.argv[1])); sys.exit(0 if any(s.get('mango') for s in i.get('instancias',{}).values()) else 1)" "$f/info.json" 2>/dev/null; then
            echo "· $n ya tiene mangos"; continue
        fi
        "$PY" "$YH/preetiquetar.py" --modelo "$MANGO_PT" --completar-mango "$n"
        "$PY" "$YH/revisar.py" "$D/convertidos/${n}_mango" --n 24 | tail -1
    done
}

fase2() {
    titulo "FASE 2: unir todo"
    "$PY" "$YH/unir_datasets.py" --nombre herramientas_obra --incluir-sin-mango \
        --max handal="${MAX_HANDAL:-15000}" || exit 1
    titulo "FASE 2: entrenar herramientas_obra"
    local m="${MODELO2:-$MANGO_PT}"
    [ -f "$m" ] || m="yolo26m-seg.pt"
    MODELO="$m" EPOCAS="${EPOCAS2:-150}" "$YH/entrenar.sh" herramientas_obra
}

case "${1:-}" in descargar|resumen|convertir|fase1|completar|fase2|todo)
    [ -x "$PY" ] || { echo "❌ Falta venv_yolo: ./yolo_herramientas/instalar.sh"; exit 1; } ;;
esac
case "${1:-}" in
    descargar) descargar ;;
    resumen) resumen ;;
    convertir) convertir ;;
    fase1) fase1 ;;
    completar) completar ;;
    fase2) fase2 ;;
    todo) convertir; fase1; completar; fase2 ;;
    *) sed -n '3,24p' "$YH/publicos.sh" | sed 's/^# \?//'; exit 2 ;;
esac
