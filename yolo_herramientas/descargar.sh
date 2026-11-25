#!/usr/bin/env bash
# Descarga de los datasets públicos a $HERRAMIENTAS_DATOS (por defecto ~/datasets_herramientas).
#
#   ./yolo_herramientas/descargar.sh paco        # anotaciones de PACO-LVIS (las imágenes las baja convertir_paco.py)
#   ./yolo_herramientas/descargar.sh handal      # HANDAL desde Google Drive: solo herramientas, sin depth/
#   ./yolo_herramientas/descargar.sh roboflow    # proyectos de Roboflow de clases.yaml (hace falta ROBOFLOW_API_KEY)
#   ./yolo_herramientas/descargar.sh ninja       # Dataset Ninja de clases.yaml (ahora ninguno: METU-ALET está roto)
#   ./yolo_herramientas/descargar.sh open_images # Open Images V7: solo las imágenes con herramientas (FiftyOne)
#   ./yolo_herramientas/descargar.sh obra        # Small Size Construction Tools (Zenodo): 4 zip, ~110 GB,
#                                                #   a la vez (OBRA_PARTES="1 2": solo esos; OBRA_PARALELO=1: de uno en uno)
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
    # Solo las categorías de herramientas, sin depth/ y borrando cada zip al descomprimirlo
    # (descargar_handal.py --listar para ver qué hay; --conservar-zips para no borrarlos)
    "$PY" -c "import gdown" 2>/dev/null || "$PY" -m pip install -q gdown \
        || { echo "❌ no se pudo instalar gdown"; return 1; }
    cd "$RAIZ/yolo_herramientas" && "$PY" descargar_handal.py
}

roboflow() {
    [ -n "${ROBOFLOW_API_KEY:-}" ] || { echo "❌ export ROBOFLOW_API_KEY=… (app.roboflow.com → Settings → API Keys)"; exit 1; }
    cd "$RAIZ/yolo_herramientas" && "$PY" descargar_roboflow.py
}

ninja() {
    local n
    n=$("$PY" -c "import sys; sys.path.insert(0, '$RAIZ/yolo_herramientas')
from comun import cargar_config; print(len(cargar_config().get('dataset_ninja') or []))")
    [ "$n" = 0 ] && { echo "· ningún dataset de Dataset Ninja en clases.yaml (METU-ALET está fuera: su descarga está rota)"; return 0; }
    # Sin --upgrade: dataset-tools fija xmltodict 0.13 y, si se reinstala cada vez, choca con FiftyOne
    "$PY" -c "import dataset_tools" 2>/dev/null || "$PY" -m pip install -q dataset-tools \
        || echo "⚠️  no se pudo instalar dataset-tools: solo valdrá la descarga a mano"
    cd "$RAIZ/yolo_herramientas" && "$PY" descargar_ninja.py
}

open_images() {
    "$PY" -c "import fiftyone" 2>/dev/null || "$PY" -m pip install -q fiftyone || { echo "❌ no se pudo instalar fiftyone"; exit 1; }
    cd "$RAIZ/yolo_herramientas" && "$PY" descargar_open_images.py
}

obra() {
    # Zenodo 6530106 (CC BY 4.0). Los zip que falten se bajan A LA VEZ (Zenodo da ~1 MB/s por conexión;
    # OBRA_PARALELO=1 para uno detrás de otro). Se puede cortar con Ctrl+C: wget -c sigue por donde iba.
    # Al acabar cada zip se comprueba su MD5 y se marca con un .ok para no repetirlo.
    local d="$DATOS/obra"
    mkdir -p "$d" && cd "$d" || exit 1
    exec 9>"$d/.descargando"            # que no lo bajen dos terminales a la vez (estropearían los zip)
    flock -n 9 || { echo "⏳ El de obra ya se está bajando en otra terminal: aquí se salta"; return 0; }
    declare -A md5=([1]=e87dec52ce299be886ae733a9ee496dd [2]=765ac6da3420841673fdc181b01f39ad
                    [3]=7fd534712155229800adf3bc825e7fc8 [4]=d80cb8715b1107a762551ee2d559acf5)
    declare -A gb=([1]=18.9 [2]=26.3 [3]=40.8 [4]=23.5)
    local partes="${OBRA_PARTES:-$("$PY" -c "import sys; sys.path.insert(0, '$RAIZ/yolo_herramientas')
from comun import cargar_config; print(*cargar_config()['obra']['partes'])")}"
    local pendientes=() n
    for n in $partes; do
        if [ -f "DATA$n.zip.ok" ]; then echo "✅ DATA$n.zip ya está"; else pendientes+=("$n"); fi
    done
    [ ${#pendientes[@]} -eq 0 ] && { echo "Siguiente: $PY $RAIZ/yolo_herramientas/convertir_obra.py --inspeccionar"; return 0; }
    local url="https://zenodo.org/records/6530106/files"
    local opts=(-c --tries=0 --retry-connrefused --retry-on-http-error=429,500,502,503,504
                --waitretry=15 --read-timeout=120)
    if [ "${OBRA_PARALELO:-}" = 1 ]; then
        for n in "${pendientes[@]}"; do
            echo "⬇ DATA$n.zip (${gb[$n]} GB) → $d"
            wget "${opts[@]}" -O "DATA$n.zip" "$url/DATA$n.zip?download=1"
        done
    else
        echo "⬇ ${#pendientes[@]} zip a la vez → $d   (Ctrl+C para parar; al relanzar sigue)"
        echo "   el detalle de cada uno, en $d/DATAn.log"
        local pids=() p
        for n in "${pendientes[@]}"; do
            wget "${opts[@]}" -o "DATA$n.log" -O "DATA$n.zip" "$url/DATA$n.zip?download=1" &
            pids+=($!)
        done
        trap 'kill "${pids[@]}" 2>/dev/null; echo; echo "⏸ Parado: al relanzar sigue por donde iba"; exit 130' INT TERM
        local t0 b0 b t vivos linea
        t0=$(date +%s); b0=0
        for n in "${pendientes[@]}"; do b0=$((b0 + $(stat -c %s "DATA$n.zip" 2>/dev/null || echo 0))); done
        while :; do
            vivos=0
            for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && vivos=1; done
            linea="" ; b=0
            for n in "${pendientes[@]}"; do
                t=$(stat -c %s "DATA$n.zip" 2>/dev/null || echo 0); b=$((b + t))
                linea+="DATA$n $(awk -v x="$t" 'BEGIN{printf "%.1f", x/1e9}')/${gb[$n]} GB · "
            done
            printf "\r   %s%s MB/s   " "$linea" "$(awk -v x=$((b - b0)) -v s=$(( $(date +%s) - t0 + 1 )) 'BEGIN{printf "%.1f", x/s/1e6}')"
            [ "$vivos" = 0 ] && break
            sleep 15
        done
        echo
        for p in "${pids[@]}"; do wait "$p"; done
        trap - INT TERM
    fi
    local mal=0
    for n in "${pendientes[@]}"; do
        local z="DATA$n.zip"
        echo "   comprobando MD5 de $z…"
        if echo "${md5[$n]}  $z" | md5sum -c --quiet - 2>/dev/null; then touch "$z.ok"; rm -f "DATA$n.log"; echo "✅ $z"
        else echo "❌ $z no está completo o está corrupto (mira DATA$n.log): relanza para seguir"; mal=1; fi
    done
    [ "$mal" = 0 ] && echo "Siguiente: $PY $RAIZ/yolo_herramientas/convertir_obra.py --inspeccionar"
    return $mal
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
    obra) obra ;;
    coco) coco ;;
    *) sed -n '2,11p' "$0" | sed 's/^# \?//'; exit 2 ;;
esac
