#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════════════════════
#  cuRobo V2 (NVIDIA, Apache 2.0) para el modo autónomo — 17-nov
#  PC del robot: RTX 5080 (Blackwell, sm_120), driver 580 (CUDA 13.0), Ubuntu 24.04.
#
#  Se usa el MISMO Python con el que se lanza Skill_Shield_Nivel_6.py. Activa antes ese entorno y,
#  desde la raíz del proyecto:
#
#      bash instalar_curobo.sh              # instala SOLO lo que falte y luego prueba
#      bash instalar_curobo.sh --comprobar  # NO instala nada: dice qué está y, si todo está listo, prueba
#      bash instalar_curobo.sh --tests      # además, los tests propios de cuRobo (tardan)
#
#  Lo que mira, en orden (lo que ya está, no se toca):
#    1. GPU y driver;
#    2. PyTorch con kernels para la RTX 5080 (sm_120)   → si falta: PyTorch con CUDA 13.0;
#    3. cuRobo V2 importable                            → si falta: ./externos/curobo (versión fija) + pip -e;
#    4. configuración del robot (curobo_fr5/fr5_orca.yml, más nueva que el MJCF) → si falta: la genera (GPU);
#    5. con todo listo: probar_curobo.py (sin mover el robot).
# ═══════════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")"
CUROBO_COMMIT="824f063"          # NVlabs/curobo main del 16-nov-2026 (la API que usa curobo_fr5/)
PY="${PYTHON:-python}"
COMPROBAR=0; TESTS=0
for a in "$@"; do
    case "$a" in
        --comprobar) COMPROBAR=1 ;;
        --tests) TESTS=1 ;;
        *) echo "Opción desconocida: $a"; exit 2 ;;
    esac
done
FALTA=()

ok()   { echo "   ✅ $*"; }
mal()  { echo "   ❌ $*"; }

echo "🐍 Python: $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"

echo "1) GPU"
if nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >/tmp/gpu.txt 2>/dev/null; then
    ok "$(cat /tmp/gpu.txt)"
else
    mal "nvidia-smi no funciona: sin GPU no hay cuRobo"; exit 1
fi

echo "2) PyTorch con soporte para la RTX 5080 (sm_120)"
if $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and 'sm_120' in torch.cuda.get_arch_list() else 1)" 2>/dev/null; then
    ok "$($PY -c "import torch; print('torch', torch.__version__, '· CUDA', torch.version.cuda)")"
else
    mal "PyTorch no está o no trae kernels para la RTX 5080"
    FALTA+=("torch")
fi

echo "3) cuRobo"
if $PY -c "import curobo" 2>/dev/null; then
    ok "cuRobo $($PY -c 'import curobo; print(curobo.__version__)') ya instalado: no se toca"
else
    mal "cuRobo no está instalado"
    FALTA+=("curobo")
fi

echo "4) Configuración del robot para cuRobo"
YML="curobo_fr5/fr5_orca.yml"
MJCF="fairino_description/mjcf/fairino_mjcf.xml"
if [ -f "$YML" ] && [ "$YML" -nt "$MJCF" ]; then
    ok "$YML ya existe (más nueva que el MJCF): no se rehace"
elif [ -f "$YML" ]; then
    mal "$YML es más vieja que el MJCF: hay que rehacerla"; FALTA+=("robot")
else
    mal "falta $YML"; FALTA+=("robot")
fi

if [ ${#FALTA[@]} -gt 0 ]; then
    if [ $COMPROBAR -eq 1 ]; then
        echo ""; echo "⛔ Falta: ${FALTA[*]}. Con --comprobar no se instala nada (lánzalo sin --comprobar para instalarlo)."
        exit 1
    fi
    echo ""; echo "🔧 Instalo solo lo que falta: ${FALTA[*]}"
    for f in "${FALTA[@]}"; do
        case "$f" in
            torch)
                echo "   → PyTorch con CUDA 13.0"
                $PY -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130 || exit 1
                $PY -c "import torch; assert torch.cuda.is_available() and 'sm_120' in torch.cuda.get_arch_list(); print('   ✅ torch', torch.__version__, torch.version.cuda)" || exit 1 ;;
            curobo)
                echo "   → cuRobo V2 (${CUROBO_COMMIT}) en ./externos/curobo"
                mkdir -p externos
                [ -d externos/curobo/.git ] || git clone https://github.com/NVlabs/curobo externos/curobo || exit 1
                git -C externos/curobo fetch --depth 50 origin main >/dev/null 2>&1 || true
                git -C externos/curobo checkout -q "${CUROBO_COMMIT}" || exit 1
                $PY -m pip install -e "externos/curobo[cu13]" || exit 1
                $PY -c "import curobo; print('   ✅ cuRobo', curobo.__version__)" || exit 1 ;;
            robot)
                echo "   → esferas de colisión del FR5 + OrcaHand (GPU)"
                $PY -m skill_shield.autonomo.curobo_fr5.robot || exit 1 ;;
        esac
    done
fi

echo ""; echo "🧭 Todo listo: pruebas (sin mover el robot)"
if [ $TESTS -eq 1 ]; then
    echo "🧪 Tests de cuRobo (pueden tardar varios minutos)…"
    $PY -m pytest --pyargs curobo.tests -q -x || echo "⚠️  Algún test de cuRobo falla: mira el error antes de seguir"
fi
$PY -m pytest -q tests/test_curobo.py tests/test_mapa_tsdf.py tests/test_comprobador_lote.py || echo "⚠️  Algún test del proyecto falla"
$PY probar_curobo.py
