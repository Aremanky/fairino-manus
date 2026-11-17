#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════════════════════
#  Instalar cuRobo V2 (NVIDIA, Apache 2.0) para el modo autónomo — 17-nov
#  PC del robot: RTX 5080 (Blackwell, sm_120), driver 580 (CUDA 13.0), Ubuntu 24.04.
#
#  Se instala en el MISMO Python con el que se lanza Skill_Shield_Nivel_6.py (el planificador corre
#  ahí). Activa antes ese entorno (conda/venv) y lanza, desde la raíz del proyecto:
#
#      bash instalar_curobo.sh            # instala + configura el robot + prueba
#      bash instalar_curobo.sh --tests    # además, los tests de cuRobo (tardan)
#
#  Qué hace:
#    1. comprueba la GPU y el driver;
#    2. comprueba que PyTorch trae kernels para la RTX 5080 (sm_120); si no, instala PyTorch con CUDA 13.0;
#    3. descarga cuRobo V2 en ./externos/curobo (versión fija, la revisada) y lo instala (pip -e);
#    4. genera la configuración de cuRobo del FR5 + OrcaHand (esferas de colisión, en la GPU);
#    5. lanza probar_curobo.py.
# ═══════════════════════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"
CUROBO_COMMIT="824f063"          # NVlabs/curobo main del 16-nov-2026 (la API que usa curobo_fr5/)
PY="${PYTHON:-python}"

echo "🐍 Python: $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"
$PY - <<'EOF'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
    sys.exit(f"✗ cuRobo necesita Python 3.10-3.13 (este es {sys.version.split()[0]})")
EOF

echo "🟩 GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "✗ nvidia-smi no funciona"; exit 1; }

echo "🔥 PyTorch:"
if ! $PY - <<'EOF'
import sys
try:
    import torch
except ImportError:
    sys.exit(1)
ok = torch.cuda.is_available() and "sm_120" in torch.cuda.get_arch_list()
print(f"   torch {torch.__version__} (CUDA {torch.version.cuda}), GPU: {torch.cuda.is_available()}, "
      f"arquitecturas: {torch.cuda.get_arch_list() if torch.cuda.is_available() else '-'}")
sys.exit(0 if ok else 1)
EOF
then
    echo "   ↪ Esta versión no trae kernels para la RTX 5080 (sm_120): instalo PyTorch con CUDA 13.0"
    $PY -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
    $PY -c "import torch; assert torch.cuda.is_available() and 'sm_120' in torch.cuda.get_arch_list(); print('   ✓ torch', torch.__version__, torch.version.cuda)"
fi

echo "📦 cuRobo V2 (${CUROBO_COMMIT}):"
mkdir -p externos
if [ ! -d externos/curobo/.git ]; then
    git clone https://github.com/NVlabs/curobo externos/curobo
fi
git -C externos/curobo fetch --depth 50 origin main >/dev/null 2>&1 || true
git -C externos/curobo checkout -q "${CUROBO_COMMIT}"
$PY -m pip install -e "externos/curobo[cu13]"
$PY -c "import curobo; print('   ✓ cuRobo', curobo.__version__)"

if [ "${1:-}" = "--tests" ]; then
    echo "🧪 Tests de cuRobo (pueden tardar varios minutos)…"
    $PY -m pytest --pyargs curobo.tests -q -x || echo "⚠️  Algún test de cuRobo falla: mira el error antes de seguir"
fi

echo "🤖 Configuración del robot para cuRobo (esferas de colisión en la GPU)…"
$PY -m skill_shield.autonomo.curobo_fr5.robot

echo "🧭 Pruebas con el FR5:"
$PY probar_curobo.py
echo "✅ Listo. Para usarlo: en config.py, AUTONOMO_TAREA_PLANIFICADOR = \"curobo\" (ya viene así)."
