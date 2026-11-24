#!/usr/bin/env bash
# Crea venv_yolo (en la raíz del repo) con PyTorch para la RTX 5080, Ultralytics y lo demás.
#   ./yolo_herramientas/instalar.sh
set -uo pipefail
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"
VENV="venv_yolo"
PY="$RAIZ/$VENV/bin/python"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"   # el mismo que arrancar_nivel6.sh (driver 580 = CUDA 13.0)

if [ ! -x "$PY" ]; then
    command -v python3.12 >/dev/null || { echo "❌ falta python3.12 (sudo apt install python3.12 python3.12-venv)"; exit 1; }
    python3.12 -m venv "$VENV" || exit 1
fi
"$PY" -m pip install --upgrade pip
echo "── PyTorch para la GPU ──"
"$PY" -m pip install --upgrade torch torchvision --index-url "$TORCH_INDEX" || { echo "❌ falló PyTorch"; exit 1; }
echo "── Ultralytics y demás ──"
"$PY" -m pip install -r yolo_herramientas/requirements.txt || { echo "❌ falló requirements.txt"; exit 1; }
echo "── DepthAI (para probar el modelo en vivo con la OAK) ──"
if "$PY" -m pip install depthai==2.32.0.0; then
    if ! "$PY" -c "import depthai as dai, numpy as np; f = dai.ImgFrame(); f.setData(np.zeros(12, np.uint8))" 2>/dev/null; then
        echo "   depthai 2.x no va con este numpy: bajo numpy a 1.x"
        "$PY" -m pip install "numpy<2"
    fi
else
    echo "⚠️  no se ha podido instalar depthai: probar_oak.py no irá (el resto sí)"
fi
echo "── Comprobación ──"
"$PY" - <<'EOF'
import torch, ultralytics
ok = torch.cuda.is_available()
print(f"torch {torch.__version__} · CUDA {torch.version.cuda} · GPU: {torch.cuda.get_device_name(0) if ok else 'NO'}")
print(f"ultralytics {ultralytics.__version__}")
if ok:
    x = torch.randn(256, 256, device="cuda"); (x @ x).sum().item()
    print("✅ la GPU calcula")
else:
    print("❌ PyTorch no ve la GPU: el entrenamiento iría en CPU (días). Revisa el driver (nvidia-smi).")
EOF
echo
echo "Listo. Actívalo con:  source $VENV/bin/activate"
