#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════════════════════
#  TOPP-RA (MIT) para los tramos ARTICULARES del modo autónomo — 18-nov
#  Con él, cada tramo articular va lo más rápido que permiten la velocidad y la aceleración de CADA
#  joint (AUTONOMO_TAREA_VEL_DEG_S, AUTONOMO_TAREA_ACEL_DEG_S2), también en las esquinas del camino.
#  Sin él, el modo autónomo sigue funcionando con el trapecio de siempre.
#
#      bash instalar_toppra.sh              # instala si falta y lo prueba
#      bash instalar_toppra.sh --comprobar  # no instala nada: solo dice si está y lo prueba
#
#  Usa el Python del entorno del proyecto (venv_nivel6) o el de la variable PYTHON. No toca la versión de
#  numpy (el proyecto necesita numpy < 2 por mediapipe y depthai).
# ═══════════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-./venv_nivel6/bin/python}"
[ -x "$PY" ] || PY=python
COMPROBAR=0; [ "${1:-}" = "--comprobar" ] && COMPROBAR=1
echo "🐍 Python: $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"
NUMPY=$($PY -c "import numpy; print(numpy.__version__)")
echo "   numpy $NUMPY (se queda como está)"

if $PY -c "import toppra" 2>/dev/null; then
    echo "✅ toppra $($PY -c 'import toppra, importlib.metadata as m; print(m.version("toppra"))' 2>/dev/null) ya instalado"
elif [ $COMPROBAR -eq 1 ]; then
    echo "❌ toppra no está instalado (lánzalo sin --comprobar para instalarlo)"; exit 1
else
    echo "🔧 Instalando toppra…"
    if ! $PY -m pip install toppra "numpy==$NUMPY"; then
        echo "   No hay paquete ya compilado para este Python: lo compilo."
        echo "   (si falla aquí: sudo apt install build-essential python3.12-dev)"
        $PY -m pip install cython "numpy==$NUMPY" || exit 1
        $PY -m pip install --no-build-isolation toppra "numpy==$NUMPY" || exit 1
    fi
fi

echo "🧪 Prueba (un tramo de 6 joints, 15 °/s y 30 °/s²):"
$PY - <<'PY'
import math, sys
import numpy as np
sys.path.insert(0, ".")
from skill_shield.autonomo.ejecucion import tiempos_toppra
qs = np.radians([0, 0, 0, 0, 0, 0]) + np.radians([30, -10, 5, 0, 20, 60]) * np.linspace(0, 1, 40)[:, None]
r = tiempos_toppra(qs, math.radians(15), math.radians(30))
if r is None:
    sys.exit("❌ TOPP-RA no ha dado trayectoria")
T, tr = r
ts = np.linspace(0, T, 400)
v = np.abs(np.diff(np.array([tr(t) for t in ts]), axis=0)).max() / (ts[1] - ts[0])
print(f"   ✅ duración {T:.2f} s · velocidad máx. {math.degrees(v):.1f} °/s (límite 15) · acaba en su sitio: "
      f"{np.allclose(tr(T), qs[-1], atol=1e-3)}")
PY
echo "Listo. AUTONOMO_TAREA_TOPPRA = True en config.py (ya lo está) para usarlo."
