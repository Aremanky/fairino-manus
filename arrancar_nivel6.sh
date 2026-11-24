#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════════════════════
#  Skill Shield Nivel 6 — ARRANCAR. Un solo script (18-nov): antes también instalar_curobo.sh e
#  instalar_toppra.sh. Cada vez que se lanza COMPRUEBA y solo instala lo que falte (lo que ya está
#  no se toca):
#    0. calibraciones de antes del 24-nov que sigan en la raíz → calibraciones/datos/;
#    1. submódulo de los táctiles (orca_core_touch);
#    2. entorno virtual venv_nivel6 (Python 3.12) con requirements.txt (se vuelve a pasar si cambia);
#    3. cuRobo V2 (NVIDIA): PyTorch para la GPU (RTX 5080, sm_120), cuRobo en ./externos/curobo
#       (versión fija) y la configuración del robot (curobo_fr5/fr5_orca.yml);
#    4. TOPP-RA (tramos articulares del modo autónomo).
#  cuRobo y TOPP-RA son opcionales: si no se pueden instalar se avisa y el programa arranca igual
#  (el modo autónomo planifica con el planificador propio y usa el trapecio de siempre).
#
#      ./arrancar_nivel6.sh                 # comprueba, instala lo que falte y arranca
#      ./arrancar_nivel6.sh --comprobar     # solo dice qué falta (no instala nada ni arranca)
#      ./arrancar_nivel6.sh --instalar      # comprueba e instala lo que falte, sin arrancar
#      ./arrancar_nivel6.sh --probar        # además prueba cuRobo y TOPP-RA (sin mover el robot)
#      ./arrancar_nivel6.sh --tests-curobo  # además, los tests propios de cuRobo (varios minutos)
# ═══════════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

VENV="venv_nivel6"
PY="$DIR/$VENV/bin/python"
SELLO="$VENV/.requirements.sha256"          # requirements.txt con el que se instaló el entorno
CUROBO_COMMIT="824f063"                     # NVlabs/curobo main del 16-nov-2026 (la API que usa curobo_fr5/)
CUROBO_DIR="externos/curobo"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"   # driver 580 = CUDA 13.0
YML="curobo_fr5/fr5_orca.yml"
MJCF="fairino_description/mjcf/fairino_mjcf.xml"

COMPROBAR=0; ARRANCAR=1; PROBAR=0; TESTS_CUROBO=0
for a in "$@"; do
    case "$a" in
        --comprobar) COMPROBAR=1; ARRANCAR=0 ;;
        --instalar) ARRANCAR=0 ;;
        --probar) PROBAR=1 ;;
        --tests-curobo) PROBAR=1; TESTS_CUROBO=1 ;;
        -h|--help) sed -n '3,19p' "$0" | sed -E 's/^# ? ?//'; exit 0 ;;
        *) echo "Opción desconocida: $a   (ayuda: $0 --help)"; exit 2 ;;
    esac
done

ok()    { echo "   ✅ $*"; }
mal()   { echo "   ❌ $*"; }
aviso() { echo "   ⚠️  $*"; }
FALTA=()         # lo que falta (lo dice --comprobar)
AVISOS=()        # lo opcional que no se ha podido instalar: el programa arranca igual
PROBAR_CUROBO=0  # se acaba de instalar algo de cuRobo: se prueba
PROBAR_TOPPRA=0

# Estado de PyTorch, cuRobo y TOPP-RA en el entorno, en una sola llamada (importar torch tarda unos s)
estado_python() {
    "$PY" - <<'PY' 2>/dev/null
import importlib.util
try:
    import torch
    if not torch.cuda.is_available():
        t = "no_cuda"
    else:
        cap = torch.cuda.get_device_capability(0)
        t = "ok" if f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list() else f"sin_sm_{cap[0]}{cap[1]}"
    print(f"torch\t{t}\t{torch.__version__} · CUDA {torch.version.cuda}")
except Exception as e:
    print(f"torch\tno\t{type(e).__name__}")
try:
    # el cuRobo de NVIDIA trae curobo.motion_planner (lo usa curobo_fr5/planificador.py). El paquete «curobo»
    # de PyPI NO es el de NVIDIA (es un paquete vacío de 1,6 kB): import curobo funciona pero no sirve
    if importlib.util.find_spec("curobo") is None:
        c = "no"
    elif importlib.util.find_spec("curobo.motion_planner") is None:
        c = "falso"
    else:
        c = "ok"
except Exception as e:
    c = "roto"
print(f"curobo\t{c}")
try:
    import importlib.metadata as m
    import toppra  # noqa: F401
    print(f"toppra\tok\t{m.version('toppra')}")
except Exception:
    print("toppra\tno")
PY
}
campo() { echo "$ESTADO" | awk -F'\t' -v k="$1" -v n="$2" '$1 == k {print $n}'; }

# ─────────────────────────────── 0. calibraciones en su carpeta ───────────────────────────────
# (24-nov) Los resultados de las calibraciones viven en calibraciones/datos/. Los que sigan en la raíz (los que
# no estaban en git: muestras, verificaciones, la autocorrección, la extrínseca A↔C, copias…) se mueven allí. Si ya hay uno
# con el mismo nombre en calibraciones/datos/, no se toca ninguno y se avisa (manda el de calibraciones/datos/).
echo "0) Calibraciones (calibraciones/datos/)"
VIEJAS=()
for f in camara_robot.* camara_robot_verificacion.* camara_a_robot.* camara_a_robot_verificacion.* \
         calib_posturas.json* calib_posturas_a.json* calib_muestras.json* calib_muestras_a.json* \
         camaras_extrinsecas*.json* manus_calibracion.json*; do
    [ -f "$f" ] && VIEJAS+=("$f")
done
if [ ${#VIEJAS[@]} -eq 0 ]; then
    ok "en su sitio"
elif [ $COMPROBAR -eq 1 ]; then
    aviso "en la raíz (se moverán a calibraciones/datos/ al arrancar): ${VIEJAS[*]}"
else
    mkdir -p calibraciones/datos
    for f in "${VIEJAS[@]}"; do
        if [ -e "calibraciones/datos/$f" ]; then
            aviso "$f está en la raíz y en calibraciones/datos/: se usa el de calibraciones/datos/ (borra el otro)"
        else
            mv "$f" calibraciones/datos/ && ok "$f → calibraciones/datos/"
        fi
    done
fi

# ─────────────────────────────── 1. submódulo de los táctiles ───────────────────────────────
echo "1) Submódulo de los táctiles (orca_core_touch, lo usa skill_shield/hardware/puente_tactil.py)"
if [ -n "$(ls -A orca_core_touch 2>/dev/null)" ]; then
    ok "ya está"
elif [ $COMPROBAR -eq 1 ]; then
    mal "falta"; FALTA+=("orca_core_touch")
else
    echo "   📦 Descargando el submódulo orca_core_touch…"
    if git submodule update --init orca_core_touch; then ok "descargado"
    else aviso "no se ha podido descargar (los táctiles no irán)"; AVISOS+=("orca_core_touch"); fi
fi

# ─────────────────────────────── 2. entorno virtual ───────────────────────────────
echo "2) Entorno virtual ($VENV, Python 3.12)"
REQ_HASH="$(sha256sum requirements.txt | cut -d' ' -f1)"
if [ ! -x "$PY" ]; then
    if [ $COMPROBAR -eq 1 ]; then
        mal "el entorno virtual no existe"; FALTA+=("entorno")
    else
        echo "   ⚙️  El entorno virtual no existe: lo creo y descargo las dependencias…"
        if ! command -v python3.12 >/dev/null 2>&1; then
            mal "no está python3.12 (sudo apt install python3.12 python3.12-venv)"; exit 1
        fi
        python3.12 -m venv "$VENV" || { mal "no se ha podido crear (sudo apt install python3.12-venv)"; exit 1; }
        if ! "$PY" -m pip install -r requirements.txt; then
            rm -rf "$VENV"                        # a medias no sirve: la próxima vez se vuelve a crear entero
            mal "falló pip install -r requirements.txt (el entorno a medias se ha borrado)"; exit 1
        fi
        echo "$REQ_HASH" > "$SELLO"
        ok "entorno creado"
    fi
elif [ "$(cat "$SELLO" 2>/dev/null)" = "$REQ_HASH" ]; then
    ok "ya existe y tiene lo de requirements.txt"
elif [ $COMPROBAR -eq 1 ]; then
    mal "requirements.txt ha cambiado desde la última instalación"; FALTA+=("requirements")
else
    echo "   🔧 requirements.txt ha cambiado (o no se sabe con cuál se instaló): instalo solo lo que falte…"
    if "$PY" -m pip install -r requirements.txt; then
        echo "$REQ_HASH" > "$SELLO"; ok "al día"
    else
        aviso "falló pip install -r requirements.txt: arranco con lo que había"; AVISOS+=("requirements")
    fi
fi

# ─────────────────────────────── 3 y 4. cuRobo y TOPP-RA ───────────────────────────────
if [ -x "$PY" ]; then
    echo "🐍 $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
    ESTADO="$(estado_python)"

    echo "3) cuRobo (planificador en la GPU NVIDIA)"
    if ! GPU="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null)"; then
        aviso "nvidia-smi no funciona: sin GPU NVIDIA no hay cuRobo (el modo autónomo planifica en la CPU)"
        AVISOS+=("cuRobo: sin GPU")
    else
        ok "GPU: $GPU"
        # 3a. PyTorch con kernels para esta GPU
        CUROBO_SIGUE=1
        if [ "$(campo torch 2)" = "ok" ]; then
            ok "PyTorch $(campo torch 3)"
        elif [ $COMPROBAR -eq 1 ]; then
            mal "PyTorch no está o no trae kernels para esta GPU ($(campo torch 2))"; FALTA+=("torch")
        else
            echo "   🔧 PyTorch no está o no trae kernels para esta GPU ($(campo torch 2)): PyTorch con CUDA 13.0…"
            if "$PY" -m pip install --upgrade torch torchvision --index-url "$TORCH_INDEX"; then
                ESTADO="$(estado_python)"
            fi
            if [ "$(campo torch 2)" = "ok" ]; then ok "PyTorch $(campo torch 3)"; PROBAR_CUROBO=1
            else aviso "PyTorch sigue sin funcionar con la GPU ($(campo torch 2)): sin cuRobo"
                 AVISOS+=("cuRobo: PyTorch"); CUROBO_SIGUE=0; fi
        fi
        # 3b. cuRobo V2 (el de NVIDIA, versión fija)
        if [ $CUROBO_SIGUE -eq 1 ]; then
            C="$(campo curobo 2)"
            if [ "$C" = "ok" ]; then
                ok "cuRobo ya instalado: no se toca"
            elif [ $COMPROBAR -eq 1 ]; then
                mal "cuRobo no está instalado ($C)"; FALTA+=("curobo")
            else
                echo "   🔧 cuRobo no está instalado ($C): cuRobo V2 (${CUROBO_COMMIT}) en ./${CUROBO_DIR}…"
                if [ "$C" = "falso" ] || [ "$C" = "roto" ]; then
                    "$PY" -m pip uninstall -y curobo >/dev/null 2>&1 && echo "   🧹 quitado el paquete «curobo» que había"
                fi
                mkdir -p externos
                if { [ -d "$CUROBO_DIR/.git" ] || git clone https://github.com/NVlabs/curobo "$CUROBO_DIR"; } \
                   && { git -C "$CUROBO_DIR" fetch --depth 50 origin main >/dev/null 2>&1 || true; } \
                   && git -C "$CUROBO_DIR" checkout -q "$CUROBO_COMMIT" \
                   && "$PY" -m pip install -e "${CUROBO_DIR}[cu13]"; then
                    ESTADO="$(estado_python)"
                fi
                if [ "$(campo curobo 2)" = "ok" ]; then ok "cuRobo instalado"; PROBAR_CUROBO=1
                else aviso "no se ha podido instalar cuRobo: el modo autónomo planifica en la CPU"
                     AVISOS+=("cuRobo"); CUROBO_SIGUE=0; fi
            fi
        fi
        # 3c. configuración del robot para cuRobo (esferas de colisión del FR5 + OrcaHand)
        if [ $CUROBO_SIGUE -eq 1 ] && [ "$(campo curobo 2)" = "ok" ]; then
            if [ -f "$YML" ] && ! [ "$MJCF" -nt "$YML" ]; then
                ok "$YML ya existe (no es más vieja que el MJCF)"
            elif [ $COMPROBAR -eq 1 ]; then
                mal "falta $YML o es más vieja que el MJCF"; FALTA+=("robot")
            else
                echo "   🔧 $YML falta o es más vieja que el MJCF: esferas de colisión del FR5 + OrcaHand (GPU)…"
                if "$PY" -m skill_shield.autonomo.curobo_fr5.robot; then ok "$YML hecha"; PROBAR_CUROBO=1
                else aviso "no se ha podido generar $YML"; AVISOS+=("cuRobo: $YML"); fi
            fi
        fi
    fi

    echo "4) TOPP-RA (tramos articulares del modo autónomo)"
    if [ "$(campo toppra 2)" = "ok" ]; then
        ok "toppra $(campo toppra 3) ya instalado"
    elif [ $COMPROBAR -eq 1 ]; then
        mal "toppra no está instalado"; FALTA+=("toppra")
    else
        # no se toca la versión de numpy: el proyecto necesita numpy < 2 (mediapipe y depthai)
        NUMPY="$("$PY" -c 'import numpy; print(numpy.__version__)')"
        echo "   🔧 Instalando toppra (numpy $NUMPY se queda como está)…"
        if ! "$PY" -m pip install toppra "numpy==$NUMPY"; then
            echo "   No hay paquete ya compilado para este Python: lo compilo"
            echo "   (si falla aquí: sudo apt install build-essential python3.12-dev)"
            "$PY" -m pip install cython "numpy==$NUMPY" && \
                "$PY" -m pip install --no-build-isolation toppra "numpy==$NUMPY"
        fi
        ESTADO="$(estado_python)"
        if [ "$(campo toppra 2)" = "ok" ]; then ok "toppra $(campo toppra 3) instalado"; PROBAR_TOPPRA=1
        else aviso "no se ha podido instalar toppra: tramos articulares con el trapecio de siempre"; AVISOS+=("TOPP-RA"); fi
    fi
fi

# ─────────────────────────────── resumen de --comprobar ───────────────────────────────
if [ $COMPROBAR -eq 1 ]; then
    echo ""
    if [ ${#FALTA[@]} -eq 0 ]; then echo "✅ Está todo. (--probar para probar cuRobo y TOPP-RA)"; exit 0; fi
    echo "⛔ Falta: ${FALTA[*]}. Lánzalo sin --comprobar para instalarlo."; exit 1
fi

# ─────────────────────────────── pruebas (sin mover el robot) ───────────────────────────────
if [ $PROBAR -eq 1 ] || [ $PROBAR_CUROBO -eq 1 ]; then
    if [ "$(campo curobo 2)" = "ok" ]; then
        echo ""; echo "🧭 Pruebas de cuRobo (sin mover el robot)"
        if [ $TESTS_CUROBO -eq 1 ]; then
            echo "🧪 Tests de cuRobo (pueden tardar varios minutos)…"
            "$PY" -m pytest --pyargs curobo.tests -q -x || aviso "algún test de cuRobo falla: mira el error antes de seguir"
        fi
        "$PY" -m pytest -q tests/test_curobo.py tests/test_mapa_tsdf.py tests/test_comprobador_lote.py \
            || aviso "algún test del proyecto falla"
        "$PY" diagnosticos/probar_curobo.py || aviso "probar_curobo.py ha fallado"
    elif [ $PROBAR -eq 1 ]; then
        aviso "cuRobo no está: no hay nada que probar"
    fi
fi
if [ $PROBAR -eq 1 ] || [ $PROBAR_TOPPRA -eq 1 ]; then
    if [ "$(campo toppra 2)" = "ok" ]; then
        echo ""; echo "🧪 TOPP-RA: un tramo de 6 joints a 15 °/s y 30 °/s²"
        "$PY" - <<'PY' || aviso "la prueba de TOPP-RA ha fallado (el modo autónomo usará el trapecio)"
import math, sys
import numpy as np
sys.path.insert(0, ".")
from skill_shield.autonomo.ejecucion import tiempos_toppra
qs = np.radians([30, -10, 5, 0, 20, 60]) * np.linspace(0, 1, 40)[:, None]
r = tiempos_toppra(qs, math.radians(15), math.radians(30))
if r is None:
    sys.exit("   ❌ TOPP-RA no ha dado trayectoria")
T, tr = r
ts = np.linspace(0, T, 400)
v = np.abs(np.diff(np.array([tr(t) for t in ts]), axis=0)).max() / (ts[1] - ts[0])
print(f"   ✅ duración {T:.2f} s · velocidad máx. {math.degrees(v):.1f} °/s (límite 15) · acaba en su sitio: "
      f"{np.allclose(tr(T), qs[-1], atol=1e-3)}")
PY
    fi
fi

if [ ${#AVISOS[@]} -gt 0 ]; then
    echo ""; echo "⚠️  Sin: ${AVISOS[*]} (el programa funciona sin ello; mira los mensajes de arriba)"
fi
[ $ARRANCAR -eq 1 ] || { echo "Listo (sin arrancar)."; exit 0; }
[ -x "$PY" ] || { mal "no hay entorno virtual: no puedo arrancar"; exit 1; }

gnome-terminal --title="CÁMARA IA - Skill Shield Nivel 6" -- bash -c "
cd '$DIR'
'$PY' Skill_Shield_Nivel_6.py
exec bash"
