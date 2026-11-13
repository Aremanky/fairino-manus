#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Submódulo de los sensores táctiles (lo necesita puente_tactil.py)
if [ -z "$(ls -A orca_core_touch 2>/dev/null)" ]; then
echo "📦 Descargando el submódulo orca_core_touch..."
git submodule update --init orca_core_touch
fi

if [ ! -d "venv_nivel6" ]; then
echo "El entorno virtual no existe"
echo "⚙️ Configurando el entorno virtual y descargando dependencias..."
python3.12 -m venv venv_nivel6
./venv_nivel6/bin/pip install -r requirements.txt
echo "✅ Entorno listo."
fi

gnome-terminal --title="CÁMARA IA - Skill Shield Nivel 6" -- bash -c "
cd '$DIR'
./venv_nivel6/bin/python Skill_Shield_Nivel_6.py
exec bash"
