#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "venv_nivel6" ]; then
echo "El entorno virtual no existe"
echo "⚙️ Configurando el entorno virtual y descargando dependencias..."
python3.12 -m venv venv_nivel6
source venv_nivel6/bin/activate
pip install -r requirements.txt
pip install opencv-python depthai==2.32.0.0 mediapipe==0.10.14 mujoco torch lib
echo "✅ Entorno listo."
fi

gnome-terminal --title="CÁMARA IA - Skill Shield Nivel 6" -- bash -c "
cd '$DIR'
./venv_nivel6/bin/python Skill_Shield_Nivel_6.py
exec bash"
