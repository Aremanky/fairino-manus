# brazo-robotico


Creear entorno
```bash
python3 -m venv venv_nivel6
```
Activar entorno
```bash
source venv_nivel6/bin/activate
```
Descargar dependencias
```bash
pip install -r requirements.txt
```
Luego escribir esto
```bash
pip install opencv-python depthai mediapipe=0.10.14 mujoco torch
```
En caso de que mediapipe te diga que no fufa en su versión escribir esto
```bash
sudo rm -rf venv_nivel6
python3.12 -m venv venv_nivel6
source venv_nivel6/bin/activate
pip install -r requirements.txt
pip install opencv-python depthai mediapipe=0.10.14 mujoco torch
```

---
O simplemente arrancar con la opción
```bash
./arrancar_nivel6.sh
```

---
Lo más probable es que falle, pero es bueno para ir depurando el .txt

> [!NOTE]  
> Falta depurarlo probando en otros ordenadores, es decir, ver cuales módulos son intrínsecos del SO

> [!WARNING]  
> Si te da fallo al descargar dependencias ve al requirements.txt y borra de esa lista el modulo que te da problemas, debería de funcionar todo sin ese modulo ya que algunos son intrínsecos del SO y para ti no serán necesarios

> [!CAUTION]
> si el que te da error es PyGObject instala esto y no lo elimines de la lista
> ```bash
> sudo apt update && sudo apt install libcairo2-dev libgirepository1.0-dev
> ```

---
## Modos de arranque (menú del principio)

| Modo | Mano (OrcaHand) | Brazo |
|---|---|---|
| 💻 Simulación Virtual | solo MuJoCo | cámaras (solo MuJoCo) |
| 🤖 Mano Física + Gemelo Digital | guante/cámara → mano real por USB | cámaras |
| 🎚️ Gemelo Digital (sliders) | guante/cámara → mano real por USB | **ventana de sliders** |

### Mano Física + Gemelo Digital (teleoperación con el FR5 físico)
- Arranca el FR5 real (bloque `TELEOP_REAL_*` de `config.py`; `TELEOP_REAL_FR5_SIMULACRO = True` para probar sin que se mueva).
- **[A]** + T-pose: calibra. El robot **no** se mueve.
- **Pedal** (o tecla **P** con `TELEOP_FR5_PEDAL = "tecla"`): ARMA → el FR5 entra despacio hacia tu postura y te sigue. Otra pisada, **[S]**, **[A]**, **[I]** o **[Q]** → desarmado y quieto.
- Sin pedal configurado (`TELEOP_FR5_PEDAL = None`) el FR5 real no se arma nunca.

### Gemelo Digital (sliders)
- Se abre una ventana con un slider por articulación, en **grados de la controladora** (los mismos que la WebApp del FR5).
- **[A]** en la ventana de la cámara activa la mano (como siempre). El brazo NO sigue a las cámaras.
- **ARMAR** en la ventana de sliders: el brazo (simulado y FR5 real) sigue a los sliders, con velocidad limitada.
  Solo se arma si el robot está dentro del rango de los sliders (si no, llévalo con la WebApp).
- **STOP / Esc / Espacio / cerrar la ventana** → desarmado: el FR5 deja de recibir ServoJ. La seta física sigue siendo la parada de verdad.
- Ajustes en `config.py`, bloque `SLIDERS_*` (simulacro, velocidades, rangos).
- Necesita Tkinter en el entorno: `sudo apt install python3.12-tk`.
- Probar la ventana sola, sin robot: `python -m skill_shield.interfaz.sliders_brazo`

SDK de FAIRINO en `fairino/Robot.py`: versión V4.0.0 (SDK V2.2.8), la del controlador V4.0.0.x.
