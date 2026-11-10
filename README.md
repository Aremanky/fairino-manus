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
| 🎚️ Gemelo Digital (sliders) | guante/cámara → mano real por USB | **ventana de sliders** |
| 🤖 Mano Física + Gemelo Digital | guante/cámara → mano real por USB | cámaras |
| 🧠 Funcionamiento autónomo | *(en desarrollo)* | *(en desarrollo)* |

### Funcionamiento autónomo (en desarrollo)
- Última opción del menú. Fija `MODO_EJECUCION = "AUTONOMO"` y salta al paquete `skill_shield/autonomo/`, con su propio bucle.
- **Paso 2 (ahora):** cámara de cabeza + gemelo en MuJoCo copiando la pose real del FR5 + estado de la OrcaHand y sus táctiles + web. **El robot no se mueve**: `fr5_servo` va siempre en simulacro en este modo y no se arma nunca. La OrcaHand se vigila en solo lectura (no se activa el par).
- **Web (FastAPI)** en `http://<IP del PC>:8080/` desde cualquier equipo de la red. Necesita `./venv_nivel6/bin/pip install fastapi uvicorn`. Si no carga desde otro equipo: `sudo ufw allow 8080/tcp`.
  - Lectura abierta: imagen de la cámara, FR5, OrcaHand, táctiles por dedo, gemelo.
  - Mandos con contraseña (`AUTONOMO_WEB_CLAVE`, **cambiadla**): pedal, opción «Pedal obligatorio» y **Reiniciar tras colisión** (borrar error → disable → enable).
  - **Pedal de la web:** botón grande (o barra espaciadora) que hay que mantener pulsado. Si se suelta, se corta la red o se cierra la pestaña, en ≤ 0,35 s el pedal cuenta como suelto. Con «Pedal obligatorio» desactivado se permite moverse sin pedal (se puede bloquear con `AUTONOMO_PERMITIR_SIN_PEDAL = False`). Siempre arranca activado.
- **Gemelo en la web:** pestañas *Cámara · Gemelo · Las dos*. El gemelo se dibuja sin ventana (`render_gemelo.py`, solo la escena, sin los menús del visor): arrastrar para girar, Mayús+arrastrar para mover, rueda o +/− para acercar, y vistas Inicial/Frente/Lado/Arriba. La vista la comparten todos los que miran. Con `AUTONOMO_VENTANA_MUJOCO = False` ya no se abre la ventana de MuJoCo en el PC. Necesita OpenGL sin ventana (`AUTONOMO_SIM_GL = "egl"`; si falla prueba osmesa y glfw).
- **Superponer gemelo** (con la calibración del paso 4): el gemelo visto desde la cámara calibrada, encima de la imagen real. Si la calibración está bien, el robot virtual tapa al real.
- Cámara de cabeza = la antigua cámara B (`AUTONOMO_CAM_MXID`).
- Para salir: cerrar la ventana de MuJoCo o Ctrl+C.
- Pruebas por separado: `python -m skill_shield.autonomo.camara --ventana` · `python -m skill_shield.autonomo.web`
- Cerrar el menú con la X o Esc equivale a pulsar este botón (es el «Cancelar» de zenity).

### Herramientas con marcadores ArUco (paso 5)
`skill_shield/autonomo/herramientas.py` — cada herramienta (y el punto de uso y el de dejada) lleva un marcador ArUco 4x4. Catálogo en `AUTONOMO_HERRAMIENTAS` (id, nombre, tipo, lado impreso, dónde queda la herramienta respecto al marcador, tamaño).
1. `python -m skill_shield.autonomo.herramientas --generar` → PNG de cada marcador en `marcadores/`. Imprímelos a 300 ppp al 100 %, mide el cuadrado negro y pégalos planos sin recortar el margen blanco.
2. `python -m skill_shield.autonomo.herramientas --probar` → cámara de cabeza en vivo: pose de cada herramienta, temblor y error (cierra antes el modo autónomo).
3. En el modo autónomo: tarjeta **Herramientas** en la web (las que se ven, estables o no, y su posición en el mundo) con **Elegir** para marcar la herramienta objetivo: se pinta en naranja en el gemelo y deja de contar como obstáculo en el mapa. Las herramientas también se dibujan sobre la imagen de la cámara y en la ventana de MuJoCo.
- La distancia de cada marcador se corrige con la profundidad estéreo (con marcadores de 3 cm a 0,8 m, solvePnP solo falla 1-3 cm; con el estéreo, unos mm) y las poses ambiguas (marcador casi de frente) se deciden con el plano que mide el estéreo.
- Hace falta la calibración para la posición en el mundo; sin ella se ven respecto a la cámara y no se pueden elegir.
- Marcadores de 3-4 cm hasta ~1 m de la cámara; más lejos, más grandes.

### Mapa de obstáculos (paso 7)
`skill_shield/autonomo/mapa.py` — con la calibración hecha, el proceso de la cámara convierte la profundidad en cubos de 2 cm (marco del MJCF) a 4 Hz: quita el **robot** (sus cápsulas con la pose real + 3 cm), el **suelo** y lo que se pida excluir (la herramienta que se va a coger). Cada cubo es *ocupado* (se ven puntos en él varias veces), *libre* (la cámara ve a través) o *desconocido* (tapado, o más de 30 s sin verse: el robot es móvil).
- En la web: indicador **Obstáculos** y, en el gemelo, los cubos en naranja (botón *Obstáculos* para ocultarlos; *Borrar mapa* con contraseña, p. ej. tras mover la peana). Con *Superponer gemelo*, los cubos deben caer encima de los objetos reales.
- Para esquivar: `ColisionMapa(modelo, mapa)` añade al modelo de colisión de la teleoperación los choques contra el mapa (lo tapado debajo/detrás de algo cuenta como obstáculo, `AUTONOMO_MAPA_DESCONOCIDO_OCUPADO`). Lo usará el planificador (paso 8).
- Lo que tapa el propio brazo no se ve: se mira con el brazo apartado y el mapa lo recuerda.
- Ajustes: bloque `AUTONOMO_MAPA_*` de `config.py` (tamaño, zona, frecuencia, olvido).

### Calibración cámara de cabeza → robot (paso 4)
`calibrar_camara_robot.py` — tablero ChArUco en la mano y la cámara fija en el soporte. Resultado: `camara_robot.json` (`T_mundo_cam` en el marco del MJCF). Como la cámara va en la misma peana que el robot, **mover la peana no estropea la calibración**.
1. `python calibrar_camara_robot.py --generar-tablero` → imprime `tablero_calib_robot.png` a 300 ppp, mide el cuadrado (30 mm) y móntalo plano en la mano.
2. `python calibrar_camara_robot.py --auto` → **automática**: pon el robot (arrastre) donde la cámara vea el tablero y **mantén la P** con la ventana delante. Explora con giros pequeños, planifica ~20 posturas seguras (modelo de colisión + tablero visible), las recorre y calibra (~5 min). Criterio: error medio ≤ 5 mm y giro ≥ 30°.
3. `python calibrar_camara_robot.py --verificar` → comprueba en 4 posturas guardadas (`--aqui`: en la postura actual, sin moverlo). Si falla, algo se ha movido → `--repetir` (repite las posturas guardadas) o `--auto`.
4. `--manual` (mueves tú el robot y pulsas ESPACIO) y `--recalcular` (con las muestras guardadas) siguen disponibles.
El robot solo se mueve con el pedal y a ≤ 15 °/s. Antes de mover, se comprueba cada trayecto con el modelo de colisión: **revisa que el soporte y la mesa (`TELEOP_FR5_OBSTACULOS`) del modelo coinciden con la realidad**. Cierra antes el modo autónomo: la cámara y el robot tienen que estar libres.

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
