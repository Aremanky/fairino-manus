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
`requirements.txt` ya lleva todo (opencv-contrib, depthai, mediapipe 0.10.14, mujoco, torch, fastapi…).
**No instales `opencv-python` aparte**: choca con `opencv-contrib-python` y desaparece `cv2.aruco`.

Si mediapipe dice que no hay versión para tu Python, el entorno tiene que ser de Python 3.12:
```bash
rm -rf venv_nivel6
python3.12 -m venv venv_nivel6
source venv_nivel6/bin/activate
pip install -r requirements.txt
```
Submódulo de los táctiles (una vez): `git submodule update --init orca_core_touch`

---
O simplemente arrancar con la opción
```bash
./arrancar_nivel6.sh
```

Paquetes del sistema (no van por pip): `sudo apt install zenity python3.12-tk`

Pruebas sin hardware (no hace falta robot, mano, guante ni cámaras), en `tests/`:
```bash
python -m pytest tests -q
```

---
## Modos de arranque (menú del principio)

| Modo | Mano (OrcaHand) | Brazo |
|---|---|---|
| 💻 Simulación Virtual | solo MuJoCo | cámaras (solo MuJoCo) |
| 🎚️ Gemelo Digital (sliders) | guante/cámara → mano real por USB | **ventana de sliders** |
| 🤖 Mano Física + Gemelo Digital | guante/cámara → mano real por USB | cámaras |
| 🎥 Grabación (teleoperación + objetos) | guante → mano real por USB | cámaras A + C; la B es la cámara de cabeza (objetos y obstáculos) |
| 🧠 Funcionamiento autónomo | *(en desarrollo)* | *(en desarrollo)* |

### Grabación (teleoperación + objetos) — para grabar demos con el mundo
La teleoperación de «Mano Física + Gemelo» (mismo pedal, mismo FR5 con `TELEOP_REAL_*`) más lo que ve el modo autónomo con la cámara de cabeza (`skill_shield/ia/cabeza.py`):
- **Cámaras:** A y C cruzadas para teleoperar (pon el MxId de la C en `CAM_C_MXID`; sin ella, solo la A) y la **B como cámara de cabeza**, con el mismo proceso que el autónomo: herramientas con ArUco, pelotas y mapa de obstáculos. Necesita la calibración cámara de cabeza → robot (`camara_robot.json`).
- **Extrínseca A↔C** en su propio archivo (`GRABACION_EXTRINSECAS_ARCHIVO`, no pisa la A↔B del modo normal): se calcula sola con el cuerpo al pulsar `[A]`, o con el tablero: `python calibrar_extrinseca.py --mxid-b <MxId de C> --salida camaras_extrinsecas_ac.json`.
- **Objetivo automático:** el objeto que se puede coger más cercano a la palma; mientras se graba, el fijado al empezar el episodio. Se pinta en naranja en la ventana de MuJoCo, junto al resto de objetos y los obstáculos cercanos a la mano (cubos naranjas).
- **El brazo frena** contra los obstáculos del mapa (la IK lo deja lo más cerca posible sin chocar; en la ventana de la cámara sale `FRENADO`). Alrededor del objetivo (`GRABACION_ZONA_OBJETO_M`) la mesa no frena —si no, no se podría coger— y en su lugar nada baja de la base del objeto. `GRABACION_FRENAR = False` para que solo se vean.
- **Agarre asistido** (`skill_shield/ia/agarre_asistido.py`, `GRABACION_AGARRE_ASISTIDO`): así se graban las demos buenas.
  1. `[A]` + T-pose como siempre. `[O]` cambia el objeto a coger (por defecto el más cercano a la mano).
  2. `[L]`: el planificador del autónomo calcula cómo llevar la mano a la **postura de agarre** de ese objeto (la misma a la que llega la Tarea antes de bajar). Con el **pedal pisado** el brazo va; al soltarlo se para. Al llegar, un segundo quieto para que la cámara mida bien el objeto → `LISTO`.
  3. Abre la mano y **pisa el pedal**: modo **relativo** con embrague —el robot copia tus desplazamientos y giros a la mitad (`GRABACION_ESCALA_POS`, `GRABACION_ESCALA_GIRO`), no tu postura— y **empieza a grabarse** el episodio. Coge el objeto y súbelo unos cm.
  4. **Suelta el pedal**: el robot se para y el episodio se guarda. `[M]` si salió mal. Volver a pisar = relativo **sin grabar** (para dejar el objeto y apartarte). `[L]` otro agarre, `[K]` vuelta a la teleoperación normal, `[B]` bloquea el giro (solo traslación).
  Así cada demo empieza donde empezará el agarre aprendido: tras el planificador. La mano la sigue llevando el guante.
- **Entrenamiento:** los episodios llevan el objeto (con su orientación si es una herramienta) y si el brazo iba frenado. Con las pelotas, la red mide la mano en el **marco de agarre**: centro de la pelota, vertical del mundo y el eje lateral de la mano al empezar (`--marco mundo` para los ejes del mundo, como antes). Así el mismo gesto hecho desde otro lado de la mesa le parece igual. Los que pasan más del 20 % del tiempo frenados no se usan (`IA_MAX_FRENADO`, o `--max-frenado 1`). La red ve el objeto, no los obstáculos: de esos se encarga el filtro de seguridad.
- Tres OAK-D a la vez: cada una en un puerto USB3, mejor en controladores distintos.

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
1. Tablero: ahora es el de la calibración extrínseca (7x5, cuadro 20 mm, marcador 12,5 mm, en `CALIB_ROBOT_TABLERO`), pegado a una placa rígida y fijado a la mano. Con otro tablero, pon sus medidas reales ahí (`--generar-tablero` crea uno nuevo). **El tamaño se comprueba solo:** al acabar de explorar, el programa compara lo que el robot ha movido la brida con lo que la cámara ha visto moverse al tablero; si el cuadrado no mide lo que dice config (±3 %), se para antes de planificar nada y dice cuánto mide.
2. `python calibrar_camara_robot.py --auto` → **automática**: pon el robot (arrastre) donde la cámara vea el tablero, **con la muñeca cómoda y nada apretado** (esa postura es la referencia de todo lo demás), y **mantén la P** con la ventana delante. Explora con giros pequeños, planifica ~20 posturas seguras (tablero visible, sin choques y dentro de lo que `fr5_servo` deja pasar), las recorre y calibra (~5 min). Criterio: error medio ≤ 5 mm y giro ≥ 30°.
   - **Protección de la mano, del tablero y de lo que sobresale (10-nov):** con las piezas reales del gemelo (`skill_shield/autonomo/choque_mano.py`) se miden los huecos de la mano y de la placa del tablero con el hombro, el brazo, el antebrazo y el giro del J4 del robot, con el soporte (columna, placas, pie), con la cabeza y la cámara, con las cajas de `TELEOP_FR5_*` y con el suelo. **Ningún hueco puede cerrarse más de 1 cm respecto al de la postura de inicio** (`CALIB_MANO_ACERCAR_MAX`), salvo que siga por encima de su objetivo (mano-robot: `CALIB_MANO_HOLGURA` + `CALIB_MANO_SALIENTES`; tablero-robot: `CALIB_TABLERO_HOLGURA`; con el entorno: `CALIB_ENTORNO_HOLGURA`), y nunca baja de su mínimo. Así no se estruja lo que sale de la mano y no está en el gemelo (conector, cables). Los tres ejes de la muñeca van, además, con la amplitud recortada (`CALIB_AUTO_AMPLITUD_DEG`).
   - **Tres barreras:** (1) el planificador solo elige posturas y trayectos que cumplen lo anterior; (2) antes de cada movimiento se comprueba el trayecto entero desde donde está el robot DE VERDAD (si no vale, **no se mueve** y salta esa postura); (3) mientras el pedal está pisado se vigila su postura real en cada vuelta: si se acerca a algo más de lo comprobado, o deja de seguir el trayecto que se le pide, se para y la calibración termina («Calibración detenida: …» dice qué pasa). Si lo mueves a mano con el pedal suelto, al volver a pisar comprueba el trayecto desde donde lo hayas dejado. Si deja de llegar la postura del robot, no lo mueve hasta que vuelve.
   - Si el robot, con el pedal pisado, se queda parado sin llegar (límites blandos de la controladora más estrechos que `FR5_LIMITES_BLANDOS_DEG`), a los `CALIB_SIN_AVANCE_S` segundos deja esa postura y sigue con la siguiente.
   - Lo que el programa NO ve: los cables (cuánto dan de sí al girar la muñeca) y, durante la exploración (~1 min, giros de ≤ 15°), dónde está montado el tablero: hasta que acaba de explorar solo cuenta con una envolvente de `CALIB_AUTO_RADIO_MANO` (13 cm) alrededor de la mano. Empieza con el tablero a más de un palmo de todo y mira los cables en las primeras posturas.
   - Para revisar posturas guardadas sin mover nada: `python -m skill_shield.autonomo.choque_mano` (tabla J4×J5: `--mapa`).
3. `python calibrar_camara_robot.py --verificar` → comprueba en 4 posturas guardadas (`--aqui`: en la postura actual, sin moverlo). Si falla, algo se ha movido → `--repetir` (repite las posturas guardadas) o `--auto`. Los dos se mueven con las mismas tres barreras, con la primera postura guardada como referencia y el tablero donde lo dejó la última calibración: **si has cambiado el tablero de sitio, usa `--auto`**.
4. `--manual` (mueves tú el robot y pulsas ESPACIO; avisa en pantalla si la mano se acerca al robot o al soporte) y `--recalcular` (con las muestras guardadas) siguen disponibles.
El robot solo se mueve con el pedal y a ≤ `CALIB_ROBOT_VEL_MAX_DEG_S` (10 °/s). **Revisa que el soporte y la mesa (`TELEOP_FR5_OBSTACULOS`) del modelo coinciden con la realidad**: lo que no esté en el modelo, no se evita. Cierra antes el modo autónomo: la cámara y el robot tienen que estar libres. Pruebas sin robot: `python -m pytest tests/test_calibracion_robot.py -q`.

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

---
## IA: grabar, entrenar y reproducir (G / E / I)
`skill_shield/ia/` — la red (ACT) aprende de demostraciones. Desde el 16-nov **ve el mundo**: el objeto visto desde la mano, los táctiles y la postura real de la mano y del brazo (antes era «ciega»: solo repetía el gesto).

**Datos fuera del repo:** `~/skill_shield_datos/episodios/<tarea>/` y `~/skill_shield_datos/modelos/` (`IA_DATOS_DIR` o la variable `SKILL_SHIELD_DATOS` para cambiarlo). Mientras ahí no haya nada se siguen usando los `episodios/` del repo. Para moverlos: `mkdir -p ~/skill_shield_datos && mv episodios modelos ~/skill_shield_datos/ && git rm -r --cached episodios modelos`.

### Grabar (`[G]`, `[M]`)
- `[G]` empieza y termina un episodio (30 Hz). Al empezar dice qué del mundo se graba: `brazo real ✓ · mano real ✓ · táctiles ✓ · objeto ✓ (Pelota de tenis)`.
- **Objeto:** el que se puede coger más cercano a la palma (a menos de `IA_OBJETO_RADIO_M`) la primera vez que se ve. En «Mano Física + Gemelo Digital» lo ve la **cámara de cabeza** (la del modo autónomo, calibrada con `calibrar_camara_robot.py`), que se lanza sola si la teleoperación no la usa: pon `MULTICAMARA_ACTIVA = False` (teleoperación solo con la cámara A + guante). En el gemelo con sliders no se abre ninguna cámara. En simulación, los objetos de la mesa del taller `[3]`.
- **Cómo grabar un agarre:** empieza con la mano ya encima del objeto (unos 10–15 cm), con el objeto bien visible; acércate, cógelo y súbelo; termina con `[G]`. Variad la posición del objeto y la dirección de llegada (orientativo: ~50 demos por tarea). Alguna demo con un fallo **corregido** ayuda.
- `[M]`: el último episodio no salió bien → se marca como **fallido** y no se entrena con él (otra vez `[M]` lo desmarca).

### Entrenar (`[E]`, o en la terminal)
```bash
python -m skill_shield.ia.act coger_pelota                    # → modelos/coger_pelota.pth
python -m skill_shield.ia.act coger_pelota coger_destornillador --modelo agarre   # multitarea
python -m skill_shield.ia.act --todas                         # cada tarea, su modelo ([Shift+E])
```
- Decide solo qué usar según lo que haya en los episodios (lo imprime): con objeto, la acción es **«dónde poner la mano respecto al objeto»** (la IK analítica la pasa a joints), así el gesto se traslada si el objeto está en otro sitio. Sin objeto, joints como antes. Opciones: `--accion`, `--ventana`, `--pasos`, `--sin-tactil`, `--sin-objeto`, `--incluir-fallidos`, `--cpu`.
- Separa episodios enteros para validación, guarda checkpoints en `modelos/<nombre>/` y elige el mejor con un **ensayo en lazo cerrado** (la red reproduce cada episodio de validación viendo sus propias posturas). Informe en `modelos/<nombre>/informe.json`.
- Comparar después: `python -m skill_shield.ia.evaluar <modelo> --checkpoints`.
- GPU: con la RTX 50xx hace falta torch con CUDA 12.8 (`pip install torch --index-url https://download.pytorch.org/whl/cu128`). En la CPU también entrena (más lento).

### Reproducir (`[I]`)
- Elige el modelo (y la tarea, si es multitarea). Con el FR5 conectado arranca y para con el **pedal**; al arrancar **fija el objeto** que hay junto a la palma (luego la mano lo tapa). Se para sola cuando la red dice que ha terminado (`IA_PROGRESO_FIN`).
- Cada acción pasa por el **filtro de seguridad** (`politica.FiltroSeguridad`): límites, velocidad máxima (`IA_V_MAX_BRAZO`), choques, saltos de la IK (`IA_SALTO_MAX_RAD`) y la mano lejos del objeto (`IA_MAX_DIST_OBJETO_M`). Si algo no cuadra, para y lo dice en el panel.
- Los modelos antiguos (ciegos) se siguen reproduciendo como antes.
- Pruebas sin hardware: `python -m pytest tests/test_ia.py -q` (las del modelo necesitan torch).

SDK de FAIRINO en `fairino/Robot.py`: versión V4.0.0 (SDK V2.2.8), la del controlador V4.0.0.x.
