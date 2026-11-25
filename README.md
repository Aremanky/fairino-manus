# Skill Shield Nivel 6 — FR5 + OrcaHand (fairino-manus)

Teleoperación y funcionamiento autónomo de un brazo **Fairino FR5** con una **OrcaHand** en la brida,
montados en una peana móvil (soporte de hombros), con un **gemelo digital en MuJoCo**.

- **Teleoperación:** el brazo sigue al operario visto por cámaras **Luxonis OAK-D Pro** (o a una ventana
  de sliders) y los dedos siguen al guante **MANUS** (o a la cámara). El FR5 real solo se mueve con el
  pedal de hombre muerto.
- **Grabación y entrenamiento:** las demostraciones se graban y una red **ACT** aprende a repetirlas
  (`skill_shield/ia/`).
- **Funcionamiento autónomo:** con la cámara de cabeza, el robot detecta objetos (pelotas, herramientas
  con ArUco), hace un mapa de obstáculos, planifica (cuRobo en la GPU si está) y ejecuta la tarea
  mandada desde una **web** abierta a la red local.

Controladora del FR5: V4.0.0.x · SDK de FAIRINO en `fairino/Robot.py` (V4.0.0, SDK V2.2.8).

---
## Instalación y arranque

```bash
./arrancar_nivel6.sh                 # comprueba, instala SOLO lo que falte y arranca
./arrancar_nivel6.sh --comprobar     # solo dice qué falta
./arrancar_nivel6.sh --instalar      # instala lo que falte, sin arrancar
./arrancar_nivel6.sh --probar        # además prueba cuRobo y TOPP-RA (sin mover el robot)
```
El script se encarga de: el submódulo de los táctiles (`orca_core_touch`), el entorno `venv_nivel6`
(Python 3.12) con `requirements.txt` (lo vuelve a pasar si cambia), **cuRobo** (PyTorch con CUDA 13,
cuRobo en `externos/curobo`) y **TOPP-RA**. cuRobo y TOPP-RA son opcionales: si no se pueden instalar,
avisa y arranca igual.

A mano, si se prefiere:
```bash
python3.12 -m venv venv_nivel6 && source venv_nivel6/bin/activate
pip install -r requirements.txt
git submodule update --init orca_core_touch
python Skill_Shield_Nivel_6.py
```
- Paquetes del sistema: `sudo apt install zenity python3.12-tk`.
- **No instales `opencv-python` aparte**: choca con `opencv-contrib-python` y desaparece `cv2.aruco`.
- Todos los ajustes están en `config.py`, agrupados por bloques (`TELEOP_*`, `SLIDERS_*`, `GRABACION_*`,
  `IA_*`, `AUTONOMO_*`, `CALIB_*`, `FR5_*`, `ORCA_HW_*`…).
- Contraseña de la web del modo autónomo: variable `SKILL_SHIELD_WEB_CLAVE` o `clave_web.txt`.

---
## Hardware

| Pieza | Para qué | Dónde se configura |
|---|---|---|
| Fairino FR5 | brazo (ServoJ cada 8 ms en su propio proceso, `skill_shield/hardware/fr5_servo.py`) | `FR5_*` (carga, TCP, límites; IP `FR5_IP`, por defecto 192.168.58.2) |
| OrcaHand | mano, por USB (`skill_shield/hardware/orca_hardware.py`, `orca_mano/`) | `PUERTO_HARDWARE`, `ORCA_HW_*` |
| Táctiles de la OrcaHand | fuerza en las yemas (`skill_shield/hardware/puente_tactil.py`) | `ORCA_TACTIL_*` |
| Guante MANUS Core 3 | dedos del operario (`skill_shield/mano/manus_guante.py`, `calibraciones/calibrar_manus.py`) | `FUENTE_DEDOS`, `MANUS_*` |
| OAK-D Pro **A** | teleoperación y, en el modo autónomo, cámara fija de la mesa | `CAM_A_MXID`, `AUTONOMO_CAM_A_*` |
| OAK-D Pro **B** | cámara de cabeza del robot (objetos y obstáculos) | `CAM_B_MXID`, `AUTONOMO_CAM_MXID` |
| OAK-D Pro **C** | teleoperación (cruzada con la A) en «Grabación» | `CAM_C_MXID` |
| Pedal | hombre muerto; de momento la tecla **P** mantenida | `TELEOP_FR5_PEDAL` |

Varias OAK-D a la vez: cada una en un puerto USB3, mejor en controladores distintos. Para ver sus MxId:
`python -c "from skill_shield.vision.multicamara import listar_camaras as l; print(l())"`.

---
## Modos (menú de arranque)

| Botón | Mano | Brazo | FR5 real |
|---|---|---|---|
| 💻 Simulación Virtual | solo MuJoCo | cámaras | no (salvo `FR5_REAL_ACTIVO`) |
| 🎚️ Gemelo Digital (sliders) | guante → OrcaHand | ventana de sliders | sí, con ARMAR (`SLIDERS_*`) |
| 🤖 Mano Física + Gemelo Digital | guante → OrcaHand | cámaras (A + `CAM_B_MXID`) | sí, con el pedal (`TELEOP_REAL_*`) |
| 🎥 Grabación | guante → OrcaHand | cámaras A + C; B = cabeza | sí, con el pedal (`TELEOP_REAL_*`) |
| 🧠 Funcionamiento autónomo | la mueve la tarea | lo mueve la tarea | sí, con el pedal de la web (`AUTONOMO_*`) |
| ✖ Salir | | | |

Cerrar la ventana con la **X**, **Esc** o **Intro** es «Salir»: no arranca nada.

En todos los modos con FR5, el gemelo enseña la **pose real** del brazo y un brazo «fantasma» con el
objetivo. La seta física sigue siendo la parada de verdad.

### Mano Física + Gemelo Digital (teleoperación)
- **[A]** + T-pose: calibra. El robot **no** se mueve.
- **Pedal** (P mantenida): mientras se mantiene, el FR5 entra despacio hacia tu postura y te sigue
  (≤ `TELEOP_REAL_FR5_VEL_MAX_DEG_S`). Al soltarlo, o con **[S]**, **[A]**, **[I]** o **[Q]**: quieto.
- `TELEOP_REAL_FR5_SIMULACRO = True` para probar sin que se mueva. Sin pedal configurado
  (`TELEOP_FR5_PEDAL = None`) el FR5 real no se arma nunca.
- La IK evita el soporte, el suelo y las cajas de `TELEOP_FR5_OBSTACULOS` (la mesa).

### Gemelo Digital (sliders)
- Una ventana con un slider por articulación, en **grados de la controladora** (los de la WebApp).
- **[A]** en la ventana de la cámara activa la mano. Las cámaras no se abren (`SLIDERS_SIN_CAMARAS`).
- **ARMAR**: el brazo (gemelo y FR5 real) sigue a los sliders con la velocidad limitada. Solo se arma
  si el robot está dentro del rango de los sliders. **STOP / Esc / Espacio / cerrar la ventana** →
  desarmado.
- **Límite del J4** (`SLIDERS_LIMITE_J4`, `skill_shield/interfaz/limite_j4.py`): la mano nunca se acerca
  al giro del J4 a menos de `SLIDERS_J4_HOLGURA` (5 cm, unos |J5| ≤ 120°), porque eso rompe una pieza
  de la mano. Se comprueba el camino entero; si no pasa, el brazo se queda en la última postura válida
  y la ventana lo dice. Se prepara unos segundos al arrancar.
- Probar la ventana sola, sin robot: `python -m skill_shield.interfaz.sliders_brazo`.

### Grabación (teleoperación + objetos)
La teleoperación de arriba más lo que ve el modo autónomo con la cámara de cabeza (`skill_shield/ia/cabeza.py`):
- **Cámaras:** A y C cruzadas para teleoperar (sin `CAM_C_MXID`, solo la A) y la B como cabeza:
  herramientas con ArUco, pelotas y mapa de obstáculos. Extrínseca A↔C en `GRABACION_EXTRINSECAS_ARCHIVO`
  (sola con el cuerpo al pulsar `[A]`, o con el tablero: `python calibraciones/calibrar_extrinseca.py
  --mxid-b <MxId de C> --salida camaras_extrinsecas_ac.json`, que se guarda en `calibraciones/datos/`).
- **Objetivo:** el objeto que se puede coger más cercano a la palma (`[O]` cambia de objeto).
- **El brazo frena** contra los obstáculos del mapa (`GRABACION_FRENAR`), salvo alrededor del objetivo.
- **Agarre asistido** (`GRABACION_AGARRE_ASISTIDO`): `[L]` lleva la mano a la postura de agarre con el
  planificador del autónomo (con el pedal); al llegar, pisar el pedal pasa a modo **relativo** y empieza
  a grabarse; al soltarlo se guarda. `[M]` marca el episodio como fallido, `[K]` vuelve a la
  teleoperación normal, `[B]` bloquea el giro.

---
## Visión: YOLO de herramientas de obra y mango

`yolo_herramientas/` — segmentación de cada herramienta de obra e instalaciones (25 clases: taladro,
destornillador, martillo, amoladora…) y de su mango con YOLO26-seg (entorno aparte, `venv_yolo`).
Datasets públicos (HANDAL, PACO, Small Size Construction Tools, METU-ALET, Open Images, Roboflow), fotos
propias con la cámara de cabeza, unión, entrenamiento en la RTX 5080 y prueba en vivo: los pasos y comandos, en `yolo_herramientas/LEEME.md`.

---
## IA: grabar, entrenar y reproducir

`skill_shield/ia/` — red ACT que ve el objeto (desde la mano), los táctiles y la postura real de mano y brazo.
Datos fuera del repo: `~/skill_shield_datos/episodios/<tarea>/` y `~/skill_shield_datos/modelos/`
(`IA_DATOS_DIR` o `SKILL_SHIELD_DATOS`).

- **[G]** empieza/termina un episodio (30 Hz). **[M]**: el último salió mal (no se entrena con él).
- **[E]** entrena en segundo plano, o en la terminal:
  ```bash
  python -m skill_shield.ia.act coger_pelota                              # → modelos/coger_pelota.pth
  python -m skill_shield.ia.act coger_pelota coger_destornillador --modelo agarre   # multitarea
  python -m skill_shield.ia.act --todas
  python -m skill_shield.ia.evaluar <modelo> --checkpoints
  ```
- **[I]** reproduce (con el FR5, arranca y para con el pedal). Cada acción pasa por el filtro de
  seguridad (límites, velocidad, choques, saltos de la IK, distancia al objeto).

---
## Funcionamiento autónomo

`skill_shield/autonomo/` — su propio bucle (`modo.py`) con procesos vigilados que se reinician solos
si se caen (cámaras, web, gemelo de la web).

- **Web** en `http://<IP del PC>:8080/` desde cualquier equipo de la red (`sudo ufw allow 8080/tcp` si
  no carga). Lectura abierta (cámaras, gemelo, FR5, OrcaHand, táctiles, mapa); los mandos piden la
  contraseña.
- **Tarea:** se elige el objetivo (pelota o herramienta con ArUco) y se pulsa **Planificar** →
  **Ejecutar**. Pelota: la coge en pinza desde arriba, la aprieta un poco y la deja en el punto de
  dejada. Herramienta: finge el agarre, la «usa» en el punto de uso y la deja. Si el camino directo
  choca, busca un rodeo.
- **Pedal de la web:** botón grande (o barra espaciadora) que hay que mantener; si se suelta o se corta
  la red, en ≤ 0,35 s se para. Con «Pedal obligatorio» desactivado (`AUTONOMO_PERMITIR_SIN_PEDAL`) la
  tarea avanza sin pedal mientras la web siga en marcha; al reiniciarse la web vuelve a ser obligatorio.
- **FR5:** con `AUTONOMO_FR5_SIMULACRO = False` la tarea mueve el robot real (≤ `AUTONOMO_FR5_VEL_MAX_DEG_S`);
  se aborta si la controladora da error o si el robot no sigue lo mandado. Si el FR5 se apaga, se
  reconecta solo (`AUTONOMO_FR5_RECONECTAR`) y nunca se arma solo.
- **Sin FR5** (apagado, sin conexión o `AUTONOMO_CON_FR5 = False`): la tarea se planifica desde la pose
  del gemelo y se hace **solo en el gemelo** (arranca en `AUTONOMO_GEMELO_REPOSO_DEG`). Un plan hecho
  así nunca mueve el robot real, aunque luego se conecte.
- **OrcaHand:** en solo lectura hasta la primera tarea; entonces se conecta con par y, si se cae, se
  reconecta sola (`AUTONOMO_MANO_REINTENTO_S`). Si se cae con una tarea en marcha, la tarea se aborta.
- **Cámaras:** cabeza = B (`AUTONOMO_CAM_MXID`); A fija mirando la mesa (`AUTONOMO_CAM_A`), que mide la
  pelota de cerca al planificar y al afinar y aporta su profundidad al mapa.
- **Mapa de obstáculos** (`mapa.py`): TSDF de la profundidad, sin el robot, el suelo ni el objetivo.
  **Escena fija** (`escena_fija.py`): mesas y muebles escaneados una vez como cajas (`escena_fija.json`).
- **Planificador:** comprueba siempre la mano entera contra el brazo (giro del J4 incluido), el soporte,
  las mesas y el mapa (`choque_mano.py`, `mundo_tarea.py`); cuRobo con varias soluciones a la vez si está
  instalado; TOPP-RA para los tramos articulares si está.
- Para salir: cerrar la ventana de MuJoCo o Ctrl+C.
- Probar por separado: `python -m skill_shield.autonomo.camara --ventana` · `python -m skill_shield.autonomo.web`.

---
## Calibraciones

Todo lo de calibración está en `calibraciones/`: los scripts, sus resultados en `calibraciones/datos/` y los
tableros para imprimir en `calibraciones/tableros/`. Qué es cada uno y cuándo repetirlo: `calibraciones/Leeme.txt`.
Se lanzan desde la raíz del repo:

| Qué | Cómo | Resultado (en `calibraciones/datos/`) |
|---|---|---|
| Cámara de cabeza → robot | `python calibraciones/calibrar_camara_robot.py --auto` (tablero ChArUco 5x4 en la mano, P mantenida) · comprobar con `--verificar` | `camara_robot.json` |
| Cámara A → robot | `python calibraciones/calibrar_camara_robot.py --camara a --auto` | `camara_a_robot.json` |
| Calibración a mano | `--manual` (arrastre en la WebApp + ESPACIO) o `--manual --sliders` (con la ventana de sliders y el pedal) · `--recalcular` con las muestras guardadas | |
| Escala de la profundidad | `python calibraciones/calibrar_escala_profundidad.py` | valor de `AUTONOMO_CAM_ESCALA_PROF` (se copia a `config.py`) |
| Extrínseca entre cámaras | `python calibraciones/calibrar_extrinseca.py` (tablero ChArUco 7x5) | `camaras_extrinsecas*.json` |
| Guante MANUS | `python calibraciones/calibrar_manus.py` | `manus_calibracion.json` |
| Carga de la brida | `python diagnosticos/identificar_carga.py` (`--leer` = la de la controladora) | `FR5_CARGA_*` |

En la calibración, el robot solo se mueve con el pedal, a ≤ `CALIB_ROBOT_VEL_MAX_DEG_S`, y cada trayecto
se comprueba antes (mano y tablero contra el brazo, el soporte, la cabeza, las cajas y el suelo; bloque
`CALIB_*`). Generar el tablero: `--generar-tablero` (queda en `calibraciones/tableros/`; imprimir al 100 %,
medir un cuadro con calibre). Marcadores de las herramientas: `python -m skill_shield.autonomo.herramientas
--generar` (y `--probar`). Cierra antes el modo autónomo: la cámara y el robot tienen que estar libres.
Las calibraciones que aún estén en la raíz (de antes de esta carpeta) las mueve solo `./arrancar_nivel6.sh`.

---
## Diagnósticos (`diagnosticos/`)

- `ajustar_gemelo.py` — que el gemelo coincida con el robot montado (giro de la base, instalación).
- `medir_pelota.py --hoja` y luego `--diametro 0.070 --lado 0.035` — error real de la medida de la pelota.
- `resumen_intentos.py [--desde AAAA-MM-DD]` — % de éxito de la tarea (cada intento se apunta en `diagnosticos/intentos.jsonl`).
- `ver_estorbo.py [plan_fallido_….npz]` — qué cubos del mapa no dejan coger la pelota.
- `banco_planes.py --guardar antes.json` / `--comparar antes.json` — comparar planes antes y después de un cambio.
- `estimar_cdm_mano.py` — primera estimación del centro de masas de la mano.
- `probar_curobo.py [--n 50]` — cuRobo con el FR5 + OrcaHand en la GPU, sin mover el robot (lo lanza `./arrancar_nivel6.sh --probar`).
- `probar_haptica_tactil.py [--dedos | --umbral | --mock]` — táctiles de la OrcaHand → vibración del guante, sin el programa principal.
- `python -m skill_shield.autonomo.choque_mano [--mapa]` — huecos de la mano en posturas guardadas.

---
## Utilidades (`utilidades/`)

- `python utilidades/episodio_poses.py utilidades/episodio_poses.txt [--grabar TAREA]` — el brazo va de pose en
  pose con Enter y la mano se teleopera con el guante (las poses salen del botón «Imprimir pose» de la ventana de
  sliders).

---
## Pruebas

Sin hardware (ni robot, ni mano, ni guante, ni cámaras):
```bash
python -m pytest tests -q
```
Las de cuRobo y algunas de la IA necesitan la GPU / torch y se saltan si no están.

---
## Estructura

```
Skill_Shield_Nivel_6.py      programa principal: menú y bucle de teleoperación / grabación / IA
config.py                    TODOS los ajustes
arrancar_nivel6.sh           instala lo que falte y arranca
calibraciones/               scripts de calibración (calibrar_*.py) + Leeme.txt
  datos/                     sus resultados: camara_robot.json, camaras_extrinsecas.json, manus_calibracion.json…
  tableros/                  tableros ChArUco para imprimir (5x4 de la palma, 7x5 de la extrínseca)
skill_shield/
  autonomo/                  modo autónomo: modo.py (bucle), tarea.py, ejecucion.py, planificador,
                             choque_mano.py, mapa.py, camara.py, camara_a.py, web.py + estatico/
  autonomo/curobo_fr5/       integración con cuRobo
  teleop_fr5/                teleoperación del FR5: cinemática, IK, colisiones, pedal
  hardware/                  fr5_servo (proceso ServoJ), orca_hardware (OrcaHand por USB: envío continuo,
                             arranque suave), puente_tactil (táctiles; lo lanza el guante / el modo autónomo)
  interfaz/                  ventana de sliders, límite del J4, HUD
  ia/                        grabación, ACT, política, agarre asistido, cámara de cabeza
  vision/ brazo/ mano/       cámaras OAK-D, MediaPipe, retargeting del brazo y de la mano
  simulacion/                entornos de MuJoCo (taller, taladro)
diagnosticos/                herramientas de diagnóstico y pruebas con hardware (probar_*.py)
utilidades/                  episodio por poses (episodio_poses.py + .txt)
yolo_herramientas/           YOLO de herramientas y mango: datasets, conversión, entrenamiento (LEEME.md)
tests/                       pruebas sin hardware
fairino_description/         modelo MJCF del FR5 + soporte + OrcaHand (lo usa todo)
orcahand_description-main/, taladro/   mallas de la OrcaHand y del taladro (las incluye el MJCF)
curobo_fr5/                  URDF y configuración del robot para cuRobo (generados desde el MJCF)
orca_mano/                   config.yaml + calibration.yaml de la OrcaHand (los lee orca_hardware)
orca_core-main/, orca_core_touch/   SDK de la OrcaHand (viejo para los motores; nuevo, submódulo, para los táctiles)
MANUS_Core_3.2.1_SDK_Linux/  SDK del guante
fairino/                     SDK del FR5
episodios/, modelos/         datos antiguos de la IA (los nuevos van a ~/skill_shield_datos)
marcadores/                  marcadores ArUco de las herramientas para imprimir
docs/                        manual de Skill Shield Nivel 6 (PDF)
web_windows/                 ejecutable de Windows de la web (gemelo sin robot): SkillShield_Web.exe,
                             mujoco.dll, modelo/, web/, licencias/ y su código en fuente/
```
