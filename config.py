#!/usr/bin/env python3
"""
Skill Shield Nivel 6 - Panel de Configuración Global
===================================================
Modifica los parámetros de este archivo para alterar la simulación o el hardware
sin necesidad de tocar el código lógico principal.
"""
import os
import math
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── CONFIGURACIÓN DE ENTORNO Y RUTAS ──────────────────────────────────────────
MODO_EJECUCION = "SIMULACION"  # Opciones: "SIMULACION" | "GEMELO_SLIDERS" | "HARDWARE_REAL" | "AUTONOMO" (lo elige el menú)

# Puerto mano
PUERTO_HARDWARE = "/dev/ttyUSB0"

# Ruta al MJCF del brazo ARCTOS + mano OrcaHand ya fusionados
RUTA_XML  = os.path.join(_BASE_DIR, "arctos_description-main",
                          "arctos_urdf_description", "urdf", "arctos_mjcf.xml")

# Ruta al logo IDI / Grupo Elecnor (Apuntando a la nueva carpeta Nivel6)
RUTA_LOGO = os.path.join(_BASE_DIR, "orcahand_description-main", "v2", "I+D+I_DENOMIN_GRUPO_RGB.png")

# ── PARÁMETROS DE LA CÁMARA OAK-D Y PROYECCIÓN ────────────────────────────────
RES            = 300    # (LEGADO) resolución cuadrada antigua. Ya no se usa en el teleop;
                        # se deja por si algún script auxiliar (p.ej. test_ia_mano.py) la importa.

# Formato 4:3 (SENSOR COMPLETO). El sensor de color de la OAK-D (IMX378) es 4:3.
# Al pedir 1080p (16:9) el firmware RECORTA franjas arriba y abajo: se pierde ~24 %
# de altura de visión. Pidiendo 12 MP (4056x3040, 4:3) y escalando en el ISP se
# conserva TODO el sensor → VFOV pasa de ~42° a ~55° (≈ +30 % de altura).
# Esto es lo que te permite levantar el brazo sin que la mano se salga del cuadro.
ANCHO_CAM      = 640    # píxeles de ancho del fotograma (color y profundidad)
ALTO_CAM       = 480    # píxeles de alto  (4:3 → sensor completo, sin recorte vertical)
ESCALA_VENTANA = 1.35   # Zoom de la ventana OpenCV (640x480 * 1.35 = 864x648)
F_PX           = 465.0  # Focal en px a 640 de ancho. SOLO es respaldo: al arrancar se lee
                        # la focal real de la calibración de la cámara.
                        # (El HFOV no cambia al pasar a 4:3, así que esta focal sigue valiendo.)

# Modo de sensor del color:
#   True  → 12 MP + setIspScale: FOV vertical COMPLETO (recomendado, brazo levantado)
#   False → 1080p clásico (16:9, recortado arriba/abajo). Solo si el ISP te da problemas.
CAM_FOV_COMPLETO = True
CAM_FPS          = 30     # fija los FPS del sensor; si tu bucle va lento, bájalo a 20

# ── DETECCIÓN DE CUERPO (MediaPipe Pose) ──────────────────────────────────────
MOSTRAR_ESQUELETO_CUERPO = True   # True = dibuja el esqueleto del cuerpo en la ventana
POSE_COMPLEJIDAD         = 1      # 0 = rápido | 1 = equilibrado | 2 = más preciso y lento
POSE_CONF_DETECCION      = 0.6
POSE_CONF_SEGUIMIENTO    = 0.45   # BAJADO: con el bucle a ~12 FPS un 0.6 hace que Pose
                                  # pierda el "lock" y reinicie la detección constantemente
POSE_MEDIR_FPS           = True   # imprime los FPS reales del bucle de teleoperación

# ── DETECCIÓN DE PELOTA POR COLOR HSV ─────────────────────────────────────────
COLORES_HSV = {
    "pelota_amarilla": ((22,  80, 100), (38, 255, 255)),
    "pelota_roja":     ((0,   95,  95), (10, 105, 105)),
    "verde":           ((36,  50,  70), (89, 255, 255)),
    "pelota_azul":     ((94,  80,  20), (126, 255, 255)),
}
COLOR_PELOTA       = "pelota_amarilla"
MIN_AREA_PELOTA    = 800
N_FRAMES_DEBOUNCE  = 10

# ── PUNTOS DE APARICIÓN (SPAWN) DINÁMICOS POR POSE ────────────────────────────
PUNTOS_SPAWN = {
    "arriba": {
        "pelota":         {"pos": ( 0.0, 0.08, 0.30), "euler": (0.0, 0.0, 0.0)},
        "destornillador": {"pos": ( 0.0, 0.08, 0.30), "euler": (1.5708, 0.0, -1.70)},
        "soporte":        {"pos": ( 0.0, 0.09, 0.17), "euler": (0.0, 0.0, -0.3)}
    },
    "izquierda": {
        "pelota":         {"pos": ( 0.0, -10, 0.20),  "euler": (0.0, 0.0, 0.0)},
        "destornillador": {"pos": ( 0.185, 0.035, 0.0), "euler": (0.0, 0.0, 0.0)},
        "soporte":        {"pos": ( 0.0, 0.0, 0.10),  "euler": (0.0, 0.0, 0.0)}
    },
    "abajo": {
        "pelota":         {"pos": ( 0.0, 0.2, -0.05), "euler": (0.0, 0.0, 0.0)},
        "destornillador": {"pos": ( -0.005, 0.2, -0.05), "euler": (1.5708, 0.0, -1.70)},
        "soporte":        {"pos": ( 0.0, 0.0, 0.10),  "euler": (0.0, 0.0, 0.0)}
    },
    "derecha": {
        "pelota":         {"pos": ( 0.0, -0.20, 0.25), "euler": (0.0, 0.0, 0.0)},
        "destornillador": {"pos": ( 0.02, -0.20, 0.0), "euler": (1.5708, 0.0, -1.70)},
        "soporte":        {"pos": ( 0.0, 10.0, -10.10), "euler": (0.0, 0.0, 0.0)}
    }
}

# ── BRAZO ARCTOS: JOINTS, LÍMITES Y OFFSET DE ACTUADORES ──────────────────────
# En arctos_mjcf.xml el <actuator> define PRIMERO los 6 del brazo y DESPUÉS
# los 17 de la mano. OFFSET_MANO es cuánto hay que desplazar cualquier índice
# "de mano" (0..16, los mismos de siempre) para caer en el actuador correcto
# de datos.ctrl[] una vez cargado el modelo fusionado.
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

LIMITES_CTRL_BRAZO = [
    (-3.14, 3.14),   # joint1 (base)
    (-2.5,  2.5),    # joint2 (hombro)
    (-2.5,  2.5),    # joint3 (codo)
    (-3.14, 3.14),   # joint4 (muñeca 1)
    (-3.14, 3.14),   # joint5 (muñeca 2)
    (-3.14, 3.14),   # joint6 (muñeca 3)
]

# Postura de reposo del brazo mientras no haya IK conectada (paso 1) o cuando
# no se esté siguiendo la cámara. AJUSTA ESTOS VALORES en el visor MuJoCo a
# una postura cómoda, sin autocolisión con la mano ni con la mesa.
POSICION_STANDBY_BRAZO = [0.0, -0.3, 0.6, 0.0, 0.0, 0.0]

OFFSET_MANO = len(ARM_JOINTS)  # = 6

# ── ACTUADORES Y LÍMITES ANATÓMICOS DE SEGURIDAD (MANO) ───────────────────────
N_ACT = 17

LIMITES_CTRL = [
    (-1.134,  0.610),  #  0: muñeca
    (-0.20,   0.50),   #  1: meñique abd
    ( 0.0,    1.80),   #  2: meñique mcp
    ( 0.0,    1.80),   #  3: meñique pip
    (-0.20,   0.50),   #  4: anular abd
    ( 0.0,    1.80),   #  5: anular mcp
    ( 0.0,    1.80),   #  6: anular pip
    ( 0.0,    0.0),    #  7: corazón abd (fijo)
    ( 0.0,    1.80),   #  8: corazón mcp
    ( 0.0,    1.80),   #  9: corazón pip
    (-0.50,   0.50),   # 10: índice abd
    ( 0.0,    1.80),   # 11: índice mcp
    ( 0.0,    1.80),   # 12: índice pip
    (-0.785,  1.85),   # 13: pulgar oposición/CMC
    (-0.35,   1.35),   # 14: pulgar abd
    (-0.15,   1.80),   # 15: pulgar mcp  ← permite extensión activa
    ( 0.0,    1.80),   # 16: pulgar ip
]

POSICION_STANDBY = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    -0.0979, 0.641,  0.00,  0.175
]

# ── PARÁMETROS BIOMECÁNICOS Y FILTROS ─────────────────────────────────────────
UMBRAL_BLOQUEO    = 0.50
UMBRAL_LIBERACION = 0.10

# Ganancias globales
FUERZA_NUDILLOS   = 1.35
FUERZA_PUNTAS     = 1.45
FUERZA_ABD        = 1.50

# Ganancias INDIVIDUALES por dedo (multiplican sobre la global)
# 1.0 = sin cambio respecto a la ganancia global
# Sube un dedo concreto sin tocar el resto
GANANCIA_DEDOS = {
    "menique_mcp":  1.00,   # ctrl[2]
    "menique_pip":  1.00,   # ctrl[3]
    "anular_mcp":   1.10,   # ctrl[5]
    "anular_pip":   1.10,   # ctrl[6]
    "corazon_mcp":  1.05,   # ctrl[8]
    "corazon_pip":  1.05,   # ctrl[9]
    "indice_mcp":   1.00,   # ctrl[11] — referencia
    "indice_pip":   1.00,   # ctrl[12]
}

# ── UMBRALES Y GANANCIAS DEL PULGAR ───────────────────────────────────────────
# CRÍTICO: estas 4 constantes son usadas directamente en retarget_a_ctrl()
UMBRAL_EXT_PULGAR_MCP = 0.15   # por debajo → zona de extensión activa
UMBRAL_EXT_PULGAR_IP  = 0.25   # por debajo → filtra temblor de punta
PULGAR_GANANCIA_MCP   = 1.70   # amplifica flexión del nudillo
PULGAR_GANANCIA_IP    = 1.90   # amplifica flexión de la punta

# ── UMBRALES DE PINZA PREDEFINIDA (CON HISTÉRESIS) ────────────────────────────
PINZA_UMBRAL_ON = {
    "indice": 0.030, "corazon": 0.040, "anular": 0.040, "menique": 0.035,
    "corazon_anular": 0.040, "indice_corazon": 0.040, "anular_menique": 0.040
}

PINZA_UMBRAL_OFF = {
    "indice": 0.040, "corazon": 0.050, "anular": 0.050, "menique": 0.045,
    "corazon_anular": 0.050, "indice_corazon": 0.050, "anular_menique": 0.050
}

_UMBRAL_COMBO_ON   = {"corazon": 0.035, "anular": 0.040}
_UMBRAL_COMBO_OFF  = {"corazon": 0.045, "anular": 0.050}
_UMBRAL_COMBO2_ON  = {"indice": 0.040, "corazon": 0.045}
_UMBRAL_COMBO2_OFF = {"indice": 0.050, "corazon": 0.055}
_UMBRAL_COMBO3_ON  = {"anular": 0.040, "menique": 0.040}
_UMBRAL_COMBO3_OFF = {"anular": 0.050, "menique": 0.050}

PINZA_DIST_MAX = 0.120

# ── DETECCIÓN DE LA MANO CON RECORTE (arregla las pinzas con el cuerpo en cuadro) ─
# Con la cámara abierta a todo el cuerpo pasan dos cosas malas para MediaPipe Hands:
#   1) Se ven LAS DOS manos y, con max_num_hands=1, puede engancharse a la que NO es
#      (la que cuelga quieta) → los dedos del robot no siguen a los tuyos y la pinza
#      nunca se activa.
#   2) A ~2 m la mano ocupa ~40 px: las yemas se estiman con errores de varios cm y la
#      distancia pulgar–dedo casi nunca baja del umbral de pinza (3-4 cm).
# Solución: MediaPipe Pose dice DÓNDE está tu mano (la del BRAZO_HUMANO) y Hands se
# ejecuta solo sobre un recorte cuadrado alrededor de ella, ampliado a MANO_RECORTE_PX.
MANO_RECORTE        = True    # False = comportamiento anterior (Hands en toda la imagen)
MANO_RECORTE_HD     = True    # True = el recorte sale de la imagen del sensor a 1352x1014
                              # (2,1x más píxeles). Necesita CAM_FOV_COMPLETO = True y USB3;
                              # si la cámara va lenta o da errores de enlace, pon False.
MANO_RECORTE_PX     = 256     # lado (px) al que se amplía el recorte antes de pasarlo a Hands
MANO_RECORTE_FACTOR = 3.0     # tamaño del recorte = FACTOR × (muñeca → nudillos) en la imagen
MANO_RECORTE_MIN    = 0.12    # lado mínimo del recorte, como fracción del alto de la imagen

# Diagnóstico de pinzas: imprime la distancia del pulgar a cada yema (cm), el umbral,
# si la pinza está activa y de dónde sale la mano (recorte / imagen completa).
DEBUG_PINZA        = True
DEBUG_PINZA_CADA_S = 0.5

# ── LANDMARKS MEDIAPIPE INDICES ──────────────────────────────────────────────
LM_MUNECA      = 0
LM_INDICE_MCP  = 5
LM_CORAZON_MCP = 9
LM_ANULAR_MCP  = 13
LM_MENIQUE_MCP = 17

# ── LANDMARKS MEDIAPIPE POSE (cuerpo) ─────────────────────────────────────────
# "IZQ"/"DER" son el lado del PROPIO SUJETO (no de la imagen).
PS_HOMBRO_IZQ, PS_HOMBRO_DER = 11, 12
PS_CODO_IZQ,   PS_CODO_DER   = 13, 14
PS_MUNECA_IZQ, PS_MUNECA_DER = 15, 16
PS_CADERA_IZQ, PS_CADERA_DER = 23, 24

# ── ÁNGULOS DEL BRAZO HUMANO (paso 3: de momento solo se calculan y se muestran) ─
BRAZO_HUMANO         = "derecho"   # "derecho" | "izquierdo" — brazo que se va a imitar
POSE_VISIBILIDAD_MIN = 0.35        # por debajo, MediaPipe "adivina" el punto → se ignora
                                   # (0.5 era demasiado estricto: al girar el torso o
                                   #  levantar el brazo, el hombro lejano baja de 0.5 y
                                   #  el brazo entero se descartaba)
DEBUG_ANGULOS_BRAZO  = True        # imprime los ángulos en la terminal
DEBUG_ANGULOS_CADA_S = 0.5         # cada cuántos segundos imprimir

# De dónde sale la PROFUNDIDAD (eje que se aleja de la cámara) de los puntos del brazo:
#   "estereo"   → profundidad REAL medida por las dos cámaras de la OAK-D (recomendado)
#   "mediapipe" → profundidad ESTIMADA por la red neuronal a partir de una sola imagen
# En la terminal se imprimen SIEMPRE las dos, para poder compararlas.
BRAZO_FUENTE_3D = "estereo"
PROF_RADIO_PX   = 4      # radio (px) del parche donde se mide la profundidad de cada articulación
                         # (subido de 5: el antebrazo es fino y un parche 11x11 se queda
                         #  sin píxeles válidos en cuanto hay un hueco del estéreo)
PROF_PERCENTIL  = 15     # percentil bajo = "el punto más cercano" → evita coger el fondo tras el brazo
PROF_MIN_PIXELES = 4     # píxeles con profundidad válida mínimos en el parche (antes 6 fijo)
PROF_Z_MIN      = 0.25   # m — por debajo, el estéreo no puede medir (línea base de la OAK-D)
PROF_Z_MAX      = 5.0    # m
PROF_DISPERSION_Z_MAX = 1.4   # m — cuánto puede alejarse una articulación de la mediana
                              # del resto antes de descartarla. Con 0.8 m, al ESTIRAR el
                              # brazo hacia la cámara la muñeca quedaba fuera y el brazo
                              # se perdía justo cuando más lo necesitas.

# Estéreo con precisión sub-píxel: sin ello la profundidad va "a escalones" de varios cm.
STEREO_SUBPIXEL = True

# Post-procesado del mapa de profundidad EN LA PROPIA CÁMARA (no cuesta CPU).
# Rellena los agujeros del brazo, que son la causa nº1 de "se pierde a ratos".
STEREO_FILTROS      = True
STEREO_HOLE_FILLING = 2       # radio de relleno de huecos (0 = desactivado, 2-4 típico)
STEREO_TEMPORAL     = True   # filtro temporal: rellena más, pero deja ESTELA en
                              # movimientos rápidos. Actívalo solo si te mueves despacio.
STEREO_EXTENDED     = False   # disparidad extendida: baja la distancia mínima a ~15 cm.
                              # En algunas versiones de depthai choca con subpixel; si al
                              # activarlo peta, pon STEREO_SUBPIXEL = False.

# ── ROBUSTEZ CUANDO EL BRAZO APUNTA A LA CÁMARA ───────────────────────────────
# Con el brazo estirado hacia la cámara, el codo queda tapado por la mano/antebrazo
# y MediaPipe lo coloca en un sitio falso (p.ej. a la altura del ombligo).
# Se detecta porque las LONGITUDES de los huesos no cuadran, y entonces el codo
# se reconstruye con geometría a partir de hombro y muñeca.
#   · LONG_BRAZO_MANUAL = None            → las longitudes se aprenden solas (mediana)
#   · LONG_BRAZO_MANUAL = (0.30, 0.26)    → tus medidas reales en metros (brazo, antebrazo)
LONG_BRAZO_MANUAL = None
LONG_BRAZO_TOL    = 0.25   # un hueso puede desviarse ±25 % de su longitud habitual

# Los huesos se aprenden AL PULSAR [A]: durante ~1 s hay que mantener el brazo
# estirado en T-pose (así se ve entero y sin acortarse por la perspectiva).
CALIB_HUESOS_FRAMES = 30   # fotogramas válidos que se promedian

# Cuando el frame actual NO puede calcular ángulos del brazo (profundidad ruidosa
# en un solo punto, oclusión de un instante...), en vez de congelar el brazo en
# su última posición comandada se sigue usando el último ángulo VÁLIDO durante
# hasta este tiempo (segundos). Pasado ese tiempo, se asume que el brazo se ha
# perdido de verdad y se deja de actualizar. Sube este valor si ves "paradas"
# breves pero molestas; bájalo si el brazo tarda en reaccionar a una pérdida real.
ANG_BRAZO_HOLD_S = 0.9   # SUBIDO de 0.4: a ~12 FPS reales, 0.4 s son solo 5 fotogramas,
                         # así que cualquier bache del estéreo congelaba el brazo.

# Los hombros se miden en el PECHO (a esta fracción del camino hacia el centro del torso)
# y no en el propio hombro: si el brazo apunta a la cámara, la mano tapa el hombro en el
# mapa de profundidad y se leería la profundidad de la mano, deformando el marco del torso.
HOMBRO_PROF_HACIA_TORSO = 0.6

# ── QUE EL BRAZO NO SE "PIERDA" POR UN HUECO DEL ESTÉREO ──────────────────────
# Antes, si UNA sola articulación (casi siempre la muñeca o el codo: son finos y se
# mueven) no tenía profundidad estéreo válida en ese fotograma, se descartaba el
# brazo entero → "sin datos" aunque la cámara lo viera perfectamente.
# Ahora cada articulación prueba, por orden:
#   E  → profundidad estéreo en su parche normal
#   E+ → estéreo en un parche el doble de grande (sale de un hueco pequeño)
#   M  → profundidad RELATIVA de MediaPipe anclada a un punto que sí tiene estéreo
#        (hombro/cadera): z = z_ancla_estéreo + (z_mp_punto − z_mp_ancla)
# Así solo se cae a "sin datos" si no hay estéreo ni en hombros ni en caderas.
PROF_RELLENO_MEDIAPIPE = True

# ── DATASET E IA (ARCHIVOS) ───────────────────────────────────────────────────
ARCHIVO_DATASET = "movimientos_mano.csv"
ARCHIVO_PESOS   = "cerebro_orca.pth"

# ── MAPEO MUÑECA HUMANA → OBJETIVO DEL EFECTOR DEL BRAZO ARCTOS ──────────────
# Punto (en metros, en el frame BASE del ARCTOS) donde debe estar el efector
# final (Link_6_1) cuando la muñeca humana está en su posición NEUTRA
# (la que tenga en el instante en que actives la cámara con [A]).
# Ajusta esto a un punto cómodo y alcanzable del brazo (ni muy cerca de la
# base ni estirado al límite).
ARM_OBJETIVO_NEUTRO = (0.30, 0.0, 0.30)   # (X adelante, Y lateral, Z altura)

# Cuántos metros se mueve el efector del brazo por cada metro que se mueve
# la muñeca humana respecto a la cámara. 1.0 = movimiento 1:1.
ARM_ESCALA_MOVIMIENTO = (1.0, 1.0, 1.0)   # (X, Y, Z) del robot

# La cámara entrega (x_m, y_m, z_m) con: x = lateral (+derecha),
# y = vertical (+hacia ABAJO, porque el eje Y de imagen crece hacia abajo),
# z = profundidad (+alejándose de la cámara).
# Aquí decides qué eje de CÁMARA alimenta cada eje del ROBOT, y con qué signo.
# Son valores de partida — muy probablemente tocará invertir algún signo tras
# la primera prueba (ver más abajo cómo hacerlo).
#
# AJUSTE (tras la primera prueba con el brazo): el mapeo "intuitivo" de
# arriba (adelante←profundidad, lateral←x) resultó estar CRUZADO en la
# práctica — mover la muñeca hacia la cámara movía el brazo a la derecha, y
# moverla a la izquierda lo movía hacia delante. Así que aquí abajo la
# profundidad ("z") alimenta ahora el lateral del robot, y el lateral de
# cámara ("x") alimenta el "adelante" del robot.
#
# Los signos son una PRIMERA HIPÓTESIS para que "izquierda → delante" y
# "hacia la cámara → derecha" salgan en el sentido correcto — pruébalo y, si
# alguno sale invertido (p.ej. "hacia delante" te lleva hacia atrás), cambia
# solo el signo de esa fila (+1.0 ↔ -1.0), no el eje de cámara.
ARM_MAPEO_EJES = {
    "robot_x": ("x", 1.0),   # "adelante" del robot ← lateral de la cámara
    "robot_y": ("z", 1.0),   # lateral del robot ← profundidad de la cámara
    "robot_z": ("y", -1.0),   # altura del robot ← vertical de cámara (invertido)
}

# Seguridad: nunca apuntar el efector a más de este radio desde la base,
# ni por debajo de esta altura (evita objetivos de IK imposibles o que
# choquen contra la mesa).
ARM_RADIO_MAX  = 0.55   # metros
ARM_Z_MINIMA   = 0.05   # metros

# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — IMITACIÓN DEL BRAZO HUMANO POR ÁNGULOS ARTICULARES (ARCTOS)
# ══════════════════════════════════════════════════════════════════════════════
# "imitacion" → el brazo copia los ÁNGULOS del brazo humano (joint1..3), sin IK.
# "ik"        → el modo anterior: la punta del brazo persigue tu muñeca (IK).
MODO_BRAZO = "imitacion"

# Cómo se convierte cada ángulo humano (en RADIANES) en el ángulo de un joint:
#
#       q_joint = offset + signo * escala * angulo_humano
#
# Sale de la geometría del arctos_mjcf.xml (con todos los q = 0 el brazo apunta
# HACIA ARRIBA y el antebrazo horizontal hacia delante; "delante" del robot = -Y):
#   · joint1 (giro de base, eje +Z): q>0 gira el brazo hacia la IZQUIERDA del robot.
#       Tu azimut es + hacia FUERA (tu derecha) → signo -1  (tu derecha = su derecha).
#   · joint2 (hombro, eje +X): q=0 brazo vertical hacia arriba; q crece hacia delante y abajo.
#       Tu elevación es 0 con el brazo colgando y 180° con él arriba → q2 = π - elevación.
#   · joint3 (codo, eje -X): con q≈1.52 el antebrazo sigue la línea del brazo (codo
#       estirado); q baja al flexionar → q3 = 1.52 - flexión.
#
# Si al probar un joint se mueve al revés, cambia SOLO su signo (+1 ↔ -1).
# Si un joint sale "descentrado" (p.ej. el codo estirado no queda recto), retoca su offset.
ARM_IMITACION = {
    #  joint      (ángulo humano, signo, offset [rad],  escala)
    "joint1": ("azimut",    -1.0, 0.0,      1.0),
    "joint2": ("elevacion", -1.0, math.pi,  1.0),
    "joint3": ("codo",      -1.0, 1.52,     1.0),
}

# Límites de SEGURIDAD para la imitación (rad). Más estrechos que los del actuador:
#   · joint2 máx 2.3 evita que el brazo baje del todo y golpee la mesa
#   · joint3 evita plegar el codo contra el propio brazo
IMITAR_LIMITES = {
    "joint1": (-3.14, 3.14),
    "joint2": (-2.5, 2.5),
    "joint3": (-2.5, 2.5),
}

IMITAR_VEL_MAX   = 1.5   # (antes 2.5; con física en tiempo real era demasiado brusco) rad/s: velocidad máxima de cada joint (protege de saltos)

# ── T-POSE COMO REFERENCIA DE POSTURA (arregla "el brazo se queda abajo") ─────
# La calibración de huesos solo aprendía LONGITUDES. Ahora, en esos mismos ~30
# fotogramas de T-pose, también se mide el az/elev/codo que ve la cámara y se
# compensa la diferencia con la T-pose ideal, de modo que
#       tu T-pose  ==  T-pose del brazo simulado
# aunque la cámara esté inclinada, el hombro se mida algo alto, o dejes caer
# un poco el codo al calibrar.
IMITAR_CALIBRAR_TPOSE = True
T_POSE_HUMANA = (90.0, 90.0, 0.0)   # (azimut, elevación, codo) en GRADOS de una T-pose
                                     # perfecta: brazo horizontal, hacia fuera, estirado.
                                     # Solo se usa si la T-pose medida no es válida y en
                                     # el modo "absoluto".

# CÓMO SE RELACIONA TU T-POSE CON EL BRAZO SIMULADO:
#   "relativo" → la pose en la que está el brazo simulado al pulsar [A] (tu
#                POSICION_STANDBY_BRAZO, la "T-pose" de reposo del sim) SE CORRESPONDE
#                con tu T-pose medida. El brazo NO se mueve al calibrar; luego copia
#                tus movimientos como incrementos respecto a esa pose:
#                    q = q_reposo + signo·escala·(ángulo_humano − ángulo_T_medido)
#   "absoluto" → el brazo copia tus ángulos directamente (q = offset + signo·escala·ángulo),
#                así que tu T-pose real equivale a la T-pose GEOMÉTRICA del robot
#                (brazo horizontal a su derecha) y el sim salta a ella.
IMITAR_REFERENCIA = "absoluto"

# Solo modo "absoluto": al pulsar [A] llevar el brazo simulado a su T-pose geométrica.
IMITAR_TPOSE_SIM_AL_CALIBRAR = True

AZIMUT_ELEV_MIN = 35.0

# ── GRAVEDAD DE LA MANO ──────────────────────────────────────────────────────
# En arctos_mjcf.xml los links del brazo llevan gravcomp="1", pero los cuerpos de
# la OrcaHand (dentro del <include>) NO lo heredan: gravcomp es un atributo de
# cada <body>, no de la clase. Resultado: la mano pesa sobre joint2/joint3 y el
# brazo "cuelga" por debajo de lo ordenado (kp=500/300 no basta para masas de kg).
# True = se compensa la gravedad de TODOS los cuerpos que cuelgan de mano_mount.
HAND_GRAVCOMP = True
IMITAR_SUAVIZADO = 3     # fotogramas de media móvil sobre los ángulos objetivo

# ── PASO 5 — MUÑECA DEL ARCTOS: SOLO PRONOSUPINACIÓN ─────────────────────────
PRONO_JOINT    = "joint6"   # "joint6" (solo mueve la mano) | "joint4"
PRONO_SIGNO    = -1.0        # si gira al revés → -1.0
PRONO_ESCALA   = 1.2
PRONO_LIMITE   = 1.5        # rad (~86°) a cada lado del neutro
PRONO_VEL_MAX  = 2.0        # (antes 3.0) rad/s
PRONO_SUAVIZADO = 0.35      # 0..1 (filtro exponencial; más bajo = más suave)
JOINT5_FIJO    = 0.0        # muñeca del ARCTOS recta: la flexión la hace la OrcaHand

# Vertical para medir el brazo:
#   "caderas" → línea cadera→hombro (bien de pie; SENTADO la mesa tapa las caderas)
#   "camara"  → vertical de la cámara (la OAK-D está fija; la T-pose corrige su inclinación)
ARRIBA_REF = "camara"

# Si la muñeca se mide más lejos que brazo+antebrazo por encima de este margen,
# es una mala medida (mano delante del cuerpo): NO se fuerza el codo a recto.
CODO_ESTIRADO_MARGEN = 0.05     # 5 %

# ── PASO 2 — HÍBRIDO: LA MANO DEL ROBOT VA DONDE VA TU MANO (distancias OAK-D) ──
# 0.0 = solo ángulos (como hasta ahora) · 1.0 = la mano del robot sigue a tu mano
# Intermedio (p.ej. 0.6) = mezcla. Se alterna en vivo con [H].
HIBRIDO_PESO        = 1.0
HIBRIDO_ESCALA      = 1.0    # >1: el robot llega proporcionalmente más lejos que tú
HIBRIDO_ALCANCE_MAX = 0.95   # nunca estirar el ARCTOS del todo (singularidad)
HIBRIDO_SUAVIZADO   = 4      # media móvil (fotogramas) sobre el punto objetivo
HIBRIDO_IK_ITER     = 15     # iteraciones de IK por fotograma
HIBRIDO_IK_PASO     = 0.12   # rad máx por iteración

# ── FÍSICA EN TIEMPO REAL (mejora 1, idea del proyecto TRON 2) ───────────────
# True: la simulación avanza en un hilo propio al ritmo del reloj real y el visor
#       se refresca a FISICA_HZ, independientemente de los FPS de la cámara.
# False: comportamiento anterior (4 mj_step por fotograma → ~0.08x a 8 FPS).
FISICA_EN_HILO     = True

# PERFIL DEL ORDENADOR — cambia SOLO esta línea al pasar a producción.
#   "puesto"     → tu ordenador de trabajo (más modesto): menos carga para el visor
#   "produccion" → ordenador bueno: valores óptimos
PERFIL_EQUIPO = "puesto"

if PERFIL_EQUIPO == "produccion":
    FISICA_HZ          = 60     # refrescos por segundo del visor / avances de la física
    FISICA_RETRASO_MAX = 0.2    # s: retraso máximo que se recupera de golpe
else:  # "puesto"
    FISICA_HZ          = 30     # la mitad de refrescos del visor: libera CPU/GPU
    FISICA_RETRASO_MAX = 0.1    # si se atrasa, se descarta antes en vez de "acelerar"

# ── VALIDACIÓN DE PROFUNDIDAD CON MEDIAPIPE (mejora 2, idea del TRON 2) ──────
# Si el estéreo del codo o de la muñeca se aleja más de esto (m) de la profundidad
# que predice MediaPipe (anclada al hombro), se considera una mala lectura (fondo,
# torso...) y no se usa. 0 = desactivado (comportamiento anterior).
#   · más bajo (0.15) → más estricto: descarta más estéreo, confía más en MediaPipe
#   · más alto (0.30) → más permisivo
PROF_TOL_MEDIAPIPE = 0.12

# ── PASO 3 — DIRECCIONES + IK CON CODO Y MUÑECA (ideas del TRON 2) ───────────
# "codo_muneca": tus DIRECCIONES de brazo y antebrazo → dónde deben estar el codo
#                y la muñeca del ARCTOS (con SUS longitudes) → IK que persigue ambos.
# "muneca":      el híbrido anterior (solo la muñeca, escalada por el alcance).
# [H] sigue alternando entre el híbrido elegido aquí y "solo ángulos".
HIBRIDO_MODO       = "codo_muneca"
HIBRIDO_PESO_CODO  = 0.15           # (antes 0.4; 7-oct: el codo medido era poco fiable y tiraba de
                                    # la postura) importancia del codo frente a la muñeca (0 = solo muñeca)
HIBRIDO_CODO_RECTO = (10.0, 30.0)   # grados de flexión: <10 el codo no cuenta, >30 cuenta entero
ZONA_MUERTA_CODO   = (5.0, 25.0)    # grados: <5 se trata como brazo recto, hasta 25 se atenúa

# ── PASO 5 — FILTRO ONE EURO (idea del TRON 2) ──────────────────────────────
# En el modo "codo_muneca" filtra tus DIRECCIONES de brazo y antebrazo y sustituye
# a las medias móviles (HIBRIDO_SUAVIZADO e IMITAR_SUAVIZADO), que retrasaban siempre.
FILTRO_ONE_EURO   = True
FILTRO_MIN_CUTOFF = 0.4     # (antes 0.8) Hz: más bajo = menos temblor en reposo (pero más retraso)
FILTRO_BETA       = 0.3     # (antes 0.7) más alto = menos retraso en movimientos rápidos (pero más temblor)

# ── PASO 6 — RESCATE DE LA IK (idea del TRON 2) ─────────────────────────────
# Si la IK lleva IK_RESCATE_TIEMPO segundos con error > IK_RESCATE_ERROR, se prueba
# desde otras posturas de partida y se cambia a la mejor si mejora de verdad.
IK_RESCATE_ERROR  = 0.08    # m (error ponderado codo+muñeca)
IK_RESCATE_TIEMPO = 0.5     # s
IK_RESCATE_MEJORA = 0.7     # la alternativa debe dejar el error por debajo del 70 %
IK_RESCATE_ITER   = 30      # iteraciones por postura de partida
# Probar también la configuración "girada" (base +180°, hombro hacia atrás), que dobla
# el codo hacia el OTRO lado. Solo funciona si IMITAR_LIMITES["joint2"] empieza por
# debajo de -1.0 (comprueba antes en el visor que el hombro puede ir hacia atrás sin chocar).
IK_RESCATE_GIRADA = False   # (antes True) PASO 7: la solución "girada" gira la base
                            # ~180° de golpe → con herramienta es un salto enorme.
                            # Vuelve a True solo si ves que la IK se atasca a menudo.

# Correcciones del rescate (28-sep) — evitan las "vueltas" de la base:
IK_RESCATE_ESPERA  = 1.5    # s mínimos entre dos cambios de solución (sin ping-pong)
IK_PENALIZA_BASE   = 0.04   # m de "coste" por cada rad que haya que girar la base
IK_PENALIZA_GIRADA = 0.06   # m de "coste" extra por usar la configuración girada

# ── CORRECCIONES 28-sep ─────────────────────────────────────────────────────
# Fusión de direcciones: el brazo y el antebrazo usan la dirección del ESTÉREO
# solo si se desvía menos de esto (grados) de la de MediaPipe; si no, MediaPipe.
# Evita codos "doblados" falsos. 0 = desactivado (solo estéreo).
FUSION_DIR_MAX_DEG = 30.0

# Giro de muñeca:
#   "mundo"    → la mano del robot se orienta IGUAL que la tuya en el espacio
#                (funciona con cualquier postura que elija la IK). Recomendado.
#   "relativo" → como antes: giro relativo al neutro, con PRONO_ESCALA.
PRONO_MODO     = "mundo"
PRONO_Z_ESCALA = 1.6        # MediaPipe Hands "aplasta" la profundidad de la mano

# ── OBJETIVO DE LA IK: PALMA DE LA ORCAHAND (28-sep) ─────────────────────────
# "palma"  → la palma de la OrcaHand va donde va TU palma, teniendo en cuenta las
#            proporciones (brazo + antebrazo + mano) tuyas y del ARCTOS + OrcaHand.
# "muneca" → como antes: se persigue el muñón (mano_mount) y se ignora la mano.
HIBRIDO_OBJETIVO    = "palma"
HUMANO_MUNECA_PALMA = 0.08   # m: de tu muñeca al centro de tu palma (aprox. adulto)

# ── ESCENARIO TALADRO — TECLA [4] ───────────────────────────────────────────
# Antes, una vez: python taladro/preparar_taladro.py  (+ include en arctos_mjcf.xml)
TALADRO_POS_SOPORTE   = (-0.45, -0.20, 0.25)  # m: centro de la cara SUPERIOR de la peana
TALADRO_TECHO_XY      = (0.0, 0.0)            # m: centro del techo (encima del robot)
TALADRO_TECHO_Z       = None    # m: altura del techo; None = automática (ver holgura)
TALADRO_TECHO_HOLGURA = 0.16    # m por encima de la palma con el brazo vertical
TALADRO_ASISTENCIA_AGARRE = True   # el taladro "no pesa" mientras la mano lo agarra
TALADRO_CONTACTOS_AGARRE  = 3      # contactos mano↔taladro para considerarlo agarrado

# ── DEMO AUTOMÁTICA DEL TALADRO — TECLA [5] ─────────────────────────────────
# El brazo coge el taladro de la peana (TALADRO_POS_SOPORTE) con la OrcaHand
# DERECHA por el lado DERECHO de la empuñadura, lo lleva al techo, aprieta el
# gatillo más fuerte con el índice y hace un agujero. Necesita el bloque
# <equality> de arctos_mjcf.xml (welds taladro_en_mano / taladro_en_peana).
TALADRO_DEMO_AGUJERO_XY    = (-0.15, -0.10)  # m: dónde se hace el agujero (dentro del techo)
TALADRO_DEMO_SEP_AGUJEROS  = 0.05     # m: cada vez que repites la demo, el agujero se desplaza
TALADRO_DEMO_TECHO_Z       = None     # m; None = automática: lo más alto que llega la BROCA
                                      # (brazo + OrcaHand + taladro) menos el margen
TALADRO_DEMO_MARGEN_TECHO  = 0.05     # m de margen hasta el alcance máximo (brazo sin estirar del todo)
TALADRO_DEMO_PROFUNDIDAD   = 0.04    # m que entra la broca en el techo
TALADRO_DEMO_BAJO_TECHO    = 0.08     # m bajo el techo desde donde sube la broca en vertical
TALADRO_DEMO_APROX         = 0.12     # m: la mano llega por el lado derecho del mango desde aquí
TALADRO_DEMO_ALTURA_PASO   = 0.12     # m que se levanta el taladro al cogerlo / dejarlo
TALADRO_DEMO_ALTURA_SEGURA = 0.20     # m sobre el mango por los que la mano se mueve al acercarse

# Dónde queda el eje del mango respecto a los nudillos de la OrcaHand (m):
#   (hacia la punta de los dedos, hacia fuera de la palma, hacia el índice)
# Al segundo valor se le suma el radio del mango. Si el mango queda metido en la
# palma sube el 2º; si los dedos no lo rodean, baja el 1º; si el índice choca con
# la cabeza del taladro, baja el 3º (negativo).
TALADRO_DEMO_AGARRE_OFFSET = (-0.017, -0.01, 0.00)

# Posturas de la mano (índices de ctrl 0..16, los de LIMITES_CTRL). Lo no indicado = 0.
TALADRO_DEMO_MANO_ABIERTA = {0: 0.0, 13: -0.0979, 14: 0.641, 15: 0.0, 16: 0.175}
TALADRO_DEMO_MANO_AGARRE  = {0: 0.0,
                             2: 1.25, 3: 1.10,     # meñique
                             5: 1.25, 6: 1.10,     # anular
                             8: 1.25, 9: 1.10,     # corazón
                             11: 1.00, 12: 0.90,   # índice (sobre el gatillo, sin apretar)
                             13: 0.35, 14: 0.45, 15: 0.70, 16: 0.90}   # pulgar rodeando
# "Apretar más fuerte el botón": solo cambia el índice. El motor arranca al pasar
# el 70 % del camino entre el agarre y este valor.
TALADRO_DEMO_MANO_GATILLO = {11: 1.60, 12: 1.50}

TALADRO_DEMO_RPM        = 240     # giro de la broca (visual; más alto parece girar al revés en pantalla)
TALADRO_DEMO_VIBRACION  = 0.003   # rad de vibración de la muñeca mientras perfora (0 = sin)
TALADRO_DEMO_VELOCIDAD  = 1.0     # >1 demo más rápida, <1 más lenta
TALADRO_DEMO_TIEMPOS = {          # s de cada fase (a TALADRO_DEMO_VELOCIDAD = 1)
    "subir": 1.5, "acercar": 3.0, "bajar": 1.5, "entrar": 2.0, "cerrar": 1.5,
    "levantar": 1.5, "al_techo": 3.5, "acercar_broca": 2.0,
    "gatillo": 0.7, "arrancar": 0.8, "perforar": 4.0, "sacar": 2.0, "soltar_gatillo": 0.6,
    "volver": 3.5, "dejar": 1.5, "abrir": 1.0, "retirar": 1.5, "reposo": 3.0,
}
TALADRO_DEMO_VEL_MAX    = 1.0     # rad/s máx. de cualquier joint del brazo (alarga la fase si hace falta)
TALADRO_DEMO_TOL_AGARRE = 0.035   # m: si al cerrar la mano el mango está más lejos, NO se agarra
                                  # (la mano se retira y se imprime cuánto se ha desviado)
TALADRO_DEMO_CAMARA = {"azimut": 25, "elevacion": -12, "distancia": 2.0}   # vista durante [5]
TALADRO_DEMO_BRAZO_CINEMATICO = True   # True: el brazo sigue el plan EXACTO (animación); los dedos
                                       # y el taladro siguen siendo físicos. False: servos + integral
TALADRO_DEMO_CONTACTOS_MIN = 2      # contactos dedos↔taladro esperados al cerrar la mano (aviso)
TALADRO_DEMO_SOLDADURA_RIGIDEZ = 0.004   # s: solref de las soldaduras taladro↔mano/peana en la demo
                                       # (más bajo = más rígida; mínimo 2×timestep = 0.004)
TALADRO_DEMO_KI = 2.0             # 1/s: corrección integral si el brazo se queda corto (0 = sin)
TALADRO_DEMO_SIN_AUTOCOLISION = True   # en la demo, los eslabones del brazo no chocan con la
                                       # mano ni con el taladro (evita atascos por roces de mallas)

# ══════════════════════════════════════════════════════════════════════════════
# PASO 7 — BRAZO PRECISO Y SIN SALTOS (para maniobrar con herramientas)
# ══════════════════════════════════════════════════════════════════════════════
# 7.1 COMANDO CONTINUO. La cámara da un objetivo nuevo solo 10-30 veces por segundo
#     y antes se escribía tal cual en ctrl → tirón en cada fotograma y parada hasta el
#     siguiente ("escalera"). Ahora el hilo de física, ANTES DE CADA mj_step (500/s),
#     lleva ctrl hacia el objetivo con velocidad y aceleración limitadas (igual que la
#     demo [5] del taladro). False = comportamiento anterior (para comparar).
SUAVE_ACTIVO     = True
SUAVE_INTERPOLAR = True   # reparte cada objetivo entre dos fotogramas: movimiento continuo,
                          # a cambio de ~1 fotograma de retraso. False = solo los límites.
#                  joint1  joint2  joint3  giro muñeca (PRONO_JOINT)
SUAVE_VEL_MAX  = (1.2,    1.2,    1.5,    2.0)    # rad/s
SUAVE_ACEL_MAX = (12.0,   12.0,   15.0,   20.0)   # rad/s² — más bajo = arranques y frenadas más suaves

# 7.2 SIN SALTOS EN LA MEDIDA
#   · Filtro de saltos: si el objetivo de la palma/codo del robot salta más de
#     SALTO_VEL_MAX·Δt + SALTO_MARGEN, ese fotograma es un error de medida y se ignora.
#     Si el salto se mantiene, el umbral crece con el tiempo y se acepta. 0 = desactivado.
SALTO_VEL_MAX = 1.0       # m/s (espacio del robot; tu mano ≈ ×0.75)
SALTO_MARGEN  = 0.03      # m
#   · Zona muerta: la palma del robot no se mueve por temblores menores que esto.
ZONA_MUERTA_PALMA = 0.005 # m (0 = desactivada)
#   · Fusión estéreo/MediaPipe sin interruptor: en FUSION_DIR_MAX_DEG ± esto (grados)
#     las dos direcciones se mezclan poco a poco en vez de saltar de una a otra.
FUSION_DIR_TRANSICION = 10.0
#   · IK con memoria: girar mucho un joint para ganar poca precisión "cuesta" (m por rad).
#     Evita los giros bruscos de la base cuando la mano pasa cerca de la vertical.
IK_PESO_CONTINUIDAD = 0.03

# 7.3 MODO FINO [F] y CONGELAR [C] — para maniobrar con herramientas
#   [F] el robot se mueve FINO_ESCALA veces lo que te mueves tú, desde donde esté
#       (1 cm tuyo = 3.5 mm del robot): más precisión y el temblor igual de reducido.
#   [C] el brazo se queda quieto (p.ej. con el taladro apoyado en el techo). Al quitarlo,
#       en modo fino sigue desde ahí; en modo normal vuelve a imitarte en FINO_SALIDA_S.
FINO_ESCALA      = 0.35
FINO_ESCALA_GIRO = 0.5    # lo mismo para el giro de muñeca
FINO_FACTOR_VEL  = 0.5    # en modo fino el comando continuo va a la mitad de vel./acel.
FINO_SALIDA_S    = 1.5    # s MÍNIMOS de transición suave al volver a la imitación normal
FINO_SALIDA_VEL  = 0.15   # m/s máx. de esa transición: si el robot quedó lejos, tarda más
IK_RESCATE_EN_FINO = False  # el rescate de la IK (cambiar de postura) nunca en fino/congelado

# 7.4 PROPORCIONES DEL ROBOT (el ARCTOS no tiene tus medidas)
#   · PALMA_CENTRO_REAL: la IK persigue el centro REAL de la palma de la OrcaHand
#     (antes un punto ~3 cm más fuera, casi en los nudillos).
#   · CODO_COMPATIBLE: el ARCTOS mide 0.28 m de hombro a codo y ~0.53 m de codo a palma
#     (tú ~0.30 y ~0.35). El objetivo del codo se coloca donde PUEDE estar con esas
#     medidas (en el lado de tu codo), para que no tire de la palma y esta no se pase.
#   False en cualquiera de los dos = comportamiento anterior.
PALMA_CENTRO_REAL = True
CODO_COMPATIBLE   = True
# ══════════════════════════════════════════════════════════════════════════════
# FLUIDEZ (30-sep) — MANO FÍSICA SIN TROMPICONES Y CÁMARA MÁS FLUIDA
# ══════════════════════════════════════════════════════════════════════════════
# ── Mano física (orca_hardware.py) ──
# La cámara da un objetivo nuevo solo ~10-15 veces por segundo. Antes se mandaba tal
# cual a los servos DESDE el bucle de la cámara → tirón y parada en cada fotograma
# (la "escalera" del PASO 7.1, pero en la mano real). Ahora un hilo propio manda un
# comando CONTINUO a ORCA_HW_HZ: interpola entre fotogramas y limita velocidad y
# aceleración de cada articulación (mismo algoritmo que GeneradorSuave del brazo).
ORCA_HW_HZ            = 50     # envíos/s a los servos (17 motores a 1 Mbps ≈ 1-2 ms por envío)
ORCA_HW_INTERPOLAR    = True   # reparte cada objetivo entre dos fotogramas (≈1 fotograma de retraso)
ORCA_HW_VEL_MAX       = 6.0    # rad/s máx. por articulación (≈340°/s). Más bajo = más suave pero más retraso
ORCA_HW_ACEL_MAX      = 60.0   # rad/s² máx. — más bajo = arranques y frenadas más suaves
ORCA_HW_LIMITES_JOINT = {      # excepciones por articulación: "joint": (vel. máx, acel. máx)
    "wrist": (3.0, 30.0),      # la muñeca mueve toda la mano: más despacio
}
ORCA_HW_DEBUG_TIEMPOS = True   # imprime los envíos/s reales y lo que tarda el SDK en cada envío
ORCA_HW_DEBUG_CADA_S  = 5.0

# ── Cámara ──
# Hands en PARALELO: el recorte de la mano se calcula con la Pose del fotograma
# ANTERIOR (la del actual aún no existe) y MediaPipe Hands corre en otro hilo mientras
# el principal hace Pose y el brazo. Así el coste de Hands casi desaparece del bucle.
# El margen del recorte (MANO_RECORTE_FACTOR) cubre lo que se mueve la mano entre dos
# fotogramas; si en ese recorte no hay mano, se sigue como antes (recorte con la Pose
# actual → imagen completa). False = todo en serie, como antes.
MANO_EN_PARALELO = True

# Perfil del bucle: cada PERFIL_BUCLE_CADA_S segundos imprime cuántos ms se van en cada
# fase (cámara, pose, brazo, mano...). Sirve para ver QUÉ frena la cámara en cada
# ordenador. Coste despreciable; pon False cuando ya no lo necesites.
PERFIL_BUCLE        = True
PERFIL_BUCLE_CADA_S = 3.0

# ═══════════════════════════════════════════════════════════════════════════
# DOS CÁMARAS OAK-D — pega este bloque al final de config.py
# ═══════════════════════════════════════════════════════════════════════════
MULTICAMARA_ACTIVA  = True   # True → abre la cámara B y fusiona el brazo (False = comportamiento de siempre)
CAM_A_MXID          = "1944301001AAFA1700"    # MxId de la cámara principal (None = la primera libre). Fíjalos cuando ya funcione.
CAM_B_MXID          = "1944301081B5FA4800"    # MxId de la cámara secundaria. Se ven con:  python -c "from skill_shield.vision.multicamara import listar_camaras as l; print(l())"
EXTRINSECAS_ARCHIVO = "camaras_extrinsecas.json"   # lo genera calibrar_extrinseca.py
MULTICAM_SYNC_MAX_S = 0.04    # máx. desfase (s) entre el fotograma de A y el de B para emparejarlos
MULTICAM_MOSTRAR_B  = True    # ventana de depuración "Camara B"
CAM_B_IR_MA         = 800     # proyector IR de la B (mA)

# Fusión: incertidumbre de cada medida (σ_z = K·z², lateral = σ_pix·z/f) y rechazo de discrepancias
FUSION_SIGMA_PIX    = 4.0     # error típico de MediaPipe en la imagen (px)
FUSION_SIGMA_Z_K    = 0.006   # σ_z = 0,006·z² → 2,4 cm a 2 m (súbelo si tu estéreo es peor)
FUSION_K_FUENTE     = {"E": 1.0, "E+": 1.6, "L": 3.0, "M": 6.0}   # fiabilidad relativa de cada fuente de profundidad
FUSION_GATE_CHI2    = 16.0    # por encima, las dos cámaras "discrepan" y se elige UNA (no se promedia)
FUSION_LR_MARGEN_M  = 0.10    # detecta izquierda/derecha intercambiados en la vista de perfil
FUSION_MEMORIA_S    = 0.5     # validez del último punto bueno para desempatar

# ═══════════════════════════════════════════════════════════════════════════
# REVISIÓN 7-OCT — IMITACIÓN MÁS NATURAL (vídeo 12:41)
# ═══════════════════════════════════════════════════════════════════════════
# 1) ESPACIO DE TRABAJO. Tu hombro está a ~1,4 m del suelo y el del ARCTOS a 0,29 m:
#    con tu brazo por debajo de la horizontal, el objetivo de la palma caía bajo el
#    suelo y se recortaba (error de 50-80 cm, IK desesperada, choques con el suelo).
#    Ahora la mitad de abajo se COMPRIME al rango que el robot sí alcanza: de la
#    horizontal hacia arriba 1:1; hacia abajo, proporcional hasta estos mínimos.
#    Si en la escena el ARCTOS va sobre una peana o una mesa, se adapta solo (se mide
#    desde la altura real de su hombro).
REUBICAR_OBJETIVOS = True    # False = comportamiento anterior (recorte a ARM_Z_MINIMA)
COMPRIMIR_ABAJO    = True
PALMA_Z_MIN        = 0.15    # m: altura mínima del centro de la palma (los dedos cuelgan algo más)
CODO_Z_MIN         = 0.12    # m: altura mínima del codo (también lo usa limites_humanos.py)

# 2) CODO SOLO CUANDO SE MIDE BIEN. En el vídeo el codo salía casi siempre reconstruido
#    ("c:R", CODO ESTIMADO) y su objetivo quedaba a 22-34 cm: tiraba de la postura.
#    Con el codo dudoso su peso en la IK se multiplica por HIBRIDO_CODO_DUDOSO.
HIBRIDO_CODO_SOLO_FIABLE = True
HIBRIDO_CODO_DUDOSO      = 0.0   # 0 = el codo dudoso no cuenta (manda solo la palma)

# 3) BASE ESTABLE CON EL BRAZO VERTICAL (colgando o arriba). Ahí tu dirección horizontal
#    es ruido (azimut 162° con el brazo colgando) y la base saltaba hasta 100°.
#    Por debajo de VERTICAL_HORIZ[0] de componente horizontal se usa la dirección actual
#    del robot; por encima de [1], la tuya; en medio se mezclan.
VERTICAL_HORIZ = (0.35, 0.60)   # ≈ elevación 20°–37° (por debajo de ~20° el azimut ya es ruido)
BASE_BLOQUEO_W = 0.3         # con menos de esta confianza en tu horizontal, joint1 no se mueve

# 4) LÍMITES HUMANOS (skill_shield/brazo/limites_humanos.py). Dos niveles:
#    · sobre TU medida: lo que un brazo no puede hacer (codo > 145°, antebrazo hacia
#      atrás con rotación imposible del húmero, brazo metido en el torso) se proyecta
#      al rango anatómico antes de dárselo al robot;
#    · sobre el ROBOT: joint1 en tu rango de azimut, joint2 sin bajar el codo al suelo,
#      joint3 sin plegarse más de CODO_FLEX_MAX respecto al codo recto.
LIMITES_HUMANOS      = True
HOMBRO_AZ_MIN        = -45.0   # ° cruzando el brazo por delante del pecho
HOMBRO_AZ_MAX        = 130.0   # ° abriendo hacia atrás con el brazo alto
HOMBRO_EXT_ELEV      =  55.0   # ° por debajo de esta elevación sí se puede llevar el brazo atrás
HOMBRO_AZ_LIBRE_ELEV =  20.0   # ° con el brazo casi colgando el azimut no se limita (es ruido)
CODO_FLEX_MAX        = 145.0   # ° flexión máxima del codo
HUMERO_ROT_EXT       =  95.0   # ° rotación externa máx. del húmero
HUMERO_ROT_INT       = 110.0   # ° rotación interna máx. del húmero
Q2_MIN               = -0.5    # rad: el hombro del robot solo se pasa un poco de la vertical
# Distancia hombro→palma del robot según TU flexión de codo (ley del coseno con SUS
# longitudes): brazo recto → robot recto. Sin esto, el objetivo al 95 % del alcance
# obligaba al robot a doblar el codo ~45° con tu brazo estirado.
RETARGET_FIEL        = True

# ═══════════════════════════════════════════════════════════════════════════
# GUANTES MANUS (skill_shield/mano/manus_guante.py · calibrar_manus.py)  — v2, 8-oct
# ═══════════════════════════════════════════════════════════════════════════
# SUSTITUYE al bloque "GUANTES MANUS" actual de config.py (desde FUENTE_DEDOS hasta el
# final de MANUS_GANANCIA_FLEXION). Lo que cambia respecto al anterior:
#   · desaparecen MANUS_FLEXION_DIRECTA y MANUS_GANANCIA_FLEXION (el mapeo ya es por rangos)
#   · aparecen MANUS_RANGO_ROBOT, MANUS_GANANCIA y los parámetros de pinza
FUENTE_DEDOS = "manus"      # "camara" → MediaPipe Hands | "manus" → guante (la muñeca y el brazo
                            #   siguen saliendo de las cámaras)
MANUS_MODO   = "integrado"  # "integrado" = sin Dashboard (CIERRA el Dashboard antes, o el dongle
                            #   queda ocupado) | "local" = MANUS Core en este PC | "remoto"
MANUS_LADO   = "derecho"    # mano que se teleopera
MANUS_SDK_RUTA = os.path.join(_BASE_DIR, "MANUS_Core_3.2.1_SDK_Linux", "Python")
MANUS_CALIBRACION = os.path.join(_BASE_DIR, "manus_calibracion.json")   # lo genera calibrar_manus.py
MANUS_TIMEOUT_S = 0.3       # s sin datos del guante → los dedos se quedan donde estén
MANUS_TAU_S     = 0.025     # s: suavizado de los dedos (0 = sin suavizado)
# Vibración: fuerza de contacto en cada yema de la OrcaHand (simulación) → ese dedo del guante
MANUS_HAPTICA       = True
MANUS_HAPTICA_HZ    = 30.0
MANUS_FUERZA_MIN    = 0.05  # N
MANUS_FUERZA_MAX    = 3.0   # N
MANUS_VIBRACION_MAX = 1.0

# ── Rangos del ROBOT (rad) en los extremos de TU mano: (cuando estás en tu mínimo, cuando
# estás en tu máximo). Tus ángulos en esos extremos los mide calibrar_manus.py. Si un dedo
# va al revés, intercambia los dos valores de su par; si no cierra del todo, sube el máximo;
# si choca, bájalo. Claves = índice de ctrl de la mano (0..16, como LIMITES_CTRL).
# Sentidos de la OrcaHand (cinemática del MJCF):
#   abd de los dedos: + = la yema va HACIA EL PULGAR (índice + = abre; anular/meñique + = cierran)
#   t-abd (14):  + = pulgar tumbado en el plano de la palma ; − = pulgar perpendicular a la palma
#   t-cmc (13):  + = pulgar barre hacia la base del meñique (oposición) ; − = hacia fuera
MANUS_RANGO_ROBOT = {
    # flexión (mano abierta → puño). Hardware: mcp hasta 100° (1.745 rad), pip hasta 107° (1.867)
    2:  (0.0, 1.70),  3:  (0.0, 1.80),      # meñique mcp, pip
    5:  (0.0, 1.70),  6:  (0.0, 1.80),      # anular
    8:  (0.0, 1.70),  9:  (0.0, 1.80),      # corazón
    11: (0.0, 1.70),  12: (0.0, 1.80),      # índice
    # abducción (dedos JUNTOS → SEPARADOS). Los "juntos" son la convergencia de retargeting.py
    10: (-0.27, 0.45),                      # índice
    4:  (0.225, -0.25),                     # anular
    1:  (0.42, -0.25),                      # meñique
    7:  (0.0, 0.0),                         # corazón: fijo
    # pulgar
    14: (0.90, -0.25),     # CMC_SPREAD:  pulgar en el plano de la palma → perpendicular a ella
    13: (-0.60, 0.576),    # CMC_STRETCH: pulgar extendido hacia fuera → cruzado hacia el meñique
    15: (0.0, 1.50),       # MCP: recto → doblado
    16: (0.175, 1.50),     # IP:  recto → doblado
}
# Ganancia extra sobre la fracción de recorrido (1.0 = tu extremo = extremo del robot). Con 1.15
# llegas al extremo del robot con el 87 % de tu recorrido: útil si con el guante no aprietas el
# puño del todo o quieres pinzas más "decididas". Claves: nombres de actuador.
MANUS_GANANCIA = {
    "menique_mcp": 1.00, "menique_pip": 1.00, "anular_mcp": 1.00, "anular_pip": 1.00,
    "corazon_mcp": 1.00, "corazon_pip": 1.00, "indice_mcp": 1.00, "indice_pip": 1.00,
    "indice_abd": 1.00, "anular_abd": 1.00, "menique_abd": 1.00,
    "pulgar_cmc": 1.00, "pulgar_abd": 1.00, "pulgar_mcp": 1.00, "pulgar_ip": 1.00,
}
# ── Pinzas con el guante ──
# Cuando tu mano se parece (RMS en grados sobre pulgar + dedo) a una de las 7 pinzas grabadas en
# calibrar_manus.py, el comando se mezcla con la pose de contacto del robot (PINZA_CTRL_CONTACTO,
# retargeting.py): 0 % por encima de OFF, 100 % por debajo de ON, continuo entre medias.
MANUS_PINZA_ACTIVA  = True
MANUS_PINZA_ON_DEG  = 7.0    # súbelo si la pinza "parpadea"; bájalo si entra en pinza sin querer
MANUS_PINZA_OFF_DEG = 14.0


# ═══════════════════════════════════════════════════════════════════════════
# RECORRIDO COMPLETO DE LA MANO (8-oct, 2ª ronda) — pega este bloque AL FINAL de config.py
# ═══════════════════════════════════════════════════════════════════════════
# Al ir al final MANDA sobre los LIMITES_CTRL y MANUS_RANGO_ROBOT de más arriba (no hace
# falta borrarlos). Qué arregla:
#   · los 4 dedos no tocaban la palma ni se estiraban del todo (sim y mano real)
#   · el pulgar no se alejaba de la palma ni se apoyaba en ella
#   · el corazón no se movía hacia los lados (estaba fijo a 0 en tres sitios)
#   · las abducciones de la mano real iban al revés (cambio mío de la 1ª ronda)
# Después: python calibrar_manus.py  (hay 2 poses nuevas del corazón).

# 1) Límites de seguridad de la mano = rango COMPLETO del modelo (ctrlrange del MJCF), que es
#    también el ROM del SDK de la mano real. Antes: flexiones desde 0 (en el MJCF, con 0° los
#    dedos quedan 3-8° doblados: no podían estirarse), abducciones −0.20…0.50 y corazón fijo.
LIMITES_CTRL = [
    (-1.134,  0.610),    #  0: muñeca (sin cambios)
    (-0.5236, 0.5236),   #  1: meñique abd
    (-0.4363, 1.7453),   #  2: meñique mcp
    (-0.2618, 1.8675),   #  3: meñique pip
    (-0.4712, 0.4712),   #  4: anular abd
    (-0.4363, 1.7453),   #  5: anular mcp
    (-0.2618, 1.8675),   #  6: anular pip
    (-0.4712, 0.4712),   #  7: corazón abd   ← antes (0, 0): fijo
    (-0.4363, 1.7453),   #  8: corazón mcp
    (-0.2618, 1.8675),   #  9: corazón pip
    (-0.4363, 0.5236),   # 10: índice abd
    (-0.4363, 1.7453),   # 11: índice mcp
    (-0.2618, 1.8675),   # 12: índice pip
    (-0.7854, 0.5760),   # 13: pulgar cmc (oposición)
    (-0.3142, 0.9599),   # 14: pulgar abd
    (-0.4363, 1.7453),   # 15: pulgar mcp
    (-0.2618, 1.8675),   # 16: pulgar ip
]

# 2) Guante → robot: (robot en TU mínimo, robot en TU máximo), rad. Extremos = los del modelo,
#    medidos con su cinemática y sus mallas.
MANUS_RANGO_ROBOT = {
    # flexión: dedo RECTO de verdad → puño con la yema TOCANDO la palma
    2:  (-0.14, 1.70),  3:  (0.0, 1.80),    # meñique mcp, pip
    5:  (-0.05, 1.70),  6:  (0.0, 1.80),    # anular
    8:  (-0.05, 1.70),  9:  (0.0, 1.80),    # corazón
    11: (-0.07, 1.70),  12: (0.0, 1.80),    # índice
    # abducción: dedos JUNTOS → SEPARADOS (abanico completo del modelo)
    10: (-0.27, 0.52),                      # índice
    4:  (0.225, -0.45),                     # anular
    1:  (0.42, -0.50),                      # meñique
    7:  (-0.40, 0.40),                      # corazón: hacia el anular → hacia el índice
    # pulgar: rango completo
    14: (0.96, -0.31),     # t-abd: tumbado en el plano de la palma → perpendicular a ella
    13: (-0.785, 0.576),   # t-cmc: hacia fuera (la 'L') → cruzado hacia el meñique
    15: (0.0, 1.745),      # MCP: recto → doblado del todo
    16: (0.0, 1.867),      # IP:  recto → doblado del todo
}

# Márgenes: qué parte de TU recorrido ya cuenta como extremo (inicio, final). Con (0.10, 0.20)
# el robot cierra el puño del todo con el 80 % de tu puño de calibración y se estira del todo
# con tu mano "casi" abierta. Claves: "flex", "abd", "pulgar" o el nombre de un actuador
# (p. ej. "indice_pip": (0.10, 0.30)). (0, 0) = como antes.
MANUS_MARGEN = {"flex": (0.10, 0.20), "abd": (0.10, 0.15), "pulgar": (0.10, 0.15)}

# Poses intermedias: el robot pasa EXACTAMENTE por aquí cuando repites esa pose (rad).
MANUS_ROBOT_INTERMEDIO = {
    7:  {"abierta": 0.0},        # corazón recto con la mano abierta
    13: {"pulgar_perp": -0.16},  # pulgar perpendicular = lo más lejos de la palma
    14: {"pulgar_perp": -0.31},
    15: {"pulgar_cruz": 1.40},   # pulgar cruzado = yema APOYADA en la palma
}

# 3) Mano física (orca_hardware.py)
ORCA_HW_SIGNOS             = {}      # tu mano va en el MISMO sentido que el sim (comprobado)
ORCA_HW_CIERRE_COMPLETO    = True    # "cerrado" del sim → tope de flexión calibrado + sobrecarrera
ORCA_HW_SOBRECARRERA_DEG   = 4.0     # grados más allá del tope de flexión (empuja con max_current)
ORCA_HW_APERTURA_EXTRA_DEG = 5.0     # dedo recto del sim → 5° más estirado en la mano real (no el pulgar)
ORCA_HW_APERTURA_COMPLETA  = False   # True = "abierto" del sim → tope de EXTENSIÓN (hiperextiende)
ORCA_HW_TRAMO_FIEL         = (0.25, 0.60)   # zona central sin estirar (fracción del recorrido)

# ═══════════════════════════════════════════════════════════════════════════
# POSE DEL BRAZO AL PULSAR [A] (9-oct) — pega este bloque AL FINAL de config.py
# ═══════════════════════════════════════════════════════════════════════════
# Los 6 joints del ARCTOS (rad) a los que va el brazo al pulsar [A], en vez de la T-pose.
# Se queda ahí mientras calibras la T-pose (~2 s) y, cuando empieza a imitarte, sale desde
# ahí (la muñeca del ARCTOS vuelve a JOINT5_FIJO). None = como antes (T-pose).
# Sacada de la foto del brazo real con la cinemática y las mallas del ARCTOS (el modelo
# superpuesto a la foto coincide en base, hombro, codo y antebrazo):
#   joint1 +90°  brazo girado a SU izquierda (la cara de la base con las rejillas mira a la
#                cámara; desde la vista inicial del visor se ve igual que en la foto)
#   joint2 −21°  hombro inclinado hacia atrás
#   joint3 −34°  antebrazo casi horizontal (13° hacia abajo)
#   joint4   0°
#   joint5 −77°  muñeca doblada hasta que la brida mira al suelo
#   joint6   0°  (en la foto no se ve el giro de la brida)
POSE_BRAZO_ACTIVAR = [1.571, -0.37, -0.60, 0.0, 0.0, 0.0]

# Velocidad / aceleración del comando continuo de joint4 y joint5 (como SUAVE_VEL_MAX)
SUAVE_VEL_MUNECA_ARCTOS  = 1.5    # rad/s
SUAVE_ACEL_MUNECA_ARCTOS = 15.0   # rad/s²
# ══════════════════════════════════════════════════════════════════════════════
# HÁPTICA REAL (15-oct): yemas de la OrcaHand Touch FÍSICA → vibración del MANUS
# Pega este bloque al FINAL de config.py (los valores del final mandan).
# ══════════════════════════════════════════════════════════════════════════════
# De dónde sale la vibración de cada dedo del guante:
#   "real" → sensores táctiles de la mano física (lo que pediste; la simulación NO vibra)
#   "sim"  → contactos de la mano simulada (lo de antes)
#   "auto" → real si llegan datos de los sensores; si no, la simulación
MANUS_HAPTICA_FUENTE = "real"

# Puente de los sensores (puente_tactil.py, proceso aparte con el orca_core NUEVO).
ORCA_TACTIL_ACTIVO    = True                                     # GuanteManus lo lanza y lo cierra solo
ORCA_TACTIL_CORE_RUTA = os.path.join(_BASE_DIR, "orca_core_touch")   # git clone ... orca_core_touch
ORCA_TACTIL_PUERTO    = "auto"            # o el puerto de los sensores, p. ej. "/dev/ttyACM0"
                                          # (NUNCA el de los motores: PUERTO_HARDWARE se excluye siempre)
ORCA_TACTIL_UDP       = "127.0.0.1:47311" # puerto local por el que llegan las fuerzas
ORCA_TACTIL_MAX_EDAD_S = 0.15             # datos más viejos que esto = sin datos → no vibra

# Qué SENSOR (nombre de orca_core) va en cada dedo FÍSICO de tu OrcaHand. orca_core los nombra
# en otro orden que tu cableado: medido el 15-oct (apretando anular vibraba corazón, etc.).
# El meñique se deduce (es el que queda): compruébalo apretándolo.
ORCA_TACTIL_SENSOR_DE_DEDO = {
    "pulgar":  "index",
    "indice":  "ring",
    "corazon": "thumb",
    "anular":  "middle",
    "menique": "pinky",
}

# Fuerza → vibración (por dedo). El sensor mide en pasos de 0.1 N (0 … 25.5 N).
#   < 0.3 N → nada · 0.3 N → lo MÍNIMO que se nota · crece en LÍNEA RECTA · ≥ 20 N → al máximo
MANUS_TACTIL_MEDIDA = "normal"            # "normal" = solo la presión (fz) · "total" = |f| (incluye roce)
MANUS_TACTIL_F_ON   = 0.3                 # N: aquí empieza a vibrar
MANUS_TACTIL_F_MAX  = 20.0                # N: aquí llega al máximo (y se queda); o por dedo físico:
# MANUS_TACTIL_F_MAX = {"pulgar": 20.0, "indice": 20.0, "corazon": 20.0, "anular": 20.0, "menique": 20.0}
MANUS_TACTIL_VIBRACION_MIN = 0.02         # vibración a 0.3 N: la MÍNIMA que notas. Búscala con
                                          #   python probar_haptica_tactil.py --umbral
MANUS_TACTIL_GAMMA  = 1.0                 # 1.0 = lineal
MANUS_VIBRACION_MAX = 1.0
MANUS_HAPTICA_TAU_SUBIDA = 0.02           # s: al apretar responde casi al instante
MANUS_HAPTICA_TAU_BAJADA = 0.08           # s: al soltar se apaga sin parpadeos
MANUS_HAPTICA_HZ    = 30.0                # máx. envíos de vibración al guante por segundo
MANUS_HAPTICA_SIEMPRE = False             # True: vibra aunque el guante no mande ([S]/standby, modo IA...)
MANUS_HAPTICA_DEBUG = False               # True: imprime fuerza y vibración por dedo cada 0.5 s


# ═══════════════════════════════════════════════════════════════════════════
# BRAZO: ARCTOS <-> FAIRINO FR5 (26-oct) — añadido por aplicar_fr5.py
# ═══════════════════════════════════════════════════════════════════════════
# Al ir al final MANDA sobre los valores del ARCTOS de más arriba.
# "ARCTOS" = todo exactamente como antes · "FR5" = brazo FAIRINO FR5.
ROBOT_BRAZO = "FR5"

if ROBOT_BRAZO == "FR5":
    # Modelo: lo genera crear_fairino_mjcf.py (mismo arctos_mjcf.xml con el FR5 en vez del ARCTOS)
    RUTA_XML = os.path.join(_BASE_DIR, "fairino_description", "mjcf", "fairino_mjcf.xml")

    # Rangos reales del FR5: ±175° · -265°..+85° · ±160° · -265°..+85° · ±175° · ±175°
    LIMITES_CTRL_BRAZO = [
        (-3.0543, 3.0543),   # joint1 (base)
        (-4.6251, 1.4835),   # joint2 (hombro)  — 0 = brazo horizontal, -π/2 = vertical
        (-2.8274, 2.8274),   # joint3 (codo)    — 0 = antebrazo recto
        (-4.6251, 1.4835),   # joint4 (muñeca 1)
        (-3.0543, 3.0543),   # joint5 (muñeca 2)
        (-3.0543, 3.0543),   # joint6 (muñeca 3, giro de la brida)
    ]

    # Muñeca "recta" (la OrcaHand sigue la línea del antebrazo): joint4 = 0 y joint5 = +90°.
    # (El FR5 tiene muñeca tipo UR; en el ARCTOS la recta era joint5 = 0.)
    JOINT5_FIJO = 1.5708

    # Reposo: brazo vertical, antebrazo 25° por debajo de la horizontal, muñeca recta
    POSICION_STANDBY_BRAZO = [0.0, -1.5708, 2.0, 0.0, 1.5708, 0.0]

    # Ángulos humanos → joints. Calculado con la cinemática del FR5 para que las
    # direcciones sean LAS MISMAS que con el ARCTOS (delante del robot = -Y, T-pose = -X):
    #   joint1 = π/2 - azimut     (T-pose → 0, brazo al frente → π/2)
    #   joint2 = π/2 - elevación  (horizontal → 0, arriba → -π/2, colgando → +π/2)
    #   joint3 = -codo            (codo recto → 0; T-pose + codo 90° → antebrazo hacia arriba)
    # Si al probar un joint se mueve al revés, cambia SOLO su signo (+1 ↔ -1).
    ARM_IMITACION = {
        "joint1": ("azimut",    -1.0, math.pi / 2, 1.0),
        "joint2": ("elevacion", -1.0, math.pi / 2, 1.0),
        "joint3": ("codo",      -1.0, 0.0,         1.0),
    }
    IMITAR_LIMITES = {
        "joint1": (-3.0543, 3.0543),
        "joint2": (-2.10, 1.4835),   # -2.10 = 0.5 rad más allá de la vertical, hacia atrás
        "joint3": (-2.8274, 2.8274),
    }

    # Pose al pulsar [A]: None = T-pose del robot (joint1..3 = 0, brazo horizontal).
    # La del ARCTOS salía de una foto del brazo real; para el FR5 se puede sacar igual.
    POSE_BRAZO_ACTIVAR = None

    # Espacio de trabajo. El hombro del FR5 está a 0.152 m sobre su base (d1 oficial; el modelo
    # antiguo decía 0.155) (el del ARCTOS a
    # 0.29) y el brazo es mucho más largo (0.425 + 0.395 m). Con CODO_Z_MIN = 0 el codo puede
    # bajar hasta la altura de la peana (~21° por debajo de la horizontal).
    CODO_Z_MIN    = 0.0
    ARM_RADIO_MAX = 0.90


# ═══════════════════════════════════════════════════════════════════════════
# SOPORTE DEL BRAZO: COLUMNA CON HOMBROS A 45° (27-oct)
# ═══════════════════════════════════════════════════════════════════════════
# El FR5 va en la placa del HOMBRO DERECHO del soporte nuevo, inclinado 45° hacia fuera y
# hacia abajo (fairino_description/mjcf/decoracion/crear_soporte_hombros.py). Su HOMBRO
# (eje de joint2) estaba en (0, 0, 0.155) con la peana (5-nov: con el d1 real, 152 mm, queda en
# (0.0021, 0, 0.1571): la base no se mueve y el hombro está 3 mm más cerca); el suelo baja a
# z = -0.975 (hombro a 1.13 m del suelo, como el de una persona).
#
# Se DETECTA SOLO leyendo fairino_mjcf.xml: si base_link tiene la base girada, se aplica
# todo este bloque; si se vuelve a poner la peana (base sin girar), no se aplica nada y
# todo queda exactamente como antes. El mapeo ángulos humanos → joints con la base
# inclinada está en skill_shield/brazo/base_brazo.py.
def _leer_soporte(ruta):
    """(base_girada, z del suelo) según el MJCF."""
    import xml.etree.ElementTree as _ET
    girada, suelo = False, None
    try:
        for _g in _ET.parse(ruta).getroot().iter():
            if _g.tag == "body" and _g.get("name") == "base_link" and _g.get("quat"):
                _q = [float(_v) for _v in _g.get("quat").split()]
                girada = abs(abs(_q[0]) / math.sqrt(sum(_v * _v for _v in _q)) - 1.0) > 1e-9
            elif _g.tag == "geom" and _g.get("name") == "floor":
                suelo = float(_g.get("pos", "0 0 0").split()[2])
    except Exception as _e:
        print(f"⚠ config: no puedo leer {ruta} ({_e})")
    return girada, suelo


SOPORTE_HOMBROS, SOPORTE_SUELO_Z = _leer_soporte(RUTA_XML)
SOPORTE_HOMBROS = SOPORTE_HOMBROS and ROBOT_BRAZO == "FR5"

if SOPORTE_HOMBROS:
    if SOPORTE_SUELO_Z is None:
        SOPORTE_SUELO_Z = -0.975

    # Reposo: brazo a lo largo del eje de la base (hacia fuera y abajo, 45°, como en la
    # foto), antebrazo hacia delante y ~12° hacia abajo, muñeca recta. [S] y el final de [5].
    POSICION_STANDBY_BRAZO = [0.0, -1.5708, 1.35, 0.0, 1.5708, 0.0]

    # Alturas mínimas: antes eran respecto a la PEANA (z = 0, casi a la altura del hombro);
    # ahora el brazo cuelga como el de una persona y el límite es el SUELO (y el bastidor
    # del soporte, que llega a 0.23 m del suelo). Con el hombro a 1.13 m ya casi no hace
    # falta "comprimir" la parte de abajo: tu brazo colgando ≈ su brazo colgando.
    PALMA_Z_MIN  = SOPORTE_SUELO_Z + 0.25   # centro de la palma (los dedos cuelgan algo más)
    CODO_Z_MIN   = SOPORTE_SUELO_Z + 0.30
    ARM_Z_MINIMA = SOPORTE_SUELO_Z + 0.20

    # joint1 solo se bloquea en la IK con la palma a menos de esto (°) del eje de la base
    BASE_BLOQUEO_ANG = 5.0

    # Peana del taladro de [4]/[5]: misma dirección y altura respecto al hombro que antes,
    # pero más lejos (0.72 m en horizontal en vez de 0.49). Con el brazo colgando del hombro,
    # en (-0.45, -0.20) su columna quedaba justo donde se mueve el brazo y la demo [5] la
    # atravesaba al subir y al volver a reposo.
    TALADRO_POS_SOPORTE = (-0.66, -0.29, 0.25)

    # Demo [5]: con la base inclinada, a 5 cm del alcance máximo la muñeca (joint6) llega a
    # su tope de ±175° al perforar. Con 10 cm el techo queda ~5 cm más bajo y sobra muñeca.
    TALADRO_DEMO_MARGEN_TECHO = 0.10

    # Mesa de [3]: con el brazo colgando, a la altura a la que trabaja una mano (0.55 m por
    # debajo del hombro, 0.58 m sobre el suelo), delante y a la derecha del robot: tu brazo
    # hacia delante-derecha y abajo con el codo doblado. (Antes: (-0.8, 0.0, 0.10), a la
    # altura de la peana.) No se pone justo a 45° hacia fuera-abajo: ahí apunta el eje de
    # joint1 y la base tendría que girar mucho para pequeños movimientos de la mano.
    # Las patas quedan fuera del bastidor del soporte. La usa entorno_taller.py.
    TALLER_POS_MESA = (-0.45, -0.45, -0.40)
    # Cámara del visor al pulsar [3] (la de antes quedaba detrás de la columna)
    TALLER_CAMARA = {
        "lookat":    (0.6 * TALLER_POS_MESA[0], 0.6 * TALLER_POS_MESA[1], TALLER_POS_MESA[2] + 0.20),
        "distancia": 1.6,
        "azimut":    60,
        "elevacion": -22,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TELEOPERACIÓN FR5 v2 (27-oct): muñeca + codo con las DOS cámaras → IK analítica de 6 ejes
# (skill_shield/teleop_fr5). Pegado al FINAL de config.py: manda sobre lo de arriba.
# ═══════════════════════════════════════════════════════════════════════════
# True  → el brazo lo mueve el módulo nuevo (joint1..6 a la vez; el antiguo PASO 2-7 queda
#         desactivado, incluida la pronosupinación por joint6).
# False → todo exactamente como antes.
TELEOP_FR5_V2 = True

# Escala humano → robot. None = automática: tu brazo estirado = TELEOP_FR5_FRACCION del
# alcance del FR5+OrcaHand (≈1,65 con un brazo de 0,56 m). Un valor fijo (p. ej. 1.2) mueve
# el robot 1,2 m por cada metro tuyo: menos amplificación, menos alcance.
TELEOP_FR5_ESCALA   = None
TELEOP_FR5_FRACCION = 0.85

# Altura mínima (m, mundo del MJCF) de la articulación de la muñeca de la OrcaHand. None =
# suelo + 0.25 con el soporte de hombros, o ARM_Z_MINIMA con la peana.
TELEOP_FR5_Z_MIN = None

# Límites articulares extra (rad) por joint (0..5), estrechando los reales del FR5 (ya con 5°
# de margen). Ej.: {1: (-2.4, 1.0)} impide que el hombro suba más de lo que sube un humano.
TELEOP_FR5_LIMITES = {2: (-2.53, 2.53)}      # codo: ±145° como el humano

# Velocidades (rad/s) y aceleraciones (rad/s²) del comando continuo con v2 (el FR5 admite
# 3,15 rad/s). Con la escala 1,65 el robot debe ir 1,65 veces más rápido que tu mano.
TELEOP_FR5_V_MAX     = 3.0      # tope del propio módulo (lo que no llega, se queda atrás, nunca salta)
TELEOP_FR5_SUAVE_VEL = 2.5      # GeneradorSuave (hilo de física), por joint
TELEOP_FR5_SUAVE_ACEL = 20.0

# ── ARMADO CON PEDAL Y ENTRADA LENTA (5-nov) ───────────────────────────────────────────
# Antes, al acabar la T-pose el robot iba SOLO hacia tu postura a 1,2 rad/s (v_transicion de
# la IK; TELEOP_FR5_V_MAX no limitaba ese arranque). Ahora:
#   · [A] + T-pose solo CALIBRA: el robot no se mueve (tampoco va a la T-pose del simulador).
#   · Al acabar, "CALIBRADO - pisa el PEDAL": el robot sigue quieto hasta que lo armas.
#   · Armado: entra hacia tu postura a TELEOP_FR5_V_ENTRADA rad/s (cada joint) y, cuando está
#     a menos de TELEOP_FR5_ENTRADA_FIN_RAD, pasa a seguirte a TELEOP_FR5_V_MAX.
#   · Desarmar (pedal otra vez / soltarlo en hombre muerto, [S] o [A]) → quieto. Al volver a
#     armar, entra otra vez despacio desde donde esté.
# Pedal: "tecla" = pedal USB que hace de teclado (o la tecla de abajo en la ventana de la
# cámara), cada pisada alterna · "evdev" = se lee de /dev/input y permite HOMBRE MUERTO
# (pip install evdev y usuario en el grupo input) · None = sin pedal, se arma al calibrar
# (solo para el simulador). Ver skill_shield/teleop_fr5/pedal.py.
TELEOP_FR5_V_ENTRADA         = 0.15     # rad/s (≈ 8,6 °/s): pedido 0,1-0,2
TELEOP_FR5_ENTRADA_FIN_RAD   = 0.05     # rad (≈ 3°) por joint: aquí acaba la entrada lenta
# 9-nov: HOMBRE MUERTO → el brazo se mueve SOLO mientras se pisa / se mantiene la tecla.
#   "teclado": mantener la P en la ventana de la cámara (mientras no haya pedal físico)
#   "evdev":   pedal USB físico (TELEOP_FR5_PEDAL_EVDEV)
#   "tecla":   cada pulsación alterna (OpenCV no ve cuándo se suelta): solo como reserva
TELEOP_FR5_PEDAL             = "teclado"  # "teclado" | "evdev" | "tecla" | None
TELEOP_FR5_PEDAL_TECLAS      = (ord("p"), ord("P"), 112)   # modo "tecla": códigos de cv2.waitKeyEx
TELEOP_FR5_PEDAL_EVDEV       = None     # modo "evdev": "/dev/input/by-id/...", parte del nombre,
                                        # o None = buscarlo solo (nombre con "pedal", "foot"...)
TELEOP_FR5_PEDAL_EVDEV_CODIGO = None    # código de la tecla del pedal (None = cualquiera)
TELEOP_FR5_PEDAL_HOMBRE_MUERTO = True   # modo "evdev": armado solo MIENTRAS se pisa
TELEOP_FR5_PEDAL_EXCLUSIVO   = True     # modo "evdev": tomarlo en exclusiva (no escribe teclas)
TELEOP_FR5_PEDAL_TECLADO_CODIGO = 25    # modo "teclado": código evdev de la tecla (25 = KEY_P)

# ── OBSTÁCULOS Y ZONAS PROHIBIDAS DE LA IK (5-nov) ─────────────────────────────────────
# Antes la IK solo evitaba el suelo y el soporte de hombros. Ahora también estas CAJAS
# (skill_shield/teleop_fr5/obstaculos.py): el robot se aparta de ellas a menos de 7 cm y no
# acepta posturas que las atraviesen (margen 3 cm). Coordenadas en metros, en el MUNDO del
# simulador: hombro del FR5 en ~(0, 0, 0.157), el robot mira hacia −Y, T-pose hacia −X,
# z hacia arriba, suelo en SOPORTE_SUELO_Z. Se ven en el visor como cajas translúcidas
# (naranja = objeto, rojo = zona prohibida) con TELEOP_FR5_OBSTACULOS_VER.
# ⚠ MEDIRLAS EN EL LABORATORIO: lo que tenga posición None NO se protege (se avisa al arrancar).
TELEOP_FR5_OBSTACULOS_VER = False
_MESA = globals().get("TALLER_POS_MESA", (-0.45, -0.45, -0.40))      # (x, y, z del tablero)
_PEANA = globals().get("TALADRO_POS_SOPORTE", (-0.66, -0.29, 0.25))  # (x, y, z de la cara de arriba)
TELEOP_FR5_OBSTACULOS = [
    # desde_suelo=True: del suelo hasta z_sup. Por defecto, donde están en el simulador ([3] y [4]).
    dict(nombre="mesa",  xy=_MESA[:2],  semi_xy=(0.18, 0.28),     z_sup=_MESA[2],  desde_suelo=True),
    # La peana del simulador está DENTRO del espacio de trabajo (de ahí coge el robot el taladro):
    # activa en ese sitio bloquea media teleoperación (probado con el diagnóstico de la IK). Ponla
    # en su sitio REAL y actívala; si hay que coger cosas de encima, deja z_sup algo más baja.
    dict(nombre="peana", xy=_PEANA[:2], semi_xy=(0.0825, 0.0825), z_sup=_PEANA[2], desde_suelo=True,
         activo=False),
    # caja suelta: dict(nombre="armario", centro=(x, y, z), semi=(sx, sy, sz)),
]
TELEOP_FR5_CAMARAS = [
    # cuerpo de la OAK-D (semi-lados) + trípode hasta el suelo (caja de radio_tripode)
    dict(nombre="camara_A", xyz=None, semi=(0.06, 0.06, 0.04), tripode=True, radio_tripode=0.15),
    dict(nombre="camara_B", xyz=None, semi=(0.06, 0.06, 0.04), tripode=True, radio_tripode=0.15),
]
TELEOP_FR5_ZONAS_PROHIBIDAS = [
    # Espacio reservado aunque no haya nada: puesto del operador, paso de personas...
    # dict(nombre="operador", centro=(x, y, z), semi=(sx, sy, sz)),
]

# ── LÍMITES BLANDOS EN LA CONTROLADORA DEL FR5 (5-nov) ───────────────────────────────────
# Tope de seguridad independiente del programa: fr5_servo los ESCRIBE en la controladora al
# conectar (SetLimitPositive/Negative) y los comprueba leyéndolos (GetJointSoftLimitDeg).
# Si un comando los supera, el FR5 se para solo aunque el programa falle. En FR5_SIMULACRO
# solo se leen y se comparan (no se cambia nada en el robot).
# Por defecto = los topes duros de la IK de teleoperación + 5° (la teleoperación nunca debería
# tocarlos), dentro de los rangos del FR5. Grados, en el convenio de la controladora.
FR5_FIJAR_LIMITES_BLANDOS = True
FR5_LIMITES_BLANDOS_DEG = (
    (-175.0, 175.0),    # joint1 (base)
    (-265.0,  85.0),    # joint2 (hombro)
    ( -10.0, 150.0),    # joint3 (codo): solo hacia el lado "humano"
    (-105.0,  85.0),    # joint4 (muñeca 1)
    (   5.0, 175.0),    # joint5 (muñeca 2)
    (-175.0, 175.0),    # joint6 (brida)
)

# Fusión de cámaras (ver fusion_muneca_codo.Parametros): ángulo del antebrazo respecto al rayo
# de la cámara por debajo del cual esa cámara NO ve bien codo y muñeca (escorzo / palma delante)
TELEOP_FR5_ESCORZO_DEG = (20.0, 40.0)
TELEOP_FR5_HOLD_S      = 0.8    # s sin ver una articulación antes de darla por perdida

# Pronosupinación: "hands" = dirección índice→meñique de MediaPipe Hands (cámara A);
# "fija" = se mantiene la del inicio (útil si Hands va mal con el guante puesto)
TELEOP_FR5_PRONO = "hands"
TELEOP_FR5_ROLL_RANGO_DEG = 120.0   # pronosupinación máxima a cada lado del neutro de la calibración

# ── POSTURA HUMANA DEL ROBOT (27-oct, 2ª ronda) ──────────────────────────────────────────
# De las 8 configuraciones con las que el FR5 llega a un mismo punto solo una se parece a un
# brazo humano; las otras tienen la muñeca volteada (joint5 < 0, joint4 girado ~180°), el codo
# al revés o el hombro dado la vuelta, y al pasar de una a otra el brazo "da vueltas". Con
# True el robot solo adopta posturas humanas (ik_teleop.PosturaHumana):
#   · codo (joint3) solo hacia el lado del bíceps (se decide solo; TELEOP_FR5_CODO_SIGNO = ±1 lo fija), 0…145°
#   · muñeca: |joint4| ≤ MUNECA_DEG[0], |joint5 − 90°| ≤ MUNECA_DEG[1]  (0 / 90° = mano alineada)
#   · hombro: el brazo del robot dentro del cono del hombro humano (HOMBRO_AZ_* de arriba)
# Si una orientación de la mano no es alcanzable con una muñeca humana, el robot sigue la
# posición exacta y deja la orientación lo más cerca posible (HUD: APROX), sin dar la vuelta.
TELEOP_FR5_POSTURA_HUMANA = True
TELEOP_FR5_CODO_SIGNO     = None
TELEOP_FR5_MUNECA_DEG     = (80.0, 60.0)

TELEOP_FR5_DEBUG = True          # imprime estado/rama/error cada DEBUG_ANGULOS_CADA_S

# ── CORRECCIONES TRAS EL ANÁLISIS DEL VÍDEO (28-oct, 2ª parte) ─────────────────────────────
# Todo automático, sin medir nada a mano:
#  · TELEOP_FR5_AUTO_EXTRINSECA: al pulsar [A] se comprueba que A y B coinciden con los puntos
#    de tu cuerpo; si no (el registro del 28-oct dio 36-46 cm: alguna cámara se había movido),
#    la extrínseca se recalcula sola con tu cuerpo (unos segundos moviendo los brazos) y se
#    guarda en camaras_extrinsecas.json (la anterior queda como .bak_*). Ya no hace falta el
#    tablero ni calibrar_extrinseca.py. TELEOP_FR5_VIGILAR_EXTRINSECA: si durante la
#    teleoperación A y B dejan de coincidir (alguien mueve una cámara), se recalibra sola.
#  · TELEOP_FR5_TRIANGULAR: hombro, codo y muñeca salen de los RAYOS de las dos cámaras (56° y
#    2 m entre ellas); el estéreo de cada OAK queda como apoyo (falla sobre ropa oscura).
#  · TELEOP_FR5_VERTICAL_AUTO / TELEOP_FR5_ARRIBA_REF = "calibrada": la vertical sale de tu
#    tronco en la T-pose, no del eje Y de la cámara A (que puede estar inclinada).
#  · TELEOP_FR5_IK = "diferencial": IK que parte de la postura anterior, una sola rama, sin
#    transiciones; la posición de la muñeca manda y orientación/codo/límites humanos son
#    coste. "analitica" = la de antes.
#  · TELEOP_FR5_SIN_PIPELINE_ANTIGUO: con la v2 no se calcula el brazo antiguo (FusorBrazo,
#    ángulos del HUD): era trabajo duplicado en cada fotograma.
TELEOP_FR5_AUTO_EXTRINSECA     = True
TELEOP_FR5_VIGILAR_EXTRINSECA  = True
TELEOP_FR5_TRIANGULAR          = True
TELEOP_FR5_VERTICAL_AUTO       = True
TELEOP_FR5_ARRIBA_REF          = "calibrada"
TELEOP_FR5_IK                  = "diferencial"
TELEOP_FR5_ALCANCE_MAX         = 0.88       # fracción máx. del alcance del FR5 que se pide (antes 0,92)
TELEOP_FR5_SIN_PIPELINE_ANTIGUO = True
# Fluidez (28-oct, tarde): la fusión + IK de la v2 corren en un hilo aparte, solapadas con la
# espera de la cámara y MediaPipe. El comando del brazo sale 1 fotograma más tarde (~35 ms).
# False = en el propio bucle, como antes.
TELEOP_FR5_EN_PARALELO = True

# ── DIAGNÓSTICO: separar medida de IK (28-oct) ─────────────────────────────────────────
# [R] con la imitación activa graba TELEOP_FR5_DIAG_S segundos quieto y saca el informe de la
# medida (ruido, huesos, ángulos, A contra B, vertical) en terminal y en diagnosticos/*.npz.
# La etiqueta va en el nombre del archivo: cámbiala para grabar otras poses ("colgando",
# "arriba"...) y comparar la elevación medida con la real (≈10° / 90° / 170°).
# La parte de la IK se prueba sin hardware:  python -m skill_shield.teleop_fr5.diagnostico ik
TELEOP_FR5_DIAG_S        = 10.0
TELEOP_FR5_DIAG_ETIQUETA = "tpose"

# ═══════════════════════════════════════════════════════════════════════════
# FR5 REAL: proceso fr5_servo — ServoJ cada 8 ms (5-nov)
# ═══════════════════════════════════════════════════════════════════════════
# skill_shield/hardware/fr5_servo.py. Durante la teleoperación, el comando del brazo del
# simulador (ctrl de joint1..6, ya suavizado por el generador del PASO 7.1) se publica en cada
# mj_step y un PROCESO aparte lo manda al FR5 con ServoJ a ritmo fijo.
#   · El simulador NO es el estado real (5-nov): al arrancar, al entrar en teleoperación y en
#     cada vuelta mientras el brazo está DESARMADO, MuJoCo (y el generador suave) se copian de
#     la postura LEÍDA del FR5; la IK sale de ahí.
#   · Solo se manda ServoJ con el brazo ARMADO (pedal, ver TELEOP_FR5_PEDAL); al armar empieza
#     desde la pose real. Si aun así el simulador estuviera lejos, se acerca a
#     FR5_VEL_APROX_DEG_S hasta estar a < FR5_DESFASE_MAX_DEG.
#   · Desarmar, [S], [A] o salir de teleoperación (IA, demo, [Q]) → ServoMoveEnd: quieto.
# Probarlo SOLO, sin cámaras:  python -m skill_shield.hardware.fr5_servo [--real] [--oscilar 2]
FR5_REAL_ACTIVO = False          # True = arrancar fr5_servo también en SIMULACION (en los modos con
                                 # hardware se arranca solo: TELEOP_REAL_* / SLIDERS_* más abajo)
FR5_SIMULACRO   = True           # True = conecta y lee la pose, pero NO manda ServoJ (1ª prueba)
FR5_IP          = "192.168.58.2" # IP de fábrica de la controladora (el PC en 192.168.58.x)
FR5_SDK_RUTA    = None           # carpeta que CONTIENE fairino/ (SDK oficial, no está en PyPI)
FR5_PERIODO_S   = 0.008          # 8 ms entre ServoJ (es también el cmdT)
FR5_CMD_TYPE    = 0              # 0 = XML-RPC · 1 = UDP (si ServoJ por XML-RPC tarda > 8 ms)
FR5_VEL_MAX_DEG_S    = 60.0      # tope por joint al imitar (subir cuando todo esté probado)
FR5_ACEL_MAX_DEG_S2  = 300.0
FR5_VEL_APROX_DEG_S  = 10.0      # acercamiento inicial a la pose del simulador
FR5_DESFASE_MAX_DEG  = 5.0
FR5_MARGEN_LIMITES_DEG = 3.0     # margen respecto a los topes del FR5
FR5_WATCHDOG_S  = 0.25           # sin comandos nuevos durante esto → mantiene la pose
FR5_ERRORES_MAX = 10             # ServoJ fallidos seguidos → deja de enviar
# MJCF → controladora: q_FR5[°] = SIGNO · grados(q_MJCF) + OFFSET. El MJCF sale del URDF
# oficial (mismos rangos y sentidos que la DH, ver cinematica.py), así que debería ser la
# identidad, pero HAY QUE COMPROBARLO joint a joint con el robot real en FR5_SIMULACRO.
FR5_SIGNO       = (1, 1, 1, 1, 1, 1)
FR5_OFFSET_DEG  = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
FR5_PRIORIDAD_RT = 50            # SCHED_FIFO si hay permisos (si no, prioridad normal)
FR5_CPU          = None          # núcleo fijo para el proceso (None = cualquiera)
# ── COLISIONES FALSAS (9-nov) ──────────────────────────────────────────────────────────
# La controladora detecta una colisión cuando el par que mide en un joint no cuadra con el que
# espera según su modelo. Ese modelo necesita saber CUÁNTO PESA lo que cuelga de la brida y
# CÓMO está montada la base respecto a la gravedad: si cree que el robot está en el suelo sin
# carga y en realidad está inclinado 135° con la mano puesta, con el brazo estirado J2 da
# "colisión" sin tocar nada. Valores en diagnosticos/ajustar_gemelo.py (apartado 3).
# None = no tocar lo que ya tenga la controladora. Solo se escriben fuera de SIMULACRO.
# Comprobación: en modo arrastre de la WebApp, al soltar el brazo debe quedarse QUIETO.
FR5_CARGA_KG            = 2.2     # mano + adaptador, pesados el 10-nov
FR5_CARGA_CDM_MM        = None    # centro de masas en la brida, mm (x, y, z). Sin medir: sacadlo con
                                  # la identificación de carga de la WebApp y ponedlo aquí
# 10-nov: herramienta (TCP) = centro de la palma de la ORCA, sacado del modelo de MuJoCo
# (mano_mount + OFFSET_LOCAL_PALMA_CENTRO pasado al marco de la brida): ~195 mm a lo largo del eje
# de la brida. Comprobadlo con un metro (cara de la brida → centro de la palma) y corregid z.
# None = no tocar el que tenga la controladora. Como la carga, solo se escribe fuera de SIMULACRO.
FR5_TCP_MM              = (0.0, 2.9, 195.0, 0.0, 0.0, 0.0)   # (x, y, z) mm, (rx, ry, rz) °
FR5_HERRAMIENTA_ID      = 1       # nº del sistema de herramienta en la controladora (1-14)
FR5_ANGULO_INSTALACION  = None    # (inclinación, giro) en grados, p. ej. (135.0, 90.0)
FR5_NIVEL_COLISION      = None    # 6 niveles 1-10, más alto = menos sensible, p. ej. (5, 7, 5, 4, 4, 4)
FR5_ESTRATEGIA_COLISION = None    # 0 = pausa con error (lo normal) · 2 = para con error
FR5_PARAR_SI_COLISION   = True    # colisión/fallo con el ServoJ en marcha → dejar de enviar YA

# ── MODO HARDWARE_REAL (9-nov): teleoperación con cámaras + FR5 FÍSICO ──────────────────────
# "Mano Física (USB) + Gemelo Digital" del menú. El FR5 se arranca en este modo aunque
# FR5_REAL_ACTIVO sea False. Seguridad, igual que hasta ahora con el pedal:
#   [A] + T-pose calibra (el robot NO se mueve) → PEDAL arma: entra despacio
#   (TELEOP_FR5_V_ENTRADA) hacia tu postura y te sigue → PEDAL otra vez, [S], [A], [I], [Q]
#   o salir de la teleoperación → desarmado (ServoMoveEnd, quieto).
# Con el FR5 real el brazo SOLO se arma con TELEOP_FR5_V2 = True y un pedal ("tecla"/"evdev");
# con TELEOP_FR5_PEDAL = None no se arma nunca.
TELEOP_REAL_CON_FR5           = True    # arrancar fr5_servo en este modo
TELEOP_REAL_FR5_SIMULACRO     = False   # True = conecta y lee, pero NO manda ServoJ (prueba en seco)
TELEOP_REAL_FR5_VEL_MAX_DEG_S = 30.0    # tope de fr5_servo en este modo (°/s, por joint);
                                        # subir hacia FR5_VEL_MAX_DEG_S cuando esté probado

# ── GEMELO DIGITAL Y BRAZO FANTASMA (9-nov) ───────────────────────────────────────────────
# Con el FR5 conectado, el brazo de MuJoCo enseña la pose REAL del robot (no el comando) y
# un brazo FANTASMA translúcido enseña el OBJETIVO (IK de teleoperación o sliders), con la
# punta objetivo en verde y la punta real en rojo. Sin FR5: brazo simulado + fantasma.
GEMELO_POSE_REAL      = True          # brazo de MuJoCo = pose leída del FR5
FANTASMA_ACTIVO       = True          # dibujar el brazo fantasma y las dos puntas
FANTASMA_CUERPO_PUNTA = "mano_mount"  # cuerpo del MJCF que se usa como punta del brazo
GEMELO_DEBUG          = True          # cada 2 s: pose real vs simulador vs comando (terminal)

# ── EPISODIO POR POSES (9-nov): episodio_poses.py ───────────────────────────────────────
# Brazo de pose en pose con Enter (poses en episodio_poses.txt) + mano con el guante MANUS.
EPISODIO_FR5_VEL_MAX_DEG_S = 7.0   # °/s por joint (la 1ª ida a la inicial va a FR5_VEL_APROX_DEG_S)
EPISODIO_FR5_SIMULACRO     = False  # True = conecta y lee, pero NO mueve el FR5

# ── MODO GEMELO_SLIDERS (9-nov): mano con guante + BRAZO con una ventana de sliders ─────────
# Se elige en el menú de arranque. La mano funciona igual que en HARDWARE_REAL; el brazo NO
# sale de las cámaras sino de skill_shield/interfaz/sliders_brazo.py. Los sliders van en
# grados de la controladora (como la WebApp). Flujo: [A] activa la mano; ARMAR en la ventana
# mueve el brazo; STOP / Esc / Espacio / cerrar la ventana → desarmado (FR5 quieto).
# El FR5 se arranca en este modo aunque FR5_REAL_ACTIVO sea False (si SLIDERS_CON_FR5).
SLIDERS_CON_FR5           = True    # arrancar fr5_servo en este modo
SLIDERS_FR5_SIMULACRO     = False   # True = conecta y lee, pero NO manda ServoJ (prueba en seco)
SLIDERS_FR5_VEL_MAX_DEG_S = 30.0    # tope de fr5_servo en este modo (°/s, por joint)
SLIDERS_SUAVE_VEL         = 0.6     # GeneradorSuave: rad/s por joint (~34 °/s) × % de la ventana
SLIDERS_SUAVE_ACEL        = 3.0     # rad/s²
SLIDERS_VELOCIDAD_INICIAL = 0.3     # % de velocidad con el que abre la ventana (0.05-1.0)
SLIDERS_MARGEN_DEG        = 5.0     # margen respecto a los topes del FR5 / límites blandos
SLIDERS_LIMITES_DEG       = None    # ((min, max) × 6) en grados para fijarlos a mano; None = auto

# ═══════════════════════════════════════════════════════════════════════════
# IA: GRABAR BRAZO + MANO, ENTRENAR Y REPRODUCIR (5-nov)
# ═══════════════════════════════════════════════════════════════════════════
# [G] graba un episodio de la SIMULACIÓN (6 joints del FR5 + 17 de la OrcaHand: posición y
# comando) a ritmo fijo, desde el hilo de la física (skill_shield/ia/grabador.py). [E] entrena
# en otro proceso (python -m skill_shield.ia.act <tarea>). [I] reproduce en la simulación y,
# si están conectados y ARMADOS, en el FR5 y la mano reales. La IA es "ciega": aprende el
# movimiento (historial de posturas → comandos siguientes), no ve objetos.
IA_FPS              = 30.0    # fotogramas por segundo de la grabación (y de la reproducción)
IA_RECORTAR_QUIETOS = True    # quitar el rato quieto del principio y del final de cada episodio
IA_MISMA_TAREA      = True    # [G] no vuelve a pedir el nombre: [Shift+G] para cambiar de tarea
IA_V_MAX_BRAZO      = 1.0     # rad/s: tope por joint del brazo al reproducir (además de fr5_servo)
IA_PESO_EXP         = 0.15    # ensamblado temporal: más alto = más peso a la predicción reciente


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONAMIENTO AUTÓNOMO (10-nov) — última opción del menú (skill_shield/autonomo/)
# ═══════════════════════════════════════════════════════════════════════════
# Paso 2 de la hoja de ruta: cámara de cabeza + gemelo en MuJoCo + web, SIN mover el robot
# (fr5_servo va siempre en simulacro en este modo, diga lo que diga FR5_SIMULACRO).
AUTONOMO_CON_FR5      = True             # arrancar fr5_servo (solo lectura) para el gemelo
AUTONOMO_CAM_MXID     = "1944301081B5FA4800"   # cámara de CABEZA = la antigua cámara B
AUTONOMO_CAM_VENTANA  = False            # True = además de la web, ventana de OpenCV con la imagen
AUTONOMO_CAM_IR_MA    = 800              # proyector IR de puntos (mejor profundidad en la mesa lisa)
AUTONOMO_JPEG_CALIDAD = 80               # calidad del vídeo de la web (más alto = más ancho de banda)
AUTONOMO_WEB_HOST     = "0.0.0.0"        # "0.0.0.0" = abierta a la red local · "127.0.0.1" = solo este PC
AUTONOMO_WEB_PUERTO   = 8080
AUTONOMO_HZ           = 60.0             # refresco del gemelo
AUTONOMO_PYTHON_PERCEPCION = None        # intérprete de OTRO venv para la cámara/percepción
                                         # (p. ej. si YOLO choca con numpy<2); None = el mismo
AUTONOMO_CON_MANO     = True             # vigilar la OrcaHand (SOLO LECTURA: no activa el par)
AUTONOMO_MANO_POSTURA_PERIODO_S = 0.1   # leer la postura de la mano (muñeca + dedos) para el gemelo y
                                         # el mapa cada estos s (solo lectura); None = no leerla
AUTONOMO_CON_TACTIL   = True             # lanzar el puente de los táctiles (si ORCA_TACTIL_ACTIVO)
# Web (FastAPI): mandos protegidos con contraseña. CAMBIADLA. None = sin contraseña (cualquiera
# de la red podría pisar el pedal o reiniciar el robot desde la web).
AUTONOMO_WEB_CLAVE    = "elecnor"
# Pedal de la web: mantener pulsado el botón grande (o la barra espaciadora). La página manda un
# latido cada 100 ms; sin latido durante AUTONOMO_PEDAL_LATIDO_S = pedal SUELTO.
AUTONOMO_PEDAL_OBLIGATORIO = True        # cómo arranca la opción «Pedal obligatorio» de la web
AUTONOMO_PERMITIR_SIN_PEDAL = True       # False = la web NO puede desactivar el pedal
AUTONOMO_PEDAL_LATIDO_S = 0.35
# Gemelo en la web (render_gemelo.py): MuJoCo dibujado sin ventana, solo la escena (sin menús)
AUTONOMO_SIM_WEB      = True             # False = sin gemelo en la web
AUTONOMO_VENTANA_MUJOCO = True           # False = sin ventana de MuJoCo en el PC (solo la web)
AUTONOMO_SIM_ANCHO    = 960              # tamaño del vídeo del gemelo
AUTONOMO_SIM_ALTO     = 540
AUTONOMO_SIM_FPS      = 20
AUTONOMO_SIM_GL       = "egl"            # OpenGL sin ventana: "egl" (GPU) · "osmesa" (CPU) · "glfw"
AUTONOMO_SIM_VISTA    = None             # vista inicial; None = la del visor (azimut 90, elevación -20…)

# Mapa de obstáculos (paso 7, skill_shield/autonomo/mapa.py): lo calcula el proceso de la cámara
# con la profundidad (hace falta la calibración cámara→robot). Vóxeles en el marco del MJCF.
AUTONOMO_MAPA          = True
AUTONOMO_MAPA_RES      = 0.02            # m: lado de cada cubo
AUTONOMO_MAPA_LIMITES  = None            # ((xmin, xmax), (ymin, ymax), (zmin, zmax)) m; None = delante
                                         # del robot: x -1.0…0.6, y -1.1…0.4, z suelo…0.6
AUTONOMO_MAPA_HZ       = 4.0             # actualizaciones por segundo
# Filtros contra obstáculos FALSOS (11-nov): la cámara de cabeza usa una profundidad «honrada» (sin
# rellenar huecos ni filtro temporal, que se inventan superficies en paredes blancas y ventanas),
# y el mapa descarta lo incoherente, lo saturado, lo demasiado cerca y los cubos sueltos.
AUTONOMO_CAM_RELLENAR_HUECOS = 0     # radio de relleno de huecos del estéreo (la teleop usa STEREO_HOLE_FILLING)
AUTONOMO_CAM_FILTRO_TEMPORAL = False
AUTONOMO_CAM_CONFIANZA = 200         # confianza del estéreo (0-255, más bajo = más estricto)
AUTONOMO_MAPA_Z_MIN    = 0.35        # m: más cerca la OAK-D no mide (lo que salga es ruido)
# Rastro del brazo (12-nov): el robot se quita del mapa en todas sus poses de los últimos BARRIDO_S
# (la profundidad llega con retraso) y los cubos por los que pasa quedan libres.
AUTONOMO_MAPA_BARRIDO_S = 0.6
# Autocorrección de la calibración con el propio brazo (12-nov, autocalib.py): con el brazo quieto
# y a la vista se compara la profundidad con sus mallas y se corrige la traslación de la cámara.
AUTONOMO_AUTOCALIB = True
AUTONOMO_AUTOCALIB_GANANCIA = 0.8            # fracción de cada medida que se aplica
AUTONOMO_AUTOCALIB_MAX_M = 0.15              # corrección máxima (más: recalibrar)
AUTONOMO_AUTOCALIB_RESIDUO_MAX_M = 0.012     # medidas peores no se usan
AUTONOMO_AUTOCALIB_QUIETO_S = 0.4
AUTONOMO_AUTOCALIB_PASO_MAX_M = 0.08        # una medida que pida mover más de esto de golpe no se usa
# (el botón «Borrar mapa» también borra la corrección: vuelve a la calibración del archivo)
AUTONOMO_MAPA_MIN_PUNTOS = 4         # puntos en un cubo en una misma imagen para contar
AUTONOMO_MAPA_MIN_GRUPO  = 4         # cubos ocupados juntos como mínimo (los sueltos son ruido)
AUTONOMO_MAPA_OLVIDO_S = 30.0            # lo que no se ve en este tiempo vuelve a «desconocido»
AUTONOMO_MAPA_MARGEN_ROBOT = 0.03        # m extra alrededor de las cápsulas del robot (se quita del mapa)
AUTONOMO_MAPA_DESCONOCIDO_OCUPADO = True # para esquivar: lo tapado (debajo/detrás de algo) cuenta como obstáculo
AUTONOMO_MAPA_VER_VISOR = True           # dibujar los obstáculos también en la ventana de MuJoCo
AUTONOMO_MAPA_MAX_DIBUJO = 6000          # máx. de vóxeles que se dibujan (el mapa completo no tiene límite)

# Herramientas con marcadores ArUco (paso 5, skill_shield/autonomo/herramientas.py). Diccionario 4x4.
# Por cada una: id del marcador, nombre, tipo ("herramienta" | "punto_uso" | "punto_dejada"),
# lado = lado del cuadrado NEGRO impreso (m), desplazamiento/giro_deg = dónde queda el centro de la
# herramienta respecto al marcador (en el marco del marcador: x derecha, y arriba, z hacia fuera),
# caja = semi-lados de la herramienta en su marco (para dibujarla y quitarla del mapa al cogerla).
# Marcadores para imprimir:  python -m skill_shield.autonomo.herramientas --generar
AUTONOMO_HERRAMIENTAS = [
    # ids desde el 20: del 0 al 16 son los del tablero de calibración (mismo diccionario 4x4)
    dict(id=20, nombre="Destornillador",  tipo="herramienta",  lado=0.093,
         desplazamiento=(0.0, 0.0, -0.015), giro_deg=(0, 0, 0), caja=(0.10, 0.016, 0.016),
         hoja=(1.2, 1.3, 2.4)),                   # la hoja cuelga de una pinza: llega más abajo
    dict(id=21, nombre="Punto de uso",    tipo="punto_uso",    lado=0.124),
    dict(id=22, nombre="Punto de dejada", tipo="punto_dejada", lado=0.124),
]
# Las hojas de los marcadores no son obstáculos (12-nov): se quita del mapa una caja del tamaño de la
# hoja alrededor de cada marcador visto; lo de debajo cuenta. Tamaño, desde el centro del marcador y en
# lados del marcador: (medio ancho, hasta arriba, hasta abajo). Cada marcador puede llevar la suya
# (clave «hoja» en AUTONOMO_HERRAMIENTAS).
AUTONOMO_MAPA_QUITAR_HOJAS = True
# Halo de las pelotas (12-nov): en el borde de una pelota el estéreo mezcla pelota y fondo y salen
# puntos sueltos alrededor y detrás de ella que la «envolvían» de cubos. Esos píxeles no van al mapa
# (lo de debajo de la pelota, la mesa, sí). Las pelotas tampoco cuentan como obstáculo.
AUTONOMO_MAPA_SIN_HALO_PELOTAS = True
AUTONOMO_MAPA_HOJA = (1.2, 1.3, 1.3)
AUTONOMO_HERRAMIENTAS_N_ESTABLE = 5      # imágenes seguidas antes de darla por buena
AUTONOMO_HERRAMIENTAS_OLVIDO_S  = 2.0    # sin verla este tiempo → deja de aparecer

# ── PELOTAS (11-nov): por color + profundidad, sin marcadores — skill_shield/autonomo/pelotas.py ──
# Cada pelota: rangos HSV de OpenCV (H 0-179, S y V 0-255; el rojo necesita dos, da la vuelta) y,
# si se sabe, su diámetro en m (descarta lo que tenga el mismo color pero otro tamaño; None = de 3 a
# 20 cm). Colores sacados de la foto del 11-nov: ajustadlos con
#     python -m skill_shield.autonomo.pelotas --calibrar      (clic en cada pelota)
AUTONOMO_PELOTAS = [
    dict(nombre="Pelota amarilla", colores=[((18, 110, 80), (28, 255, 255))], diametro=None,
         color_gemelo=(1.0, 0.85, 0.1)),
    dict(nombre="Pelota de tenis", colores=[((29, 90, 80), (38, 255, 255))], diametro=0.067,
         color_gemelo=(0.8, 0.95, 0.2)),
    dict(nombre="Pelota verde", colores=[((40, 80, 70), (60, 255, 255))], diametro=None,
         color_gemelo=(0.2, 0.8, 0.3)),
    dict(nombre="Pelota roja", colores=[((0, 120, 60), (7, 255, 255)), ((168, 120, 60), (179, 255, 255))],
         diametro=None, color_gemelo=(0.9, 0.1, 0.15)),
]
AUTONOMO_PELOTAS_N_ESTABLE = 5
AUTONOMO_PELOTAS_OLVIDO_S  = 2.0

# ═══════════════════════════════════════════════════════════════════════════
# TAREA: COGER → USAR → DEJAR (11-nov, paso 10, 1ª versión) — skill_shield/autonomo/tarea.py
# ═══════════════════════════════════════════════════════════════════════════
# Tarjeta «Tarea» de la web: Planificar → Ejecutar → el robot se mueve SOLO mientras se pisa el
# pedal de la web. Sin herramienta real: la mano finge el agarre delante del marcador.
AUTONOMO_TAREA            = True
AUTONOMO_FR5_SIMULACRO    = False   # False = la tarea mueve el FR5 REAL (con pedal) · True = solo el gemelo
AUTONOMO_FR5_VEL_MAX_DEG_S = 20.0   # tope de fr5_servo en este modo (°/s por joint; el FR5 llega a ~180)
AUTONOMO_TAREA_VEL_DEG_S  = 15.0    # velocidad del plan (°/s del joint más rápido) ≈ 8 % del máximo
AUTONOMO_TAREA_ACEL_DEG_S2 = 30.0
AUTONOMO_TAREA_DESVIO_MAX_DEG = 8.0  # si el robot real se separa más de esto de lo mandado → abortar
AUTONOMO_TAREA_MOVER_MANO = True    # False = la mano no se toca (solo se mueve el brazo)
# Distancias al marcador (m). La palma se queda DELANTE, sin tocar: al cerrar, los dedos avanzan.
AUTONOMO_TAREA_DIST_AGARRE = 0.10   # palma ↔ marcador al coger y al dejar
AUTONOMO_TAREA_DIST_PREVIA = 0.14   # desde donde se acerca en línea recta (y adonde se retira)
# Destornillador virtual, en el marco de mano_mount (dedos +Y, la palma mira a −X, meñique→índice +Z).
# Cogido como lo coge una persona: mango en el puño y varilla saliendo por el lado del índice (+Z),
# punta a ~9 cm de la mano. Al usarlo, la punta apunta al marcador del punto de uso y la mano gira
# alrededor de la varilla. (Con la varilla por el eje de la brida —PUNTA (0, 0.30, 0.0099), EJE
# (0, 1, 0)— el giro sería solo J6, pero con los marcadores cerca del robot la muñeca no llega.)
AUTONOMO_TAREA_PUNTA      = (-0.03, 0.15, 0.14)
AUTONOMO_TAREA_EJE        = (0.0, 0.0, 1.0)
AUTONOMO_TAREA_USO_PREVIA = 0.12    # punta ↔ marcador al llegar y al irse
AUTONOMO_TAREA_USO_CERCA  = 0.02    # punta ↔ marcador al «atornillar» (no llega a tocar)
AUTONOMO_TAREA_USO_LEJOS  = 0.06    # punta ↔ marcador al alejarse para deshacer el giro
AUTONOMO_TAREA_USO_GIRO_DEG = 30.0  # giro de J6 en cada «vuelta»
AUTONOMO_TAREA_USO_VECES  = 3
# No hace falta llegar de frente al marcador: si la muñeca no da (marcador cerca del robot o
# mirándolo), se prueba llegando de lado, de 20 en 20° hasta este máximo respecto a su normal.
AUTONOMO_TAREA_INCLINACION_MAX_DEG  = 60.0
AUTONOMO_TAREA_INCLINACION_PASO_DEG = 20.0
AUTONOMO_TAREA_DIRECCIONES = 8      # direcciones probadas en cada inclinación
AUTONOMO_TAREA_OPCIONES    = 3      # formas de llegar a cada sitio que se guardan (por si el camino entre ellos choca)
# Comprobaciones del plan
AUTONOMO_TAREA_USAR_MAPA  = True     # esquivar (comprobar) también el mapa de obstáculos de la cámara
AUTONOMO_TAREA_EXCLUIR_RADIO = 0.15  # m alrededor de cada MARCADOR que no cuentan como obstáculo (pelotas: su radio + 2 cm)
AUTONOMO_TAREA_MARGEN_MAPA = 0.03   # m de margen de la MANO con los obstáculos del mapa
AUTONOMO_TAREA_MARGEN_MAPA_BRAZO = 0.015  # m de margen del brazo (sus cápsulas ya son más gordas que él)
AUTONOMO_TAREA_GUARDAR_FALLOS = True  # si no hay plan, guardar pose + sitios + mapa en diagnosticos/
AUTONOMO_TAREA_PASO_DEG   = 2.0     # cada cuánto se comprueba una postura en los tramos articulares
AUTONOMO_TAREA_MARGEN_LIMITES_DEG = 2.0
# Posturas de la mano (índices de ctrl 0..16, los de LIMITES_CTRL; lo no indicado = 0)
AUTONOMO_TAREA_MANO_ABIERTA = {0: 0.0, 13: -0.0979, 14: 0.641, 15: 0.0, 16: 0.175}
AUTONOMO_TAREA_MANO_AGARRE  = {0: 0.0, 2: 1.25, 3: 1.10, 5: 1.25, 6: 1.10, 8: 1.25, 9: 1.10,
                               11: 1.25, 12: 1.10, 13: 0.35, 14: 0.45, 15: 0.70, 16: 0.90}
AUTONOMO_TAREA_T_ABRIR_S  = 1.0
AUTONOMO_TAREA_T_CERRAR_S = 1.5
# Rodear obstáculos (paso 8): si el camino directo entre dos sitios choca (y pasando por la postura
# de inicio también), se busca un rodeo en el espacio de joints (RRT-Connect + atajos).
AUTONOMO_TAREA_RODEAR       = True
AUTONOMO_TAREA_RODEO_S      = 6.0   # s máx. buscando el rodeo de un tramo
AUTONOMO_TAREA_RODEO_TOTAL_S = 25.0  # s máx. de rodeos en todo un plan
# ── Pelotas (11-nov): «coger, apretar un poco y dejar». La mano llega DESDE ARRIBA, baja hasta
# rodearla, cierra, aprieta un poco, afloja, comprueba con los táctiles (si van) y la deja apoyada
# sobre el marcador del punto de dejada. Lo que no cuenta como obstáculo: solo la pelota (+2 cm).
AUTONOMO_TAREA_PELOTA_HOLGURA_PALMA = 0.03  # m entre la pelota y el centro de la palma (sobre ella)
AUTONOMO_TAREA_PELOTA_SUBIDA = 0.12         # m que sube/baja en vertical para cogerla o dejarla
# Zona de agarre (12-nov): alrededor de la pelota (y de donde se deja) y por debajo de su parte de
# arriba, los vóxeles del mapa no cuentan (es por donde tiene que pasar la mano y donde la profundidad
# tiene más ruido); en su lugar la mano no puede bajar de la parte de abajo de la pelota.
AUTONOMO_TAREA_PELOTA_ZONA_APOYO = 0.20     # m (+ el radio), en horizontal: lo que ocupa la mano abierta
AUTONOMO_TAREA_MARGEN_APOYO = 0.005         # m por encima de la parte de abajo de la pelota
AUTONOMO_TAREA_LEJOS_HOMBRO = 1.0           # m: más lejos, el plan avisa (límite del alcance)
# Alcance del FR5 + OrcaHand medido con el modelo (12-nov), desde el hombro (eje del J2): brida 0,93 m,
# centro de la palma 1,14 m, puntas de los dedos 1,26 m (con todo estirado). Desde el plato de la base
# (15 cm por debajo del hombro) se llega a 1,29 m, pero solo estirando el brazo en la línea del J1.
# Para coger algo con la palma mirándolo, la IK encuentra posturas hasta ~1,1 m del hombro; cómodo
# (varias formas de llegar), < 1,0 m.
AUTONOMO_TAREA_ALCANCE_PALMA_M = 1.14
AUTONOMO_TAREA_LEJOS_CAMARA = 1.0           # m: más lejos, el plan avisa (profundidad con ruido)
AUTONOMO_TAREA_RUIDO_PROF_K = 0.02          # ruido de la profundidad ≈ k·d² (m): 1 cm a 0,7 m, 2,4 cm a 1,1 m
# Formas de cogerla (12-nov): se prueban todas las inclinaciones a la vez, desde arriba hasta de lado,
# y se queda la que menos mueve el brazo (con algo de preferencia por llegar derecho); primero
# rodeándola con la palma y, si no se puede, con los dedos (la palma más lejos: deja más sitio).
AUTONOMO_TAREA_PELOTA_INCLINACION_MAX_DEG = 90.0   # 90 = de lado (la palma en horizontal)
AUTONOMO_TAREA_PELOTA_PREFERIR_DERECHO = 0.3      # coste extra por radián de inclinación
AUTONOMO_TAREA_PELOTA_DEDOS_EXTRA = 0.035         # m más lejos la palma «con los dedos» (0 = no probarlo)
AUTONOMO_TAREA_PELOTA_APOYO_GRUESO = 0.06         # m bajo la pelota que aún son «la mesa con ruido»
# Afinar (12-nov): encima de la pelota se para AFINAR_S con el brazo quieto; la cámara corrige la
# calibración con el propio brazo y vuelve a medir la pelota; si se ha movido más de AFINAR_MIN_M
# se rehace la bajada (si es más de AFINAR_MAX_M, se para: la pelota se ha movido de verdad).
AUTONOMO_TAREA_PELOTA_AFINAR_S = 2.0
AUTONOMO_TAREA_AFINAR_MIN_M = 0.005
AUTONOMO_TAREA_AFINAR_MAX_M = 0.10
AUTONOMO_TAREA_MARGEN_PROPIO = 0.05          # m: cubos pegados al robot al planificar = el propio robot
AUTONOMO_TAREA_MARGEN_MAPA_MALLA = 0.02      # m entre las mallas del brazo y el mapa (el brazo va desplazado de sus cápsulas)
AUTONOMO_TAREA_PELOTA_TACTIL_N = 0.3        # N para que un dedo cuente como «la noto»
AUTONOMO_TAREA_PELOTA_TACTIL_DEDOS = 2      # dedos que la tienen que notar (si no, se para)
# Posturas (ctrl 0..16: 0 muñeca · 1-3 meñique abd/mcp/pip · 4-6 anular · 7-9 corazón · 10-12 índice
# · 13-16 pulgar cmc/abd/mcp/dip). Ajustadlas a vuestras pelotas: son un punto de partida.
AUTONOMO_TAREA_MANO_PELOTA_ABIERTA = {0: 0.0, 1: -0.20, 4: -0.10, 7: 0.0, 10: 0.15, 13: 0.0, 14: 0.9,
                                      15: 0.0, 16: 0.0}
AUTONOMO_TAREA_MANO_PELOTA_CERRADA = {0: 0.0, 1: -0.15, 2: 0.85, 3: 0.75, 4: -0.08, 5: 0.85, 6: 0.75,
                                      7: 0.0, 8: 0.85, 9: 0.75, 10: 0.10, 11: 0.85, 12: 0.75,
                                      13: 0.25, 14: 0.9, 15: 0.55, 16: 0.65}
AUTONOMO_TAREA_MANO_PELOTA_APRETAR = {0: 0.0, 1: -0.15, 2: 1.00, 3: 0.90, 4: -0.08, 5: 1.00, 6: 0.90,
                                      7: 0.0, 8: 1.00, 9: 0.90, 10: 0.10, 11: 1.00, 12: 0.90,
                                      13: 0.25, 14: 0.9, 15: 0.70, 16: 0.80}

# ═══════════════════════════════════════════════════════════════════════════
# CALIBRACIÓN CÁMARA DE CABEZA → ROBOT (10-nov, paso 4) — calibrar_camara_robot.py
# ═══════════════════════════════════════════════════════════════════════════
# Tablero ChArUco en la brida/mano: el de la calibración extrínseca (7x5, 4x4), medido el 10-nov:
# cuadro 20 mm, marcador 12,5 mm. Sus marcadores son los ids 0-16 del diccionario 4x4: por eso las
# herramientas usan ids a partir del 20, y el detector de herramientas ignora los del tablero.
# Si se cambia de tablero, poned aquí sus medidas REALES (con regla o pie de rey).
CALIB_ROBOT_TABLERO = dict(cuadros=(7, 5), lado=0.0375, marcador=0.0275, diccionario="DICT_4X4_50")
CALIB_ROBOT_ARCHIVO   = "camara_robot.json"     # resultado (T_mundo_cam, en el marco del MJCF)
CALIB_ROBOT_POSTURAS  = "calib_posturas.json"   # posturas enseñadas (joints): se reutilizan siempre
CALIB_ROBOT_MUESTRAS  = "calib_muestras.json"   # datos brutos de la última calibración (--recalcular)
CALIB_ROBOT_MIN_MUESTRAS = 15
CALIB_ROBOT_ERROR_MAX_MM = 5.0                  # criterio de la hoja de ruta
CALIB_ROBOT_VEL_MAX_DEG_S = 10.0                # tope del FR5 al ir solo a las posturas (con pedal)
CALIB_ROBOT_ESPERA_S  = 1.0                     # espera en cada postura antes de capturar
CALIB_ROBOT_VERIFICAR_N = 4                     # posturas que usa --verificar
# Calibración automática (--auto): el programa elige las posturas comprobando choques con el modelo
# de colisión de la teleoperación (soporte, suelo y TELEOP_FR5_OBSTACULOS: ¡que la mesa esté bien!)
CALIB_AUTO_POSTURAS   = 20                      # posturas de medida (además de las ~10 de explorar)
# (12-nov) Si en la calibración la OrcaHand fuera montada girada respecto al GEMELO, este ángulo
# (alrededor del eje de la brida) gira su modelo en las comprobaciones de choque de la calibración.
# Con la mano ya bien montada (como el gemelo), 0. Las posturas guardadas cuando la mano estaba al
# revés se adaptan una vez con:  python calibrar_camara_robot.py --girar-posturas 180
CALIB_MANO_GIRO_DEG = 90.0
# (13-nov) Durante la calibración la OrcaHand se conecta con par y se queda RECTA (muñeca a 0 y dedos
# estirados), para que el tablero pegado a ella no se mueva. Usa el puerto PUERTO_HARDWARE: cierra
# antes el modo autónomo y la teleoperación. Postura: {índice de ctrl 0..16: rad}; None = la mano
# abierta de la tarea (la misma que las comprobaciones de choque).
CALIB_MANO_RECTA = True
CALIB_MANO_POSTURA = None
CALIB_MANO_ESPERA_S = 3.0
CALIB_AUTO_RADIO_MANO = 0.13                    # m: la mano + el tablero (14x10 cm), para las colisiones
# Cuánto puede alejarse cada joint (J1…J6) de la postura de inicio. Los tres de la muñeca (J4, J5,
# J6) van contenidos (10-nov): son los que estrujan lo que sale de la mano y retuercen sus cables.
# En simulación la calibración sale igual de precisa que con (…, 45, 45, 70).
CALIB_AUTO_AMPLITUD_DEG = (30, 20, 25, 35, 30, 40)
CALIB_AUTO_RESET      = True                    # colisión → borrar error, disable, enable solo (luego, P otra vez)
# ── Huecos que se exigen al mover el robot solo (--auto, --repetir, --verificar), en metros ──
# Se miden con las piezas reales del gemelo (skill_shield/autonomo/choque_mano.py): la mano y la
# placa del tablero contra el hombro, el brazo, el antebrazo y el giro del J4 del robot, contra el
# # soporte (columna, placas, pie) y contra la cabeza con la cámara, las cajas de TELEOP_FR5_* y el suelo.
# REGLA PRINCIPAL: ningún hueco puede cerrarse más de CALIB_MANO_ACERCAR_MAX respecto al que había
# en la postura de inicio (la que deja el operario viendo que nada aprieta), salvo que siga siendo
# mayor que su hueco objetivo. Y nunca baja de su mínimo. Así, lo que sobresale de la mano y no
# está en el gemelo (conector, cables, cinta) no se estruja aunque no sepamos su forma.
CALIB_MANO_HOLGURA    = 0.03                    # mínimo absoluto mano ↔ robot (con la muñeca recta hay ~6 cm)
CALIB_MANO_SALIENTES  = 0.04                    # cuánto sobresale de la mano lo que NO está en el gemelo
#                                                 (objetivo mano ↔ robot = HOLGURA + SALIENTES). ¡Medidlo!
CALIB_MANO_ACERCAR_MAX = 0.01                   # cuánto puede cerrarse un hueco respecto a la postura de inicio
CALIB_TABLERO_HOLGURA = 0.04                    # hueco objetivo tablero ↔ robot (mínimo absoluto: 1,5 cm)
CALIB_TABLERO_MARGEN  = 0.025                   # placa/papel que sobresale alrededor del dibujo del tablero
CALIB_ENTORNO_HOLGURA = 0.05                    # hueco objetivo de mano y tablero con soporte, cabeza, cajas y suelo
#                                                 (mínimo absoluto: 3 cm)
CALIB_MARGEN_LIMITES_DEG = 1.0                  # las posturas se piden a ≥ 1° de lo que fr5_servo deja pasar
CALIB_SIN_AVANCE_S    = 5.0                     # pedal pisado y robot parado sin llegar → se deja esa postura
