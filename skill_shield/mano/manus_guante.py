"""
Guantes MANUS → dedos de la OrcaHand (y vibración de vuelta al guante).

Sustituye a MediaPipe Hands SOLO para los 16 actuadores de los dedos (ctrl[1..16] de
la mano). La flexión de muñeca (ctrl[0]), el giro del antebrazo y el brazo siguen
saliendo de las cámaras: los ángulos "ergonomics" de MANUS no incluyen la muñeca.

Uso desde Skill_Shield_Nivel_6.py (FUENTE_DEDOS = "manus" en config.py):

    guante = GuanteManus()          # una sola vez al arrancar el programa
    guante.iniciar()                # no bloquea: conecta en un hilo propio
    ...
    guante.aplicar = True/False     # ¿manda el guante sobre los dedos? (cámara activa)
    guante.escribir_ctrl(datos, dt) # en el hilo de física, ANTES de cada mj_step
    guante.haptica(modelo, datos)   # en el bucle principal: contactos → vibración
    ...
    guante.cerrar()

Datos del SDK (MANUS Core 3.2.1, Python):
  · callback de ergonomics: por guante, 40 ángulos en GRADOS (0-19 mano izquierda,
    20-39 derecha) en el orden de ErgonomicsDataType:
        pulgar CMCSpread, CMCStretch, MCPStretch, IPStretch,
        índice/corazón/anular/meñique: MCPSpread, MCPStretch, PIPStretch, DIPStretch
  · callback de landscape: qué guante es cada mano (side) y si tiene háptica.
  · lib.CoreSdk_VibrateFingersForGlove(glove_id, float[5])  (pulgar → meñique, 0..1)

═══════════════════════════════════════════════════════════════════════════════
v4 (15-oct) — IMITACIÓN 1:1: TU ÁNGULO = ÁNGULO DEL ROBOT
═══════════════════════════════════════════════════════════════════════════════
Antes (v2-v3) el robot NO copiaba tu mano: estiraba tu recorrido al suyo (con márgenes:
el último 20 % de tu cierre ya era su tope), pasaba a la fuerza por poses intermedias
(p. ej. MCP del pulgar a 80° al cruzarlo) y, cerca de una pinza grabada, se mezclaba con
una pose de contacto fija. Todo eso se ha QUITADO. Ahora cada actuador k es:

    q_k = q_ref_k + signo_k · ganancia_k · (θ_k − cero_k)          (θ en rad)

  · θ_k      tu ángulo (MANUS ergonomics, ya calibrado por MANUS para tu mano).
  · cero_k   tu ángulo con la mano ABIERTA (pose "abierta" de calibraciones/calibrar_manus.py): quita el
             desfase de cada sensor (p. ej. tu DIP del meñique marca 24° con el dedo recto).
  · q_ref_k  lo que hace el robot en esa pose: dedos RECTOS (mcp −4°…−8°, pip 0: así
             quedan rectos en el MJCF), dedos JUNTOS (abducciones de "juntos"), pulgar
             recto (mcp/ip 0). Ver ROBOT_REF.
  · signo_k  sale de tus poses de calibración (el sentido de cada canal de MANUS respecto
             al robot); si una pose no tiene recorrido, el habitual (SIGNO_DEFECTO).
  · ganancia 1.0 = 1:1 (config.MANUS_GANANCIA, por actuador). No hay márgenes ni topes
             forzados: solo se recorta a los límites del modelo (LIMITES_CTRL).

DOS CASOS ESPECIALES (porque la OrcaHand no es una mano humana):
  1) PIP de los 4 dedos. En la OrcaHand la falange media y la distal son UNA pieza
     (FingerTipAssembly) que gira en el PIP. Para que la PUNTA del dedo del robot vaya
     donde va la tuya, su ángulo es el de la cuerda PIP → yema de tu dedo:
         φ = atan2(sin p + ρ·sin(p+d), cos p + ρ·cos(p+d))
     con p = tu PIP, d = tu DIP (desde la mano abierta) y ρ = distal/media (≈ 0.8,
     config.MANUS_RATIO_DISTAL). Dedo recto → 0; puño (p 70°, d 65°) → ~98°.
  2) CMC del pulgar (t-cmc ← CMC_STRETCH, t-abd ← CMC_SPREAD). Son ejes distintos de los
     tuyos, así que su CERO no puede ser "pulgar relajado = reposo del robot" sin más: se
     ajusta (escala 1:1, solo el desfase) para que en tus poses del pulgar el robot quede
     lo más cerca posible de su pose equivalente: pulgar FUERA (L plana) → t-cmc −0.785,
     t-abd +0.96 · PERPENDICULAR → −0.16, −0.31 · CRUZADO → +0.576, −0.31 (ANCLAS_PULGAR).
     Es UN número por articulación (un desfase), NO una pose que se fuerce: el pulgar sigue
     moviéndose grado a grado contigo.

LO QUE NO SE PUEDE COPIAR 1:1 (y antes se "fingía" con las poses forzadas):
  · Las proporciones de la OrcaHand no son las tuyas: con tus ángulos exactos las yemas del
    robot pueden quedarse a unos mm de tocarse en una pinza, o el pulgar cruzado no
    apoyarse en la palma (tú lo haces con el CMC; el robot necesitaría doblar el MCP).
    Si quieres compensarlo, sube la ganancia de esa articulación en config.MANUS_GANANCIA
    (deja de ser 1:1 en ella, pero sigue siendo continuo: nada salta a una pose).

CALIBRACIÓN: calibraciones/calibrar_manus.py (9 poses, ~2 min) guarda "version": 4. Una calibración
v3 con "poses_crudas" se convierte sola al cargarla (python calibraciones/calibrar_manus.py --recalcular
para guardarla ya convertida). Sin calibración: ceros típicos (funciona, pero calibra).

═══════════════════════════════════════════════════════════════════════════════
v5 (15-oct, tras el vídeo) — MODO "cierre" (por defecto): TU PUÑO = PUÑO DEL ROBOT
═══════════════════════════════════════════════════════════════════════════════
Con el 1:1 puro los dedos no tocaban la palma. NO era un límite del código (el índice pedía
71° de nudillo con 100° disponibles): la OrcaHand necesita ~97° de MCP + ~103° de PIP para
apoyar la yema en la palma y tu puño son 71-84° de MCP. Y el pulgar: tú lo cruzas sobre la
palma con el CMC (tu MCP apenas se mueve, +1°), el robot necesita doblar su MCP ~80°.

config.MANUS_MAPEO = "cierre" (por defecto) | "1:1":
  · "cierre": cada flexión es una RECTA que pasa por dos puntos tuyos medidos al calibrar:
        tu mano abierta → robot recto      ·      tu puño → robot con la yema en la palma
    (pulgar: tu pulgar doblado al máximo → robot doblado al máximo). Entre medias,
    proporcional; si cierras más que al calibrar, sigue la misma recta hasta el tope
    mecánico. Sin márgenes, sin poses intermedias, sin pinzas: continuo y lineal.
    Pulgar:
      - t-cmc y t-abd: una transformación lineal de tus DOS ángulos del CMC (SPREAD,
        STRETCH) que pasa EXACTAMENTE por tus 3 poses: fuera (L plana), perpendicular y
        cruzado → las mismas poses del robot (ANCLAS_PULGAR). Así el pulgar del robot
        cruza y se apoya cuando tú lo cruzas, aunque sus ejes no sean los tuyos.
      - t-mcp: tu MCP (recto → doblado al máximo) + lo que CRUZAS el pulgar (tu CMC_STRETCH
        desde la mano abierta hacia el meñique): cruzado → MCP del robot a 80° (yema en la
        palma). Solo actúa al cruzar: con el pulgar abierto no se toca.
  · "1:1": el mapeo de v4 (tu ángulo = ángulo del robot).
  Las ABDUCCIONES de los 4 dedos van 1:1 en los dos modos.
  Lo del robot (extremos, anclas) sale SIEMPRE de config: si la calibración tiene
  "poses_crudas" el mapeo se rehace al cargar, así que cambiar MANUS_MAPEO,
  MANUS_RANGO_ROBOT o MANUS_ANCLAS_PULGAR no exige recalibrar.

═══════════════════════════════════════════════════════════════════════════════
v7 (15-oct) — HÁPTICA REAL: SENSORES DE LA ORCAHAND TOUCH FÍSICA → VIBRACIÓN DEL MANUS
═══════════════════════════════════════════════════════════════════════════════
config.MANUS_HAPTICA_FUENTE = "real" (por defecto) | "sim" | "auto":
  · "real": cada dedo del guante vibra según la fuerza que mide la YEMA del mismo dedo de
    la mano FÍSICA (sensores táctiles de la OrcaHand Touch). La simulación no interviene.
    Los sensores los lee skill_shield/hardware/puente_tactil.py en un proceso aparte (con el orca_core nuevo, ver
    su cabecera); GuanteManus lo lanza solo y recibe las fuerzas por UDP local
    (skill_shield/hardware/tactil_real.py). Sin sensores → el guante no vibra.
  · "sim": como en v6, contactos de la mano SIMULADA (clase SensorTactil).
  · "auto": real si llegan datos de los sensores; si no, la simulación.
Curva (por dedo): nada por debajo de MANUS_TACTIL_F_ON (0.3 N); de ahí a MANUS_TACTIL_F_MAX
(20 N) la vibración crece en LÍNEA RECTA desde MANUS_TACTIL_VIBRACION_MIN (lo mínimo que se
nota) hasta MANUS_VIBRACION_MAX; por encima de 20 N, al máximo. Filtrada (subida rápida,
bajada suave) y con histéresis en el umbral.
Cableado: ORCA_TACTIL_SENSOR_DE_DEDO dice qué sensor (nombre de orca_core) está en cada dedo
FÍSICO de tu mano; orca_core los nombra en otro orden que tu cableado (15-oct).
Por defecto solo vibra mientras el guante manda (aplicar = True); MANUS_HAPTICA_SIEMPRE =
True vibra siempre (p. ej. al reproducir un episodio en la mano física).
Nuevo cero de los sensores (la mano robótica sin tocar nada): guante.tactil.cero().
Para probar sin el programa principal: python diagnosticos/probar_haptica_tactil.py
"""
import json
import math
import os
import sys
import threading
import time

import numpy as np

import config as _cfg
from config import LIMITES_CTRL, N_ACT, OFFSET_MANO


def _p(nombre, defecto):
    return getattr(_cfg, nombre, defecto)


# ── Parámetros (config.py, bloque MANUS; si faltan, estos valores) ────────────
MANUS_MODO        = _p("MANUS_MODO", "local")          # "integrado" | "local" | "remoto"
MANUS_LADO        = _p("MANUS_LADO", "derecho")        # mano que se teleopera
MANUS_SDK_RUTA    = _p("MANUS_SDK_RUTA", os.path.join(
    os.path.dirname(os.path.abspath(_cfg.__file__)), "MANUS_Core_3.2.1_SDK_Linux", "Python"))
MANUS_CALIBRACION = _p("MANUS_CALIBRACION", os.path.join(
    os.path.dirname(os.path.abspath(_cfg.__file__)), "calibraciones", "datos", "manus_calibracion.json"))
MANUS_TIMEOUT_S   = float(_p("MANUS_TIMEOUT_S", 0.3))  # sin datos más tiempo → dedos quietos
MANUS_TAU_S       = float(_p("MANUS_TAU_S", 0.025))    # suavizado (constante de tiempo)
MANUS_HAPTICA     = bool(_p("MANUS_HAPTICA", True))
MANUS_HAPTICA_HZ  = float(_p("MANUS_HAPTICA_HZ", 30.0))      # envíos de vibración por segundo
MANUS_VIBRACION_MAX = float(_p("MANUS_VIBRACION_MAX", 1.0))
# ── Háptica v6 (15-oct): presión de OBJETOS sobre cada dedo → vibración de ese dedo ──
# (ver la clase SensorTactil). MANUS_FUERZA_MIN / MANUS_FUERZA_MAX de la v2 ya no se usan.
MANUS_HAPTICA_F_ON   = float(_p("MANUS_HAPTICA_F_ON", 0.3))     # N: por debajo no vibra (roces)
MANUS_HAPTICA_F_MAX  = _p("MANUS_HAPTICA_F_MAX", "auto")        # N a los que vibra al 100 %, o "auto":
MANUS_HAPTICA_F_MAX_FRAC = float(_p("MANUS_HAPTICA_F_MAX_FRAC", 0.5))  # "auto" = esta fracción de la
                                                                # fuerza MÁXIMA que puede hacer ese dedo
MANUS_VIBRACION_MIN  = float(_p("MANUS_VIBRACION_MIN", 0.25))   # al empezar a tocar (por debajo no se nota)
MANUS_HAPTICA_GAMMA  = float(_p("MANUS_HAPTICA_GAMMA", 0.6))    # <1: se distinguen mejor las presiones suaves
MANUS_HAPTICA_ZONA   = str(_p("MANUS_HAPTICA_ZONA", "dedo"))    # "dedo" (todas las falanges) | "yema"
MANUS_HAPTICA_PROPIA = _p("MANUS_HAPTICA_PROPIA", "pinza")     # contactos de la mano consigo misma:
                                                                # "pinza" = solo pulgar ↔ otro dedo (pinzas sin
                                                                # objeto) · False = ninguno · True = todos (palma...)
MANUS_HAPTICA_TAU_SUBIDA = float(_p("MANUS_HAPTICA_TAU_SUBIDA", 0.02))   # s: al apretar, casi inmediato
MANUS_HAPTICA_TAU_BAJADA = float(_p("MANUS_HAPTICA_TAU_BAJADA", 0.08))   # s: al soltar, sin parpadeos
MANUS_HAPTICA_MUESTREO_HZ = float(_p("MANUS_HAPTICA_MUESTREO_HZ", 100.0))  # lecturas de contactos/s
MANUS_HAPTICA_DEBUG  = bool(_p("MANUS_HAPTICA_DEBUG", False))   # imprime fuerza y vibración por dedo
MANUS_HAPTICA_DEBUG_CADA_S = float(_p("MANUS_HAPTICA_DEBUG_CADA_S", 0.5))
# ── Háptica v7: sensores FÍSICOS de la OrcaHand Touch (ver cabecera) ──
MANUS_HAPTICA_FUENTE = str(_p("MANUS_HAPTICA_FUENTE", "real")).lower()  # "real" | "sim" | "auto"
ORCA_TACTIL_ACTIVO   = bool(_p("ORCA_TACTIL_ACTIVO", True))      # lanzar el puente de los sensores
MANUS_TACTIL_MEDIDA  = str(_p("MANUS_TACTIL_MEDIDA", "normal")).lower()  # "normal" (fz) | "total" (|f|)
MANUS_TACTIL_F_ON    = float(_p("MANUS_TACTIL_F_ON", 0.3))       # N: aquí empieza a vibrar (el sensor da 0.1 N/paso)
MANUS_TACTIL_F_MAX   = _p("MANUS_TACTIL_F_MAX", 20.0)            # N al 100 %: número o {"pulgar": 20.0, ...}
MANUS_TACTIL_VIBRACION_MIN = float(_p("MANUS_TACTIL_VIBRACION_MIN", 0.15))  # vibración a F_ON: la mínima
                                                                # que notas (búscala con diagnosticos/probar_haptica_tactil.py --umbral)
MANUS_TACTIL_GAMMA   = float(_p("MANUS_TACTIL_GAMMA", 1.0))      # 1.0 = lineal entre F_ON y F_MAX
MANUS_HAPTICA_SIEMPRE = bool(_p("MANUS_HAPTICA_SIEMPRE", False))  # True: vibra aunque el guante no mande
DEDOS_TACTIL = ("thumb", "index", "middle", "ring", "pinky")     # nombres de los sensores en orca_core
DEDOS_FISICOS = ("pulgar", "indice", "corazon", "anular", "menique")  # orden de la vibración MANUS
# Qué SENSOR (nombre de orca_core) lleva cada dedo FÍSICO de la OrcaHand. Medido el 15-oct:
# apretando el anular vibraba el corazón, corazón→pulgar, índice→anular, pulgar→índice.
_SENSOR_DE_DEDO_DEFECTO = {"pulgar": "index", "indice": "ring", "corazon": "thumb",
                           "anular": "middle", "menique": "pinky"}


def _mapa_sensores():
    """ORCA_TACTIL_SENSOR_DE_DEDO → tupla de 5 nombres de sensor, en orden pulgar → meñique."""
    m = dict(_p("ORCA_TACTIL_SENSOR_DE_DEDO", _SENSOR_DE_DEDO_DEFECTO))
    sensores = tuple(str(m.get(d, s)) for d, s in zip(DEDOS_FISICOS, DEDOS_TACTIL))
    if sorted(sensores) != sorted(DEDOS_TACTIL):
        print(f"⚠️  [MANUS] ORCA_TACTIL_SENSOR_DE_DEDO no es una permutación de {DEDOS_TACTIL} "
              f"({sensores}): uso el orden de orca_core tal cual.")
        return DEDOS_TACTIL
    return sensores


SENSOR_DE_DEDO = _mapa_sensores()

# Ganancia por actuador (1.0 = 1:1). Claves: nombres de ACTUADORES (ver NOMBRES).
MANUS_GANANCIA    = dict(_p("MANUS_GANANCIA", {}))
# Longitud de tu falange distal / media (para el PIP del robot, ver cabecera)
MANUS_RATIO_DISTAL = float(_p("MANUS_RATIO_DISTAL", 0.8))
# v5: "cierre" (tu puño = puño del robot, lineal) | "1:1" (tu ángulo = ángulo del robot)
MANUS_MAPEO = str(_p("MANUS_MAPEO", "cierre"))

# ── Índices dentro de UNA mano (0..19), orden de ErgonomicsDataType ──────────
ERGO = {
    "THUMB_CMC_SPREAD": 0, "THUMB_CMC_STRETCH": 1, "THUMB_MCP": 2, "THUMB_IP": 3,
    "INDEX_SPREAD": 4,  "INDEX_MCP": 5,  "INDEX_PIP": 6,  "INDEX_DIP": 7,
    "MIDDLE_SPREAD": 8, "MIDDLE_MCP": 9, "MIDDLE_PIP": 10, "MIDDLE_DIP": 11,
    "RING_SPREAD": 12,  "RING_MCP": 13,  "RING_PIP": 14,  "RING_DIP": 15,
    "PINKY_SPREAD": 16, "PINKY_MCP": 17, "PINKY_PIP": 18, "PINKY_DIP": 19,
}

# ctrl de la mano → nombre legible (los mismos que GANANCIA_DEDOS / orca_hardware)
NOMBRES = {1: "menique_abd", 2: "menique_mcp", 3: "menique_pip",
           4: "anular_abd",  5: "anular_mcp",  6: "anular_pip",
           7: "corazon_abd", 8: "corazon_mcp", 9: "corazon_pip",
           10: "indice_abd", 11: "indice_mcp", 12: "indice_pip",
           13: "pulgar_cmc", 14: "pulgar_abd", 15: "pulgar_mcp", 16: "pulgar_ip"}

# Fuentes de MANUS de cada actuador. Con DOS fuentes (PIP, DIP) se usa la cuerda
# PIP → yema (ver cabecera); con una, el ángulo tal cual.
FUENTES = {
    1:  ["PINKY_SPREAD"],  2: ["PINKY_MCP"],  3: ["PINKY_PIP", "PINKY_DIP"],
    4:  ["RING_SPREAD"],   5: ["RING_MCP"],   6: ["RING_PIP", "RING_DIP"],
    7:  ["MIDDLE_SPREAD"], 8: ["MIDDLE_MCP"], 9: ["MIDDLE_PIP", "MIDDLE_DIP"],
    10: ["INDEX_SPREAD"], 11: ["INDEX_MCP"], 12: ["INDEX_PIP", "INDEX_DIP"],
    13: ["THUMB_CMC_STRETCH"],   # barrido del pulgar: hacia fuera ↔ hacia el meñique
    14: ["THUMB_CMC_SPREAD"],    # abducción palmar (pulgar fuera del plano de la palma)
    15: ["THUMB_MCP"],
    16: ["THUMB_IP"],
}

# ── Robot en tu pose de referencia ("abierta": dedos estirados y juntos) ─────
# Sentidos de la OrcaHand (comprobados con la cinemática del MJCF orcahand_right_body.xml):
#   · flexiones: + = cerrar.  · abd de los 4 dedos: + = la yema va HACIA EL PULGAR.
#   · t-abd: + = pulgar tumbado en el plano de la palma; − = perpendicular a ella.
#   · t-cmc: + = pulgar barre hacia la base del meñique; − = hacia fuera.
_ROBOT_REF_DEFECTO = {
    # dedos RECTOS (en el MJCF, a 0° quedan 3-8° doblados; rectos con estos mcp)
    2: -0.14, 5: -0.05, 8: -0.05, 11: -0.07,
    3: 0.0, 6: 0.0, 9: 0.0, 12: 0.0,
    # dedos JUNTOS (convergencia de retargeting.py; corazón recto)
    1: 0.42, 4: 0.225, 7: 0.0, 10: -0.27,
    # pulgar: MCP/IP rectos. CMC: solo sin calibración del pulgar (con ella, ANCLAS_PULGAR)
    15: 0.0, 16: 0.0, 13: -0.60, 14: 0.42,
}
ROBOT_REF = {int(k): float(v) for k, v in
             {**_ROBOT_REF_DEFECTO, **dict(_p("MANUS_ROBOT_REF", {}))}.items()}

# Sentido de cada canal de MANUS respecto al robot, si la calibración no lo puede medir.
# (Medido con tu guante el 13-oct: flexiones y abducciones van como el robot; los dos
#  canales del CMC del pulgar, al revés.)
SIGNO_DEFECTO = {**{k: 1.0 for k in range(1, 13)}, 13: -1.0, 14: -1.0, 15: 1.0, 16: 1.0}

# Robot en tus poses del pulgar: SOLO para ajustar el desfase del CMC (ver cabecera).
_ANCLAS_PULGAR_DEFECTO = {
    13: {"fuera": -0.785, "perp": -0.16, "cruz": 0.576},
    14: {"fuera": 0.96,   "perp": -0.31, "cruz": -0.31},
}
ANCLAS_PULGAR = {int(k): {str(p): float(r) for p, r in dict(v).items()}
                 for k, v in {**_ANCLAS_PULGAR_DEFECTO,
                              **dict(_p("MANUS_ANCLAS_PULGAR", {}))}.items()}

# v5 "cierre": robot cuando TÚ cierras del todo (puño; pulgar doblado al máximo). Por defecto
# el máximo de config.MANUS_RANGO_ROBOT (el mismo que usa orca_hardware para el tope real):
# mcp 1.70 + pip 1.80 = yema apoyada en la palma (medido con las mallas del MJCF).
_CIERRE_DEFECTO = {2: 1.70, 3: 1.80, 5: 1.70, 6: 1.80, 8: 1.70, 9: 1.80, 11: 1.70, 12: 1.80,
                   15: 1.745, 16: 1.867}
ROBOT_CIERRE = {k: float(max(dict(_p("MANUS_RANGO_ROBOT", {})).get(k, (v, v))))
                for k, v in _CIERRE_DEFECTO.items()}
# Pose de calibración que es "cerrado del todo" para cada uno
POSE_CIERRE = {**{k: "puno" for k in (2, 3, 5, 6, 8, 9, 11, 12)}, 15: "pulgar_flex", 16: "pulgar_flex"}
# MCP del pulgar del robot con TU pulgar cruzado sobre la palma (yema apoyada en la palma)
MCP_PULGAR_CRUZADO = float(_p("MANUS_MCP_PULGAR_CRUZADO", 1.40))
MIN_CIERRE_DEG = 15.0          # tu cierre medido menor que esto → esa articulación va 1:1
GANANCIA_MAX = 3.0             # ganancia calibrada fuera de [1/GANANCIA_MAX, GANANCIA_MAX] → aviso

# Calibración: de qué par de poses sale el SIGNO de cada actuador, y hacia dónde va el
# robot entre ellas (+1 / −1). "fuera"/"cruz" = las poses que elija poses_barrido_pulgar.
PAR_SIGNO = {
    **{k: ("abierta", "puno", 1.0) for k in (2, 3, 5, 6, 8, 9, 11, 12)},
    10: ("abierta", "separados", 1.0),          # índice se separa hacia el pulgar
    4:  ("abierta", "separados", -1.0),         # anular se separa hacia el meñique
    1:  ("abierta", "separados", -1.0),         # meñique, igual
    7:  ("corazon_anular", "corazon_indice", 1.0),
    13: ("fuera", "cruz", 1.0),                 # t-cmc sube al cruzar hacia el meñique
    14: ("fuera", "pulgar_perp", -1.0),         # t-abd baja al ponerlo perpendicular
    15: ("abierta", "pulgar_flex", 1.0),
    16: ("abierta", "pulgar_flex", 1.0),
}
MIN_RECORRIDO_SIGNO_DEG = 4.0      # por debajo, el signo medido no es fiable → SIGNO_DEFECTO

# Barrido del pulgar: poses candidatas para "fuera" (L plana) y "dentro" (cruzado), por
# orden de preferencia (13-oct: "pulgar_plano" salió igual que "pulgar_cruz").
POSES_FUERA = ("pulgar_plano", "separados", "abierta")
POSES_DENTRO = ("pulgar_cruz", "pinza_menique", "pinza_anular_menique")
MIN_BARRIDO_PULGAR_DEG = 15.0

# Ceros típicos (grados) si no hay calibración: tu mano abierta ≈ 0, salvo el CMC
_CERO_DEFECTO = {"THUMB_CMC_SPREAD": 10.0, "THUMB_CMC_STRETCH": 38.0}

# Yemas de la OrcaHand (mismos bodies que retargeting.py), en el orden del SDK:
# pulgar, índice, corazón, anular, meñique
_BODIES_YEMA = ("right_T-DP_b7429e50", "right_I-FingerTipAssembly_ec49c16c",
                "right_M-FingerTipAssembly_34afb748", "right_M-FingerTipAssembly_424a8e75",
                "right_P-FingerTipAssembly_cd219176")


def _importar_sdk():
    """manus_sdk instalado con pip o, si no, desde la carpeta del SDK (MANUS_SDK_RUTA)."""
    try:
        import manus_sdk  # noqa: F401
    except ImportError:
        if MANUS_SDK_RUTA and os.path.isdir(MANUS_SDK_RUTA) and MANUS_SDK_RUTA not in sys.path:
            sys.path.insert(0, MANUS_SDK_RUTA)
    from manus_sdk import lib, ffi, ManusSDK
    from manus_sdk.generated._enums import SDKReturnCode, AxisView, AxisPolarity, Side
    return lib, ffi, ManusSDK, SDKReturnCode, AxisView, AxisPolarity, Side


# ══════════════════════════════════════════════════════════════════════════════
# MAPEO 1:1
# ══════════════════════════════════════════════════════════════════════════════
def ganancia(k):
    return float(MANUS_GANANCIA.get(NOMBRES.get(k, ""), 1.0))


def angulo_relativo(ang_deg, fuentes, cero):
    """Tu ángulo para un actuador, en GRADOS, desde la pose de referencia (cero).
    Una fuente: θ − cero. Dos (PIP, DIP): ángulo de la cuerda PIP → yema (ver cabecera)."""
    if len(fuentes) == 1:
        return float(ang_deg[ERGO[fuentes[0]]]) - float(cero[0])
    p = math.radians(float(ang_deg[ERGO[fuentes[0]]]) - float(cero[0]))
    d = math.radians(float(ang_deg[ERGO[fuentes[1]]]) - float(cero[1]))
    r = MANUS_RATIO_DISTAL
    return math.degrees(math.atan2(math.sin(p) + r * math.sin(p + d),
                                   math.cos(p) + r * math.cos(p + d)))


def mapeo_por_defecto():
    """{ctrl: {"fuentes", "cero" (grados), "signo", "q_ref" (rad)}} sin calibrar."""
    return {k: {"fuentes": list(f), "cero": [_CERO_DEFECTO.get(x, 0.0) for x in f],
                "signo": SIGNO_DEFECTO[k], "q_ref": ROBOT_REF[k]}
            for k, f in FUENTES.items()}


def poses_barrido_pulgar(medidas):
    """(pose_fuera, pose_dentro, nota) del barrido del pulgar (CMC_STRETCH), o
    (None, None, nota) si ningún par tiene recorrido. Prefiere pulgar_plano → pulgar_cruz;
    si una de las dos está mal hecha, usa separados/abierta o la pinza del meñique."""
    i_st = ERGO["THUMB_CMC_STRETCH"]
    for dentro in POSES_DENTRO:
        if dentro not in medidas:
            continue
        for fuera in POSES_FUERA:
            if fuera not in medidas:
                continue
            d = abs(medidas[dentro][i_st] - medidas[fuera][i_st])
            if d >= MIN_BARRIDO_PULGAR_DEG:
                nota = ""
                if (fuera, dentro) != ("pulgar_plano", "pulgar_cruz"):
                    nota = (f"'pulgar_plano' → 'pulgar_cruz' no tiene recorrido: uso '{fuera}' → "
                            f"'{dentro}' ({d:.0f}° de CMC_STRETCH)")
                return fuera, dentro, nota
    return None, None, (f"ningún par de poses del pulgar cambia ≥ {MIN_BARRIDO_PULGAR_DEG:.0f}° "
                        f"de CMC_STRETCH")


def construir_mapeo(medidas, modo=None):
    """medidas: {pose: array(20) grados} → (mapeo {ctrl: {...}}, informe {ctrl: texto}).
    modo: "cierre" | "1:1" (None = config.MANUS_MAPEO). Lo usan calibraciones/calibrar_manus.py y
    cargar_calibracion (que lo rehace con las poses crudas guardadas)."""
    modo = modo or MANUS_MAPEO
    medidas = {p: np.asarray(a, dtype=float) for p, a in medidas.items()}
    mapeo, informe = mapeo_por_defecto(), {}
    if "abierta" not in medidas:
        return mapeo, {k: "⚠️  sin la pose 'abierta': ceros típicos" for k in FUENTES}
    fuera, dentro, nota_barrido = poses_barrido_pulgar(medidas)
    alias = {"fuera": fuera, "cruz": dentro}
    for k, fuentes in FUENTES.items():
        cero = [float(medidas["abierta"][ERGO[f]]) for f in fuentes]
        notas = []
        # ── signo: de tus poses; si no hay recorrido, el habitual ──
        pa, pb, dir_robot = PAR_SIGNO[k]
        pa, pb = alias.get(pa, pa), alias.get(pb, pb)
        signo = SIGNO_DEFECTO[k]
        if pa in medidas and pb in medidas:
            d = (angulo_relativo(medidas[pb], fuentes, cero)
                 - angulo_relativo(medidas[pa], fuentes, cero))
            if abs(d) >= MIN_RECORRIDO_SIGNO_DEG:
                medido = float(np.sign(d) * dir_robot)
                if medido != signo:
                    notas.append(f"⚠️  sentido AL REVÉS que lo habitual ('{pa}' → '{pb}': {d:+.0f}°): "
                                 f"uso tu medida")
                signo = medido
            else:
                notas.append(f"'{pa}' → '{pb}' solo cambia {d:+.1f}°: sentido habitual")
        else:
            notas.append(f"faltan '{pa}'/'{pb}': sentido habitual")
        # ── referencia del robot: la de la mano abierta; CMC del pulgar, por anclas ──
        q_ref = ROBOT_REF[k]
        if k in ANCLAS_PULGAR:
            g = ganancia(k)
            usadas = []
            for clave, r in ANCLAS_PULGAR[k].items():
                pose = alias.get(clave, "pulgar_perp" if clave == "perp" else clave)
                if pose in medidas:
                    th = math.radians(angulo_relativo(medidas[pose], fuentes, cero))
                    usadas.append((pose, r - signo * g * th))
            if usadas:
                q_ref = float(np.mean([c for _, c in usadas]))
                disp = max(abs(c - q_ref) for _, c in usadas)
                notas.append(f"reposo ajustado con {', '.join(p for p, _ in usadas)} "
                             f"(error máx. de las anclas {math.degrees(disp):.0f}° con ganancia {g:.2f})")
                # Diagnóstico (NO se aplica): qué ganancia encajaría mejor tus poses del pulgar.
                # Si es muy distinta de 1, el guante mide ese eje con otra escala que el robot.
                th = np.array([math.radians(angulo_relativo(medidas[p], fuentes, cero))
                               for p, _ in usadas])
                rr = np.array([ANCLAS_PULGAR[k][c] for c in ANCLAS_PULGAR[k]
                               if alias.get(c, "pulgar_perp" if c == "perp" else c) in medidas])
                if len(th) >= 2 and np.ptp(th) > math.radians(MIN_RECORRIDO_SIGNO_DEG):
                    pend = float(np.polyfit(th, rr, 1)[0])
                    g_fit = pend / signo
                    res = rr - np.polyval(np.polyfit(th, rr, 1), th)
                    if g_fit > 0 and abs(g_fit - g) > 0.25:
                        notas.append(f"ℹ️  el guante mide este eje con otra escala que el robot: con "
                                     f"ganancia {g_fit:.1f} tus poses del pulgar encajarían a "
                                     f"±{math.degrees(float(np.abs(res).max())):.0f}° "
                                     f"(MANUS_GANANCIA['{NOMBRES[k]}'] = {g_fit:.1f} y --recalcular)")
            else:
                notas.append("⚠️  sin poses del pulgar: reposo típico")
            if k == 13 and nota_barrido:
                notas.append(nota_barrido)
        mapeo[k] = {"fuentes": list(fuentes), "cero": cero, "signo": signo, "q_ref": float(q_ref)}
        informe[k] = (f"{' ⟶ '.join(fuentes) if len(fuentes) > 1 else fuentes[0]}: "
                      f"cero {', '.join(f'{c:+.1f}°' for c in cero)} · sentido "
                      f"{'+' if signo > 0 else '−'} · mano abierta → robot {q_ref:+.2f} rad"
                      + "".join(f"\n{'':20s}· {n}" for n in notas))
    if modo == "cierre":
        _aplicar_cierre(mapeo, informe, medidas, fuera, dentro)
    return mapeo, informe


# ══════════════════════════════════════════════════════════════════════════════
# v5 — MODO "cierre": tu puño = puño del robot (rectas por dos poses tuyas)
# ══════════════════════════════════════════════════════════════════════════════
def _delta(medidas, pose, fuente):
    """Tu ángulo de UNA fuente en esa pose, desde la mano abierta (grados)."""
    return float(medidas[pose][ERGO[fuente]] - medidas["abierta"][ERGO[fuente]])


def _aplicar_cierre(mapeo, informe, medidas, fuera, dentro):
    """Convierte las flexiones y el pulgar del mapeo 1:1 al modo "cierre" (ver cabecera).
    Lo que no se pueda calibrar (pose ausente o sin recorrido) se queda en 1:1 y se avisa."""
    nota = lambda k, txt: informe.__setitem__(k, informe[k] + f"\n{'':20s}· {txt}")

    # ── 1) Flexiones: recta abierta → recto, puño (o pulgar doblado) → cerrado ──
    for k, pose in POSE_CIERRE.items():
        v = mapeo[k]
        if pose not in medidas:
            nota(k, f"CIERRE: falta la pose '{pose}' → 1:1")
            continue
        th = v["signo"] * angulo_relativo(medidas[pose], v["fuentes"], v["cero"])
        if th < MIN_CIERRE_DEG:
            nota(k, f"CIERRE: en '{pose}' solo cierras {th:.0f}° → 1:1")
            continue
        g = (ROBOT_CIERRE[k] - v["q_ref"]) / math.radians(th)
        if not (1.0 / GANANCIA_MAX <= g <= GANANCIA_MAX):
            nota(k, f"⚠️  CIERRE: ganancia {g:.2f} fuera de lo razonable (¿pose '{pose}' mal hecha?): "
                    f"la limito")
            g = min(max(g, 1.0 / GANANCIA_MAX), GANANCIA_MAX)
        v["ganancia"] = float(g)
        nota(k, f"CIERRE: tu '{pose}' ({th:.0f}°) → robot {math.degrees(ROBOT_CIERRE[k]):.0f}° "
                f"(×{g:.2f} sobre el 1:1)")

    # ── 2) CMC del pulgar: lineal en (SPREAD, STRETCH), exacta por fuera/perp/cruz ──
    cero_sp = float(medidas["abierta"][ERGO["THUMB_CMC_SPREAD"]])
    cero_st = float(medidas["abierta"][ERGO["THUMB_CMC_STRETCH"]])
    poses = {"fuera": fuera, "perp": "pulgar_perp", "cruz": dentro}
    cmc_ok = False
    if all(p is not None and p in medidas for p in poses.values()):
        X = np.array([[1.0, _delta(medidas, poses[c], "THUMB_CMC_SPREAD"),
                       _delta(medidas, poses[c], "THUMB_CMC_STRETCH")] for c in ("fuera", "perp", "cruz")])
        area = abs(np.linalg.det(X[1:, 1:] - X[0, 1:])) / 2.0      # triángulo de tus 3 poses (grados²)
        if area >= 150.0:
            cmc_ok = True
            for k in (13, 14):
                r = np.array([ANCLAS_PULGAR[k][c] for c in ("fuera", "perp", "cruz")])
                q0, a_sp, a_st = np.linalg.solve(X, r)
                mapeo[k] = {"tipo": "afin", "q_ref": float(q0),
                            "fuentes": ["THUMB_CMC_SPREAD", "THUMB_CMC_STRETCH"],
                            "terminos": [{"partes": [["THUMB_CMC_SPREAD", cero_sp, float(a_sp)],
                                                     ["THUMB_CMC_STRETCH", cero_st, float(a_st)]],
                                          "coef": 1.0}]}
                informe[k] = (f"THUMB_CMC_SPREAD + THUMB_CMC_STRETCH (lineal): exacta en "
                              f"'{poses['fuera']}', 'pulgar_perp' y '{poses['cruz']}' → mano abierta = "
                              f"robot {math.degrees(q0):+.0f}°")
        else:
            for k in (13, 14):
                nota(k, f"CIERRE: tus 3 poses del pulgar casi no se distinguen ({area:.0f} grados²) → 1:1")
    else:
        for k in (13, 14):
            nota(k, "CIERRE: faltan poses del pulgar → 1:1")

    # ── 3) MCP del pulgar: tu MCP + cuánto pasa el pulgar del robot de la perpendicular
    #       hacia el meñique (solo al CRUZAR: con el pulgar abierto, perpendicular o en el
    #       puño, por fuera de los dedos, no influye) ──
    v = mapeo[15]
    if cmc_ok and "pulgar_flex" in medidas and v.get("ganancia"):
        t13 = mapeo[13]["terminos"][0]["partes"]
        q13 = lambda pose: mapeo[13]["q_ref"] + sum(c * _delta(medidas, pose, f) for f, _, c in t13)
        umbral = q13("pulgar_perp") - mapeo[13]["q_ref"]          # (sobre el reposo, rad)
        x_f = max(0.0, q13("pulgar_flex") - mapeo[13]["q_ref"] - umbral)
        x_c = max(0.0, q13(dentro) - mapeo[13]["q_ref"] - umbral)
        v1f = v["signo"] * _delta(medidas, "pulgar_flex", "THUMB_MCP")
        v1c = v["signo"] * _delta(medidas, dentro, "THUMB_MCP")
        A = np.array([[v1f, x_f], [v1c, x_c]])
        b = np.array([ROBOT_CIERRE[15] - v["q_ref"], MCP_PULGAR_CRUZADO - v["q_ref"]])
        if x_c > math.radians(MIN_CIERRE_DEG) and abs(np.linalg.det(A)) > 1.0:
            a1, a2 = np.linalg.solve(A, b)
            if a1 > 0 and a2 > 0:
                mapeo[15] = {"tipo": "afin", "q_ref": v["q_ref"], "fuentes": ["THUMB_MCP"] + [f for f, _, _ in t13],
                             "terminos": [
                                 {"partes": [["THUMB_MCP", v["cero"][0], float(a1 * v["signo"])]], "coef": 1.0},
                                 {"partes": [[f, c0, c] for f, c0, c in t13], "coef": float(a2),
                                  "bisagra": 1.0, "umbral": float(umbral)}]}
                nota(15, f"CIERRE: + al CRUZAR el pulgar: cuando el pulgar del robot pasa de la "
                         f"perpendicular hacia el meñique, su MCP se dobla hasta "
                         f"{math.degrees(MCP_PULGAR_CRUZADO):.0f}° en '{dentro}' (yema en la palma)")
            else:
                nota(15, "CIERRE: el término de cruce no sale coherente → solo tu MCP")
        else:
            nota(15, "CIERRE: sin recorrido de cruce del pulgar → solo tu MCP")


def _mapeo_desde_v4(d):
    mapeo = mapeo_por_defecto()
    for k, v in d.get("mapeo", {}).items():
        k = int(k)
        if k in FUENTES:
            mapeo[k] = dict(v)
    return mapeo


def cargar_calibracion(ruta=MANUS_CALIBRACION, verbose=True):
    """Devuelve el mapeo {ctrl: {...}} en el modo de config.MANUS_MAPEO. Con "poses_crudas"
    (calibraciones v3, v4 o v5) se REHACE al cargar: lo del robot sale siempre de config.
    Sin ellas, se usa el mapeo guardado (v4)."""
    mapeo = mapeo_por_defecto()
    if ruta and os.path.exists(ruta):
        try:
            with open(ruta, encoding="utf-8") as fh:
                d = json.load(fh)
            version = int(d.get("version", 1))
            if d.get("poses_crudas"):
                mapeo, _ = construir_mapeo(d["poses_crudas"])
                if verbose:
                    print(f"🧤 [MANUS] Calibración cargada: {os.path.basename(ruta)} "
                          f"({d.get('fecha', '?')}) · modo '{MANUS_MAPEO}'"
                          + ("  (es v3: sus pinzas y poses forzadas se ignoran)" if version < 4 else ""))
            elif version >= 4:
                mapeo = _mapeo_desde_v4(d)
                if verbose:
                    print(f"🧤 [MANUS] Calibración cargada: {os.path.basename(ruta)} "
                          f"({d.get('fecha', '?')}) · mapeo guardado (sin poses crudas)")
            elif verbose:
                print(f"⚠️  [MANUS] {os.path.basename(ruta)} es antigua y no tiene poses crudas: "
                      f"ceros típicos. Ejecuta  python calibraciones/calibrar_manus.py")
        except Exception as e:
            print(f"⚠️  [MANUS] No se pudo leer {ruta} ({e}): uso ceros típicos.")
    elif verbose:
        print("⚠️  [MANUS] Sin calibración (manus_calibracion.json): ceros típicos. "
              "Ejecuta  python calibraciones/calibrar_manus.py  (2 min).")
    return mapeo


def angulos_a_ctrl(ang_deg, mapeo):
    """20 ángulos de UNA mano (GRADOS) → array N_ACT de ctrl (rad; ctrl[0] = NaN: la muñeca
    no la da el guante). Recortado a LIMITES_CTRL (los topes mecánicos del modelo).
      · "lineal": q_ref + signo · ganancia_calibrada · MANUS_GANANCIA · (θ − cero)
      · "afin":   q_ref + MANUS_GANANCIA · Σ coef · x,   x = Σ peso·(θ − cero) de sus partes
                  (o max(0, x − umbral) si el término es de "bisagra": solo actúa a partir
                  de ese punto, p. ej. el MCP del pulgar al cruzarlo)"""
    ang_deg = np.asarray(ang_deg, dtype=float)
    out = np.full(N_ACT, np.nan)
    for k, v in mapeo.items():
        if v.get("tipo") == "afin":
            s = 0.0
            for t in v["terminos"]:
                x = sum(float(p) * (float(ang_deg[ERGO[f]]) - float(c0)) for f, c0, p in t["partes"])
                if t.get("bisagra"):
                    x = max(0.0, x - float(t.get("umbral", 0.0)))
                s += float(t["coef"]) * x
            q = v["q_ref"] + ganancia(k) * s
        else:
            th = math.radians(angulo_relativo(ang_deg, v["fuentes"], v["cero"]))
            q = v["q_ref"] + v["signo"] * float(v.get("ganancia", 1.0)) * ganancia(k) * th
        lo, hi = LIMITES_CTRL[k]
        out[k] = min(max(q, lo), hi)
    return out


class GuanteManus:
    """Conexión con MANUS (hilo propio), últimos ángulos de la mano y vibración."""

    def __init__(self, modo=MANUS_MODO, lado=MANUS_LADO, tactil=True, fuente=None):
        """tactil=False: no lanza el puente de los sensores físicos (p. ej. calibraciones/calibrar_manus.py).
        fuente: "real" | "sim" | "auto" (None = config.MANUS_HAPTICA_FUENTE)."""
        self.modo = modo
        self.lado = lado
        self.aplicar = False             # el bucle principal decide si el guante manda
        self.conectado = False
        self.mapeo = cargar_calibracion()
        if tuple(LIMITES_CTRL[7]) == (0.0, 0.0) or tuple(LIMITES_CTRL[11])[0] >= 0.0:
            print("⚠️  [MANUS] config.py aún tiene los límites antiguos (corazón fijo, dedos sin "
                  "estirar del todo): pega el bloque de config_rango_completo.py al FINAL de config.py.")
        self.pinza = None                # v4: ya no hay pinzas forzadas (se deja por el HUD)
        self.pinza_peso = 0.0
        self._lock = threading.Lock()
        self._ang = None                 # 20 ángulos (grados) de nuestra mano
        self._t_ang = 0.0
        self._n_paquetes, self._t_hz, self.hz = 0, time.perf_counter(), 0.0
        self._guantes = {}               # glove_id → (side, isHaptics)
        self._id_guante = None           # glove_id de nuestra mano (para vibrar)
        self._ctrl_suave = None          # comando que se escribe en el hilo de física
        # háptica v6: el hilo de física MIDE (escribir_ctrl), un hilo propio ENVÍA (_bucle_haptica)
        self._modelo = None              # lo da haptica() desde el bucle principal
        self._tacto = None               # SensorTactil (se crea con el modelo)
        self._t_muestra = 0.0
        self.tacto_fuerza = np.zeros(5)  # N filtrados por dedo (pulgar → meñique) — HUD / debug
        self.tacto_vibracion = np.zeros(5)
        self._activos = np.zeros(5, dtype=bool)
        self._vibrando = False
        self._hilo_haptica = None
        # háptica v7: sensores FÍSICOS (TactilReal) y de qué fuente sale la vibración ahora
        fuente = str(fuente or MANUS_HAPTICA_FUENTE).lower()
        self.fuente = fuente if fuente in ("real", "sim", "auto") else "real"
        self._usar_tactil = bool(tactil) and ORCA_TACTIL_ACTIVO and self.fuente in ("real", "auto")
        self.tactil = None               # TactilReal (se crea en iniciar)
        self.fuente_activa = "-"         # "real" | "sim" | "-" (HUD)
        self.haptica_siempre = MANUS_HAPTICA_SIEMPRE
        self._f_max_real = _f_max_tactil()
        self._activos_real = np.zeros(5, dtype=bool)
        self._hilo = None
        self._activo = False
        self._sdk = None
        self.error = ""

    # ── ciclo de vida ────────────────────────────────────────────────────────
    def iniciar(self):
        """Inicializa el SDK y conecta en segundo plano. Devuelve False si el SDK no
        se puede cargar (el programa sigue con los dedos de la cámara)."""
        try:
            (self.lib, self.ffi, ManusSDK, self.RC,
             AxisView, AxisPolarity, Side) = _importar_sdk()
        except Exception as e:
            self.error = f"SDK de MANUS no disponible: {e}"
            print(f"❌ [MANUS] {self.error}")
            return False
        self._Side = Side
        lib, ffi = self.lib, self.ffi
        r = lib.CoreSdk_InitializeIntegrated() if self.modo == "integrado" \
            else lib.CoreSdk_InitializeCore()
        if r != self.RC.Success:
            self.error = f"no se pudo inicializar el SDK ({self.RC(r).name})"
            print(f"❌ [MANUS] {self.error}")
            return False
        cs = ffi.new("CoordinateSystemVUH*")
        cs.view, cs.up = AxisView.XFromViewer, AxisPolarity.PositiveZ
        cs.handedness, cs.unitScale = Side.Right, 1.0
        lib.CoreSdk_InitializeCoordinateSystemWithVUH(cs[0], True)
        self._sdk = ManusSDK()
        self._sdk.register_landscape_callback(self._cb_landscape)
        self._sdk.register_ergonomics_callback(self._cb_ergonomics)
        self._activo = True
        self._hilo = threading.Thread(target=self._conectar, name="MANUS-conexion", daemon=True)
        self._hilo.start()
        if MANUS_HAPTICA:
            if self._usar_tactil:
                self._iniciar_tactil()
            self._hilo_haptica = threading.Thread(target=self._bucle_haptica, name="MANUS-haptica",
                                                  daemon=True)
            self._hilo_haptica.start()
            print(f"🫳 [MANUS] Háptica: fuente '{self.fuente}'"
                  + (f" · sensores físicos {MANUS_TACTIL_F_ON:.1f} N → "
                     + "/".join(f"{f:.0f}" for f in self._f_max_real)
                     + f" N  (vibración {100 * MANUS_TACTIL_VIBRACION_MIN:.0f} % → "
                     f"{100 * MANUS_VIBRACION_MAX:.0f} %, gamma {MANUS_TACTIL_GAMMA:g})"
                     + "  · sensores: " + " ".join(f"{d[:3]}←{s}" for d, s in
                                                   zip(DEDOS_FISICOS, SENSOR_DE_DEDO))
                     if self.fuente != "sim" else ""))
        print(f"🧤 [MANUS] SDK listo (modo {self.modo}); conectando en segundo plano...")
        return True

    def _conectar(self):
        lib, ffi = self.lib, self.ffi
        loopback = self.modo == "local"
        espera = 1 if loopback else 3
        avisado = False
        while self._activo and not self.conectado:
            try:
                if lib.CoreSdk_LookForHosts(espera, loopback) == self.RC.Success:
                    n = ffi.new("uint32_t*")
                    if lib.CoreSdk_GetNumberOfAvailableHostsFound(n) == self.RC.Success and n[0] > 0:
                        hosts = ffi.new("ManusHost[]", n[0])
                        if lib.CoreSdk_GetAvailableHostsFound(hosts, n[0]) == self.RC.Success \
                                and lib.CoreSdk_ConnectToHost(hosts[0]) == self.RC.Success:
                            self.conectado = True
                            print("✅ [MANUS] Conectado: esperando datos del guante...")
                            return
            except Exception as e:
                self.error = str(e)
            if not avisado:
                print("⏳ [MANUS] Sin conexión todavía"
                      + (" (¿está abierto el MANUS Core Dashboard?)" if self.modo != "integrado"
                         else " (¿dongle enchufado y guante encendido?)") + ". Reintentando...")
                avisado = True
            time.sleep(1.0)

    def _iniciar_tactil(self):
        try:
            try:
                from skill_shield.hardware.tactil_real import TactilReal
            except ImportError:
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "hardware"))
                from tactil_real import TactilReal
            self.tactil = TactilReal()
            if not self.tactil.iniciar():
                self.tactil.cerrar()
                self.tactil = None
        except Exception as e:
            print(f"⚠️  [MANUS] Sin sensores táctiles físicos: {e}")
            self.tactil = None

    def cerrar(self):
        self._activo = False
        if self._hilo_haptica is not None:
            self._hilo_haptica.join(timeout=0.5)      # apaga la vibración al salir
            self._hilo_haptica = None
        if self.tactil is not None:
            self.tactil.cerrar()
            self.tactil = None
        if self._sdk is None:
            return
        try:
            self.vibrar([0.0] * 5)
            self.lib.CoreSdk_ShutDown()
        except Exception:
            pass
        self._sdk = None
        print("🧤 [MANUS] SDK cerrado.")

    # ── callbacks del SDK (hilo del SDK: copiar y salir, nada pesado) ─────────
    def _cb_landscape(self, landscape):
        try:
            g = landscape.gloveDevices
            guantes = {int(g.gloves[i].id): (int(g.gloves[i].side), bool(g.gloves[i].isHaptics))
                       for i in range(g.gloveCount)}
            lado = self._Side.Right if self.lado == "derecho" else self._Side.Left
            with self._lock:
                self._guantes = guantes
                self._id_guante = next((gid for gid, (s, _) in guantes.items() if s == lado), None)
        except Exception as e:
            self.error = f"landscape: {e}"

    def _cb_ergonomics(self, stream):
        try:
            base = 20 if self.lado == "derecho" else 0
            elegido = None
            with self._lock:
                id_nuestro = self._id_guante
            for i in range(stream.dataCount):
                d = stream.data[i]
                vals = [float(d.data[base + k]) for k in range(20)]
                if id_nuestro is not None and int(d.id) == id_nuestro:
                    elegido = vals
                    break
                if elegido is None and any(abs(v) > 1e-6 for v in vals):
                    elegido = vals                 # sin landscape aún: el que tenga datos
            if elegido is None:
                return
            ahora = time.perf_counter()
            with self._lock:
                self._ang, self._t_ang = elegido, ahora
                self._n_paquetes += 1
                if ahora - self._t_hz >= 1.0:
                    self.hz = self._n_paquetes / (ahora - self._t_hz)
                    self._n_paquetes, self._t_hz = 0, ahora
        except Exception as e:
            self.error = f"ergonomics: {e}"

    # ── lectura ──────────────────────────────────────────────────────────────
    def angulos(self):
        """(20 ángulos en GRADOS, antigüedad en s) de tu mano, o (None, inf)."""
        with self._lock:
            if self._ang is None:
                return None, float("inf")
            return list(self._ang), time.perf_counter() - self._t_ang

    def datos_frescos(self):
        _, edad = self.angulos()
        return edad <= MANUS_TIMEOUT_S

    def ctrl_objetivo(self):
        """ctrl de la mano (N_ACT, NaN en la muñeca) según el guante, 1:1, o None."""
        ang, edad = self.angulos()
        if ang is None or edad > MANUS_TIMEOUT_S:
            return None
        return angulos_a_ctrl(ang, self.mapeo)

    def estado_txt(self):
        if self._sdk is None:
            txt = "MANUS: sin SDK"
        elif not self.conectado:
            txt = "MANUS: conectando..."
        elif not self.datos_frescos():
            txt = "MANUS: SIN DATOS (¿guante encendido?)"
        else:
            txt = f"MANUS: OK {self.hz:3.0f} Hz {MANUS_MAPEO}" + ("" if self.aplicar else " (standby)")
        if self.tactil is not None:
            txt += " " + self.tactil.estado_corto()     # sensores físicos de la OrcaHand Touch
        return txt

    # ── escritura en la simulación (hilo de física, antes de cada mj_step) ────
    def escribir_ctrl(self, datos, dt):
        """Lleva ctrl[1..16] de la mano hacia el guante con un filtro de primer orden
        (MANUS_TAU_S). Sin datos frescos o con aplicar=False no toca nada.
        Además (háptica v6) mide aquí la presión en cada dedo: este método corre en el hilo
        de física, justo antes de cada mj_step, así que los contactos se leen a
        MANUS_HAPTICA_MUESTREO_HZ y no al ritmo de la cámara."""
        if not self.aplicar:
            self._ctrl_suave = None
            return
        self._muestrear_tacto(datos)
        obj = self.ctrl_objetivo()
        if obj is None:
            return                                     # dedos quietos donde estén
        i0 = OFFSET_MANO
        if self._ctrl_suave is None:                   # arranque: desde donde está la mano
            self._ctrl_suave = np.array(datos.ctrl[i0:i0 + N_ACT], dtype=float)
        a = 1.0 - math.exp(-dt / MANUS_TAU_S) if MANUS_TAU_S > 0 else 1.0
        m = ~np.isnan(obj)                             # todo menos la muñeca
        self._ctrl_suave[m] += a * (obj[m] - self._ctrl_suave[m])
        datos.ctrl[i0 + 1:i0 + N_ACT] = self._ctrl_suave[1:]

    def restaurar_dedos(self, datos):
        """Vuelve a poner en ctrl[1..16] los dedos del guante. Se llama justo después
        de retarget_a_ctrl (cámara), que también los escribe: así la mano FÍSICA
        (construir_hardware_target) y la grabación de episodios leen los del guante."""
        if self.aplicar and self._ctrl_suave is not None:
            datos.ctrl[OFFSET_MANO + 1:OFFSET_MANO + N_ACT] = self._ctrl_suave[1:]

    # ── vibración ────────────────────────────────────────────────────────────
    def es_haptico(self):
        with self._lock:
            return self._id_guante is not None and self._guantes.get(self._id_guante, (0, False))[1]

    def vibrar(self, potencias):
        """potencias: 5 valores 0..1 (pulgar, índice, corazón, anular, meñique)."""
        with self._lock:
            gid = self._id_guante
        if self._sdk is None or not self.conectado or gid is None:
            return False
        p = self.ffi.new("float[5]")
        for i in range(5):
            p[i] = float(min(max(potencias[i], 0.0), 1.0))
        return self.lib.CoreSdk_VibrateFingersForGlove(gid, p) == self.RC.Success

    def haptica(self, modelo, datos=None):
        """Lo llama el bucle principal en cada fotograma (como antes). En v6 solo entrega el
        modelo: la presión se MIDE en el hilo de física (escribir_ctrl) y la vibración se
        ENVÍA desde un hilo propio (_bucle_haptica), así no depende de los FPS de la cámara."""
        if self._modelo is None and modelo is not None:
            self._modelo = modelo

    def _muestrear_tacto(self, datos):
        """Hilo de física: fuerzas de contacto por dedo → filtro (subida rápida, bajada suave).
        Solo con la fuente "sim", o "auto" sin datos de los sensores físicos."""
        if not MANUS_HAPTICA or self._modelo is None or self.fuente == "real":
            return
        if self.fuente == "auto" and self.fuente_activa == "real":
            return
        ahora = time.perf_counter()
        dt = ahora - self._t_muestra
        if dt < 1.0 / MANUS_HAPTICA_MUESTREO_HZ:
            return
        self._t_muestra = ahora
        dt = min(dt, 0.1)
        if self._tacto is None:
            try:
                self._tacto = SensorTactil(self._modelo)
            except Exception as e:
                print(f"⚠️  [MANUS] Háptica desactivada: no se pudo preparar el sensor táctil ({e})")
                self._tacto = False
        if not self._tacto:
            return
        f = self._tacto.fuerzas(self._modelo, datos)
        filt = self.tacto_fuerza.copy()
        tau = np.where(f > filt, MANUS_HAPTICA_TAU_SUBIDA, MANUS_HAPTICA_TAU_BAJADA)
        filt += (1.0 - np.exp(-dt / np.maximum(tau, 1e-4))) * (f - filt)
        pot, self._activos = self._tacto.potencias(filt, self._activos)
        with self._lock:
            self.tacto_fuerza, self.tacto_vibracion = filt, pot

    def _fuerzas_reales(self):
        """Fuerza (N) por dedo de los sensores FÍSICOS, pulgar → meñique, o None sin datos frescos."""
        if self.tactil is None:
            return None
        d = self.tactil.fuerzas()
        if d is None:
            return None
        f = np.zeros(5)
        for i, sensor in enumerate(SENSOR_DE_DEDO):     # i = dedo FÍSICO (pulgar → meñique)
            v = d.get(sensor)
            if not v:
                continue
            if MANUS_TACTIL_MEDIDA == "total":
                f[i] = math.sqrt(sum(float(x) * float(x) for x in v[:3]))
            else:
                f[i] = max(float(v[2]), 0.0)          # fz: componente normal (≥ 0)
        return f

    def _tacto_real(self, f, filt, dt):
        """Filtro (subida rápida, bajada suave) + curva umbral → 100 % con histéresis."""
        tau = np.where(f > filt, MANUS_HAPTICA_TAU_SUBIDA, MANUS_HAPTICA_TAU_BAJADA)
        filt = filt + (1.0 - np.exp(-dt / np.maximum(tau, 1e-4))) * (f - filt)
        pot, self._activos_real = potencias_tactil(filt, self._f_max_real, self._activos_real)
        return filt, pot

    def _bucle_haptica(self):
        """Hilo propio. Cada ciclo (MANUS_HAPTICA_MUESTREO_HZ) toma la fuerza de cada dedo
        —sensores físicos o simulación, según MANUS_HAPTICA_FUENTE— y manda la vibración
        al guante como mucho a MANUS_HAPTICA_HZ (la apaga en cuanto todo vuelve a 0)."""
        periodo = 1.0 / max(MANUS_HAPTICA_MUESTREO_HZ, MANUS_HAPTICA_HZ, 1.0)
        periodo_envio = 1.0 / max(MANUS_HAPTICA_HZ, 1.0)
        enviado = np.zeros(5)
        filt_real = np.zeros(5)
        t_envio = t_debug = 0.0
        t_prev = time.perf_counter()
        while self._activo:
            t0 = time.perf_counter()
            dt, t_prev = min(t0 - t_prev, 0.1), t0
            try:
                manda = self.haptica_siempre or self.aplicar
                f_real = self._fuerzas_reales() if self.fuente != "sim" else None
                if f_real is not None:
                    filt_real, pot_real = self._tacto_real(f_real, filt_real, dt)
                    fuente = "real"
                else:
                    filt_real[:] = 0.0
                    self._activos_real[:] = False
                    fuente = "sim" if self.fuente in ("sim", "auto") and bool(self._tacto) else "-"
                self.fuente_activa = fuente
                if fuente == "real":
                    with self._lock:
                        self.tacto_fuerza, self.tacto_vibracion = filt_real.copy(), pot_real.copy()
                    pot = pot_real if manda else np.zeros(5)
                    fz = filt_real
                else:
                    with self._lock:
                        pot = self.tacto_vibracion.copy() if (fuente == "sim" and self.aplicar) \
                            else np.zeros(5)
                        fz = self.tacto_fuerza.copy()
                activo = self.conectado and self.es_haptico()
                if not activo:
                    pot = np.zeros(5)
                apagar = enviado.any() and not pot.any()
                cambia = np.abs(pot - enviado).max() > 0.02
                if apagar or (cambia and t0 - t_envio >= periodo_envio):
                    if self.vibrar(pot):
                        enviado, t_envio = pot.copy(), t0
                if MANUS_HAPTICA_DEBUG and t0 - t_debug >= MANUS_HAPTICA_DEBUG_CADA_S \
                        and (activo or fuente == "real"):
                    t_debug = t0
                    print(f"[HAPTICA {fuente}] " + "  ".join(
                        f"{n[:3]} {f:5.2f}N→{100 * p:3.0f}%"
                        for n, f, p in zip(SensorTactil.DEDOS, fz, pot)))
            except Exception as e:
                self.error = f"haptica: {e}"
            time.sleep(max(0.0, periodo - (time.perf_counter() - t0)))
        if enviado.any():
            try:
                self.vibrar(np.zeros(5))
            except Exception:
                pass


def _f_max_tactil():
    """MANUS_TACTIL_F_MAX → array (pulgar → meñique). Acepta un número o un dict por dedo
    (claves en inglés como orca_core —thumb, index…— o en español —pulgar, indice…)."""
    v = MANUS_TACTIL_F_MAX
    out = np.full(5, 20.0)
    if isinstance(v, dict):                     # por dedo FÍSICO (pulgar, indice... o thumb, index...)
        for i, (en, es) in enumerate(zip(DEDOS_TACTIL, DEDOS_FISICOS)):
            out[i] = float(v.get(es, v.get(en, out[i])))
    else:
        out[:] = float(v)
    return np.maximum(out, MANUS_TACTIL_F_ON + 0.1)


def potencias_tactil(f, f_max, activos, f_on=None):
    """Fuerza (N) → vibración 0..1 por dedo: 0 bajo el umbral (con histéresis: se apaga al
    bajar del 70 % del umbral), y de MANUS_TACTIL_VIBRACION_MIN a MANUS_VIBRACION_MAX entre el
    umbral y f_max (recta con MANUS_TACTIL_GAMMA = 1); por encima de f_max, al máximo."""
    on = MANUS_TACTIL_F_ON if f_on is None else f_on
    activos = np.where(activos, f > 0.7 * on, f >= on)
    x = np.clip((f - on) / (f_max - on), 0.0, 1.0) ** max(MANUS_TACTIL_GAMMA, 0.05)
    pot = MANUS_TACTIL_VIBRACION_MIN + (MANUS_VIBRACION_MAX - MANUS_TACTIL_VIBRACION_MIN) * x
    return np.where(activos, pot, 0.0), activos


class SensorTactil:
    """Presión que los OBJETOS hacen sobre cada dedo de la OrcaHand simulada.

    Por qué antes vibraba "al llegar a un límite": se sumaban TODOS los contactos de la yema
    con cualquier cosa, también con la PALMA y los dedos vecinos de la propia mano. Al cerrar
    el puño (y más con MANUS_MAPEO = "cierre", que lleva las yemas a la palma) la mano se
    aprieta contra sí misma y el guante vibraba al máximo sin tocar nada. Ahora:
      · solo cuentan los contactos de un dedo con algo que NO es el robot (objetos, mesa,
        taladro...) y, con MANUS_HAPTICA_PROPIA = "pinza" (por defecto), el PULGAR contra
        otro dedo (una pinza sin objeto se siente en los dos dedos). Nunca la palma ni los
        roces laterales entre dedos. False = solo objetos; True = todo, como antes;
      · cuenta TODO el dedo (falange proximal, media/distal y su base; en el pulgar desde el
        CMC): agarrar un mango con la parte baja de los dedos también se siente.
        MANUS_HAPTICA_ZONA = "yema" vuelve a solo la punta;
      · la escala es la de CADA dedo: 100 % de vibración = MANUS_HAPTICA_F_MAX_FRAC (50 %) de
        la fuerza máxima que ese dedo puede hacer en la punta (par máximo de su actuador MCP
        / distancia del MCP a la punta). El servo de la mano es un muelle (kp): cuanto más
        cierras tu dedo por encima de lo que deja el objeto, más aprieta → más vibra.
      · curva: nada por debajo de MANUS_HAPTICA_F_ON; desde ahí, de MANUS_VIBRACION_MIN a
        MANUS_VIBRACION_MAX con exponente MANUS_HAPTICA_GAMMA (las presiones suaves se
        distinguen mejor), con histéresis para que no parpadee en el umbral.
    """
    DEDOS = ("pulgar", "indice", "corazon", "anular", "menique")
    # articulación en la base de cada dedo (todo lo que cuelga de ella es "el dedo")
    _RAIZ = ("right_t-cmc", "right_i-abd", "right_m-abd", "right_r-abd", "right_p-abd")
    _MCP = ("right_t-mcp", "right_i-mcp", "right_m-mcp", "right_r-mcp", "right_p-mcp")
    _YEMA = _BODIES_YEMA

    def __init__(self, modelo):
        import mujoco
        self._mj = mujoco
        B, J = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT
        nb = int(modelo.nbody)

        def bajo(raiz):
            """Cuerpos que cuelgan de 'raiz' (incluida)."""
            out = np.zeros(nb, dtype=bool)
            for b in range(nb):
                p = b
                while p > 0 and p != raiz:
                    p = int(modelo.body_parentid[p])
                out[b] = (p == raiz) and raiz > 0
            return out

        self.dedo_de_body = np.full(nb, -1, dtype=int)
        for i, (raiz, yema) in enumerate(zip(self._RAIZ, self._YEMA)):
            if MANUS_HAPTICA_ZONA == "yema":
                b0 = mujoco.mj_name2id(modelo, B, yema)
            else:
                j = mujoco.mj_name2id(modelo, J, raiz)
                b0 = int(modelo.jnt_bodyid[j]) if j >= 0 else mujoco.mj_name2id(modelo, B, yema)
            if b0 < 0:
                raise ValueError(f"no encuentro el dedo '{self.DEDOS[i]}' ({raiz} / {yema})")
            self.dedo_de_body[bajo(b0)] = i
        # todo el robot (brazo + mano): sus contactos consigo mismo no cuentan
        base = mujoco.mj_name2id(modelo, B, "base_link")
        if base < 0:
            base = mujoco.mj_name2id(modelo, B, "mano_mount")
        self.es_robot = bajo(base) if base > 0 else (self.dedo_de_body >= 0)

        # Fuerza máxima en la punta de cada dedo: par máx. del actuador MCP / palanca MCP → punta
        d = mujoco.MjData(modelo)
        mujoco.mj_kinematics(modelo, d)
        self.f_max = np.zeros(5)
        self.f_punta_max = np.zeros(5)
        for i, (mcp, yema) in enumerate(zip(self._MCP, self._YEMA)):
            j = mujoco.mj_name2id(modelo, J, mcp)
            by = mujoco.mj_name2id(modelo, B, yema)
            par = 1.0
            for a in range(int(modelo.nu)):
                if int(modelo.actuator_trntype[a]) == 0 and int(modelo.actuator_trnid[a, 0]) == j:
                    if int(modelo.actuator_forcelimited[a]):
                        par = float(max(abs(modelo.actuator_forcerange[a, 0]),
                                        abs(modelo.actuator_forcerange[a, 1])))
                    par *= abs(float(modelo.actuator_gear[a, 0])) or 1.0
                    break
            palanca = 0.06
            if j >= 0 and by >= 0:
                # la punta: el punto del cuerpo de la yema más lejano a su articulación (aprox.:
                # su centro de masas, prolongado lo mismo otra vez desde el PIP)
                pip = d.xpos[by]
                punta = pip + 2.0 * (d.xipos[by] - pip)
                palanca = max(float(np.linalg.norm(punta - d.xanchor[j])), 0.02)
            self.f_punta_max[i] = par / palanca
            if str(MANUS_HAPTICA_F_MAX).lower() == "auto":
                self.f_max[i] = MANUS_HAPTICA_F_MAX_FRAC * self.f_punta_max[i]
            else:
                self.f_max[i] = float(MANUS_HAPTICA_F_MAX)
        self.f_max = np.maximum(self.f_max, MANUS_HAPTICA_F_ON + 0.1)
        self._f6 = np.zeros(6)
        p = MANUS_HAPTICA_PROPIA
        self._propia = "pinza" if str(p).lower() == "pinza" else bool(p)
        print("🫳 [MANUS] Háptica por presión (objetos"
              + {"pinza": " + pinzas pulgar↔dedo", True: " + toda la mano", False: ""}[self._propia]
              + f", zona '{MANUS_HAPTICA_ZONA}'): "
              + "  ".join(f"{n} {MANUS_HAPTICA_F_ON:.1f}→{f:.1f} N" for n, f in zip(self.DEDOS, self.f_max))
              + "  (umbral → 100 %)")

    def fuerzas(self, modelo, datos):
        """Fuerza normal total (N) que recibe cada dedo, pulgar → meñique."""
        f = np.zeros(5)
        gb = modelo.geom_bodyid
        for i in range(int(datos.ncon)):
            c = datos.contact[i]
            b1, b2 = int(gb[c.geom1]), int(gb[c.geom2])
            d1, d2 = self.dedo_de_body[b1], self.dedo_de_body[b2]
            if d1 < 0 and d2 < 0:
                continue                                   # ningún dedo implicado
            if self.es_robot[b1] and self.es_robot[b2] and self._propia is not True:
                # la mano contra sí misma: solo la pinza pulgar ↔ otro dedo (si "pinza")
                if not (self._propia == "pinza" and d1 >= 0 and d2 >= 0 and d1 != d2
                        and 0 in (d1, d2)):
                    continue
            self._mj.mj_contactForce(modelo, datos, i, self._f6)
            fn = abs(float(self._f6[0]))
            if d1 >= 0:
                f[d1] += fn
            if d2 >= 0 and d2 != d1:
                f[d2] += fn
        return f

    def potencias(self, f, activos):
        """Fuerza filtrada (N) → vibración 0..1 por dedo, con histéresis en el umbral."""
        on = MANUS_HAPTICA_F_ON
        activos = np.where(activos, f > 0.7 * on, f > on)
        x = np.clip((f - on) / (self.f_max - on), 0.0, 1.0) ** max(MANUS_HAPTICA_GAMMA, 0.05)
        pot = MANUS_VIBRACION_MIN + (MANUS_VIBRACION_MAX - MANUS_VIBRACION_MIN) * x
        return np.where(activos, pot, 0.0), activos
