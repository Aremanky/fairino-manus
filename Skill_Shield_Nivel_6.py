#!/usr/bin/env python3
"""
Teleoperación OrcaHand v2 (mano derecha) con OAK-D + MediaPipe → MuJoCo
=========================================================================
Sin dependencias de ROS 2. Todo el pipeline es Python puro.

Arquitectura:
  OAK-D (DepthAI) → MediaPipe (21 landmarks) → Retargeting → MuJoCo ctrl[]

Controles de teclado:
  [A]  Activar control por cámara
  [S]  Standby (devuelve todos los actuadores a 0)
  [G]  Empezar / terminar un episodio (BRAZO + MANO a 30 Hz) en episodios/<tarea>/.
       La 1ª vez pide el nombre de la TAREA; [Shift+G] cambia de tarea.
  [E]  Entrenar la tarea actual (modelos/<tarea>.pth) en OTRO proceso: la app sigue
  [Shift+E]  Reentrenar TODAS las tareas grabadas (también en otro proceso)
  [P]  Pedal (tecla mantenida, TELEOP_FR5_PEDAL = "teclado"): arma el brazo mientras se pulsa
  [I]  Modo IA — elige una TAREA y la reproduce (brazo + mano; con el FR5 conectado,
       arranca y para con el PEDAL)
  [3]  Mostrar / ocultar el ENTORNO DE TRABAJO (mesa con alicates, destornillador,
       bombilla y fusible — definido en mesa_trabajo.xml)
  [F]  Modo FINO del brazo: el robot se mueve FINO_ESCALA veces lo que te mueves tú
       (para maniobrar con herramientas). Otra vez [F] → vuelve a imitarte suavemente
  [C]  CONGELAR / descongelar el brazo (se queda quieto; la mano sigue funcionando)
  [R]  (TELEOP_FR5_V2) Registro de diagnóstico: TELEOP_FR5_DIAG_S s quieto (T-pose) →
       ruido de hombro/codo/muñeca, huesos, ángulos y A contra B (diagnosticos/*.npz)
  [Q]  Salir

Dependencias: requirements.txt

Uso:
  ./venv_nivel6/bin/python Skill_Shield_Nivel_6.py     (o ./arrancar_nivel6.sh)
"""
# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURA DEL PROYECTO (refactor en módulos — el comportamiento NO cambia)
# ══════════════════════════════════════════════════════════════════════════════
# Este archivo contiene SOLO el bucle principal main(). Todo lo demás vive en el
# paquete skill_shield/, agrupado por responsabilidad:
#
#   skill_shield/
#   ├── vision/       detectores.py (MediaPipe) · camara_oak.py (pipeline DepthAI)
#   │                 profundidad_3d.py · recorte_mano.py
#   ├── mano/         biomecanica.py · retargeting.py (landmarks → ctrl de la OrcaHand)
#   ├── brazo/        ik.py · hibrido.py · ik_codo_muneca.py · control_suave.py
#   │                 angulos_humano.py (Pose + estéreo → ángulos del brazo humano)
#   ├── simulacion/   objetos.py · entorno_taller.py [3] · entorno_taladro.py [4]
#   │                 demo_taladro.py [5] · hilo_fisica.py
#   ├── ia/           act.py (modelo ACT, episodios, entrenamiento, carga)
#   ├── hardware/     sincronizacion.py (array de motores para la mano física)
#   ├── interfaz/     hud_logo.py · menu.py · perfil.py
#   └── comun/        suavizador.py
#
# config.py y orca_hardware.py siguen en la raíz, sin cambios.
# ══════════════════════════════════════════════════════════════════════════════
import os
import sys
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import depthai as dai
import numpy as np
import mujoco
import mujoco.viewer
import math
import time
from contextlib import nullcontext
from collections import deque
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIÓN DEL PANEL DE CONFIGURACIÓN GLOBAL (NIVEL 6)
# ══════════════════════════════════════════════════════════════════════════════
from config import *
from orca_hardware import OrcaHardwareWrapper

# ── Módulos del proyecto (paquete skill_shield) ──────────────────────────────
from skill_shield.vision.detectores import (
    detector_mano, detector_mano_recorte, detector_pose, mp_draw, mp_hands, mp_pose, mp_styles,
)
# — Dos cámaras OAK-D (MULTICAMARA_ACTIVA en config.py) —
from skill_shield.vision.multicamara import (
    abrir_dispositivo, activar_ir, CamaraSecundaria, cargar_extrinsecas, comprobar_usb,
    CAM_A_MXID, CAM_B_MXID,
)
from skill_shield.vision.fusion_brazo import FusorBrazo, calcular_angulos_brazo_multicamara
from skill_shield.ia.act import (
    _listar_modelos_entrenados, _listar_tareas_grabadas, _nombre_seguro, cargar_ia,
    EPISODIOS_DIR, MODELOS_DIR,
)
from skill_shield.comun.suavizador import SuavizadorND
from skill_shield.interfaz.hud_logo import (
    _cargar_logo, _dibujar_logo_mujoco, _preparar_logo_mujoco, _superponer_logo,
)
from skill_shield.hardware.sincronizacion import construir_hardware_target, limitar_salto_hardware
from skill_shield.simulacion.entorno_taller import (
    EntornoTaller, TALLER_CAMARA, TALLER_ENCUADRAR_CAMARA,
)
from skill_shield.simulacion.entorno_taladro import EntornoTaladro
from skill_shield.simulacion.demo_taladro import DemoTaladro
from skill_shield.brazo.ik import landmarks_a_3d, OFFSET_LOCAL_MUNON, resolver_ik_brazo
from skill_shield.brazo.hibrido import preparar_hibrido
from skill_shield.brazo.ik_codo_muneca import (
    _eje_codo_humano_torso, _equivalente_en_rango, _perp, coste_ik, enderezar, es_girada,
    FiltroOneEuro, lateral_mano_humana_mundo, objetivo_mano_hibrido, objetivos_codo_muneca,
    objetivos_codo_palma, peso_codo_humano, q6_para_orientacion, rescatar_ik,
    resolver_ik_codo_muneca, semilla_girada,
    bloqueo_base, codo_fiable,                       # 7-oct: base estable + codo fiable
)
# — 7-oct: límites humanos de hombro y codo (limites_humanos.py, LIMITES_HUMANOS en config.py) —
from skill_shield.brazo.limites_humanos import (
    completar_hib, LIMITES_HUMANOS, LimitesHumanos, objetivos_codo_palma_fiel,
    RETARGET_FIEL, semilla_codo_espejo,
)
from skill_shield.brazo.control_suave import (
    aplicar_modo_brazo, CanalFino, FiltroSaltos, fusionar_direccion, GeneradorSuave,
    limitar_alcance_palma, ZonaMuerta,
)
from skill_shield.brazo.angulos_humano import (
    _fmt_ang, _lm_hand, _unitario, _wrap, angulos_humano_a_q_brazo, angulos_humano_a_q_brazo_rel,
    calcular_angulos_brazo_estereo, calcular_angulos_brazo_humano, calcular_pronosupinacion,
    corregir_con_tpose, EstimadorLongitudesBrazo, limitar_velocidad_articular,
)
from skill_shield.vision.recorte_mano import caja_recorte_mano, detectar_mano_en_recorte
from skill_shield.mano.biomecanica import calcular_flexion_muneca
from skill_shield.mano.retargeting import retarget_a_ctrl
from skill_shield.vision.camara_oak import crear_pipeline_oak
from skill_shield.simulacion.hilo_fisica import HiloFisica
from skill_shield.mano.manus_guante import GuanteManus      # guantes MANUS (FUENTE_DEDOS)
from skill_shield.hardware.fr5_servo import FR5Servo, sincronizar_mujoco   # FR5 real: ServoJ cada 8 ms
from skill_shield.interfaz.perfil import PerfilBucle



# ══════════════════════════════════════════════════════════════════════════════
# BUCLE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── MENÚ DE ARRANQUE CON ZENITY (compatible con Wayland/GNOME) ──────────
    # zenity es la herramienta de diálogos nativa de GNOME, siempre visible.
    # Si no está instalada: sudo apt install zenity
    # La misma pregunta de siempre, con CUATRO botones (dos --extra-button), en este orden:
    #   [Extra 1] 💻 Simulación             → SIMULACION
    #   [Extra 2] 🎚️ Gemelo (sliders)        → GEMELO_SLIDERS: mano física + guante igual que
    #             HARDWARE_REAL, pero el BRAZO se mueve con la ventana de sliders
    #             (skill_shield/interfaz/sliders_brazo.py)
    #   [OK]      🤖 Mano Física + Gemelo   → HARDWARE_REAL
    #   [Cancel]  🧠 Funcionamiento autónomo → AUTONOMO (10-nov; paquete skill_shield/autonomo/)
    # zenity devuelve 1 tanto para Cancel como para los botones extra; los extra se distinguen
    # porque escriben su etiqueta en stdout.
    # OJO: cerrar la ventana (X / Esc) equivale a Cancel → hoy entra en AUTONOMO. Mientras el
    # modo autónomo no mueva nada da igual; cuando mueva el FR5, exigir siempre el pedal.
    import subprocess
    global MODO_EJECUCION
    _BOTON_SIMULACION = "💻  Simulación Virtual (MuJoCo)"
    _BOTON_SLIDERS    = "🎚️  Gemelo Digital (sliders)"
    try:
        resultado = subprocess.run(
            [
                'zenity', '--question',
                '--title=Skill Shield Nivel 6',
                '--text=¿Qué modo de operación deseas iniciar?',
                f'--extra-button={_BOTON_SIMULACION}',                  # 1º
                f'--extra-button={_BOTON_SLIDERS}',                     # 2º
                '--ok-label=🤖  Mano Física (USB) + Gemelo Digital',     # 3º
                '--cancel-label=🧠  Funcionamiento autónomo',            # 4º (el último)
                '--width=420',
                '--height=50',
            ],
            capture_output=True
        )
        stderr_txt = resultado.stderr.decode(errors="ignore").strip() if resultado.stderr else ""
        stdout_txt = resultado.stdout.decode(errors="ignore").strip() if resultado.stdout else ""

        if resultado.returncode == 1 and stdout_txt == _BOTON_SIMULACION:
            # Botón extra 1 → Simulación
            MODO_EJECUCION = "SIMULACION"
        elif resultado.returncode == 1 and stdout_txt == _BOTON_SLIDERS:
            # Botón extra 2 → gemelo digital con el brazo en sliders
            MODO_EJECUCION = "GEMELO_SLIDERS"
        elif resultado.returncode == 0:
            # Botón OK → mano física + gemelo digital normal
            MODO_EJECUCION = "HARDWARE_REAL"
        elif resultado.returncode == 1 and not stdout_txt and not stderr_txt:
            # Botón Cancelar (el último) → funcionamiento autónomo
            MODO_EJECUCION = "AUTONOMO"
        else:
            # Cualquier otro código, o zenity escribió algo en stderr → no sabemos si el
            # usuario llegó a ver la ventana. Por seguridad, NUNCA asumimos hardware real.
            if stderr_txt:
                print(f"⚠️  Zenity no pudo mostrar la ventana correctamente: {stderr_txt}")
            print("   Por seguridad, arrancando en SIMULACION.")
            MODO_EJECUCION = "SIMULACION"
    except FileNotFoundError:
        print("⚠️  zenity no encontrado → arrancando en SIMULACION por defecto.")
        print("   Para instalar: sudo apt install zenity")
        MODO_EJECUCION = "SIMULACION"
    except Exception as e:
        print(f"⚠️  Error inesperado al lanzar el selector de modo ({e}).")
        print("   Por seguridad, arrancando en SIMULACION.")
        MODO_EJECUCION = "SIMULACION"

    print(f"🚀 Iniciando Skill Shield Nivel 6 en modo: {MODO_EJECUCION}")

    # ── FUNCIONAMIENTO AUTÓNOMO: paquete aparte con su propio bucle (no usa el de abajo) ──
    # Se sale de main() ANTES de arrancar fr5_servo, la mano o las cámaras: el paquete
    # arranca las suyas (FR5 en simulacro, cámara de cabeza, web y gemelo en MuJoCo).
    if MODO_EJECUCION == "AUTONOMO":
        from skill_shield.autonomo.modo import ejecutar as ejecutar_autonomo
        ejecutar_autonomo(globals())
        return
    modo_sliders = MODO_EJECUCION == "GEMELO_SLIDERS"
    mano_fisica  = MODO_EJECUCION in ("HARDWARE_REAL", "GEMELO_SLIDERS")   # OrcaHand por USB
    # 13-nov: en GEMELO_SLIDERS el brazo va con los sliders y los dedos con el guante MANUS: las
    # cámaras OAK-D no hacen falta y NO se abren (SLIDERS_SIN_CAMARAS = False → como antes).
    sin_camaras  = modo_sliders and bool(globals().get("SLIDERS_SIN_CAMARAS", True))

    # ── FR5 REAL (proceso fr5_servo, ServoJ cada 8 ms) ──
    # Se arranca AQUÍ, antes del visor, las cámaras y los hilos: el proceso se crea con fork.
    # 9-nov: en los dos modos con hardware se arranca aunque FR5_REAL_ACTIVO sea False, cada uno
    # con su simulacro y velocidad máxima:
    #   · GEMELO_SLIDERS → SLIDERS_FR5_*   (se arma con ARMAR en la ventana de sliders)
    #   · HARDWARE_REAL  → TELEOP_REAL_*   (teleoperación con cámaras; se arma con el PEDAL)
    # En SIMULACION solo si FR5_REAL_ACTIVO = True (como antes).
    fr5 = None
    _cfg_fr5 = dict(globals())
    _pref_fr5 = {"GEMELO_SLIDERS": "SLIDERS", "HARDWARE_REAL": "TELEOP_REAL"}.get(MODO_EJECUCION)
    _fr5_por_modo = _pref_fr5 is not None and bool(globals().get(f"{_pref_fr5}_CON_FR5", True))
    if _fr5_por_modo:
        _cfg_fr5["FR5_SIMULACRO"] = bool(globals().get(f"{_pref_fr5}_FR5_SIMULACRO", False))
        _cfg_fr5["FR5_VEL_MAX_DEG_S"] = min(float(globals().get("FR5_VEL_MAX_DEG_S", 60.0)),
                                            float(globals().get(f"{_pref_fr5}_FR5_VEL_MAX_DEG_S", 30.0)))
    if globals().get("FR5_REAL_ACTIVO", False) or _fr5_por_modo:
        fr5 = FR5Servo.desde_config(_cfg_fr5)
        if not fr5.iniciar():
            print("   El programa sigue SOLO en simulación (el FR5 no recibe nada).")

    # ── GEMELO_SLIDERS: ventana de sliders del brazo (proceso aparte, también con fork) ──
    sliders = None
    if modo_sliders:
        from skill_shield.interfaz.sliders_brazo import (
            SlidersBrazo, SIN_FR5, FR5_ESPERA, FR5_ENVIANDO, FR5_ERROR)
        sliders = SlidersBrazo.desde_config(globals())
        if not sliders.iniciar():
            print("   GEMELO_SLIDERS sin ventana: la mano funciona, el brazo NO se mueve.")
        elif fr5 is None or not fr5.conectado:
            sliders.mensaje("Sin conexión con el FR5: los sliders solo mueven el simulador.")
    fr5_al_inicio = fr5 is not None and fr5.conectado   # (si falla luego, los sliders desarman)
    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────

    # — Cargar modelo MuJoCo (UNA SOLA VEZ) —
    modelo = mujoco.MjModel.from_xml_path(RUTA_XML)
    datos  = mujoco.MjData(modelo)
    # ── Reflejo del suelo: apagado ──
    for _i in range(modelo.nmat):
        modelo.mat_reflectance[_i] = 0.0
    visor  = mujoco.viewer.launch_passive(modelo, datos)

    # ── INICIALIZACIÓN DEL HARDWARE REAL ──
    mano_real = None
    if mano_fisica:
        mano_real = OrcaHardwareWrapper(puerto=PUERTO_HARDWARE)
        mano_real.conectar()
    # ───────────────────────────────────────

    # ── Configurar perspectiva inicial de la cámara ──
    with visor.lock():
        # VISOR_VISTA_INICIAL (config.py): la misma que usa el gemelo de la web
        visor.cam.lookat[:] = VISOR_VISTA_INICIAL["mirar"]
        visor.cam.distance  = VISOR_VISTA_INICIAL["distancia"]
        visor.cam.azimuth   = VISOR_VISTA_INICIAL["azimut"]
        visor.cam.elevation = VISOR_VISTA_INICIAL["elevacion"]
    # ─────────────────────────────────────────────────────────────────────────

    print(f"✅ Modelo MuJoCo cargado: {modelo.nu} actuadores, {modelo.nsensor} sensores.")

    # ── Cargar logo IDI / Grupo Elecnor para el HUD ───────────────────────────
    # Se generan tres variantes:
    #   · _logo_bgr / _logo_alpha        → ventana de teleoperación (750×750 px)
    #   · _logo_bgr_sm / _logo_alpha_sm  → panel de modo autónomo  (180×450 px)
    #   · _logo_mj_px / _logo_mj_w/h     → píxeles RGB para mjr_drawPixels
    _logo_bgr,    _logo_alpha    = _cargar_logo(RUTA_LOGO, alto_px=70)
    _logo_bgr_sm, _logo_alpha_sm = _cargar_logo(RUTA_LOGO, alto_px=35)
    _logo_mj_px, _logo_mj_w, _logo_mj_h = _preparar_logo_mujoco(RUTA_LOGO, alto_px=80)
    if _logo_bgr is not None:
        print(f"✅ Logo IDI cargado ({_logo_bgr.shape[1]}×{_logo_bgr.shape[0]} px).")
    if _logo_mj_px is not None:
        print(f"✅ Logo IDI para MuJoCo HUD preparado ({_logo_mj_w}×{_logo_mj_h} px).")
    # ─────────────────────────────────────────────────────────────────────────

    if modelo.nu != OFFSET_MANO + N_ACT:
        print(f"⚠  ADVERTENCIA: El modelo tiene {modelo.nu} actuadores "
              f"pero el código espera {OFFSET_MANO + N_ACT} "
              f"({OFFSET_MANO} de brazo + {N_ACT} de mano). Revisa LIMITES_CTRL.")

    # Vista (NO copia) sobre la porción de datos.ctrl[] que corresponde a la
    # mano: escribir en ctrl_mano[i] escribe directamente en
    # datos.ctrl[OFFSET_MANO + i]. Así todo el código de mano de más abajo
    # sigue usando los mismos índices 0..16 de siempre, sin tocarlo.
    ctrl_mano = datos.ctrl[OFFSET_MANO: OFFSET_MANO + N_ACT]

    # ── Índices articulares del BRAZO, para la IK (calculados una sola vez) ──
    dof_idx_brazo  = [modelo.jnt_dofadr[mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_JOINT, j)]
                       for j in ARM_JOINTS]
    qpos_idx_brazo = [modelo.jnt_qposadr[mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_JOINT, j)]
                       for j in ARM_JOINTS]
    # Body del efector, para leer su posición REAL al calibrar con [A]
    mano_mount_id = mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_BODY, "mano_mount")
    # ── Gravedad de la mano: gravcomp NO se hereda de mano_mount a sus hijos ──
    print(f"⚖️  Masa colgando de mano_mount: {modelo.body_subtreemass[mano_mount_id]:.2f} kg")
    if HAND_GRAVCOMP:
        _n_comp = 0
        for _b in range(1, modelo.nbody):
            _p = _b
            while _p > 0 and _p != mano_mount_id:
                _p = modelo.body_parentid[_p]
            if _p == mano_mount_id:
                modelo.body_gravcomp[_b] = 1.0
                _n_comp += 1
        print(f"⚖️  Gravedad compensada en {_n_comp} cuerpos de la mano (HAND_GRAVCOMP).")
    # — Entorno de trabajo [3]: se carga escondido —
    taller = EntornoTaller(modelo, datos)
    taladro = EntornoTaladro(modelo, datos)
    demo_taladro = DemoTaladro(modelo, taladro)      # [5] demo automática del taladro

    # Aplicar posición de standby inicial (brazo fijo por ahora, sin IK)
    for i, v in enumerate(POSICION_STANDBY_BRAZO):
        datos.ctrl[i] = float(v)
    for i, v in enumerate(POSICION_STANDBY):
        ctrl_mano[i] = float(v)
    mujoco.mj_step(modelo, datos)

    # — FR5 REAL (5-nov): el simulador NO es el estado real. Con el robot conectado, el brazo
    #   de MuJoCo se pone en la postura LEÍDA del FR5 (al arrancar, al entrar en teleoperación y
    #   en cada vuelta mientras está desarmado), así la IK y el generador suave salen de ahí. —
    idx_ctrl_brazo = [mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{_j}")
                      for _j in ARM_JOINTS]
    if min(idx_ctrl_brazo) < 0:
        idx_ctrl_brazo = list(range(6))

    def _sincronizar_con_fr5(motivo=None):
        """MuJoCo ← postura real del FR5. False si no hay lectura reciente."""
        _q_real = fr5.q_real_rad() if fr5 is not None else None
        if _q_real is None:
            return False
        sincronizar_mujoco(modelo, datos, _q_real, qpos_idx_brazo, dof_idx_brazo, idx_ctrl_brazo)
        if motivo:
            print(f"🔗 MuJoCo sincronizado con el FR5 real ({motivo}): q = ["
                  + ", ".join(f"{_v:+.2f}" for _v in _q_real) + "] rad")
        return True

    if fr5 is not None:
        with visor.lock():
            if not _sincronizar_con_fr5("arranque"):
                print("⚠ FR5: sin postura real → el simulador arranca en POSICION_STANDBY_BRAZO")

    # — 9-nov: brazo FANTASMA (objetivo) + puntas objetivo/real en el visor (solo visual) —
    fantasma = None
    _t_debug_gemelo = [0.0]    # último print del diagnóstico del gemelo
    if globals().get("FANTASMA_ACTIVO", True):
        try:
            from skill_shield.interfaz.fantasma import FantasmaBrazo
            fantasma = FantasmaBrazo(modelo, ARM_JOINTS,
                                     cuerpo_punta=globals().get("FANTASMA_CUERPO_PUNTA", "mano_mount"))
        except Exception as _e_f:
            print(f"⚠ Brazo fantasma desactivado ({_e_f})")
            fantasma = None

    # — Suavizadores —
    suav_dedos = SuavizadorND(n_dim=N_ACT - 1, ventana=4)  # ctrl[1..16]

    # — Estado compartido de control —
    estado: dict = {
        "camara_activa":    False,
        "muneca_bloqueada": False,
        "pinza_previa":     False,
        "ultima_pos_palma": 0.0,
        "vec_neutro":       None,   # Vector muñeca→MCP calibrado
        "eje_lateral":      None,   # Eje de bisagra de la muñeca
        "calc_muneca":      0.0,
        "dedo_activo":      "indice",
        "modo_pinza":       False,
        "dedo_bloqueado":   None,     # fija el dedo mientras la pinza está activa
        "ratio_pinza":      0.0,      # factor continuo 0.0 (abierta) → 1.0 (cerrada)
        "pinza_tocando":    False,
        "pinza_habilitada": True,
    }
    # — Variables de dataset e IA —
    tarea_actual    = None # nombre de la tarea/maniobra que se está grabando ahora
    # 5-nov: grabadora BRAZO + MANO a ritmo fijo desde el hilo de la física (formato 2)
    from skill_shield.ia.grabador import GrabadorEpisodio
    grabador = GrabadorEpisodio.desde_modelo(
        modelo, ARM_JOINTS, OFFSET_MANO, N_ACT, fps=float(globals().get("IA_FPS", 30.0)),
        robot=str(globals().get("ROBOT_BRAZO", "?")),
        recortar=bool(globals().get("IA_RECORTAR_QUIETOS", True)))

    def _zenity(args):
        """Diálogo de zenity (texto devuelto) o None si no hay zenity / se cancela."""
        try:
            r = subprocess.run(["zenity"] + args, capture_output=True, text=True, timeout=600)
            return r.stdout.strip() if r.returncode == 0 else ""
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def _pedir_tarea(actual):
        """Nombre de la tarea a grabar (diálogo; si no hay zenity, por terminal)."""
        ya = _listar_tareas_grabadas()
        txt = _zenity(["--entry", "--title=Grabar tarea", "--width=420",
                       "--text=Nombre de la tarea (ya grabadas: " + (", ".join(ya) or "ninguna") + ")",
                       f"--entry-text={actual or ''}"])
        if txt is None:                                   # sin zenity: por terminal
            print(f"\n📝 Tareas ya grabadas: {', '.join(ya) or '(ninguna todavía)'}")
            txt = input(f"   Nombre de la tarea [{actual or 'tarea_1'}]: ").strip()
            return _nombre_seguro(txt) if txt else (actual or _nombre_seguro("tarea_1"))
        return _nombre_seguro(txt) if txt else actual     # cancelar = la misma (o ninguna)

    def _elegir(titulo, opciones):
        """Una opción de la lista (diálogo; si no hay zenity, por terminal) o None."""
        txt = _zenity(["--list", f"--title={titulo}", "--column=Tarea", "--height=400",
                       "--width=360"] + list(opciones))
        if txt is None:
            for _k, _n in enumerate(opciones, start=1):
                print(f"   [{_k}] {_n}")
            e = input(f"   Elige nº (1-{len(opciones)}, vacío = cancelar): ").strip()
            return opciones[int(e) - 1] if e.isdigit() and 1 <= int(e) <= len(opciones) else None
        return txt.split("|")[0] if txt else None

    def _entrenar_en_segundo_plano(args):
        """Entrena en OTRO proceso (python -m skill_shield.ia.act …): la app sigue funcionando.
        La salida va a la terminal y a modelos/entrenamiento_<…>.log."""
        os.makedirs(MODELOS_DIR, exist_ok=True)
        log = os.path.join(MODELOS_DIR, f"entrenamiento_{'_'.join(args).lstrip('-')}.log")
        print(f"🧠 Entrenando en segundo plano: python -m skill_shield.ia.act {' '.join(args)}  "
              f"(progreso en {log})")
        with open(log, "w") as _f:
            subprocess.Popen([sys.executable, "-m", "skill_shield.ia.act"] + list(args),
                             stdout=_f, stderr=subprocess.STDOUT,
                             cwd=os.path.dirname(os.path.abspath(__file__)))

    def _parar_grabacion(motivo=""):
        """Cierra el episodio en curso (si lo hay) y lo guarda en episodios/<tarea>/."""
        if not grabador.activo:
            return
        _ruta, _n = grabador.parar(os.path.join(EPISODIOS_DIR, tarea_actual or "tarea_1"))
        if _ruta:
            print(f"💾 Episodio guardado{(' (' + motivo + ')') if motivo else ''}: {_ruta} "
                  f"({_n} fotogramas, {_n / grabador.fps:.1f} s)")
    modo_ia_real  = False
    modo_demo_taladro = False   # [5] demo automática: coger taladro y agujerear el techo
    red_neuronal  = None

    pipeline = None if sin_camaras else crear_pipeline_oak()
    salir    = False
    # Resultados "vacíos" de MediaPipe para el modo sin cámaras (nadie detectado)
    _SIN_POSE = SimpleNamespace(pose_landmarks=None, pose_world_landmarks=None)
    _SIN_MANO = SimpleNamespace(multi_hand_landmarks=None, multi_hand_world_landmarks=None)

    # — Guante MANUS: dedos de la OrcaHand desde el guante (la muñeca sigue con la cámara) —
    guante = None
    if globals().get("FUENTE_DEDOS", "camara") == "manus":
        guante = GuanteManus()
        if not guante.iniciar():
            print("⚠️  [MANUS] Sin guante y SIN cámaras (gemelo con sliders): los dedos NO se moverán."
                  if sin_camaras else "⚠️  [MANUS] Sigo con los dedos de la CÁMARA.")
            guante = None

    print("\n  Controles: [A] Activar | [S] Standby | [G] Grabar | "
          "[E] Entrenar tarea actual | [Shift+E] Entrenar todas | [P] Pedal | [I] Modo IA (red neuronal) | "
          "[3] Taller | [4] Taladro | [5] Demo taladro | [R] Diagnóstico T-pose (v2) | [X] Reset errores FR5 | [Q] Salir\n")

    NOMBRE_VENTANA = "OrcaHand Teleop OAK-D + MuJoCo"  # sin em dash —
    cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NOMBRE_VENTANA, int(ANCHO_CAM * ESCALA_VENTANA), int(ALTO_CAM * ESCALA_VENTANA))

    # El logo IDI se dibuja directamente dentro del visor MuJoCo mediante
    # mjr_drawPixels (ver _dibujar_logo_mujoco). No se abre ninguna ventana
    # adicional de OpenCV para el logo.

    while not salir:

        # ══════════════════════════════════════════════════
        # ESTADO A2 — MODO IA REAL (Aprendizaje por demostración)
        # Usa el modelo ACT entrenado con [E] a partir de tus episodios
        # grabaciones [G]. En cada paso: mantiene un historial de los últimos
        # fotogramas de postura (en vez de un reloj global), predice la acción
        # (ctrl) y la aplica — así distingue "voy abriendo" de "voy cerrando"
        # aunque la postura instantánea coincida.
        # ══════════════════════════════════════════════════
        if modo_ia_real:
            # 5-nov: BRAZO + MANO (modelos de formato 2) o solo mano (formato 1, como antes).
            #  · La física avanza a TIEMPO REAL (antes 4 mj_step = 8 ms por fotograma de ~33 ms:
            #    la mano iba 4 veces más lenta que al grabar y la red veía posturas retrasadas).
            #  · Los comandos se interpolan entre fotogramas paso a paso de la física.
            #  · Con el FR5 conectado: empieza PARADO, con el simulador copiando la postura real;
            #    el PEDAL arranca (ServoJ desde la postura real) y para. Sin FR5: arranca solo y
            #    el pedal pausa.
            from skill_shield.ia.act import PoliticaACT
            from skill_shield.teleop_fr5.pedal import Pedal
            cv2.destroyAllWindows()
            cv2.namedWindow("Modo IA Real", cv2.WINDOW_AUTOSIZE)

            dt_objetivo = float(red_neuronal["dt_objetivo"])
            n_sub = max(1, int(round(dt_objetivo / float(modelo.opt.timestep))))
            politica = PoliticaACT(red_neuronal, peso_exp=float(globals().get("IA_PESO_EXP", 0.15)))

            # — qué joint del simulador es cada columna del modelo —
            _por_nombre = {n: (qa, ci) for n, qa, ci in zip(grabador.nombres, grabador.qadr, grabador.cidx)}
            if red_neuronal.get("formato", 1) == 2:
                _nombres_ia = [c[2:] for c in red_neuronal["col_estado"]]       # "q:joint1" → "joint1"
                _acc_ia = [c[2:] for c in red_neuronal["col_accion"]]
                _faltan = [n for n in set(_nombres_ia + _acc_ia) if n not in _por_nombre]
            else:                                                              # solo mano
                _nombres_ia = _acc_ia = list(grabador.nombres[len(ARM_JOINTS):])
                _faltan = []
            if _faltan:
                print(f"❌ Modo IA: el modelo usa articulaciones que este robot no tiene: {_faltan}")
                modo_ia_real = False
                cv2.destroyWindow("Modo IA Real")
                continue
            qadr_ia = [_por_nombre[n][0] for n in _nombres_ia]
            cidx_ia = np.array([_por_nombre[n][1] for n in _acc_ia])
            es_brazo = np.array([n in ARM_JOINTS for n in _acc_ia])
            lim_ia = np.array([LIMITES_CTRL_BRAZO[ARM_JOINTS.index(n)] if n in ARM_JOINTS
                               else LIMITES_CTRL[int(c) - OFFSET_MANO] for n, c in zip(_acc_ia, cidx_ia)],
                              dtype=float)
            paso_max_brazo = float(globals().get("IA_V_MAX_BRAZO", 1.0)) * dt_objetivo
            try:                                     # colisiones del brazo (las de la IK de teleop)
                col_ia = teleop_v2.ik.col if teleop_v2 is not None else None
            except NameError:
                col_ia = None
            con_brazo = bool(es_brazo.any())
            print(f"🧠 Modo IA (ACT) — {'BRAZO + MANO' if con_brazo else 'solo MANO'}: "
                  f"{len(_acc_ia)} articulaciones, {1 / dt_objetivo:.0f} Hz, {n_sub} pasos de física "
                  "por fotograma (tiempo real).")

            usar_hardware_real = (mano_fisica
                                  and mano_real and mano_real.conectado)
            ultimo_hardware_target = construir_hardware_target(datos) if usar_hardware_real else None
            fr5_ia = fr5 if (fr5 is not None and fr5.conectado and con_brazo) else None
            if usar_hardware_real:
                print("🤖 Mano física: reproduce lo mismo que la simulación.")
            if fr5_ia is not None:
                print("🦾 FR5 REAL conectado: la reproducción empieza PARADA. Pisa el PEDAL para "
                      "arrancar (el robot sale de su postura real) y otra vez para parar.")
            pedal_ia = Pedal.desde_config(globals())
            en_marcha = fr5_ia is None                   # sin FR5 arranca solo
            fr5_ia_enviando = False
            n_bloqueos = 0

            def _salir_ia():
                if fr5_ia is not None:
                    fr5_ia.habilitar(False)
                pedal_ia.cerrar()
                if usar_hardware_real:
                    for _i, _v in enumerate(POSICION_STANDBY):
                        ctrl_mano[_i] = float(_v)
                    mano_real.enviar_posiciones(construir_hardware_target(datos))
                cv2.destroyWindow("Modo IA Real")

            if fr5_ia is not None:
                _sincronizar_con_fr5("modo IA")
            politica.reiniciar([float(datos.qpos[a]) for a in qadr_ia])

            while modo_ia_real and not salir:
                t_inicio_paso = time.perf_counter()

                # — teclas y pedal —
                key = cv2.waitKey(1) & 0xFF
                _ev = pedal_ia.tecla(key) or pedal_ia.actualizar()
                if _ev == "armar" and not en_marcha:
                    if fr5_ia is not None:
                        _sincronizar_con_fr5()           # parte EXACTAMENTE de la postura real
                    politica.reiniciar([float(datos.qpos[a]) for a in qadr_ia])
                    if fr5_ia is not None:
                        fr5_ia.publicar(datos.ctrl[idx_ctrl_brazo])
                        fr5_ia.habilitar(True)
                        fr5_ia_enviando = True
                    en_marcha = True
                    print("▶ Modo IA: EN MARCHA")
                elif _ev == "desarmar" and en_marcha:
                    if fr5_ia is not None:
                        fr5_ia.habilitar(False)
                        fr5_ia_enviando = False
                    en_marcha = False
                    print("⏸ Modo IA: PARADO")
                if key == ord("s"):
                    modo_ia_real = False
                    _salir_ia()
                    break
                if key == ord("q"):
                    salir = True
                    _salir_ia()
                    break
                if not visor.is_running():
                    salir = True
                    _salir_ia()
                    break

                if en_marcha:
                    # — 1. Predecir + ensamblado temporal —
                    estado_ia = [float(datos.qpos[a]) for a in qadr_ia]
                    accion = np.clip(politica.paso(estado_ia), lim_ia[:, 0], lim_ia[:, 1])
                    ctrl_ant = np.array([float(datos.ctrl[c]) for c in cidx_ia])
                    # — 2. Seguridad del brazo: velocidad limitada y sin colisiones —
                    if con_brazo:
                        accion[es_brazo] = ctrl_ant[es_brazo] + np.clip(
                            accion[es_brazo] - ctrl_ant[es_brazo], -paso_max_brazo, paso_max_brazo)
                        if col_ia is not None:
                            _q_brazo = np.array([float(datos.ctrl[i]) for i in idx_ctrl_brazo])
                            for _n, _v in zip(np.array(_acc_ia)[es_brazo], accion[es_brazo]):
                                _q_brazo[ARM_JOINTS.index(_n)] = _v
                            if not col_ia.libre(_q_brazo):
                                accion[es_brazo] = ctrl_ant[es_brazo]      # no se mueve
                                n_bloqueos += 1
                    # — 3. Física a tiempo real, con el comando interpolado —
                    for _k in range(1, n_sub + 1):
                        _c = ctrl_ant + (accion - ctrl_ant) * (_k / n_sub)
                        for _j, _ci in enumerate(cidx_ia):
                            datos.ctrl[int(_ci)] = float(_c[_j])
                        mujoco.mj_step(modelo, datos)
                        if fr5_ia_enviando:
                            fr5_ia.publicar(datos.ctrl[idx_ctrl_brazo])
                    if usar_hardware_real:
                        objetivo_hw = limitar_salto_hardware(construir_hardware_target(datos),
                                                             ultimo_hardware_target)
                        mano_real.enviar_posiciones(objetivo_hw)
                        ultimo_hardware_target = objetivo_hw
                    # — red de seguridad: la simulación se ha desestabilizado —
                    _q_chk = np.array([float(datos.qpos[a]) for a in qadr_ia])
                    if not np.all(np.isfinite(_q_chk)) or np.max(np.abs(_q_chk)) > 10.0:
                        print("⚠️  Modo IA: la simulación se ha desestabilizado → STANDBY.")
                        modo_ia_real = False
                        for _i, _v in enumerate(POSICION_STANDBY):
                            ctrl_mano[_i] = float(_v)
                        mujoco.mj_forward(modelo, datos)
                        _salir_ia()
                        break
                elif fr5_ia is not None:
                    _sincronizar_con_fr5()                 # parado: el simulador = el robot real
                visor.sync()
                _dibujar_logo_mujoco(visor, _logo_mj_px, _logo_mj_w, _logo_mj_h)

                # --- HUD ---
                panel = np.zeros((175, 460, 3), dtype=np.uint8)
                titulo_hud = f"MODO IA (ACT) - {'BRAZO + MANO' if con_brazo else 'MANO'}"
                if fr5_ia is not None:
                    titulo_hud += "  [FR5 REAL]"
                if usar_hardware_real:
                    titulo_hud += "  [MANO REAL]"
                cv2.putText(panel, titulo_hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                            (0, 100, 255) if (fr5_ia is not None or usar_hardware_real) else (0, 255, 255), 2)
                cv2.putText(panel, "EN MARCHA" if en_marcha else "PARADO - pisa el PEDAL", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0) if en_marcha else (0, 165, 255), 2)
                cv2.putText(panel, f"Ritmo {dt_objetivo * 1000:.0f} ms | bloque {red_neuronal['chunk']} | "
                                    f"historial {red_neuronal['ventana']}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                if n_bloqueos:
                    cv2.putText(panel, f"Brazo frenado por colision: {n_bloqueos} veces", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
                cv2.putText(panel, "[P/pedal] marcha/paro  |  [S] Salir al teleop  |  [Q] Cerrar app",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                _superponer_logo(panel, _logo_bgr_sm, _logo_alpha_sm, margen=5, posicion="bottom-right")
                cv2.imshow("Modo IA Real", panel)

                # — ritmo real: cada fotograma dura lo mismo que al grabar —
                elapsed = time.perf_counter() - t_inicio_paso
                if elapsed < dt_objetivo:
                    time.sleep(dt_objetivo - elapsed)

        # ══════════════════════════════════════════════════
        # ESTADO A3 — DEMO AUTOMÁTICA DEL TALADRO [5]
        # Sin cámara: el brazo coge el taladro, lo lleva al techo, aprieta el
        # gatillo y hace un agujero. La física avanza aquí al ritmo del reloj real.
        # ══════════════════════════════════════════════════
        elif modo_demo_taladro:
            cv2.destroyAllWindows()
            NOMBRE_DEMO = "Demo Taladro"
            cv2.namedWindow(NOMBRE_DEMO, cv2.WINDOW_AUTOSIZE)
            if taller.disponible and taller.visible:
                taller.ocultar(modelo, datos)               # un escenario cada vez
            if not demo_taladro.iniciar(modelo, datos, visor):
                modo_demo_taladro = False
                cv2.destroyWindow(NOMBRE_DEMO)
                continue
            ts_demo  = float(modelo.opt.timestep)
            deuda    = 0.0
            t_ant    = time.perf_counter()
            periodo  = 1.0 / FISICA_HZ
            while modo_demo_taladro and not salir:
                t0 = time.perf_counter()
                deuda = min(deuda + (t0 - t_ant), FISICA_RETRASO_MAX)
                t_ant = t0
                n_pasos = int(deuda / ts_demo)
                for _ in range(n_pasos):
                    demo_taladro.paso(modelo, datos, ts_demo)   # ctrl de brazo y mano
                    mujoco.mj_step(modelo, datos)
                deuda -= n_pasos * ts_demo

                if visor.is_running():
                    visor.sync()
                    _dibujar_logo_mujoco(visor, _logo_mj_px, _logo_mj_w, _logo_mj_h)
                else:
                    salir = True

                # --- HUD ---
                panel = np.zeros((200, 470, 3), dtype=np.uint8)
                cv2.putText(panel, "DEMO TALADRO (brazo + OrcaHand derecha)", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(panel, f"Fase: {demo_taladro.fase}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                rpm = demo_taladro.omega * 60.0 / (2 * math.pi)
                cv2.putText(panel, f"Motor: {'ON ' if demo_taladro.motor else 'OFF'}  ({rpm:4.0f} rpm)",
                            (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0) if demo_taladro.motor else (160, 160, 160), 1)
                cv2.putText(panel, f"Indice (gatillo): {datos.ctrl[OFFSET_MANO + 11]:.2f} rad",
                            (10, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                if demo_taladro.error:
                    cv2.putText(panel, demo_taladro.error, (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                                0.42, (0, 0, 255), 1)
                cv2.putText(panel, "Agujero:", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 200), 1)
                cv2.rectangle(panel, (85, 124), (285, 140), (90, 90, 90), 1)
                cv2.rectangle(panel, (85, 124), (85 + int(200 * demo_taladro.progreso), 140),
                              (0, 165, 255), -1)
                txt_teclas = ("[5] Repetir  |  [S] Volver al teleop  |  [Q] Cerrar"
                              if demo_taladro.terminada else "[S] Cancelar y volver al teleop  |  [Q] Cerrar")
                cv2.putText(panel, txt_teclas, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                _superponer_logo(panel, _logo_bgr_sm, _logo_alpha_sm, margen=5, posicion="bottom-right")
                cv2.imshow(NOMBRE_DEMO, panel)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    demo_taladro.terminar(modelo, datos)
                    modo_demo_taladro = False
                    cv2.destroyWindow(NOMBRE_DEMO)
                elif key == ord("5") and demo_taladro.terminada:
                    demo_taladro.iniciar(modelo, datos, visor)
                    t_ant = time.perf_counter()
                elif key == ord("q"):
                    salir = True

                espera = periodo - (time.perf_counter() - t0)
                if espera > 0:
                    time.sleep(espera)

        # ══════════════════════════════════════════════════
        # ESTADO B — TELEOPERACIÓN HUMANA (cámara activa)
        # ══════════════════════════════════════════════════
        else:
            if sin_camaras:
                print("🎚️  Gemelo con sliders SIN cámaras: dedos con el guante MANUS, brazo con los "
                      "sliders ([A] activa la mano). Para abrirlas: SLIDERS_SIN_CAMARAS = False")
            else:
                print("⚡ Conectando OAK-D (puede tardar ~2 s)...")

            with (nullcontext() if sin_camaras else
                  abrir_dispositivo(pipeline, CAM_A_MXID, excluir=(CAM_B_MXID,))) as device, \
                    (CamaraSecundaria(excluir=(device.getMxId(),))
                     if MULTICAMARA_ACTIVA and device is not None else nullcontext()) as cam_b:
                # (estas no dependen de la cámara)
                buf_hd  = deque(maxlen=6)
                t_ult_print_pinza = 0.0
                origen_mano = "-"
                buf_depth = deque(maxlen=6)   # paquetes de profundidad recientes
                _fps_n, _fps_t0, _fps_val = 0, time.perf_counter(), 0.0
                if device is None:              # 13-nov: GEMELO_SLIDERS sin cámaras
                    q_color = q_hd = q_depth = None
                    usar_hd  = False
                    f_px_cam = F_PX
                    _t_frame_sc = time.perf_counter()   # ritmo del bucle sin cámara
                else:
                    # ── FLUIDEZ: ¿USB3? Con color + recorte HD + profundidad a 30 FPS se
                    # mandan ~108 MB/s (~860 Mbit/s); por USB2 (máx. 480 Mbit/s) no caben
                    # y los fotogramas llegan tarde o se pierden. ──
                    _usb = comprobar_usb(device, "A")
                    if _usb is not None and _usb not in (dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS):
                        print("   Mientras tanto, pon MANO_RECORTE_HD = False.")
                    # ── PROYECTOR IR DE PUNTOS (OAK-D Pro) ────────────────────────
                    # Añade textura donde la piel/manga no tiene ninguna: es la causa
                    # nº1 de que el estéreo del BRAZO (fino, poca textura, 1.5-2 m)
                    # dé profundidades disparatadas. Sin efecto en el color/RGB.
                    activar_ir(device, "A", CAM_A_IR_MA)

                    q_color = device.getOutputQueue("color", maxSize=1, blocking=False)
                    usar_hd = MANO_RECORTE and MANO_RECORTE_HD and CAM_FOV_COMPLETO
                    q_hd    = device.getOutputQueue("color_hd", maxSize=4, blocking=False) \
                              if usar_hd else None
                    # maxSize=4: guardamos varios mapas de profundidad para poder elegir
                    # el que corresponde EN EL TIEMPO al fotograma de color (ver bucle).
                    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

                    # Focal REAL de la cámara para este tamaño de fotograma (viene de la
                    # calibración de fábrica). Si falla, se usa F_PX de config.py.
                    try:
                        _K = device.readCalibration().getCameraIntrinsics(
                            dai.CameraBoardSocket.CAM_A, ANCHO_CAM, ALTO_CAM)
                        f_px_cam = float(_K[0][0])
                        print(f"📷 Focal calibrada OAK-D: {f_px_cam:.1f} px "
                              f"(fotograma {ANCHO_CAM}x{ALTO_CAM})")
                    except Exception as e_cal:
                        f_px_cam = F_PX
                        print(f"⚠️  No se pudo leer la calibración ({e_cal}); uso F_PX={F_PX}")

                # ── DOS CÁMARAS: extrínsecos A↔B y fusor del brazo ──
                fusor_b = None
                ent_b   = None
                if cam_b is not None:
                    _ext = cargar_extrinsecas(device.getMxId(), cam_b.mxid)
                    if _ext is not None:
                        fusor_b = FusorBrazo(_ext[0], _ext[1], BRAZO_HUMANO)
                        print(f"🔀 Fusión A+B activa (calibración {_ext[2].get('metodo', '?')}, "
                              f"{_ext[2].get('n_muestras', '?')} muestras)")
                    else:
                        print("⚠️  Cámara B abierta SIN extrínsecos: se muestra, pero NO se fusiona. "
                              "Ejecuta  python calibrar_extrinseca.py")

                # ── TELEOP FR5 v2 (27-oct): muñeca + codo con las dos cámaras → IK de 6 ejes ──
                # skill_shield/teleop_fr5. Sustituye a los PASOS 2-7 del brazo (que quedan
                # desactivados con TELEOP_FR5_V2 = True): fusión robusta a la palma delante de
                # una cámara, límites humanos reales, IK analítica (8 configuraciones) y cambio
                # de configuración planificado cuando la actual no puede seguir.
                teleop_v2 = None
                # (GEMELO_SLIDERS: el brazo NO sale de las cámaras → ni v2 ni pipeline antiguo)
                if globals().get("TELEOP_FR5_V2", False) and MODO_BRAZO == "imitacion" and sliders is None:
                    from skill_shield.teleop_fr5.teleop_brazo import TeleopBrazoFR5, ConfigTeleop
                    from skill_shield.teleop_fr5.fusion_muneca_codo import (
                        Camara as _CamV2, Intrinsecos as _IntrV2, ObservacionCamara as _ObsV2,
                        Parametros as _ParFusionV2)
                    from skill_shield.teleop_fr5.retargeting import (
                        ConfigRetargeting as _CfgRetV2, LimitesAnatomicos as _LimAnatV2)
                    from skill_shield.teleop_fr5.ik_teleop import (
                        ParametrosIK as _ParIKV2, Entorno as _EntornoV2, entorno_soporte_hombros)
                    try:
                        intr_v2_a = _IntrV2.desde_K(_K, ANCHO_CAM, ALTO_CAM)
                    except NameError:
                        intr_v2_a = _IntrV2.desde_focal(f_px_cam, ANCHO_CAM, ALTO_CAM)
                    _cams_v2 = {"A": _CamV2("A")}
                    intr_v2_b = None
                    _auto_ext_v2 = bool(globals().get("TELEOP_FR5_AUTO_EXTRINSECA", True))
                    if cam_b is not None and (fusor_b is not None or _auto_ext_v2):
                        # 28-oct: con la extrínseca automática, B entra aunque no haya
                        # camaras_extrinsecas.json (o no corresponda): se calcula con tu cuerpo al pulsar [A]
                        if fusor_b is not None:
                            _cams_v2["B"] = _CamV2("B", _ext[0], _ext[1])
                        else:
                            _cams_v2["B"] = _CamV2("B")
                            print("📐 Cámara B sin extrínseca válida: se calculará sola con tu cuerpo al pulsar [A]")
                        intr_v2_b = (_IntrV2.desde_K(cam_b.K, ANCHO_CAM, ALTO_CAM) if cam_b.K is not None
                                     else _IntrV2.desde_focal(cam_b.f_px, ANCHO_CAM, ALTO_CAM))
                    _mxid_a_v2 = device.getMxId()
                    _mxid_b_v2 = cam_b.mxid if cam_b is not None else None

                    def _guardar_ext_v2(_R, _t, _inf):
                        from skill_shield.teleop_fr5.autocalib import guardar_extrinsecas
                        from skill_shield.vision.multicamara import _ruta_extrinsecas
                        _r = guardar_extrinsecas(_ruta_extrinsecas(), _R, _t, _inf, _mxid_a_v2, _mxid_b_v2)
                        print(f"💾 Extrínseca A↔B guardada en {os.path.basename(_r)} (la anterior, en .bak_*)")
                    if globals().get("SOPORTE_HOMBROS", False):
                        _ent_v2 = entorno_soporte_hombros(SOPORTE_SUELO_Z)
                        _z_min_v2 = SOPORTE_SUELO_Z + 0.25
                    else:
                        _ent_v2 = _EntornoV2(z_suelo=0.0)
                        _z_min_v2 = ARM_Z_MINIMA
                    if globals().get("TELEOP_FR5_Z_MIN") is not None:
                        _z_min_v2 = TELEOP_FR5_Z_MIN
                    # 5-nov: mesa, peana, cámaras y zonas prohibidas (skill_shield/teleop_fr5/obstaculos.py)
                    from skill_shield.teleop_fr5.obstaculos import cajas_desde_config, dibujar_en_visor
                    _ent_v2.cajas = cajas_desde_config(globals(), _ent_v2.z_suelo)
                    if globals().get("TELEOP_FR5_OBSTACULOS_VER", True) and not globals().get("_obst_dibujados"):
                        with visor.lock():
                            dibujar_en_visor(visor, _ent_v2)
                        globals()["_obst_dibujados"] = True
                    teleop_v2 = TeleopBrazoFR5(
                        RUTA_XML, _cams_v2,
                        cfg=ConfigTeleop(brazo=BRAZO_HUMANO, longitudes_manual=LONG_BRAZO_MANUAL,
                                         n_calibracion=CALIB_HUESOS_FRAMES,
                                         ik=str(globals().get("TELEOP_FR5_IK", "diferencial")),
                                         auto_extrinseca=_auto_ext_v2,
                                         vigilar_extrinseca=bool(globals().get("TELEOP_FR5_VIGILAR_EXTRINSECA", True)),
                                         vertical_auto=bool(globals().get("TELEOP_FR5_VERTICAL_AUTO", True)),
                                         # 5-nov: calibrar no mueve el robot; se arma con el pedal
                                         requiere_armado=TELEOP_FR5_PEDAL is not None,
                                         v_entrada=globals().get("TELEOP_FR5_V_ENTRADA", 0.15),
                                         entrada_fin_rad=float(globals().get("TELEOP_FR5_ENTRADA_FIN_RAD", 0.05))),
                        al_calibrar_extrinseca=_guardar_ext_v2,
                        par_fusion=_ParFusionV2(vis_min=POSE_VISIBILIDAD_MIN, z_min=PROF_Z_MIN, z_max=PROF_Z_MAX,
                                                prof_radio_px=PROF_RADIO_PX, prof_percentil=PROF_PERCENTIL,
                                                prof_min_pix=PROF_MIN_PIXELES,
                                                escorzo_ang=tuple(globals().get("TELEOP_FR5_ESCORZO_DEG", (20.0, 40.0))),
                                                hold_s=float(globals().get("TELEOP_FR5_HOLD_S", 0.8)),
                                                hombro_hacia_torso=HOMBRO_PROF_HACIA_TORSO,
                                                arriba_ref=str(globals().get("TELEOP_FR5_ARRIBA_REF", ARRIBA_REF)),
                                                triangular=bool(globals().get("TELEOP_FR5_TRIANGULAR", True))),
                        cfg_retarget=_CfgRetV2(escala=globals().get("TELEOP_FR5_ESCALA"),
                                               fraccion_alcance=float(globals().get("TELEOP_FR5_FRACCION", 0.85)),
                                               z_min=_z_min_v2, prono_escala_z=PRONO_Z_ESCALA,
                                               roll_rango_deg=float(globals().get("TELEOP_FR5_ROLL_RANGO_DEG", 120.0)),
                                               alcance_max=float(globals().get("TELEOP_FR5_ALCANCE_MAX", 0.88))),
                        lim_anatomicos=_LimAnatV2(hombro_az_min=HOMBRO_AZ_MIN, hombro_az_max=HOMBRO_AZ_MAX,
                                                  hombro_ext_elev=HOMBRO_EXT_ELEV,
                                                  hombro_az_libre_elev=HOMBRO_AZ_LIBRE_ELEV,
                                                  codo_flex_max=CODO_FLEX_MAX, humero_rot_ext=HUMERO_ROT_EXT,
                                                  humero_rot_int=HUMERO_ROT_INT, activos=LIMITES_HUMANOS),
                        par_ik=_ParIKV2(limites_extra=dict(TELEOP_FR5_LIMITES),
                                        v_seguimiento=float(globals().get("TELEOP_FR5_V_MAX", 3.0)),
                                        postura_humana=bool(globals().get("TELEOP_FR5_POSTURA_HUMANA", True)),
                                        codo_signo=globals().get("TELEOP_FR5_CODO_SIGNO"),
                                        muneca_q4_deg=float(globals().get("TELEOP_FR5_MUNECA_DEG", (80.0, 60.0))[0]),
                                        muneca_q5_deg=float(globals().get("TELEOP_FR5_MUNECA_DEG", (80.0, 60.0))[1])),
                        entorno=_ent_v2)
                    teleop_v2.cadena.comparar_con_dh()
                    print(f"🦾 TELEOP FR5 v2 activa: cámaras {list(_cams_v2)}, alcance robot "
                          f"{teleop_v2.retarget.alcance_robot:.2f} m, suelo {_ent_v2.z_suelo}")
                t_ult_print_v2 = 0.0
                t_prev_v2 = None
                # 5-nov: pedal para ARMAR el brazo tras calibrar (skill_shield/teleop_fr5/pedal.py)
                pedal_v2 = None
                if teleop_v2 is not None:
                    from skill_shield.teleop_fr5.pedal import Pedal
                    pedal_v2 = Pedal.desde_config(globals())
                    print(f"🦶 Armado del brazo: {pedal_v2.descripcion}; entrada a "
                          f"{globals().get('TELEOP_FR5_V_ENTRADA', 0.15)} rad/s")
                # 28-oct (fluidez): la v2 en un hilo aparte, solapada con la cámara y MediaPipe
                teleop_v2_par = None
                if teleop_v2 is not None and globals().get("TELEOP_FR5_EN_PARALELO", True):
                    from skill_shield.teleop_fr5.teleop_brazo import TeleopEnParalelo
                    teleop_v2_par = TeleopEnParalelo(teleop_v2)
                registro_v2 = None    # 28-oct: [R] registro quieto (skill_shield/teleop_fr5/diagnostico.py)

                angulos_brazo    = None   # últimos ángulos válidos del brazo humano
                t_ult_ang_valido = 0.0    # perf_counter() del último frame con _ang != None
                t_ult_print_ang  = 0.0
                est_brazo        = EstimadorLongitudesBrazo(LONG_BRAZO_MANUAL)
                suav_brazo       = SuavizadorND(n_dim=3, ventana=IMITAR_SUAVIZADO)
                brazo_imit_activo = False   # True cuando el brazo ya está imitando
                sujetar_pose_A   = False    # 9-oct: brazo quieto en POSE_BRAZO_ACTIVAR hasta imitar
                refs_tpose       = []       # (az, elev, codo) medidos durante la calibración
                tpose_med        = None     # mediana de refs_tpose (None = sin corrección)
                tpose_ref        = tuple(T_POSE_HUMANA)   # T-pose usada como referencia
                q_ref_sim        = None     # pose de reposo del sim (joint1-3) al empezar a imitar
                q_imit           = None     # último objetivo enviado a joint1..3 (rad)
                hib              = None     # datos del híbrido (alcance y ejes del ARCTOS)
                q_ik_prev        = None     # última solución de la IK (continuidad)
                peso_hib         = HIBRIDO_PESO
                suav_obj         = SuavizadorND(n_dim=3, ventana=HIBRIDO_SUAVIZADO)
                suav_codo        = SuavizadorND(n_dim=3, ventana=HIBRIDO_SUAVIZADO)
                ik_info          = None     # (obj. codo, obj. muñeca, peso codo) para el debug
                oe_brazo         = FiltroOneEuro()   # PASO 5: dirección del brazo
                oe_antebrazo     = FiltroOneEuro()   # PASO 5: dirección del antebrazo
                t_atasco         = None     # PASO 6: desde cuándo la IK está atascada
                t_ult_rescate    = 0.0      # PASO 6: último cambio de solución (espera entre cambios)
                lat_local        = None     # índice→meñique de la mano del robot (ejes de la mano)
                lat_cal          = []       # muestras para calibrarlo en la T-pose
                mano_girada      = False    # PASO 6: robot en configuración "girada"
                t_prev_imit      = time.perf_counter()
                t_ult_print_q    = 0.0
                # ── Paso 5: pronosupinación ──
                prono_idx    = ARM_JOINTS.index(PRONO_JOINT)   # 5 = joint6 (ctrl[5])
                roll_ref     = None      # giro neutro medido en la T-pose
                refs_roll    = []
                q_prono_ref  = float(POSICION_STANDBY_BRAZO[prono_idx])
                q_prono      = q_prono_ref
                d_prev       = 0.0
                t_prev_prono = time.perf_counter()
                t_ult_print_roll = 0.0
                estado["antebrazo"] = None
                estado["arriba"]    = None
                estado["eje_codo"]  = None

                # 5-nov: el generador (y la IK) arrancan de la postura REAL del FR5
                if fr5 is not None:
                    _sincronizar_con_fr5("entrar en teleoperación")
                    # 9-nov: avisar si con esta configuración el FR5 real no se va a poder armar
                    if sliders is None and fr5.conectado and (teleop_v2 is None or pedal_v2 is None
                                                               or pedal_v2.modo is None):
                        print("⚠ FR5 real conectado, pero NO se armará: hace falta TELEOP_FR5_V2 = True "
                              "y un pedal (TELEOP_FR5_PEDAL = \"tecla\" o \"evdev\") en config.py.")
                    elif sliders is None and fr5.conectado:
                        print(f"🦾 FR5 REAL listo para teleoperar"
                              f"{' (SIMULACRO: no se mueve)' if fr5.p.get('FR5_SIMULACRO') else ''}: "
                              f"[A] + T-pose para calibrar → {pedal_v2.descripcion} para ARMAR / DESARMAR.")
                fr5_enviando = False          # ¿el FR5 real está recibiendo ServoJ?

                # ── PASO 7 — brazo preciso y sin saltos ──
                # 7.1: el generador suave escribe ctrl de joint1..3 y del giro de muñeca
                #      antes de cada mj_step; el bucle solo le PUBLICA objetivos.
                gen_brazo = None
                if sliders is not None:
                    # GEMELO_SLIDERS: los 6 joints con velocidad moderada; la ventana la escala
                    # (gen_brazo.factor = % de velocidad de los sliders)
                    _v_sl = float(globals().get("SLIDERS_SUAVE_VEL", 0.6))
                    _a_sl = float(globals().get("SLIDERS_SUAVE_ACEL", 3.0))
                    gen_brazo = GeneradorSuave((0, 1, 2, 3, 4, 5), (_v_sl,) * 6, (_a_sl,) * 6)
                    gen_brazo.activar(datos)
                    print(f"🎚️  Brazo con SLIDERS: vel. máx {_v_sl} rad/s, acel. máx {_a_sl} rad/s² "
                          "(× el % de velocidad de la ventana)")
                elif SUAVE_ACTIVO and MODO_BRAZO == "imitacion":
                    # 9-oct: también joint4/joint5 (la pose de [A] los mueve): sin tirones
                    _extra_suave = [_i for _i in (3, 4) if _i != prono_idx]
                    if teleop_v2 is not None:
                        # v2: los 6 joints con la misma velocidad (el FR5 admite 3,15 rad/s)
                        _v_v2 = float(globals().get("TELEOP_FR5_SUAVE_VEL", 2.5))
                        _a_v2 = float(globals().get("TELEOP_FR5_SUAVE_ACEL", 20.0))
                        gen_brazo = GeneradorSuave((0, 1, 2, 3, 4, 5), (_v_v2,) * 6, (_a_v2,) * 6)
                    else:
                        gen_brazo = GeneradorSuave(
                        (0, 1, 2, prono_idx, *_extra_suave),
                        tuple(SUAVE_VEL_MAX)
                        + (float(globals().get("SUAVE_VEL_MUNECA_ARCTOS", 1.5)),) * len(_extra_suave),
                        tuple(SUAVE_ACEL_MAX)
                        + (float(globals().get("SUAVE_ACEL_MUNECA_ARCTOS", 15.0)),) * len(_extra_suave))
                    gen_brazo.activar(datos)
                    print(f"🧈 Comando continuo del brazo activo (vel. máx {SUAVE_VEL_MAX} rad/s, "
                          f"acel. máx {SUAVE_ACEL_MAX} rad/s²)")

                def _enviar_brazo(indices, valores, de_sliders=False):
                    """Objetivos del brazo: al generador suave (si está activo) o directos.
                    GEMELO_SLIDERS: solo cuentan los de la ventana de sliders; el resto (cámaras,
                    [A], [S], joint5 fijo...) se ignora para que nada más mueva el brazo."""
                    if sliders is not None and not de_sliders:
                        return
                    if gen_brazo is not None and gen_brazo.activo:
                        gen_brazo.enviar(datos, indices, valores)
                    else:
                        for _i_b, _v_b in zip(indices, valores):
                            datos.ctrl[_i_b] = float(_v_b)

                # 7.2: filtro de saltos + zona muerta; 7.3: modo fino / congelar
                filtro_saltos = FiltroSaltos()
                zm_palma      = ZonaMuerta(ZONA_MUERTA_PALMA)
                zm_codo       = ZonaMuerta(ZONA_MUERTA_PALMA)
                canal_palma   = CanalFino(FINO_ESCALA, vel_salida=FINO_SALIDA_VEL)
                canal_giro    = CanalFino(FINO_ESCALA_GIRO, angular=True, vel_salida=1.0)
                dirs_ok       = None     # últimas direcciones (brazo, antebrazo) aceptadas
                brazo_fino, brazo_congelado = False, False
                q6_obj_prev   = None     # último objetivo fiable del giro (modo mundo)
                lim_humanos   = LimitesHumanos()   # 7-oct: tu postura dentro del rango anatómico

                # — Física en tiempo real en su propio hilo (ver HiloFisica) —
                # 9-nov: GEMELO DIGITAL de verdad. Con el FR5 conectado, el brazo de MuJoCo enseña
                # la pose REAL leída del robot (fr5_servo la lee a 50 Hz), no el comando: se
                # escribe en qpos (con velocidad 0) antes de cada mj_step. El comando (ctrl) y la
                # IK no cambian; el OBJETIVO se ve en el brazo fantasma.
                _pose_real = {"t": -1.0, "q": None}

                def _brazo_a_pose_real(d_):
                    if fr5 is None or not globals().get("GEMELO_POSE_REAL", True):
                        return
                    if d_.time - _pose_real["t"] >= 0.01:        # leerla a ~100 Hz basta
                        _pose_real["t"] = d_.time
                        _pose_real["q"] = fr5.q_real_rad(max_edad_s=0.3)
                    _q_r = _pose_real["q"]
                    if _q_r is None:                             # sin lectura reciente: simulado
                        return
                    for _k in range(6):
                        d_.qpos[qpos_idx_brazo[_k]] = float(_q_r[_k])
                        d_.qvel[dof_idx_brazo[_k]] = 0.0

                def _antes_de_paso(m_, d_, dt_):
                    """Antes de cada mj_step: brazo continuo (PASO 7.1) + dedos del guante."""
                    if gen_brazo is not None:
                        gen_brazo.paso(m_, d_, dt_)
                    if guante is not None:
                        guante.escribir_ctrl(d_, dt_)
                    if fr5 is not None:                  # FR5 real: último comando del brazo
                        fr5.publicar(d_.ctrl[_idx_ctrl_fr5])
                        _brazo_a_pose_real(d_)           # 9-nov: el gemelo enseña el robot REAL
                    grabador.muestrear(d_)               # 5-nov: grabación a ritmo fijo

                # ctrl de joint1..6 (act_joint1..6 del MJCF)
                _idx_ctrl_fr5 = [mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{_j}")
                                 for _j in ARM_JOINTS]
                if min(_idx_ctrl_fr5) < 0:
                    _idx_ctrl_fr5 = list(range(6))

                fisica  = HiloFisica(modelo, datos, visor, antes_de_paso=_antes_de_paso) \
                          if FISICA_EN_HILO else None     # (5-nov: siempre: la grabadora va ahí)
                _soltar = fisica.soltar if fisica is not None else nullcontext
                if fisica is not None:
                    fisica.empezar()
                    print(f"⚙️  Física en tiempo real: hilo propio a {FISICA_HZ} Hz "
                          f"(paso {modelo.opt.timestep * 1000:.1f} ms)")
                if fr5 is not None and fisica is None:
                    print("⚠ FR5: FISICA_EN_HILO = False → no se publican comandos; "
                          "el FR5 se quedará quieto")
                # (el ServoJ al FR5 se habilita en el bucle: con la v2, solo con el brazo ARMADO)

                # — FLUIDEZ: MediaPipe Hands en su propio hilo (MANO_EN_PARALELO) y perfil —
                pool_mano    = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Hands") \
                               if (MANO_RECORTE and MANO_EN_PARALELO) else None
                pose_lm_prev = None      # Pose del fotograma anterior → recorte de la mano
                perfil       = PerfilBucle(fisica, PERFIL_BUCLE_CADA_S) if PERFIL_BUCLE else None
                if pool_mano is not None:
                    print("🧵 MediaPipe Hands en paralelo con Pose (MANO_EN_PARALELO)")

                while not modo_ia_real and not modo_demo_taladro and not salir:
                    if perfil is not None:
                        perfil.inicio()
                    # — Captura de frames (COLOR y PROFUNDIDAD SINCRONIZADOS) —
                    # Antes se hacía q_color.get() y q_depth.get() por separado: como
                    # son colas independientes, el mapa de profundidad podía ir 1-2
                    # fotogramas desfasado respecto al color. Con el brazo en
                    # movimiento eso significa leer la profundidad DEL FONDO en el
                    # píxel de la muñeca → z descartada → "brazo perdido".
                    if device is None:
                        # 13-nov: SIN CÁMARAS (GEMELO_SLIDERS): fotograma negro solo para el HUD y el
                        # teclado, a ~30 vueltas/s. La espera suelta la física (sigue en su hilo).
                        with _soltar():
                            time.sleep(max(0.0, _t_frame_sc + 1.0 / 30.0 - time.perf_counter()))
                        _t_frame_sc = time.perf_counter()
                        frame_bgr   = np.zeros((ALTO_CAM, ANCHO_CAM, 3), np.uint8)
                        frame_depth = frame_hd = _t_color = None
                    else:
                        with _soltar():                   # esperar a la cámara: la física sigue
                            _pkt_color = q_color.get()
                        frame_bgr  = _pkt_color.getCvFrame()
                        _t_color   = _pkt_color.getTimestamp()

                        while True:                       # vaciar lo que haya llegado
                            _p = q_depth.tryGet()
                            if _p is None:
                                break
                            buf_depth.append(_p)
                        if not buf_depth:                 # primer fotograma: esperar
                            buf_depth.append(q_depth.get())

                        _pkt_depth  = min(
                            buf_depth,
                            key=lambda p: abs((p.getTimestamp() - _t_color).total_seconds()))
                        frame_depth = _pkt_depth.getFrame()

                        # Imagen HD para el recorte de la mano (la más cercana en el tiempo)
                        frame_hd = None
                        if q_hd is not None:
                            while True:
                                _p = q_hd.tryGet()
                                if _p is None:
                                    break
                                buf_hd.append(_p)
                            if buf_hd:
                                _pkt_hd = min(
                                    buf_hd,
                                    key=lambda p: abs((p.getTimestamp() - _t_color).total_seconds()))
                                if abs((_pkt_hd.getTimestamp() - _t_color).total_seconds()) < 0.05:
                                    frame_hd = _pkt_hd.getCvFrame()

                    if perfil is not None:
                        perfil.marca("cámara")

                    # — Medidor de FPS reales del bucle —
                    _fps_n += 1
                    if _fps_n >= 15:
                        _ahora  = time.perf_counter()
                        _fps_val = _fps_n / max(1e-6, _ahora - _fps_t0)
                        _fps_n, _fps_t0 = 0, _ahora

                    frame_rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_rgb.flags.writeable = False   # evita una copia en MediaPipe

                    # — FLUIDEZ: imagen LIMPIA para recortar la mano. frame_bgr se usa
                    #   después para DIBUJAR (esqueleto, textos): si el recorte sale de ahí,
                    #   Hands ve las líneas del esqueleto encima de tu mano. —
                    frame_limpio = None
                    if MANO_RECORTE:
                        frame_limpio = frame_hd if frame_hd is not None else frame_bgr.copy()

                    # — FLUIDEZ: Hands YA, en otro hilo, mientras aquí se hace Pose + brazo.
                    #   El recorte sale de la Pose del fotograma ANTERIOR. Se recoge abajo,
                    #   en "Detección de mano". —
                    fut_mano, caja_par = None, None
                    if pool_mano is not None and pose_lm_prev is not None:
                        caja_par = caja_recorte_mano(pose_lm_prev, BRAZO_HUMANO,
                                                     frame_limpio.shape[1], frame_limpio.shape[0])
                        if caja_par is not None:
                            fut_mano = pool_mano.submit(detectar_mano_en_recorte,
                                                        detector_mano_recorte, frame_limpio, caja_par)


                    # — DOS CÁMARAS: la Pose de B corre en su hilo mientras A hace la suya —
                    fut_cam_b = cam_b.lanzar(_t_color) if cam_b is not None else None

                    # — Detección de CUERPO (solo visual por ahora) —
                    # Se dibuja ANTES que la mano para que los dedos queden encima.
                    if device is None:                # sin cámaras: nadie a la vista
                        resultado_pose = _SIN_POSE
                    else:
                        with _soltar():               # MediaPipe Pose: la física sigue
                            resultado_pose = detector_pose.process(frame_rgb)
                    pose_lm_prev = resultado_pose.pose_landmarks   # FLUIDEZ: recorte del siguiente fotograma
                    ent_b = cam_b.recoger(fut_cam_b) if fut_cam_b is not None else None
                    if perfil is not None:
                        perfil.marca("pose")
                    if MOSTRAR_ESQUELETO_CUERPO and resultado_pose.pose_landmarks:
                        mp_draw.draw_landmarks(
                            frame_bgr, resultado_pose.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

                    # — Ángulos del brazo humano (paso 3: solo calcular y mostrar) —
                    # 28-oct: con TELEOP_FR5_V2 el brazo lo mueve teleop_fr5 y este pipeline antiguo
                    # (FusorBrazo, ángulos, PASOS 4-7) solo duplicaba trabajo en cada fotograma.
                    if resultado_pose.pose_world_landmarks and resultado_pose.pose_landmarks and sliders is None \
                            and (teleop_v2 is None or not globals().get("TELEOP_FR5_SIN_PIPELINE_ANTIGUO", True)):
                        _calib_antes = est_brazo.calibrando   # ¿estamos aún en la T-pose de calibración?
                        _ang_mp = calcular_angulos_brazo_humano(
                            resultado_pose.pose_world_landmarks, BRAZO_HUMANO)
                        if fusor_b is not None:
                            _ang_es = calcular_angulos_brazo_multicamara(
                                fusor_b,
                                (resultado_pose.pose_landmarks, resultado_pose.pose_world_landmarks,
                                 frame_depth, f_px_cam),
                                ent_b, BRAZO_HUMANO, est_brazo)
                        else:
                            _ang_es = calcular_angulos_brazo_estereo(
                                resultado_pose.pose_landmarks, frame_depth, f_px_cam, BRAZO_HUMANO,
                                estimador=est_brazo,
                                pose_world_lm=resultado_pose.pose_world_landmarks)
                        _ang = _ang_es if BRAZO_FUENTE_3D == "estereo" else _ang_mp
                        # FUSIÓN DE DIRECCIONES (28-sep): el estéreo daba codos de 86-106°
                        # con el brazo casi recto (MediaPipe: 12-36°). Cada segmento (brazo,
                        # antebrazo) usa la dirección del estéreo SOLO si se parece a la de
                        # MediaPipe; si se desvía más de FUSION_DIR_MAX_DEG, usa MediaPipe.
                        if BRAZO_FUENTE_3D == "estereo" and _ang_es is not None \
                                and _ang_mp is not None and FUSION_DIR_MAX_DEG > 0 \
                                and "dir_brazo" in _ang_es and "dir_brazo" in _ang_mp:
                            _ang_es["fusion"] = ""
                            for _k, _et in (("dir_brazo", "B"), ("dir_antebrazo", "A")):
                                # PASO 7.2: mezcla gradual (antes: cambio de golpe → salto)
                                _ang_es[_k], _w_es = fusionar_direccion(_ang_es[_k], _ang_mp[_k])
                                if _w_es < 0.99:
                                    _ang_es["fusion"] += f"{_et}{100 * (1 - _w_es):.0f}%"
                            if _ang_es["fusion"]:          # el codo, coherente con las direcciones
                                _ang_es["codo"] = math.degrees(math.acos(float(np.clip(
                                    _ang_es["dir_brazo"] @ _ang_es["dir_antebrazo"], -1.0, 1.0))))
                        if _calib_antes and _ang is not None:
                            refs_tpose.append((_ang["azimut"], _ang["elevacion"], _ang["codo"]))

                        if _ang is not None:
                            # Con el brazo casi vertical el azimut es ruido: se
                            # conserva el último valor fiable.
                            if not _ang["azimut_fiable"] and angulos_brazo is not None:
                                _ang["azimut"]        = angulos_brazo["azimut"]
                                _ang["azimut_muneca"] = angulos_brazo["azimut_muneca"]
                            angulos_brazo    = _ang
                            t_ult_ang_valido = time.perf_counter()

                            _col = (0, 255, 0) if _ang["azimut_fiable"] else (0, 200, 255)
                            _y0  = ALTO_CAM - 62
                            for _k, _txt in enumerate((
                                    f"Brazo {BRAZO_HUMANO} [{BRAZO_FUENTE_3D}] ref:{_ang['ref']}"
                                    + ("  CODO ESTIMADO" if _ang.get("codo_estimado") else ""),
                                    f"Azimut    {angulos_brazo['azimut']:7.1f}",
                                    f"Elevacion {_ang['elevacion']:7.1f}",
                                    f"Codo      {_ang['codo']:7.1f}")):
                                cv2.putText(frame_bgr, _txt, (10, _y0 + 15 * _k),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _col, 1)
                        else:
                            cv2.putText(frame_bgr,
                                        f"Brazo {BRAZO_HUMANO}: sin datos ({BRAZO_FUENTE_3D}) "
                                        + (est_brazo.motivo_fallo[:55] if BRAZO_FUENTE_3D == "estereo" else ""),
                                        (10, ALTO_CAM - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                        if DEBUG_ANGULOS_BRAZO and \
                           (time.time() - t_ult_print_ang) >= DEBUG_ANGULOS_CADA_S:
                            t_ult_print_ang = time.time()
                            _extra = ""
                            if _ang_es is not None:
                                _v = _ang_es["vis"]
                                _extra = (f"   vis(h/c/m)={_v[0]:.2f}/{_v[1]:.2f}/{_v[2]:.2f}"
                                          f"   prof[{_ang_es['fuentes']}]"
                                          + (f"  [MP en {_ang_es['fusion']}]" if _ang_es.get("fusion") else "")
                                          + ("  [CODO ESTIMADO]" if _ang_es["codo_estimado"] else ""))
                            else:
                                _extra = f"   [SIN DATOS ESTEREO: {est_brazo.motivo_fallo or '?'}]"
                            _lon = est_brazo.longitudes
                            _extra += (f"   huesos={_lon[0]:.2f}/{_lon[1]:.2f}m" if _lon else
                                       "   huesos=aprendiendo")
                            print(f"[ANGULOS BRAZO {BRAZO_HUMANO}] "
                                  f"ESTEREO: {_fmt_ang(_ang_es)}   |   "
                                  f"MEDIAPIPE: {_fmt_ang(_ang_mp)}{_extra}")

                        # ══ PASO 4 — el brazo ARCTOS IMITA los ángulos del brazo humano ══
                        # Solo con la cámara activa ([A]) y los huesos ya calibrados. joint4-6
                        # (giro de antebrazo y muñeca) se quedan en su valor de standby.
                        #
                        # _ang_uso: si el frame actual falló (depth ruidosa, oclusión de
                        # un instante) pero tenemos un ángulo válido de hace poco (<
                        # ANG_BRAZO_HOLD_S), lo seguimos usando en vez de congelar el brazo.
                        # Si ya ha pasado más tiempo que eso, se considera que el brazo se
                        # ha perdido de verdad y no se actualiza (mejor que "adivinar").
                        if _ang is not None:
                            _ang_uso = _ang
                        elif angulos_brazo is not None and \
                                (time.perf_counter() - t_ult_ang_valido) <= ANG_BRAZO_HOLD_S:
                            _ang_uso = angulos_brazo
                        else:
                            _ang_uso = None

                        if _ang_uso is not None and "antebrazo" in _ang_uso:
                            estado["antebrazo"] = _ang_uso["antebrazo"]
                            estado["arriba"]    = _ang_uso["arriba"]
                            # Eje del codo del ARCTOS: horizontal y perpendicular al plano
                            # vertical del brazo (el que fija joint1 con tu azimut). Con el
                            # brazo vertical el azimut ya es el último fiable → no degenera.
                            _az  = math.radians(_ang_uso["azimut"])
                            _dir = (_ang_uso["lado"] * math.sin(_az) * _ang_uso["derecha"]
                                    + math.cos(_az) * _ang_uso["delante"])
                            estado["eje_codo"] = _unitario(np.cross(_ang_uso["arriba"], _dir))
                            estado["az_brazo"] = _ang_uso["azimut"]
                            estado["torso"]    = (_ang_uso["derecha"], _ang_uso["arriba"],
                                                  _ang_uso["delante"], _ang_uso["lado"])

                        if MODO_BRAZO == "imitacion" and estado["camara_activa"] \
                                and est_brazo.calibrado and _ang_uso is not None \
                                and teleop_v2 is None:   # v2: el brazo lo mueve teleop_fr5 (más abajo)
                            _t = time.perf_counter()
                            if not brazo_imit_activo:
                                # T-pose medida durante la calibración → referencia de postura
                                tpose_med = None
                                if IMITAR_CALIBRAR_TPOSE and len(refs_tpose) >= 10:
                                    _r = np.median(np.array(refs_tpose, dtype=float), axis=0)
                                    if 45.0 < _r[1] < 135.0:
                                        tpose_med = (float(_r[0]), float(_r[1]), float(_r[2]))
                                        print(f"📐 T-pose medida: az={_r[0]:.1f}°  elev={_r[1]:.1f}°  "
                                              f"codo={_r[2]:.1f}°  (ideal {T_POSE_HUMANA[0]:.0f}/"
                                              f"{T_POSE_HUMANA[1]:.0f}/{T_POSE_HUMANA[2]:.0f}) "
                                              f"→ sesgo elev {_r[1] - T_POSE_HUMANA[1]:+.1f}°")
                                    else:
                                        print(f"⚠️  T-pose medida incoherente (elev={_r[1]:.1f}°): "
                                              "no se aplica corrección. Repite [S] y [A] con el brazo horizontal.")
                                tpose_ref = tpose_med if tpose_med is not None else tuple(T_POSE_HUMANA)
                                # Arranque suave: se parte de donde YA está el brazo
                                q_imit = np.array([datos.ctrl[0], datos.ctrl[1], datos.ctrl[2]])
                                q_ref_sim = q_imit.copy()   # el reposo del sim ↔ tu T-pose (modo relativo)
                                if IMITAR_REFERENCIA == "relativo":
                                    print(f"🔗 Referencia RELATIVA: reposo del sim q1={q_ref_sim[0]:.2f} "
                                          f"q2={q_ref_sim[1]:.2f} q3={q_ref_sim[2]:.2f} ↔ tu T-pose. "
                                          "El brazo no se mueve al calibrar.")
                                suav_brazo.reset()
                                # 9-oct: se suelta la pose de [A]: joint4/joint5 vuelven a los de la
                                # teleoperación, y la geometría del híbrido se calcula YA con ellos
                                # (preparar_hibrido usa los joint4..6 que tenga el brazo ahora).
                                _q45 = (float(POSICION_STANDBY_BRAZO[3]), float(JOINT5_FIJO))
                                if sujetar_pose_A:
                                    sujetar_pose_A = False
                                    _enviar_brazo((3, 4), _q45)
                                _qi45 = qpos_idx_brazo[3:5]
                                _guarda45 = [float(datos.qpos[_i]) for _i in _qi45]
                                for _i, _v in zip(_qi45, _q45):
                                    datos.qpos[_i] = _v
                                hib = preparar_hibrido(modelo, datos, qpos_idx_brazo)
                                if LIMITES_HUMANOS:   # 7-oct: límites de hombro/codo/suelo en la IK
                                    completar_hib(modelo, datos, hib)
                                for _i, _v in zip(_qi45, _guarda45):
                                    datos.qpos[_i] = _v
                                mujoco.mj_kinematics(modelo, datos)
                                q_ik_prev = None
                                suav_obj.reset()
                                suav_codo.reset()
                                oe_brazo.reset()
                                oe_antebrazo.reset()
                                t_atasco, mano_girada = None, False
                                lat_local, lat_cal = None, []    # se recalibra en esta T-pose
                                # PASO 7: filtros de saltos, zona muerta y modo fino desde cero
                                filtro_saltos.reset(); zm_palma.reset(); zm_codo.reset()
                                canal_palma.reset(); canal_giro.reset()
                                dirs_ok, q6_obj_prev = None, None
                                brazo_fino, brazo_congelado = False, False
                                aplicar_modo_brazo(False, False, (), gen_brazo)
                                if IK_RESCATE_GIRADA and hib["lo"][1] > -1.0:
                                    print(f"ℹ️  Rescate 'codo al otro lado' inactivo: IMITAR_LIMITES['joint2'] "
                                          f"empieza en {hib['lo'][1]:.2f} (hace falta < -1.0 para inclinar "
                                          f"el hombro hacia atrás).")
                                print(f"🖐️  Híbrido listo: alcance ARCTOS = {hib['alcance']:.3f} m  "
                                      f"(brazo {hib['L_brazo']:.3f} m + antebrazo {hib['L_antebrazo']:.3f} m)"
                                      f"  modo={HIBRIDO_MODO}  objetivo={HIBRIDO_OBJETIVO} "
                                      f"(hombro→punta {hib['alcance_punta']:.3f} m, codo→punta "
                                      f"{hib['L_antebrazo_punta']:.3f} m)")
                                brazo_imit_activo = True
                                t_prev_imit = _t
                                print("🦾 Imitación del brazo ACTIVADA.")
                            _dt_imit    = min(_t - t_prev_imit, 0.1)
                            t_prev_imit = _t

                            if IMITAR_REFERENCIA == "relativo":
                                _q_obj = angulos_humano_a_q_brazo_rel(
                                    _ang_uso["azimut"], _ang_uso["elevacion"], _ang_uso["codo"],
                                    tpose_ref, q_ref_sim)
                                _el_c = _ang_uso["elevacion"] - tpose_ref[1]  # Δ respecto a tu T-pose
                            else:
                                _az_c, _el_c, _co_c = corregir_con_tpose(_ang_uso, tpose_med)
                                _q_obj = angulos_humano_a_q_brazo(
                                    math.radians(_az_c),
                                    math.radians(_el_c),
                                    math.radians(_co_c))
                            # ══ PASO 2 — HÍBRIDO: tu mano decide dónde va la mano del robot ══
                            _obj_mano = None
                            ik_info   = None
                            if peso_hib > 0.0 and hib is not None and HIBRIDO_MODO == "codo_muneca" \
                                    and "dir_brazo" in _ang_uso:
                                # PASO 3: tus DIRECCIONES → objetivos de codo y muñeca → IK
                                _d_b = _ang_uso["dir_brazo"]
                                _d_a = _ang_uso["dir_antebrazo"]
                                _t_oe = time.perf_counter()
                                _lon_h = est_brazo.longitudes or (0.30, 0.28)

                                def _objetivos(db, da):
                                    if HIBRIDO_OBJETIVO == "palma":
                                        if LIMITES_HUMANOS and RETARGET_FIEL and "L1_ef" in hib:
                                            # 7-oct: tu brazo recto = brazo del robot recto
                                            return objetivos_codo_palma_fiel(datos, db, da, hib, _lon_h)
                                        return objetivos_codo_palma(datos, db, da, hib, _lon_h)
                                    return objetivos_codo_muneca(datos, db, da, hib)

                                # PASO 7.2: FILTRO DE SALTOS, antes del One Euro (que dejaría
                                # pasar el salto creyendo que es un movimiento rápido)
                                if SALTO_VEL_MAX > 0:
                                    if _ang is None and dirs_ok is not None:
                                        _d_b, _d_a = dirs_ok      # sin medida nueva: la última BUENA
                                    elif filtro_saltos.aceptar(_objetivos(_d_b, _d_a), _t_oe) \
                                            or dirs_ok is None:
                                        dirs_ok = (_d_b, _d_a)
                                    else:
                                        _d_b, _d_a = dirs_ok      # medida imposible → la última buena
                                if FILTRO_ONE_EURO:       # PASO 5: filtro sobre las direcciones
                                    _d_b0, _d_a0 = _d_b, _d_a
                                    _d_b = _unitario(oe_brazo(_d_b, _t_oe))
                                    _d_a = _unitario(oe_antebrazo(_d_a, _t_oe))
                                    if _d_b is None or _d_a is None:
                                        _d_b, _d_a = _d_b0, _d_a0
                                _d_b, _d_a = lim_humanos(_d_b, _d_a)   # 7-oct: rango anatómico
                                _d_a = enderezar(_d_b, _d_a)
                                _w_c = peso_codo_humano(_d_b, _d_a)
                                _codo_ok = codo_fiable(_ang_uso)
                                if HIBRIDO_CODO_SOLO_FIABLE and not _codo_ok:
                                    _w_c *= HIBRIDO_CODO_DUDOSO   # 7-oct: codo dudoso → manda la palma
                                _oc, _om = _objetivos(_d_b, _d_a)
                                if not FILTRO_ONE_EURO:   # sin One Euro: medias móviles de antes
                                    _oc = suav_codo.actualizar(_oc)
                                    _om = suav_obj.actualizar(_om)
                                # PASO 7.2: zona muerta → quieto de verdad cuando estás quieto
                                _oc, _om = zm_codo(_oc), zm_palma(_om)
                                # PASO 7.3: modo fino / congelado (sobre la palma)
                                _om = canal_palma(_om, _t_oe,
                                                  limitar=lambda p: limitar_alcance_palma(datos, p, hib))
                                _modo_b = canal_palma.modo
                                if _modo_b != "normal":
                                    _w_c = 0.0            # en fino/congelado manda solo la palma
                                if _modo_b == "congelado" and q_ik_prev is not None:
                                    _q_ik, _ec_ik, _em_ik = q_ik_prev.copy(), 0.0, 0.0
                                else:
                                    _q_ik, _ec_ik, _em_ik = resolver_ik_codo_muneca(
                                        modelo, datos, _oc, _om, _w_c, hib,
                                        q_inicial=q_ik_prev if q_ik_prev is not None else _q_obj,
                                        q_ref=q_ik_prev, w_cont=IK_PESO_CONTINUIDAD,
                                        # 7-oct: brazo vertical → la base no se mueve
                                        bloqueados=bloqueo_base(hib) if q_ik_prev is not None else None)

                                # PASO 6: RESCATE — si lleva un rato atascada (o está "girada"),
                                # probar otras posturas de partida. Todo se compara por COSTE:
                                # error + penalización por girar la base + por ir "girada".
                                _err_ik = math.hypot(_w_c * _ec_ik, _em_ik)
                                _t_ik   = time.perf_counter()
                                _q1_ref = float(_q_obj[0])          # base que pide tu azimut
                                _girada = es_girada(_q_ik, _q1_ref)
                                if (_err_ik <= IK_RESCATE_ERROR and not _girada) or \
                                        (_modo_b != "normal" and not IK_RESCATE_EN_FINO):
                                    t_atasco = None               # (en fino/congelado no hay rescate)
                                elif t_atasco is None:
                                    t_atasco = _t_ik
                                elif (_t_ik - t_atasco >= IK_RESCATE_TIEMPO
                                      and _t_ik - t_ult_rescate >= IK_RESCATE_ESPERA):
                                    _semillas = [("tu postura (ángulos)", _q_obj),
                                                 ("reposo", np.array(POSICION_STANDBY_BRAZO[:3], dtype=float))]
                                    # La configuración girada solo tiene sentido con el codo
                                    # DOBLADO de verdad (con el brazo recto no aporta nada).
                                    if IK_RESCATE_GIRADA and hib["lo"][1] <= -1.0 \
                                            and _w_c >= 0.9 * HIBRIDO_PESO_CODO:
                                        _semillas += [("codo al otro lado", semilla_girada(_q_ik)),
                                                      ("codo al otro lado (ángulos)", semilla_girada(_q_obj))]
                                    # 7-oct: posturas del brazo HUMANO que la IK no encuentra
                                    # sola (p.ej. mano a la cara: brazo arriba y codo plegado)
                                    if LIMITES_HUMANOS and "q3_recto" in hib:
                                        _semillas += [("codo espejo", semilla_codo_espejo(_q_ik, hib)),
                                                      ("codo plegado arriba", np.clip(
                                                          [_q_ik[0], *angulos_humano_a_q_brazo(0.0, math.pi, 1.52)[1:3]],   # ARCTOS (q1,0,0) · FR5 (q1,-π/2,-1.52)
                                                          hib["lo"], hib["hi"]))]
                                    _q_r, _err_r, _c_r, _nom_r = rescatar_ik(
                                        modelo, datos, _oc, _om, _w_c, hib, _semillas,
                                        q_actual=_q_ik, q1_ref=_q1_ref)
                                    _c_act = coste_ik(_q_ik, _err_ik, _q_ik, _q1_ref)
                                    if _q_r is not None and _c_r < IK_RESCATE_MEJORA * _c_act:
                                        print(f"[IK RESCATE] desde '{_nom_r}': error "
                                              f"{100 * _err_ik:.1f} → {100 * _err_r:.1f} cm"
                                              f"{'  (GIRADA)' if es_girada(_q_r, _q1_ref) else ''}")
                                        _q_ik = _q_r
                                        t_ult_rescate = _t_ik
                                    t_atasco = None
                                q_ik_prev = _q_ik.copy()
                                ik_info   = (_oc, _om, _w_c)
                                _p_h = 1.0 if _modo_b != "normal" else peso_hib   # fino/congelado: solo IK
                                _q_obj = (1.0 - _p_h) * np.asarray(_q_obj) + _p_h * _q_ik
                                for _k, _jn in enumerate(("joint1", "joint2", "joint3")):
                                    _q_obj[_k] = float(np.clip(_q_obj[_k], *IMITAR_LIMITES[_jn]))
                            elif peso_hib > 0.0 and hib is not None and est_brazo.longitudes \
                                    and "hombro_muneca" in _ang_uso:
                                _obj_mano = suav_obj.actualizar(objetivo_mano_hibrido(
                                    datos, _ang_uso["hombro_muneca"],
                                    sum(est_brazo.longitudes), hib))
                                _q_ik = resolver_ik_brazo(
                                    modelo, datos, _obj_mano,
                                    dof_idx_brazo[:3], qpos_idx_brazo[:3],
                                    nombre_body_efector="mano_mount",
                                    offset_local_efector=OFFSET_LOCAL_MUNON,
                                    n_iter=HIBRIDO_IK_ITER, paso_max=HIBRIDO_IK_PASO,
                                    # parte de la solución anterior (sin saltos de postura);
                                    # la primera vez, de TU postura por ángulos
                                    q_inicial=q_ik_prev if q_ik_prev is not None else _q_obj)
                                q_ik_prev = _q_ik.copy()
                                _q_obj = (1.0 - peso_hib) * np.asarray(_q_obj) + peso_hib * _q_ik
                                for _k, _jn in enumerate(("joint1", "joint2", "joint3")):
                                    _q_obj[_k] = float(np.clip(_q_obj[_k], *IMITAR_LIMITES[_jn]))
                            if LIMITES_HUMANOS and hib is not None:
                                # 7-oct: también en modo "solo ángulos" [H]: límites humanos+suelo
                                _q_obj = np.clip(np.asarray(_q_obj, dtype=float), hib["lo"], hib["hi"])
                            if not (FILTRO_ONE_EURO and ik_info is not None):
                                _q_obj = suav_brazo.actualizar(_q_obj)   # (con One Euro ya viene filtrado)
                            if gen_brazo is not None and gen_brazo.activo:
                                # PASO 7.1: solo se publica el objetivo; el hilo de física
                                # lo alcanza de forma continua (vel. y acel. limitadas)
                                q_imit = np.asarray(_q_obj, dtype=float).copy()
                                _enviar_brazo((0, 1, 2), q_imit)
                            else:
                                q_imit = limitar_velocidad_articular(
                                    _q_obj, q_imit, _dt_imit, IMITAR_VEL_MAX)
                                for _i in range(3):
                                    datos.ctrl[_i] = float(q_imit[_i])

                            if DEBUG_ANGULOS_BRAZO and (time.time() - t_ult_print_q) >= DEBUG_ANGULOS_CADA_S:
                                t_ult_print_q = time.time()
                                _q_real = [datos.qpos[_qi] for _qi in qpos_idx_brazo[:3]]
                                _sim_real = (fisica.factor if fisica is not None
                                             else 4 * modelo.opt.timestep / max(_dt_imit, 1e-3))
                                _hold_txt = "" if _ang is not None else \
                                    f"  [HOLD {time.perf_counter() - t_ult_ang_valido:.2f}s]"
                                _modo_txt = ("" if canal_palma.modo == "normal" else
                                             f"  [{canal_palma.modo.upper()}]")
                                if SALTO_VEL_MAX > 0 and filtro_saltos.rechazados:
                                    _modo_txt += f"  [saltos filtrados: {filtro_saltos.rechazados}]"
                                if gen_brazo is not None and gen_brazo.activo:
                                    _modo_txt += (f"  [OBJ q1={q_imit[0]:5.2f} q2={q_imit[1]:5.2f} "
                                                  f"q3={q_imit[2]:5.2f}]")
                                print(f"[BRAZO IMITA] humano az={_ang_uso['azimut']:6.1f}° "
                                      f"elev={_ang_uso['elevacion']:5.1f}° ({'Δ' if IMITAR_REFERENCIA == 'relativo' else 'corr'} {_el_c:5.1f}°) "
                                      f"codo={_ang_uso['codo']:5.1f}°  →  "
                                      f"CMD q1={datos.ctrl[0]:6.2f} q2={datos.ctrl[1]:5.2f} q3={datos.ctrl[2]:5.2f}  |  "
                                      f"REAL q1={_q_real[0]:6.2f} q2={_q_real[1]:5.2f} q3={_q_real[2]:5.2f} rad  |  "
                                      f"sim/real={_sim_real:.2f}x{_hold_txt}{_modo_txt}")
                                if ik_info is not None:
                                    _oc, _om, _w_c = ik_info
                                    _ec = np.linalg.norm(_oc - datos.xanchor[hib["j_codo"]])
                                    _em = np.linalg.norm(
                                        _om - (datos.xpos[hib["b_mano"]]
                                               + datos.xmat[hib["b_mano"]].reshape(3, 3) @ hib["off_punta"]))
                                    print(f"[HIBRIDO codo+muñeca] error codo={100 * _ec:4.1f} cm "
                                          f"(peso {_w_c:.2f}{'' if _codo_ok else ', dudoso'})   "
                                          f"error muñeca={100 * _em:4.1f} cm   "
                                          f"muñeca obj=({_om[0]:.2f}, {_om[1]:.2f}, {_om[2]:.2f}) m"
                                          + (f"   base fija" if bloqueo_base(hib)[0] else "")
                                          + (f"   {lim_humanos.texto()}" if LIMITES_HUMANOS else ""))
                                if _obj_mano is not None:
                                    _bm = mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_BODY, "mano_mount")
                                    _err = np.linalg.norm(_obj_mano - datos.xpos[_bm])
                                    print(f"[HIBRIDO] objetivo=({_obj_mano[0]:.2f}, {_obj_mano[1]:.2f}, "
                                          f"{_obj_mano[2]:.2f}) m   error mano={100 * _err:4.1f} cm   "
                                          f"peso={peso_hib:.2f}")

                    # — Estado de la calibración de huesos / imitación (HUD) —  (v2: su propio HUD)
                    if MODO_BRAZO == "imitacion" and estado["camara_activa"] and teleop_v2 is None \
                            and sliders is None:
                        if est_brazo.calibrando:
                            cv2.putText(frame_bgr,
                                        f"CALIBRANDO HUESOS {est_brazo.progreso}/{est_brazo.n_objetivo}"
                                        f" - manten T-pose (brazo estirado)",
                                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                        elif est_brazo.calibrado and brazo_imit_activo:
                            if brazo_congelado:
                                _txt_b, _col_b = "BRAZO: CONGELADO [C]", (0, 165, 255)
                            elif brazo_fino:
                                _txt_b, _col_b = f"BRAZO: FINO x{FINO_ESCALA:.2f} [F]", (255, 255, 0)
                            else:
                                _txt_b, _col_b = "BRAZO: IMITANDO", (0, 255, 0)
                            cv2.putText(frame_bgr, _txt_b, (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _col_b, 1)
                        elif est_brazo.calibrado:
                            cv2.putText(frame_bgr,
                                        f"BRAZO: {'HIBRIDO' if peso_hib > 0 else 'ANGULOS'} [H]",
                                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    if perfil is not None:
                        perfil.marca("brazo")

                    # — Detección de mano —
                    # 0º) FLUIDEZ: si Hands ya se lanzó en paralelo (MANO_EN_PARALELO), se
                    #     recoge su resultado (recorte con la Pose del fotograma anterior).
                    # 1º) Si no, o no encontró mano: recorte alrededor de TU mano (la del
                    #     BRAZO_HUMANO) según la Pose de ESTE fotograma, como antes.
                    # 2º) Si Pose no ve la muñeca o en el recorte no hay mano, imagen
                    #     completa como antes (p.ej. mano cerca de la cámara sin cuerpo).
                    resultado   = None
                    origen_mano = "-"
                    _cajas_mano = []                  # (caja, ¿encontró mano?) para dibujarlas
                    if fut_mano is not None:
                        with _soltar():               # esperar al hilo de Hands: la física sigue
                            resultado = fut_mano.result()
                        _cajas_mano.append((caja_par, resultado is not None))
                        if resultado is not None:
                            origen_mano = (f"recorte{'-HD' if frame_hd is not None else ''} "
                                           f"{caja_par[2]}px (paralelo)")
                    if resultado is None and MANO_RECORTE and resultado_pose.pose_landmarks:
                        _img_rec = frame_limpio
                        _caja = caja_recorte_mano(resultado_pose.pose_landmarks, BRAZO_HUMANO,
                                                  _img_rec.shape[1], _img_rec.shape[0])
                        if _caja is not None:
                            with _soltar():           # MediaPipe Hands: la física sigue
                                resultado = detectar_mano_en_recorte(
                                    detector_mano_recorte, _img_rec, _caja)
                            if resultado is not None:
                                origen_mano = f"recorte{'-HD' if frame_hd is not None else ''} {_caja[2]}px"
                            _cajas_mano.append((_caja, resultado is not None))
                    # Recuadro(s) del recorte en la ventana (coordenadas del preview)
                    if _cajas_mano:
                        _sx = ANCHO_CAM / frame_limpio.shape[1]
                        _sy = ALTO_CAM  / frame_limpio.shape[0]
                        for _cj, _ok in _cajas_mano:
                            cv2.rectangle(frame_bgr,
                                          (int(_cj[0] * _sx), int(_cj[1] * _sy)),
                                          (int((_cj[0] + _cj[2]) * _sx),
                                           int((_cj[1] + _cj[2]) * _sy)),
                                          (255, 200, 0) if _ok else (0, 0, 255), 1)
                    if resultado is None and device is None:   # sin cámaras: no hay mano que ver
                        resultado = _SIN_MANO
                    if resultado is None:
                        with _soltar():               # MediaPipe Hands: la física sigue
                            resultado = detector_mano.process(frame_rgb)
                        if resultado.multi_hand_landmarks:
                            origen_mano = "imagen completa"
                    if perfil is not None:
                        perfil.marca("mano")
                    calc_m_hud   = 0.0

                    if resultado.multi_hand_landmarks:
                        hand_lm  = resultado.multi_hand_landmarks[0]
                        hand_wlm = resultado.multi_hand_world_landmarks[0]

                        # ══ PASO 5 — PRONOSUPINACIÓN → joint6 ══  (no con v2: lo hace la IK de 6 ejes)
                        if estado["camara_activa"] and estado["antebrazo"] is not None and teleop_v2 is None:
                            roll = calcular_pronosupinacion(hand_wlm, estado["antebrazo"],
                                                            estado["eje_codo"])
                            _t = time.perf_counter()
                            _dt = min(_t - t_prev_prono, 0.1)
                            t_prev_prono = _t
                            if roll is not None:
                                if roll_ref is None:
                                    # Calibrando: T-pose con el PULGAR HACIA ARRIBA
                                    refs_roll.append(roll)
                                    if len(refs_roll) >= 15:
                                        # media circular (no se rompe cerca de ±180°)
                                        roll_ref = float(np.angle(np.mean(np.exp(1j * np.array(refs_roll)))))
                                        q_prono_ref = float(datos.ctrl[prono_idx])
                                        q_prono, d_prev = q_prono_ref, 0.0
                                        print(f"🔄 Giro neutro calibrado: {math.degrees(roll_ref):.1f}°")
                                else:
                                    d = d_prev + _wrap((roll - roll_ref) - d_prev)   # sin saltos
                                    d = float(np.clip(d, -PRONO_LIMITE, PRONO_LIMITE))
                                    d_prev = d
                                    q_obj  = q_prono_ref + PRONO_SIGNO * PRONO_ESCALA * d
                                    # ORIENTACIÓN EN EL MUNDO: con el brazo imitado, se busca el
                                    # joint6 que deja la mano del robot como la tuya en el espacio
                                    # (no un giro relativo, que fallaba si la IK elegía otra postura).
                                    _q6_mundo = None
                                    if PRONO_MODO == "mundo" and hib is not None and brazo_imit_activo \
                                            and estado.get("torso") is not None:
                                        _lat_h = lateral_mano_humana_mundo(hand_wlm, estado["torso"], hib)
                                        if _lat_h is not None and lat_local is None:
                                            # Calibración (justo tras [A], T-pose y pulgar arriba):
                                            # tu mano ahora ≡ mano del robot en su T-pose.
                                            _m = _perp(_lat_h, hib["eje6_tpose"])
                                            if _m is not None:
                                                lat_cal.append(_m)
                                            if len(lat_cal) >= 10:
                                                _m = _perp(np.mean(lat_cal, axis=0), hib["eje6_tpose"])
                                                if _m is not None:
                                                    lat_local = hib["R_mano_tpose"].T @ _m
                                                    print("🖐️  Orientación de la mano calibrada (modo mundo)")
                                        elif _lat_h is not None:
                                            _q6_mundo = q6_para_orientacion(
                                                datos, hib, lat_local, _lat_h,
                                                float(datos.qpos[qpos_idx_brazo[prono_idx]]))
                                    if _q6_mundo is not None:
                                        q_obj = _equivalente_en_rango(_q6_mundo, q_prono,
                                                                      *LIMITES_CTRL_BRAZO[prono_idx])
                                        q6_obj_prev = q_obj
                                    elif PRONO_MODO == "mundo" and lat_local is not None \
                                            and q6_obj_prev is not None:
                                        # PASO 7.2: el modo mundo no es fiable en ESTE fotograma →
                                        # se mantiene su último objetivo (antes se pasaba al giro
                                        # relativo, que da otro valor → tirón de muñeca)
                                        q_obj = q6_obj_prev
                                    elif hib is not None and peso_hib > 0.0 and ik_info is not None \
                                            and estado.get("az_brazo") is not None:
                                        _e_h = hib["ejes"] @ _eje_codo_humano_torso(
                                            math.radians(estado["az_brazo"]))
                                        _dot = float(_e_h @ datos.xaxis[hib["j_codo"]])
                                        if abs(_dot) > 0.3:             # zona muerta: sin parpadeos
                                            _gir = (1.0 if _dot >= 0 else -1.0) != hib["signo_codo"]
                                            if _gir != mano_girada:
                                                print(f"🔁 Robot {'GIRADO (codo al otro lado)' if _gir else 'normal'}"
                                                      f": giro de muñeca {'compensado +180°' if _gir else 'sin compensar'}")
                                            mano_girada = _gir
                                        if mano_girada:
                                            q_obj = _equivalente_en_rango(
                                                q_obj + math.pi, q_prono,
                                                *LIMITES_CTRL_BRAZO[prono_idx])
                                    # PASO 7.3: modo fino / congelado también para el giro
                                    _lo6, _hi6 = LIMITES_CTRL_BRAZO[prono_idx]
                                    q_obj = float(canal_giro(
                                        q_obj, _t,
                                        limitar=lambda v: _equivalente_en_rango(
                                            float(v[0]), q_prono, _lo6, _hi6))[0])
                                    q_suav = q_prono + PRONO_SUAVIZADO * (q_obj - q_prono)
                                    paso   = PRONO_VEL_MAX * _dt
                                    q_prono = q_prono + float(np.clip(q_suav - q_prono, -paso, paso))
                                    _enviar_brazo((prono_idx,), (q_prono,))  # PASO 7.1

                                    if DEBUG_ANGULOS_BRAZO and time.time() - t_ult_print_roll >= DEBUG_ANGULOS_CADA_S:
                                        t_ult_print_roll = time.time()
                                        print(f"[PRONO{' mundo' if _q6_mundo is not None else ''}] "
                                              f"giro={math.degrees(d):6.1f}°  →  "
                                              f"q6 CMD={q_prono:5.2f}  REAL={datos.qpos[qpos_idx_brazo[prono_idx]]:5.2f} rad")
                        if not sujetar_pose_A and teleop_v2 is None:   # 9-oct: no durante la pose de [A]
                            _enviar_brazo((4,), (JOINT5_FIJO,))   # joint5 siempre recto (v2: lo decide la IK)

                        # Dibujar esqueleto en el frame
                        mp_draw.draw_landmarks(
                            frame_bgr, hand_lm, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style())

                        # Convertir a frame de mano 3D
                        lm3d = landmarks_a_3d(hand_wlm, es_mano_izq=False)

                        # — Retargeting y control de MuJoCo —
                        if estado["camara_activa"]:
                            estado = retarget_a_ctrl(
                                modelo, datos, lm3d, hand_wlm, estado, suav_dedos)
                            if guante is not None:      # MANUS: los dedos son los del guante
                                guante.restaurar_dedos(datos)
                            calc_m_hud = estado["calc_muneca"]

                            # ── Diagnóstico de pinzas ──
                            if DEBUG_PINZA and (time.time() - t_ult_print_pinza) >= DEBUG_PINZA_CADA_S:
                                t_ult_print_pinza = time.time()
                                _dp = estado.get("dist_pulgar", {})
                                _txt = "  ".join(
                                    f"{_d}={100 * _dp[_d]:4.1f}/{100 * PINZA_UMBRAL_ON[_d]:.1f}"
                                    for _d in ("indice", "corazon", "anular", "menique") if _d in _dp)
                                print(f"\n[PINZA] cm pulgar→yema (medido/umbral): {_txt}  |  "
                                      f"activa={estado.get('modo_pinza')} "
                                      f"dedo={estado.get('dedo_bloqueado')}  "
                                      f"habilitada={estado.get('pinza_habilitada', True)}  "
                                      f"mano de: {origen_mano}")
                        # ── Resaltado de yemas en pinza ──────────────────────
                        if estado.get("modo_pinza"):
                            punta_por_dedo = {
                                "indice":  8,
                                "corazon": 12,
                                "anular":  16,
                                "menique": 20,
                            }
                            # Yema del pulgar (siempre landmark 4)
                            lm_th  = hand_lm.landmark[4]
                            px_th  = int(lm_th.x * ANCHO_CAM)
                            py_th  = int(lm_th.y * ALTO_CAM)
                            cv2.circle(frame_bgr, (px_th, py_th), 11, (0, 0, 255), -1)
                            cv2.circle(frame_bgr, (px_th, py_th), 11, (255, 255, 255), 2)

                            # Para combos multi-dedo se resaltan las dos yemas correspondientes
                            dedos_resaltar = (
                                ["corazon", "anular"]  if estado["dedo_activo"] == "corazon_anular"  else
                                ["indice",  "corazon"] if estado["dedo_activo"] == "indice_corazon"  else
                                ["anular",  "menique"] if estado["dedo_activo"] == "anular_menique"  else
                                [estado["dedo_activo"]]
                            )
                            for d in dedos_resaltar:
                                lm_tip = hand_lm.landmark[punta_por_dedo[d]]
                                px_tip = int(lm_tip.x * ANCHO_CAM)
                                py_tip = int(lm_tip.y * ALTO_CAM)
                                cv2.circle(frame_bgr, (px_tip, py_tip), 11, (0, 0, 255), -1)
                                cv2.circle(frame_bgr, (px_tip, py_tip), 11, (255, 255, 255), 2)

                        # (5-nov) La grabación ya no va aquí, dentro del "if" de MediaPipe: la hace
                        # la grabadora (skill_shield/ia/grabador.py) desde el hilo de la física,
                        # a ritmo fijo y con el brazo + la mano.


                        pct = int(min(100, max(0,
                            (-calc_m_hud / abs(UMBRAL_BLOQUEO)) * 100)))
                        if estado["muneca_bloqueada"]:
                            cv2.putText(frame_bgr,
                                        "MUNECA BLOQUEADA — extiende para liberar",
                                        (10, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
                        else:
                            cv2.putText(frame_bgr,
                                        f"Flexion muneca: {pct}%",
                                        (10, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 100), 1)

                        if estado.get("modo_pinza"):
                            cv2.putText(frame_bgr,
                                        f"PINZA -> {estado['dedo_activo'].upper()}",
                                        (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)

                        if not estado.get("pinza_habilitada", True):
                            cv2.putText(frame_bgr,
                                        "PINZAS DESACTIVADAS [Espacio]",
                                        (10, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                    elif device is None:              # sin cámaras: la mano va con el guante
                        cv2.putText(frame_bgr, "SIN CAMARAS (gemelo con sliders)", (10, 85),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        cv2.putText(frame_bgr,
                                    "Mano: guante MANUS -> OrcaHand" if estado["camara_activa"]
                                    else "Pulsa [A] para activar la mano (guante MANUS)",
                                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0) if estado["camara_activa"] else (0, 165, 255), 1)
                    else:
                        msg = ("⚠ Mano perdida — posición mantenida"
                               if estado["camara_activa"] else "Buscando mano...")
                        cv2.putText(frame_bgr, msg, (10, 85),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

                    if perfil is not None:
                        perfil.marca("dedos")

                    # ══ TELEOP FR5 v2: muñeca + codo (2 cámaras) → IK de 6 ejes → joint1..6 ══
                    if teleop_v2 is not None and estado["camara_activa"]:
                        _t_v2 = time.perf_counter()
                        _dt_v2 = None if t_prev_v2 is None else min(_t_v2 - t_prev_v2, 0.1)
                        t_prev_v2 = _t_v2
                        _obs_a = None
                        if resultado_pose.pose_landmarks:
                            _mano_px = None
                            if resultado is not None and resultado.multi_hand_landmarks:
                                _mano_px = np.array([[_l.x * ANCHO_CAM, _l.y * ALTO_CAM]
                                                     for _l in resultado.multi_hand_landmarks[0].landmark])
                            _obs_a = _ObsV2(resultado_pose.pose_landmarks, frame_depth, intr_v2_a,
                                            mano_lm=_mano_px, t=_t_v2)
                        _obs_b = None
                        if ent_b is not None and intr_v2_b is not None and ent_b[0] is not None:
                            _obs_b = _ObsV2(ent_b[0], ent_b[2], intr_v2_b, t=_t_v2)
                        _mano_v2 = None
                        if globals().get("TELEOP_FR5_PRONO", "hands") == "hands" and resultado is not None \
                                and resultado.multi_hand_world_landmarks:
                            _hw = resultado.multi_hand_world_landmarks[0]
                            _mano_v2 = (_lm_hand(_hw, LM_CORAZON_MCP) - _lm_hand(_hw, LM_MUNECA),
                                        _lm_hand(_hw, LM_MENIQUE_MCP) - _lm_hand(_hw, LM_INDICE_MCP))
                        _q_act_v2 = [float(datos.ctrl[_i]) for _i in range(6)]
                        if teleop_v2_par is not None:
                            # resultado del fotograma ANTERIOR (el hilo lo calculó mientras esta
                            # vuelta esperaba a la cámara y a MediaPipe) y se lanza el de este
                            _q_v2, _info_v2 = teleop_v2_par.recoger()
                            _alimentar_registro = registro_v2 is not None and registro_v2.activo
                            if _alimentar_registro:                 # (el hilo está parado aquí)
                                _fin_reg = registro_v2.alimentar(_t_v2, teleop_v2)
                            teleop_v2_par.lanzar({"A": _obs_a, "B": _obs_b}, _t_v2, mano=_mano_v2,
                                                 q_actual=_q_act_v2, dt=_dt_v2)
                        else:
                            _q_v2, _info_v2 = teleop_v2.actualizar(
                                {"A": _obs_a, "B": _obs_b}, _t_v2, mano=_mano_v2,
                                q_actual=_q_act_v2, dt=_dt_v2)
                            _alimentar_registro = registro_v2 is not None and registro_v2.activo
                            if _alimentar_registro:
                                _fin_reg = registro_v2.alimentar(_t_v2, teleop_v2)
                        _est_v2 = _info_v2.get("estado", "?")
                        if _alimentar_registro:
                            if _fin_reg:
                                print(f"🩺 Registro de diagnóstico terminado → {registro_v2.ruta}")
                            else:
                                cv2.putText(frame_bgr, f"REGISTRO QUIETO {registro_v2.restante(_t_v2):4.1f} s - no te muevas",
                                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        if _q_v2 is not None and _est_v2 in ("exacta", "aprox", "transicion", "bloqueado"):
                            _enviar_brazo((0, 1, 2, 3, 4, 5), [float(_v) for _v in _q_v2])
                            brazo_imit_activo = True
                        if _est_v2 == "calibrando":
                            _txt_v2 = _info_v2.get("texto") or (
                                f"CALIBRANDO HUESOS {_info_v2.get('progreso', 0)}/{CALIB_HUESOS_FRAMES}"
                                " - manten T-pose (brazo estirado)")
                            _col_v2 = (0, 200, 255)
                        elif _est_v2 == "desarmado":
                            _txt_v2 = _info_v2.get("texto", "DESARMADO")
                            _col_v2 = (0, 140, 255)
                        elif _info_v2.get("entrada"):
                            _txt_v2 = (f"BRAZO v2: ENTRANDO DESPACIO "
                                       f"({globals().get('TELEOP_FR5_V_ENTRADA', 0.15)} rad/s)")
                            _col_v2 = (255, 200, 0)
                        else:
                            _txt_v2 = (f"BRAZO v2: {_est_v2.upper()}  "
                                       f"err={100 * (_info_v2.get('error') or 0):.1f}cm"
                                       + (f"  {_info_v2['aviso']}" if _info_v2.get("aviso") else ""))
                            _col_v2 = {"exacta": (0, 255, 0), "aprox": (0, 200, 255),
                                       "transicion": (255, 200, 0)}.get(_est_v2, (0, 0, 255))
                        cv2.putText(frame_bgr, _txt_v2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _col_v2, 1)
                        cv2.putText(frame_bgr, "fusion: " + str(_info_v2.get("fuentes", ""))[:70],
                                    (10, ALTO_CAM - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                        if globals().get("TELEOP_FR5_DEBUG", True) and _t_v2 - t_ult_print_v2 >= DEBUG_ANGULOS_CADA_S:
                            t_ult_print_v2 = _t_v2
                            _qtxt = "  ".join(f"{_v:+.2f}" for _v in _q_v2) if _q_v2 is not None else "-"
                            print(f"[TELEOP v2] {_est_v2:10s} rama={_info_v2.get('rama')} "
                                  f"err={100 * (_info_v2.get('error') or 0):4.1f}cm  q=[{_qtxt}]  "
                                  f"{_info_v2.get('motivo', '')}  | {_info_v2.get('fuentes', '')}")

                    # — Teclado Avanzado (usamos waitKeyEx para captar flechas) —
                    key = cv2.waitKeyEx(1)
                    if perfil is not None:
                        perfil.marca("waitKey")

                    # 5-nov: PEDAL — arma / desarma el brazo (solo cuenta con la cámara activa)
                    if pedal_v2 is not None:
                        _ev_pedal = pedal_v2.tecla(key) or pedal_v2.actualizar()
                        if pedal_v2.modo in ("tecla", "teclado") and key in pedal_v2.teclas:
                            key = -1                     # la tecla del pedal no hace nada más
                        if _ev_pedal == "armar":
                            if estado["camara_activa"] and teleop_v2.calibrado:
                                teleop_v2.armar()
                                print("🟠 PEDAL: brazo ARMADO → entra despacio hacia tu postura")
                            else:
                                pedal_v2.desarmar()
                                print("⚠ PEDAL: no se arma sin calibrar ([A] + T-pose primero)")
                        elif _ev_pedal == "desarmar":
                            teleop_v2.desarmar()
                            print("⏸ PEDAL: brazo DESARMADO → quieto")

                    # ══ GEMELO_SLIDERS (9-nov): el brazo sigue a la ventana de sliders ══
                    # La ventana enseña la pose REAL (FR5, o simulador si no hay FR5) y, ARMADA,
                    # publica el objetivo. Aquí solo se valida y se pasa al generador suave; el
                    # bloque del FR5 de abajo arma / desarma el ServoJ igual que con el pedal.
                    if sliders is not None:
                        _fr5_ok = fr5 is not None and fr5.conectado
                        _en_linea_sl = fr5 is not None and fr5.en_linea   # (también en ERROR)
                        _q_real_sl = fr5.q_real_rad() if _en_linea_sl else None
                        if _q_real_sl is not None:
                            _q_vis_sl = sliders.a_grados(_q_real_sl)
                            sliders.publicar_real(_q_vis_sl, de_fr5=True)
                        elif not _en_linea_sl:
                            _q_vis_sl = sliders.a_grados([datos.ctrl[_i] for _i in _idx_ctrl_fr5])
                            sliders.publicar_real(_q_vis_sl, de_fr5=False)
                        else:
                            _q_vis_sl = None              # FR5 conectado pero sin lectura reciente
                        if not fr5_al_inicio:
                            sliders.estado_fr5(SIN_FR5)
                        elif not _fr5_ok:
                            sliders.estado_fr5(FR5_ERROR)
                            if sliders.armado:            # el FR5 ha fallado: el gemelo no sigue solo
                                sliders.desarmar("FR5 desconectado o con error → DESARMADO (mira la terminal).")
                        else:
                            sliders.estado_fr5(FR5_ENVIANDO if fr5_enviando else FR5_ESPERA)

                        if sliders.recien_armado():
                            if _q_vis_sl is None:
                                sliders.desarmar("No se arma: no hay lectura reciente de la pose del FR5.")
                            elif sliders.fuera_de_rango(_q_vis_sl):
                                sliders.desarmar("No se arma: " + ", ".join(f"J{_j}" for _j in sliders.fuera_de_rango(_q_vis_sl)) + " fuera "
                                                 "del rango de los sliders (muévelo con la WebApp).")
                            else:
                                _simul = _fr5_ok and fr5.p.get("FR5_SIMULACRO", True)
                                sliders.mensaje("ARMADO en SIMULACRO: el FR5 real NO se mueve "
                                                "(SLIDERS_FR5_SIMULACRO en config.py)." if _simul else "")
                                print("🟠 SLIDERS: brazo ARMADO"
                                      + (" (FR5 en simulacro)" if _simul else
                                         " → el FR5 real sigue a los sliders" if _fr5_ok else
                                         " (solo simulador)"))
                        if sliders.armado:
                            gen_brazo.factor = sliders.velocidad
                            _enviar_brazo(range(6), sliders.objetivo_rad(), de_sliders=True)
                        cv2.putText(frame_bgr,
                                    "BRAZO: SLIDERS " + ("ARMADO" if sliders.armado else "DESARMADO (quieto)"),
                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 165, 255) if sliders.armado else (200, 200, 200), 2)

                    # 5-nov: FR5 real ↔ simulador. Mientras NO está armado, el robot no recibe
                    # nada y MuJoCo (y el generador) se copian de su postura REAL en cada vuelta:
                    # lo que ves es el robot de verdad. Al armar, se sincroniza una última vez y
                    # empieza el ServoJ desde ahí (sin pinchar el robot y sin salto).
                    # 9-nov: FR5 en ERROR (colisión, fallo de ServoJ…) → todo desarmado; [X] lo recupera
                    if fr5 is not None and fr5.en_error:
                        if fr5_enviando:
                            fr5_enviando = False
                            if teleop_v2 is not None:
                                teleop_v2.desarmar()
                            if pedal_v2 is not None:
                                pedal_v2.desarmar()
                            _c_err = fr5.estado()["codigo"]
                            print(f"⚠ FR5 en ERROR (código {_c_err[0]}/{_c_err[1]}) → DESARMADO. "
                                  "Pulsa [X] (o RESET en los sliders) para borrar el error.")
                        cv2.putText(frame_bgr, "FR5 ERROR " + "/".join(str(_v) for _v in fr5.estado()["codigo"])
                                    + "  -  [X] reset", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if sliders is not None and sliders.pide_reset() and fr5 is not None:
                        fr5.resetear()
                        fr5_enviando = False
                        print("🔄 RESET del FR5 pedido desde los sliders")

                    if fr5 is not None and fr5.conectado:
                        if sliders is not None:
                            _armar_fr5 = sliders.armado     # GEMELO_SLIDERS: ARMAR de la ventana
                        else:
                            # 9-nov: con el FR5 real SOLO se arma con la teleop v2 + PEDAL. Antes, sin
                            # v2 (pipeline antiguo del ARCTOS) se armaba siempre: ya no.
                            _armar_fr5 = (teleop_v2 is not None and pedal_v2 is not None
                                          and pedal_v2.modo is not None
                                          and estado["camara_activa"] and teleop_v2.armado)
                        if not _armar_fr5 or not fr5_enviando:
                            if fr5_enviando:
                                fr5.habilitar(False)
                                fr5_enviando = False
                            if _sincronizar_con_fr5() and gen_brazo is not None and gen_brazo.activo:
                                _factor_gen = gen_brazo.factor   # (conserva el modo fino)
                                gen_brazo.activar(datos)         # el generador sale de la postura real
                                gen_brazo.factor = _factor_gen
                            if _armar_fr5:
                                fr5.publicar(datos.ctrl[_idx_ctrl_fr5])
                                fr5.habilitar(True)              # ServoMoveStart desde la postura real
                                fr5_enviando = True

                    # 9-nov: brazo FANTASMA = OBJETIVO (lo que la IK / los sliders le piden al
                    # generador) y puntas objetivo (verde) / real (roja). El brazo sólido es el
                    # robot REAL (con el FR5 conectado) o el simulado.
                    if fantasma is not None:
                        _q_f = [float(datos.ctrl[_i]) for _i in _idx_ctrl_fr5]
                        if gen_brazo is not None and gen_brazo.activo:
                            for _k_g, _i_g in enumerate(gen_brazo.idx):
                                if _i_g in _idx_ctrl_fr5:
                                    _q_f[_idx_ctrl_fr5.index(_i_g)] = float(gen_brazo.obj_fin[_k_g])
                        try:
                            _d_f = fantasma.dibujar(visor, _q_f, punta_real=datos.xpos[fantasma.punta].copy())
                            if _d_f is not None and _d_f > 0.01:
                                cv2.putText(frame_bgr, f"objetivo - real: {100 * _d_f:4.1f} cm",
                                            (10, ALTO_CAM - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                            (0, 255, 255), 1)
                        except Exception as _e_f:
                            print(f"⚠ Brazo fantasma desactivado ({_e_f})")
                            fantasma = None

                    # 9-nov: diagnóstico del gemelo — real (FR5) vs simulador vs comando, en grados
                    if fr5 is not None and globals().get("GEMELO_DEBUG", True) \
                            and time.perf_counter() - _t_debug_gemelo[0] >= 2.0:
                        _t_debug_gemelo[0] = time.perf_counter()
                        _e_g = fr5.estado()
                        _fmt_g = lambda _v: "[" + ", ".join(f"{_x:7.1f}" for _x in _v) + "]"
                        _edad_g = _e_g["edad_real_s"]
                        print(f"[GEMELO] {_e_g['texto']}"
                              f"{'' if _edad_g is None else f' (lectura de hace {1000 * _edad_g:.0f} ms)'}"
                              f"\n   real FR5 (°): {_fmt_g(_e_g['q_real_deg'])}"
                              f"\n   simulador(°): {_fmt_g([math.degrees(datos.qpos[_a]) for _a in qpos_idx_brazo])}"
                              f"\n   comando  (°): {_fmt_g([math.degrees(datos.ctrl[_i]) for _i in _idx_ctrl_fr5])}")

                    if key == 32:  # 32 es el código ASCII del Espacio
                        estado["pinza_habilitada"] = not estado.get("pinza_habilitada", True)
                        estado_txt = "HABILITADA" if estado["pinza_habilitada"] else "DESHABILITADA"
                        print(f"🤏 Detección de pinza predefinida: {estado_txt}.")

                        # Si se deshabilita mientras hacías una pinza, la forzamos a soltar
                        if not estado["pinza_habilitada"]:
                            estado["modo_pinza"] = False
                            estado["dedo_bloqueado"] = None

                    elif key == ord("a") and not estado["camara_activa"]:
                        estado["camara_activa"] = True
                        suav_dedos.reset()
                        # Paso 4: empieza la calibración de huesos y la imitación arranca
                        # de nuevo cuando termine (el brazo humano debe estar en T-pose).
                        brazo_imit_activo = False
                        refs_tpose = []
                        tpose_med  = None
                        roll_ref, refs_roll, d_prev = None, [], 0.0
                        if sliders is not None:
                            print("✋ [A] Mano ACTIVA (guante → OrcaHand). El brazo va con los sliders.")
                        elif MODO_BRAZO == "imitacion":
                            est_brazo.iniciar_calibracion()
                            if teleop_v2 is not None:
                                # v2: calibra los huesos con su propia fusión y arranca la IK desde
                                # donde esté el brazo (no se mueve a la T-pose ni a POSE_BRAZO_ACTIVAR)
                                if teleop_v2_par is not None:
                                    teleop_v2_par.recoger()              # que el hilo no esté a medias
                                    teleop_v2_par.ultimo = (None, {"estado": "calibrando"})
                                teleop_v2.iniciar_calibracion()     # (también desarma)
                                teleop_v2.ik.q = None
                                t_prev_v2 = None
                                if pedal_v2 is not None:
                                    pedal_v2.desarmar()
                                print("🦾 [A] TELEOP v2: calibrando huesos (T-pose, brazo estirado) "
                                      f"~{CALIB_HUESOS_FRAMES} fotogramas. El FR5 NO se mueve: al acabar, "
                                      "pisa el pedal para armarlo.")
                            _pose_A = globals().get("POSE_BRAZO_ACTIVAR") if teleop_v2 is None else None
                            if _pose_A is not None:
                                # 9-oct: pose fija al pulsar [A] (config.POSE_BRAZO_ACTIVAR). Se
                                # sujeta mientras calibras la T-pose; al empezar a imitarte, el
                                # brazo sale de aquí y la muñeca del ARCTOS vuelve a JOINT5_FIJO.
                                _enviar_brazo(range(len(_pose_A)), [float(_v) for _v in _pose_A])
                                sujetar_pose_A = True
                                print("🦾 [A] Brazo a POSE_BRAZO_ACTIVAR: "
                                      + "  ".join(f"q{_k + 1}={float(_v):+.2f}" for _k, _v in enumerate(_pose_A))
                                      + " rad")
                            elif IMITAR_REFERENCIA == "absoluto" and IMITAR_TPOSE_SIM_AL_CALIBRAR \
                                    and teleop_v2 is None:   # 5-nov: con el FR5 calibrar NO mueve el robot
                                _qT = angulos_humano_a_q_brazo(
                                    math.radians(T_POSE_HUMANA[0]),
                                    math.radians(T_POSE_HUMANA[1]),
                                    math.radians(T_POSE_HUMANA[2]))
                                _enviar_brazo((0, 1, 2), _qT)   # PASO 7.1: va a la T-pose sin tirón
                            if est_brazo.manual is None:
                                print("🦴 Calibrando huesos: pon el BRAZO ESTIRADO en T-pose "
                                      f"y no te muevas (~{CALIB_HUESOS_FRAMES} fotogramas)...")
                        if resultado.multi_hand_world_landmarks:
                            v, e = calcular_flexion_muneca(resultado.multi_hand_world_landmarks[0])
                            estado["vec_neutro"]  = v
                            estado["eje_lateral"] = e
                        if device is not None:
                            print("✅ Cámara ACTIVA — muñeca calibrada en posición neutra "
                                  "(el brazo parte de su posición actual).")

                    elif key in (ord("x"), ord("X")):
                        # 9-nov: recuperación rápida tras una colisión (falsa o no) o un fallo
                        if fr5 is not None and fr5.en_linea:
                            fr5.resetear()
                            fr5_enviando = False
                            if teleop_v2 is not None:
                                teleop_v2.desarmar()
                            if pedal_v2 is not None:
                                pedal_v2.desarmar()
                            if sliders is not None:
                                sliders.desarmar("RESET del FR5: vuelve a ARMAR.")
                            print("🔄 [X] RESET del FR5 (ResetAllError) → vuelve a armar")
                        else:
                            print("ℹ [X] No hay FR5 conectado")

                    elif key == ord("s"):
                        estado["camara_activa"] = False
                        estado["muneca_bloqueada"] = False
                        if teleop_v2 is not None:            # 5-nov: hay que volver a armar
                            if teleop_v2_par is not None:
                                teleop_v2_par.recoger()
                            teleop_v2.desarmar()
                            if pedal_v2 is not None:
                                pedal_v2.desarmar()
                        _parar_grabacion("antes de standby")
                        suav_dedos.reset()
                        # PASO 7.1: el brazo vuelve al reposo de forma continua (sin tirón)
                        sujetar_pose_A = False
                        brazo_fino, brazo_congelado = False, False
                        canal_palma.reset(); canal_giro.reset()
                        aplicar_modo_brazo(False, False, (), gen_brazo)
                        _enviar_brazo(range(len(POSICION_STANDBY_BRAZO)), POSICION_STANDBY_BRAZO)
                        for i, v in enumerate(POSICION_STANDBY):
                            ctrl_mano[i] = float(v)
                        print("⏸ STANDBY — posición de reposo restaurada (brazo + mano).")

                    elif key == ord("h"):
                        peso_hib = 0.0 if peso_hib > 0.0 else (HIBRIDO_PESO or 1.0)
                        q_ik_prev = None
                        suav_obj.reset()
                        suav_codo.reset()
                        oe_brazo.reset()
                        oe_antebrazo.reset()
                        t_atasco = None
                        filtro_saltos.reset(); zm_palma.reset(); zm_codo.reset()
                        canal_palma.reset(); canal_giro.reset()
                        dirs_ok = None
                        brazo_fino, brazo_congelado = False, False
                        aplicar_modo_brazo(False, False, (), gen_brazo)
                        print(f"🖐️  Brazo: {('HÍBRIDO (' + HIBRIDO_MODO + ')') if peso_hib > 0 else 'solo ÁNGULOS'}"
                              f"  (peso={peso_hib:.2f})")

                    elif key in (ord("f"), ord("F")):
                        # PASO 7.3 — MODO FINO: el robot se mueve FINO_ESCALA veces lo que tú
                        if not (brazo_imit_activo and estado["camara_activa"] and peso_hib > 0.0
                                and HIBRIDO_MODO == "codo_muneca"):
                            print("⚠ Modo fino: solo con el brazo IMITANDO en híbrido 'codo_muneca' "
                                  "([A] + T-pose; [H] si estás en 'solo ángulos').")
                        else:
                            brazo_fino = not brazo_fino
                            aplicar_modo_brazo(brazo_fino, brazo_congelado,
                                               (canal_palma, canal_giro), gen_brazo)
                            print(f"🎯 Modo FINO ACTIVADO: el brazo se mueve x{FINO_ESCALA:.2f} lo que tú, "
                                  "desde donde está." if brazo_fino else
                                  "↩️  Modo fino desactivado: el brazo vuelve a imitarte (suavemente).")

                    elif key in (ord("c"), ord("C")):
                        # PASO 7.3 — CONGELAR: el brazo se queda quieto (la mano sigue)
                        if not (brazo_imit_activo and estado["camara_activa"] and peso_hib > 0.0
                                and HIBRIDO_MODO == "codo_muneca"):
                            print("⚠ Congelar: solo con el brazo IMITANDO en híbrido 'codo_muneca'.")
                        else:
                            brazo_congelado = not brazo_congelado
                            aplicar_modo_brazo(brazo_fino, brazo_congelado,
                                               (canal_palma, canal_giro), gen_brazo)
                            print("🧊 Brazo CONGELADO (la mano sigue funcionando). [C] para seguir."
                                  if brazo_congelado else
                                  ("▶️  Brazo descongelado: sigue en modo fino DESDE AQUÍ." if brazo_fino
                                   else "▶️  Brazo descongelado: vuelve a imitarte (suavemente)."))

                    elif key in (ord("r"), ord("R")) and teleop_v2 is not None:
                        # 28-oct: separar medida de IK — graba la salida de Fusor2Cam estando quieto
                        if not estado["camara_activa"]:
                            print("🩺 [R] Activa antes la cámara con [A] (la fusión v2 solo corre activa).")
                        elif registro_v2 is not None and registro_v2.activo:
                            registro_v2.activo = False
                            print("🩺 [R] Registro de diagnóstico cancelado.")
                        else:
                            from skill_shield.teleop_fr5.diagnostico import RegistroQuieto
                            registro_v2 = RegistroQuieto(float(globals().get("TELEOP_FR5_DIAG_S", 10.0)),
                                                         etiqueta=str(globals().get("TELEOP_FR5_DIAG_ETIQUETA", "tpose")),
                                                         longitudes_manual=LONG_BRAZO_MANUAL)
                            registro_v2.iniciar(time.perf_counter())
                            print(f"🩺 [R] Registro de diagnóstico: {registro_v2.duracion:.0f} s QUIETO "
                                  f"(T-pose, brazo estirado). [R] otra vez para cancelar.")

                    elif key == ord("g") and estado["camara_activa"]:
                        # 5-nov: brazo + mano a ritmo fijo (grabador.py). El nombre de la tarea
                        # se pide con la física SUELTA (antes input() la congelaba).
                        if not grabador.activo:
                            if fisica is None:
                                print("⚠ [G] necesita FISICA_EN_HILO = True (la grabadora va en "
                                      "el hilo de la física).")
                            else:
                                if tarea_actual is None or not globals().get("IA_MISMA_TAREA", True):
                                    with _soltar():
                                        tarea_actual = _pedir_tarea(tarea_actual)
                                if tarea_actual:
                                    if teleop_v2 is not None and not teleop_v2.armado:
                                        print("ℹ️  El brazo está DESARMADO: se grabará quieto "
                                              "hasta que pises el pedal.")
                                    grabador.iniciar(tarea_actual, datos)
                                    print(f"🔴 GRABANDO brazo + mano a {grabador.fps:.0f} Hz "
                                          f"(tarea '{tarea_actual}'). [G] para terminar el episodio.")
                        else:
                            _parar_grabacion()

                    elif key == ord("G"):
                        # 5-nov: [Shift+G] cambiar de tarea (las siguientes [G] graban en la nueva)
                        _parar_grabacion("cambio de tarea")
                        with _soltar():
                            tarea_actual = _pedir_tarea(tarea_actual)
                        print(f"📝 Tarea para las próximas grabaciones: '{tarea_actual}'")

                    elif key == ord("e"):
                        # Entrena SOLO la última tarea usada (rápido, para iterar
                        # sobre una maniobra concreta sin tocar las demás)
                        if tarea_actual is None:
                            print("❌ No hay ninguna tarea seleccionada todavía. "
                                  "Graba con [G] primero.")
                        else:
                            # 5-nov: en otro proceso (antes congelaba cámaras y física minutos)
                            _parar_grabacion("antes de entrenar")
                            _entrenar_en_segundo_plano([tarea_actual])

                    elif key == ord("E"):
                        # Mayúscula: reentrena TODAS las tareas grabadas (más lento,
                        # útil tras cambiar algo estructural del modelo)
                        _parar_grabacion("antes de entrenar")
                        _entrenar_en_segundo_plano(["--todas"])

                    elif key == ord("3"):
                        # [3] — Toggle entorno de trabajo (mesa + herramientas)
                        if not taller.disponible:
                            print("⚠ El entorno de trabajo no está en el modelo "
                                  "(falta el include de mesa_trabajo.xml).")
                        else:
                            # 27-oct: un escenario cada vez, como hace [4]: con el soporte de
                            # hombros la mesa baja y ocupa el sitio de la peana del taladro
                            if not taller.visible and taladro.disponible and taladro.visible:
                                taladro.ocultar(modelo, datos)
                            if taller.alternar(modelo, datos):
                                print("🧰 Entorno de trabajo MOSTRADO (alicates, destornillador, "
                                      "bombilla y fusible recolocados).")
                                if TALLER_ENCUADRAR_CAMARA:
                                    visor.cam.lookat[:]  = TALLER_CAMARA["lookat"]
                                    visor.cam.distance   = TALLER_CAMARA["distancia"]
                                    visor.cam.azimuth    = TALLER_CAMARA["azimut"]
                                    visor.cam.elevation  = TALLER_CAMARA["elevacion"]
                            else:
                                print("⬛ Entorno de trabajo OCULTADO.")

                    elif key == ord("4"):
                        # [4] — Escenario taladro (techo + peana + taladro)
                        if not taladro.disponible:
                            print("⚠ El escenario taladro no está en el modelo (ejecuta "
                                  "taladro/preparar_taladro.py y añade el include).")
                        else:
                            if taller.disponible and taller.visible:
                                taller.ocultar(modelo, datos)     # uno cada vez
                            if taladro.alternar(modelo, datos):
                                print("🔩 Escenario taladro MOSTRADO: coge el taladro por la "
                                      "empuñadura y llévalo al techo.")
                                visor.cam.lookat[:]  = (0.5 * TALADRO_POS_SOPORTE[0],
                                                        0.5 * TALADRO_POS_SOPORTE[1], 0.55)
                                visor.cam.distance   = 1.9
                                visor.cam.elevation  = -12
                            else:
                                print("⬛ Escenario taladro OCULTADO.")

                    elif key == ord("5"):
                        # [5] — Demo automática: coger el taladro, llevarlo al techo,
                        #       apretar el gatillo y hacer un agujero (sin cámara)
                        if not demo_taladro.ok:
                            print("⚠ La demo del taladro no está disponible (mira los avisos "
                                  "del arranque).")
                        else:
                            _parar_grabacion("antes de la demo")
                            estado["camara_activa"] = False
                            modo_demo_taladro = True
                            break

                    elif key == ord("i"):
                        # Modo IA real: elige qué tarea/maniobra quieres que reproduzca
                        modelos_disponibles = _listar_modelos_entrenados()
                        if not modelos_disponibles:
                            print("❌ No hay ninguna tarea entrenada todavía. "
                                  "Graba con [G] y entrena con [E] primero.")
                        else:
                            with _soltar():               # 5-nov: sin congelar la física
                                tarea_elegida = _elegir("Tarea a reproducir", modelos_disponibles)
                            if tarea_elegida:
                                ruta_pesos = os.path.join(MODELOS_DIR, f"{tarea_elegida}.pth")
                                red_neuronal = cargar_ia(ruta_pesos)
                                if red_neuronal is not None:
                                    print(f"🎯 Tarea seleccionada: '{tarea_elegida}'")
                                    _parar_grabacion("antes del modo IA")
                                    modo_ia_real = True
                                    estado["camara_activa"] = False
                                    break
                            else:
                                print("   Cancelado.")

                    elif key == ord("q"):
                        salir = True

                    # — Guante MANUS: manda sobre los dedos solo con la cámara ACTIVA ([A]) —
                    if guante is not None:
                        guante.aplicar = estado["camara_activa"]
                        guante.haptica(modelo, datos)      # contactos de las yemas → vibración

                    # — Paso de simulación MuJoCo —
                    if visor.is_running():
                        # Si estamos en modo hardware, enviamos los datos a los servos
                        if mano_fisica and mano_real and mano_real.conectado:
                            hardware_target = construir_hardware_target(datos)
                            mano_real.enviar_posiciones(hardware_target)
                        # Escenario taladro: agarre asistido (si está visible)
                        if taladro.disponible:
                            taladro.actualizar(modelo, datos)

                        # Mueve el Gemelo Digital en ambos modos. Con FISICA_EN_HILO
                        # lo hace HiloFisica al ritmo del reloj real; si no, como antes.
                        if fisica is None:
                            for _ in range(4):
                                if gen_brazo is not None:        # PASO 7.1
                                    gen_brazo.paso(modelo, datos, modelo.opt.timestep)
                                if guante is not None:           # dedos del guante MANUS
                                    guante.escribir_ctrl(datos, modelo.opt.timestep)
                                mujoco.mj_step(modelo, datos)
                            visor.sync()
                        # ── Logo IDI — HUD inferior centrado del visor MuJoCo ──
                        _dibujar_logo_mujoco(visor, _logo_mj_px, _logo_mj_w, _logo_mj_h)
                    else:
                        salir = True  # Ventana MuJoCo cerrada por el usuario

                    if perfil is not None:
                        perfil.marca("teclas+mano+sim")

                    # — HUD de estado —
                    est_str = "ACTIVO" if estado["camara_activa"] else "STANDBY"
                    c_hud   = (0, 255, 0) if estado["camara_activa"] else (0, 0, 255)
                    cv2.putText(frame_bgr, est_str, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_hud, 2)
                    # FPS reales: si bajan de ~15, MediaPipe Pose pierde el seguimiento
                    # y tendrás parpadeo del esqueleto por mucho que ajustes lo demás.
                    cv2.putText(frame_bgr, f"{_fps_val:4.1f} FPS", (10, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 255, 0) if _fps_val >= 15 else (0, 165, 255), 1)
                    if guante is not None:
                        _txt_g = guante.estado_txt()
                        cv2.putText(frame_bgr, _txt_g, (ANCHO_CAM - 290, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0, 255, 0) if "OK" in _txt_g else (0, 165, 255), 1)
                    if grabador.activo:
                        cv2.circle(frame_bgr, (ANCHO_CAM - 20, 20), 8, (0, 0, 255), -1)
                        cv2.putText(frame_bgr, f"REC {grabador.segundos:4.1f}s", (ANCHO_CAM - 110, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    cv2.imshow(
                        NOMBRE_VENTANA,
                        _superponer_logo(
                            cv2.resize(frame_bgr,
                                       (int(ANCHO_CAM * ESCALA_VENTANA),
                                        int(ALTO_CAM * ESCALA_VENTANA))),
                            _logo_bgr, _logo_alpha,
                            margen=12, posicion="top-left"))
                    if cam_b is not None and MULTICAM_MOSTRAR_B:
                        cam_b.dibujar_y_mostrar(
                            pose_lm=ent_b[0] if ent_b else None,
                            texto=(f"B {'OK' if ent_b else 'sin cuerpo'}"
                                   + (f" | {fusor_b.resumen}" if fusor_b is not None else "")))
                    if perfil is not None:
                        perfil.marca("ventana")
                        perfil.informe()

                # Salimos de teleoperación (modo autónomo / IA / salir): esos
                # modos avanzan la física por su cuenta → se para el hilo.
                if pool_mano is not None:            # FLUIDEZ: cerrar el hilo de Hands
                    pool_mano.shutdown(wait=True)
                    pool_mano = None
                if fr5 is not None:                  # FR5 real: ServoMoveEnd y quieto
                    fr5.habilitar(False)
                if fantasma is not None:             # fuera de teleoperación no hay objetivo
                    fantasma.ocultar(visor)
                if sliders is not None and sliders.armado:   # al volver, hay que ARMAR de nuevo
                    sliders.desarmar("Has salido de la teleoperación → DESARMADO.")
                if pedal_v2 is not None:
                    pedal_v2.cerrar()
                if fisica is not None:
                    fisica.parar()
                    fisica = None
                if gen_brazo is not None:
                    gen_brazo.desactivar()

    # — Limpieza —
    cv2.destroyAllWindows()
    detector_mano.close()
    detector_mano_recorte.close()
    detector_pose.close()
    if visor.is_running():
        visor.close()

    if mano_real is not None:
        mano_real.desconectar()
    if guante is not None:
        guante.cerrar()
    _parar_grabacion("al salir")
    grabador.esperar()                     # que terminen de escribirse los episodios
    if sliders is not None:
        sliders.parar()
    if fr5 is not None:
        e_fr5 = fr5.estado()
        fr5.parar()
        print(f"🦾 fr5_servo: {e_fr5['ciclos']} ServoJ, periodo medio "
              f"{e_fr5['periodo_medio_ms']:.2f} ms (máx {e_fr5['periodo_max_ms']:.2f}), "
              f"ServoJ {e_fr5['servoj_medio_ms']:.2f} ms, retrasos {e_fr5['retrasos']}, "
              f"errores {e_fr5['errores_servoj']}")

    print("👋 Fin del programa.")


if __name__ == "__main__":
    main()