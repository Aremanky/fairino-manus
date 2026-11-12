#!/usr/bin/env python3
"""
Calibración cámara de cabeza → robot (paso 4 de la hoja de ruta, 10-nov)
=========================================================================
Tablero ChArUco en la brida/mano + cámara de cabeza fija en el soporte («ojo fuera de la mano»).
Resultado: camara_robot.json con T_mundo_cam (marco del MJCF, el del gemelo y la IK).

La cámara viaja en la MISMA peana que el robot: al mover la peana la calibración NO cambia.
Las posturas se guardan en ángulos de joint (calib_posturas.json), así que valen siempre.

Uso (cierra antes el modo autónomo / la teleoperación: la cámara y el robot tienen que estar libres):
  python calibrar_camara_robot.py --generar-tablero   # PNG del tablero para imprimir y montar
  python calibrar_camara_robot.py --auto              # AUTOMÁTICA: pon el robot donde la cámara vea
                                                      #   el tablero y mantén el pedal; el programa
                                                      #   elige las posturas, las recorre y calibra
  python calibrar_camara_robot.py --repetir           # repite las posturas guardadas y recalibra
  python calibrar_camara_robot.py --verificar         # va a unas pocas posturas y comprueba el error
  python calibrar_camara_robot.py --verificar --aqui  # comprueba en la postura actual, sin moverlo
  python calibrar_camara_robot.py --manual            # MANUAL: mueves tú el robot (arrastre en la
                                                      #   WebApp) y pulsas ESPACIO en cada postura
  python calibrar_camara_robot.py --recalcular        # recalcula con las muestras guardadas

El robot solo se mueve MIENTRAS se mantiene el pedal de hombre muerto (TELEOP_FR5_PEDAL =
"teclado": tecla P con la ventana delante, o "evdev"); al soltarlo se queda quieto. Velocidad
máxima CALIB_ROBOT_VEL_MAX_DEG_S. Q/Esc en la ventana: parar y salir.
Si la controladora marca una colisión (falsa, casi siempre, por el peso de la mano + tablero),
se borra sola como en la WebApp (Clear → disable → enable; CALIB_AUTO_RESET) o con la tecla X;
para seguir, suelta la P y vuelve a pulsarla. Tres colisiones en la misma postura → se salta.

Calibración automática, por dentro:
  1. EXPLORAR (~1 min): pequeños giros de la muñeca y del brazo desde la postura de inicio →
     calibración aproximada (ya se sabe más o menos dónde está la cámara). Aquí se comprueba
     también que el cuadrado del tablero mide lo que dice CALIB_ROBOT_TABLERO: si no, se para.
  2. PLANIFICAR (unos segundos, sin moverse): miles de posturas candidatas; se quedan las que
     dejan el tablero bien visible, a las que el robot puede llegar (límites de fr5_servo) y en
     las que nada se acerca a nada. Se eligen ~20 muy distintas, se ordenan y se comprueba cada
     trayecto tal como lo moverá fr5_servo.
  3. MEDIR (~3-5 min): las recorre, captura en cada una y calcula la calibración definitiva.
     Las posturas se guardan para --repetir y --verificar.

Que nada choque ni se estruje (10-nov). La postura de inicio —la deja el operario, con la muñeca
cómoda y nada apretado— es la referencia: ningún hueco de la mano o del tablero con el robot, el
soporte, la cabeza con la cámara, las cajas de config o el suelo puede cerrarse más de 1 cm
respecto al que había al empezar (salvo que siga sobrando hueco), y nunca baja de su mínimo
(crear_seguridad; huecos en el bloque CALIB_* de config.py). Tres barreras:
  · el plan: posturas y trayectos comprobados sin ninguna manga;
  · antes de cada movimiento: el trayecto entero desde donde está el robot DE VERDAD (ir_a);
  · con el pedal pisado: la postura real, en cada vuelta. Si se acerca a algo más de lo comprobado
    o no sigue el trayecto que se le pide, el robot se para y la calibración termina («Calibración
    detenida: …» dice qué pasa). Si lo han movido a mano con el pedal suelto, se vuelve a
    comprobar el trayecto desde donde esté; y sin lectura de su postura, no se le mueve.
"""
import argparse
import collections
import datetime
import json
import math
import os
import sys
import time

import cv2
import numpy as np

import config as cfg
from skill_shield.autonomo import calibracion as C

RAIZ = os.path.dirname(os.path.abspath(__file__))
TABLERO = dict(getattr(cfg, "CALIB_ROBOT_TABLERO",
                       dict(cuadros=(5, 4), lado=0.03, marcador=0.022, diccionario="DICT_5X5_100")))


def _ruta(nombre_cfg, defecto):
    r = getattr(cfg, nombre_cfg, defecto)
    return r if os.path.isabs(r) else os.path.join(RAIZ, r)


ARCHIVO = _ruta("CALIB_ROBOT_ARCHIVO", "camara_robot.json")
POSTURAS = _ruta("CALIB_ROBOT_POSTURAS", "calib_posturas.json")
MUESTRAS = _ruta("CALIB_ROBOT_MUESTRAS", "calib_muestras.json")
MIN_MUESTRAS = int(getattr(cfg, "CALIB_ROBOT_MIN_MUESTRAS", 15))
ERROR_MAX = float(getattr(cfg, "CALIB_ROBOT_ERROR_MAX_MM", 5.0))
VEL_MAX = float(getattr(cfg, "CALIB_ROBOT_VEL_MAX_DEG_S", 15.0))
ESPERA_S = float(getattr(cfg, "CALIB_ROBOT_ESPERA_S", 1.0))
N_VERIFICAR = int(getattr(cfg, "CALIB_ROBOT_VERIFICAR_N", 4))
N_AUTO = int(getattr(cfg, "CALIB_AUTO_POSTURAS", 20))
RADIO_MANO = float(getattr(cfg, "CALIB_AUTO_RADIO_MANO", 0.11))
AMPLITUD = tuple(getattr(cfg, "CALIB_AUTO_AMPLITUD_DEG", (30, 20, 25, 45, 45, 70)))
AUTO_RESET = bool(getattr(cfg, "CALIB_AUTO_RESET", True))   # colisión → Clear, disable, enable solo
SIN_AVANCE_S = float(getattr(cfg, "CALIB_SIN_AVANCE_S", 5.0))   # armado y parado sin llegar → se deja esa postura
# Lo PLANIFICADO se comprueba sin manga. Al robot de verdad, que nunca queda exactamente donde se
# le pide, se le dan unos mm: al comprobar el trayecto desde donde está y, algo más, al vigilarlo.
MARGEN_TRAYECTO = 0.006         # m de manga al comprobar un trayecto desde la postura REAL, antes de moverse
MARGEN_VIGILANCIA = 0.012       # m de manga a la postura REAL mientras se mueve (va algo por detrás de la pedida)
TUBO_DEG = 3.0                  # ° que la postura REAL puede separarse del trayecto comprobado sin volver a comprobarlo
FOTOS_POR_MUESTRA = 5           # se promedia la pose del tablero en 5 imágenes seguidas
QUIETO_DEG = 0.05               # el robot cuenta como quieto si en 0,6 s se mueve menos que esto
LLEGADA_DEG = 0.3               # postura alcanzada


class Cancelado(Exception):
    pass


class Detenido(Exception):
    """La calibración no puede seguir sin que alguien mire el robot (el mensaje dice por qué)."""


class FueraDeTrayecto(Exception):
    """El robot no está en el trayecto que se comprobó (lo trata Sesion.ir_a: vuelve a comprobar).
    armado: si se ha salido con el pedal pisado (se ha desviado él) o suelto (lo han movido a mano)."""

    def __init__(self, armado):
        super().__init__()
        self.armado = bool(armado)


# ═════════════ utilidades ═════════════

def generar_tablero():
    from calibrar_extrinseca import crear_tablero
    cuadros, lado = tuple(TABLERO["cuadros"]), float(TABLERO["lado"])
    board, _ = crear_tablero(cuadros, lado, TABLERO["marcador"], TABLERO["diccionario"])
    px = int(round(lado * 1000 * 300 / 25.4))            # 300 ppp
    margen = int(px * 0.4)
    # OJO: generateImage mete el margen DENTRO del tamaño pedido y encoge el tablero para que quepa.
    # Por eso el tamaño total tiene que llevar ya los dos márgenes (si no, el cuadro sale de ~31,5 mm
    # en vez de 37,5 mm al imprimir a 300 ppp).
    img = board.generateImage((cuadros[0] * px + 2 * margen, cuadros[1] * px + 2 * margen),
                              marginSize=margen)
    salida = os.path.join(RAIZ, "tablero_calib_robot.png")
    cv2.imwrite(salida, img)
    print(f"Guardado {salida} ({cuadros[0]}x{cuadros[1]} cuadros de {lado * 1000:.0f} mm, "
          f"diccionario {TABLERO['diccionario']}).\nImprímelo a 300 ppp al 100 % (sin «ajustar a página»), "
          f"COMPRUEBA con una regla que cada cuadrado mide {lado * 1000:.0f} mm y móntalo plano y rígido "
          f"en la brida/mano (pieza impresa). Si mide otra cosa, pon el valor real en CALIB_ROBOT_TABLERO.")


def _leer_json(ruta, defecto=None):
    if not os.path.exists(ruta):
        return defecto
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _escribir_json(ruta, datos):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=1, ensure_ascii=False)


def _normalizar_posturas(lista):
    """calib_posturas.json admite q sueltas (se captura en todas) o {q, capturar}."""
    return [dict(q=list(map(float, p["q"])), capturar=bool(p.get("capturar", True))) if isinstance(p, dict)
            else dict(q=list(map(float, p)), capturar=True) for p in lista]


def calcular_y_guardar(muestras, cadena, info_cam):
    """Calibra con las muestras (FK recalculada desde q), enseña el resultado y lo guarda."""
    P = C.puntos_tablero(TABLERO["cuadros"], TABLERO["lado"])
    Tb = [cadena.fk_marcos(np.array(m["q"]))[5] for m in muestras]
    Tc = [np.array(m["T_cam_tablero"]) for m in muestras]
    giro, recorrido = C.diversidad(Tb)
    res, usadas = C.descartar_peores(Tb, Tc, P, umbral_mm=max(8.0, 1.6 * ERROR_MAX))
    print("\n── Resultado ──────────────────────────────────────────")
    for k, i in enumerate(usadas):
        print(f"   muestra {i + 1:2d}: {res['errores_mm'][k]:5.2f} mm")
    descartadas = sorted(set(range(len(muestras))) - set(usadas))
    if descartadas:
        print(f"   descartadas por error alto: {[i + 1 for i in descartadas]}")
    print(f"   método {res['metodo']} · error medio {res['error_medio_mm']:.2f} mm · "
          f"máx {res['error_max_mm']:.2f} mm · {len(usadas)} muestras")
    print(f"   diversidad: giro máx. entre posturas {giro:.0f}°, recorrido de la brida {recorrido:.0f} mm")
    T = res["T_mundo_cam"]
    print(f"   cámara en el mundo (MJCF): x={T[0, 3]:+.3f} y={T[1, 3]:+.3f} z={T[2, 3]:+.3f} m, "
          f"mirando hacia {np.round(T[:3, 2], 2).tolist()}")
    if giro < 30:
        print("⚠️  Poco giro entre posturas (< 30°): la calibración puede estar mal aunque el error "
              "parezca bajo. Añade posturas girando la muñeca.")
    ok = res["error_medio_mm"] <= ERROR_MAX and giro >= 30
    # Comprobación final del tamaño del tablero, ya con TODAS las muestras (con las de la exploración
    # sola la escala sale con ±2 %; con las ~30 de la medida, con ±0,3 %).
    try:
        escala = C.estimar_escala([Tb[i] for i in usadas], [Tc[i] for i in usadas])
        lado_mm = float(TABLERO["lado"]) * 1000
        print(f"   tamaño del tablero según el robot: cuadro ≈ {escala * lado_mm:.1f} mm "
              f"(config.py: {lado_mm:.1f} mm)")
        if abs(escala - 1.0) > 0.015:
            ok = False
            print(f"⚠️  El cuadro no mide lo que dice CALIB_ROBOT_TABLERO ({(escala - 1) * 100:+.1f} %): corrige "
                  f"lado y marcador en config.py y lanza --repetir (las poses de la cámara se miden con el tamaño del tablero).")
    except (ValueError, np.linalg.LinAlgError):
        pass
    C.guardar(ARCHIVO, res, mxid=info_cam["mxid"], K=info_cam["K"], dist=info_cam["dist"],
              ancho=info_cam["ancho"], alto=info_cam["alto"], tablero=TABLERO,
              n_muestras=len(muestras), usadas=usadas)
    print(("✅" if ok else "⚠️ ") + f" Guardado {ARCHIVO}" +
          ("" if ok else f" — NO cumple el criterio (≤ {ERROR_MAX:.0f} mm y giro ≥ 30°): repite"))
    return res, ok


# ═════════════ cámara, robot, pedal y colisiones ═════════════

def abrir_camara():
    from calibrar_extrinseca import _pipeline_charuco_hd
    from skill_shield.vision.multicamara import (abrir_dispositivo, comprobar_usb, intrinsecos_color,
                                                 listar_camaras, mxid_de)
    mxid = getattr(cfg, "AUTONOMO_CAM_MXID", None)
    print(f"📷 OAK-D disponibles: {listar_camaras()}")
    pipeline, w, h = _pipeline_charuco_hd()
    dev = abrir_dispositivo(pipeline, mxid)
    comprobar_usb(dev, "cabeza")
    K, dist = intrinsecos_color(dev, w, h)
    q = dev.getOutputQueue("color_hd", maxSize=2, blocking=False)
    info = dict(mxid=mxid_de(dev), K=K, dist=np.asarray(dist).ravel(), ancho=w, alto=h)
    print(f"📷 Cámara de cabeza {info['mxid']}: {w}x{h}, fx={K[0, 0]:.0f} fy={K[1, 1]:.0f} px")
    return dev, q, info


def arrancar_fr5(mover):
    from skill_shield.hardware.fr5_servo import FR5Servo
    c = {k: getattr(cfg, k) for k in dir(cfg) if k.startswith("FR5_")}
    c["FR5_SIMULACRO"] = not mover
    c["FR5_VEL_MAX_DEG_S"] = min(float(c.get("FR5_VEL_MAX_DEG_S", 60.0)), VEL_MAX)
    fr5 = FR5Servo.desde_config(c)
    if not fr5.iniciar():
        if fr5.estado()["texto"] == "iniciando":         # conecta pero la controladora no contesta
            fr5.parar()
            sys.exit("✗ La controladora del FR5 no contesta (se ha quedado colgada). Apágala y enciéndela, "
                     "espera a que la WebApp funcione y vuelve a probar.")
        sys.exit("✗ Sin conexión con el FR5 (o con error). Revisa la red/WebApp y vuelve a probar.")
    return fr5


def crear_pedal():
    from skill_shield.teleop_fr5.pedal import Pedal
    modo = getattr(cfg, "TELEOP_FR5_PEDAL", None)
    p = Pedal(modo=modo, teclas=getattr(cfg, "TELEOP_FR5_PEDAL_TECLAS", (ord("p"), ord("P"))),
              dispositivo=getattr(cfg, "TELEOP_FR5_PEDAL_EVDEV", None),
              codigo=getattr(cfg, "TELEOP_FR5_PEDAL_EVDEV_CODIGO", None),
              hombre_muerto=getattr(cfg, "TELEOP_FR5_PEDAL_HOMBRE_MUERTO", True),
              exclusivo=getattr(cfg, "TELEOP_FR5_PEDAL_EXCLUSIVO", True),
              codigo_teclado=getattr(cfg, "TELEOP_FR5_PEDAL_TECLADO_CODIGO", 25))
    if modo is None or not p.hombre_muerto:
        sys.exit("✗ Para que el robot se mueva solo hace falta un pedal de HOMBRE MUERTO: "
                 "TELEOP_FR5_PEDAL = \"teclado\" (tecla P mantenida) o \"evdev\" con "
                 "TELEOP_FR5_PEDAL_HOMBRE_MUERTO = True.")
    print(f"🦶 Pedal: {p.descripcion}. El robot solo se mueve MIENTRAS lo mantienes.")
    return p


def modelo_colision(cadena, T_mundo_cam=None):
    """El de la teleoperación (soporte, suelo, mesa/obstáculos de config) con la mano más gorda
    (lleva el tablero) y, si ya se sabe, la cámara de cabeza como obstáculo."""
    from skill_shield.autonomo.mapa import modelo_colision as _mc
    return _mc(cadena, cfg, T_mundo_cam, radio_mano=RADIO_MANO)


def girar_mano(ch, grados):
    """(12-nov) La mano del gemelo girada 'grados' alrededor del eje de la brida (z de wrist3), en
    un ChoqueManoBrazo ya creado: en la calibración la OrcaHand va montada dada la vuelta (180°)
    respecto al gemelo. Solo cambia los puntos de la mano (el tablero se mide aparte, en el marco
    de la brida; la cápsula de la mano es simétrica respecto a ese eje)."""
    if not grados:
        return ch
    from skill_shield.autonomo.choque_mano import reducir
    a = math.radians(float(grados))
    R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    ch.mano = np.asarray(ch.mano, float) @ R.T
    ch._objetos["mano"] = ch.mano
    ch._grueso["mano"] = reducir(ch.mano, 0.02)
    return ch


def choque_mano(cadena, T_brida_tablero=None, entorno=None):
    """Comprobación de la mano (+ el tablero) contra el propio robot y el soporte, con las piezas
    del gemelo; con 'entorno' (el del modelo de colisión), también contra la cabeza con la cámara,
    las cajas de config y el suelo. La mano, girada CALIB_MANO_GIRO_DEG alrededor de la brida
    (como va montada en la calibración)."""
    from skill_shield.autonomo.choque_mano import desde_config
    return girar_mano(desde_config(cadena, cfg, T_brida_tablero, entorno),
                      float(getattr(cfg, "CALIB_MANO_GIRO_DEG", 0.0)))


def limites_articulares(cadena):
    """Rango (rad, convenio del MJCF) en el que se pueden pedir posturas sabiendo que el robot va
    a llegar: lo que fr5_servo deja pasar (topes del FR5 y límites blandos de la controladora, con
    sus márgenes) menos CALIB_MARGEN_LIMITES_DEG, dentro de los rangos del MJCF."""
    from skill_shield.hardware.fr5_servo import PARAMETROS_POR_DEFECTO, limites_envio_rad
    p = dict(PARAMETROS_POR_DEFECTO)
    p.update({k: getattr(cfg, k) for k in PARAMETROS_POR_DEFECTO if hasattr(cfg, k)})
    lim = limites_envio_rad(p, p.get("FR5_LIMITES_BLANDOS_DEG"))
    m = math.radians(float(getattr(cfg, "CALIB_MARGEN_LIMITES_DEG", 1.0)))
    return np.column_stack([np.maximum(cadena.limites[:, 0], lim[:, 0] + m),
                            np.minimum(cadena.limites[:, 1], lim[:, 1] - m)])


def crear_seguridad(cadena, q_inicio, T_brida_tablero=None, T_mundo_cam=None):
    """Comprobador de choques para que el robot se mueva solo, con q_inicio como referencia:
    cápsulas de la teleoperación + piezas reales de la mano (y del tablero y la cámara, si ya se
    sabe dónde están). Devuelve un calibracion.Colisiones."""
    modelo = modelo_colision(cadena, T_mundo_cam)
    mano = choque_mano(cadena, T_brida_tablero, modelo.entorno).fijar_inicio(q_inicio)
    return C.Colisiones(modelo, q_inicio, mano=mano)


def posturas_exploracion(cadena, q0, col):
    """Paso ② del modo automático: los giros pequeños desde q0 que valen según 'col' y a los que
    el robot puede llegar (límites que aplica fr5_servo)."""
    return C.posturas_explorar(q0, col, limites_articulares(cadena), VEL_MAX)


def planificar_medida(cadena, q0, T_mundo_cam, T_brida_tablero, K, ancho, alto):
    """Paso ③ del modo automático: con la calibración aproximada, elige las posturas de medida.
    Devuelve (plan, info, comprobador de choques con el que se ha planificado)."""
    col = crear_seguridad(cadena, q0, T_brida_tablero, T_mundo_cam)   # ya con el tablero
    P = C.puntos_tablero(TABLERO["cuadros"], TABLERO["lado"])
    plan, info = C.planificar_posturas(q0, cadena, col, T_mundo_cam, T_brida_tablero, K, ancho, alto, P,
                                       n=N_AUTO, amplitud_deg=AMPLITUD, v_max_deg=VEL_MAX,
                                       limites=limites_articulares(cadena))
    return plan, info, col


class Robot:
    """Pose real del FR5 + si está quieto (historial de 0,6 s)."""

    def __init__(self, fr5):
        self.fr5 = fr5
        self._hist = collections.deque()

    def leer(self):
        q = self.fr5.q_real_rad(max_edad_s=0.3)
        ahora = time.monotonic()
        if q is not None:
            self._hist.append((ahora, np.array(q)))
        while self._hist and ahora - self._hist[0][0] > 0.6:
            self._hist.popleft()
        return q

    @property
    def quieto(self):
        if len(self._hist) < 5 or self._hist[-1][0] - self._hist[0][0] < 0.45:
            return False
        Q = np.array([q for _, q in self._hist])
        return float(np.degrees(np.max(Q.max(0) - Q.min(0)))) < QUIETO_DEG


# ═════════════ sesión: ventana + cámara + robot ═════════════

class Sesion:
    def __init__(self, mover):
        from calibrar_extrinseca import crear_tablero, pose_tablero
        from skill_shield.teleop_fr5.cinematica import cargar_cadena
        self._estado_inicial()
        self.mover = mover
        self.fr5 = arrancar_fr5(mover)                   # lo primero: fork antes de abrir la cámara
        self.robot = Robot(self.fr5)
        self.pedal = crear_pedal() if mover else None
        self.cadena = cargar_cadena(getattr(cfg, "RUTA_XML", None))
        self.board, self.det = crear_tablero(tuple(TABLERO["cuadros"]), float(TABLERO["lado"]),
                                             float(TABLERO["marcador"]), TABLERO["diccionario"])
        self._pose_tablero = pose_tablero
        self.dev, self.cola, self.info = abrir_camara()
        self.K, self.dist = self.info["K"], self.info["dist"]
        self.P = C.puntos_tablero(TABLERO["cuadros"], TABLERO["lado"])
        self.ventana = "Calibracion camara-robot"
        cv2.namedWindow(self.ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.ventana, int(self.info["ancho"] * 0.75), int(self.info["alto"] * 0.75))

    def _estado_inicial(self):
        """El estado que no es hardware (el mismo en la sesión real y en la simulada de las pruebas)."""
        self.armado = False
        self._t_reset = -10.0
        self._en_error = False
        self._avisado_webapp = False
        self.n_colisiones = 0
        self.mano = None                                 # ChoqueManoBrazo: avisos del modo manual
        self.seguridad = None                            # calibracion.Colisiones (lo pone cada modo)
        self.motivo_salto = ""                           # por qué ir_a no ha llevado el robot a una postura
        self._trayecto = None                            # posturas del trayecto comprobado que se está recorriendo
        self.q = None
        self.T_ct, self.txt_tab = None, ""
        self.mensaje, self.t_mensaje = "", 0.0

    def cerrar(self):
        self.fr5.habilitar(False)
        cv2.destroyAllWindows()
        if self.pedal is not None:
            self.pedal.cerrar()
        self.fr5.parar()
        self.dev.close()

    def avisar(self, txt):
        self.mensaje, self.t_mensaje = txt, time.monotonic()
        print("   " + txt)

    def _parar(self):
        """Quieto ya: deja de mandar al FR5; para seguir hay que soltar y volver a pisar el pedal."""
        self.fr5.habilitar(False)
        self.armado = False
        if self.pedal is not None:
            self.pedal.desarmar()

    def _vigilar_postura(self):
        """Mientras se le manda un objetivo al robot (trayectos y capturas), su postura REAL —no la
        pedida— tiene que seguir dentro de lo comprobado. Si no hace lo que se le pide, se para
        aquí y la calibración termina (Detenido)."""
        seg = self.seguridad
        if seg is None or self.q is None:
            return
        if not seg.libre(self.q, MARGEN_VIGILANCIA):
            motivo = seg.motivo(self.q, MARGEN_VIGILANCIA) or "la postura real no vale"
            self._parar()
            raise Detenido(f"el robot se ha salido de lo comprobado: {motivo}. Se queda parado: sepáralo a "
                           "mano (arrastre en la WebApp) y vuelve a empezar.")
        if self._trayecto is not None and float(np.degrees(np.min(np.max(
                np.abs(self._trayecto - self.q), axis=1)))) > TUBO_DEG:
            # Está bien colocado, pero no en el trayecto que se comprobó (lo han movido a mano con el
            # pedal suelto, se ha desviado…): quieto hasta que ir_a compruebe el trayecto desde aquí.
            estaba_armado = self.armado
            self.fr5.habilitar(False)
            self.armado = False
            raise FueraDeTrayecto(estaba_armado)

    def _reiniciar(self, motivo):
        """Lo mismo que se hace a mano en la WebApp: Clear → disable → enable (fr5_servo.resetear).
        No mueve el robot: para seguir hay que soltar y volver a pisar el pedal."""
        if time.monotonic() - self._t_reset < 3.0:       # ya se está reiniciando
            return
        self._t_reset = time.monotonic()
        self.fr5.resetear()
        if self.pedal is not None:
            self.pedal.desarmar()
        self.armado = False
        self.avisar(f"{motivo}: borrando el error (Clear, disable, enable)... suelta la P y vuelve a pulsarla")

    def _vigilar_errores(self, key):
        """Colisión / fallo de la controladora: se reinicia solo (CALIB_AUTO_RESET) o con la X."""
        if key in (ord("x"), ord("X")) and self.fr5.en_linea:
            self._reiniciar("X pulsada")
        if self.fr5.en_error:
            if not self._en_error:
                self._en_error = True
                self.n_colisiones += 1
                print(f"✗ El FR5 marca colisión/fallo (nº {self.n_colisiones}). "
                      + ("Lo reinicio solo." if AUTO_RESET else "Pulsa X para borrarlo."))
                if self.pedal is not None:
                    self.pedal.desarmar()
                self.armado = False
                if AUTO_RESET:                           # una vez por colisión; si no se borra, X
                    self._reiniciar("Colisión")
            elif time.monotonic() - self._t_reset > 8.0 and not self._avisado_webapp:
                self._avisado_webapp = True
                self.avisar("No se borra el error: pulsa X otra vez o bórralo en la WebApp (Clear, Disable, "
                            "Enable); el programa lo notará solo")
        else:
            self._avisado_webapp = False
            self._en_error = False

    def _pedal(self, key, puede):
        self.pedal.tecla(key)
        self.pedal.actualizar()
        nuevo = bool(self.pedal.armado) and not self.fr5.en_error and puede \
            and time.monotonic() - self._t_reset > 2.0         # dejar acabar el reinicio
        if nuevo != self.armado:
            self.armado = nuevo
            self.fr5.habilitar(nuevo)
            print("🦶 " + ("pedal PISADO → el robot se mueve" if nuevo else "pedal suelto → quieto"))

    def vuelta(self, objetivo=None, hud=()):
        """Una vuelta: lee robot y cámara, publica el objetivo y atiende el pedal. Devuelve
        (tecla, hay imagen nueva). Lanza Cancelado con Q/Esc."""
        self.q = self.robot.leer()
        if objetivo is not None:
            if self.mover:
                self._vigilar_postura()
            self.fr5.publicar(objetivo)                  # cada vuelta (watchdog de fr5_servo)
        nueva = self._mirar(hud)
        key = self._tecla(nueva)
        if self.mover:
            self._vigilar_errores(key)
            # sin lectura de la postura real no se puede vigilar: mientras no llegue, no se le mueve
            a_ciegas = self.seguridad is not None and self.q is None
            self._pedal(key, objetivo is not None and not a_ciegas)
        if key in (27, ord("q"), ord("Q")):
            raise Cancelado()
        return key, nueva

    def _tecla(self, hay_imagen):
        """Tecla pulsada en la ventana (−1 si ninguna)."""
        return cv2.waitKeyEx(1 if hay_imagen else 3)

    def _mirar(self, hud=()):
        """Última imagen de la cámara: busca el tablero (deja T_ct y txt_tab) y la enseña con el
        HUD. Devuelve False si no ha llegado ninguna imagen nueva."""
        p, pkt = self.cola.tryGet(), None
        while p is not None:
            pkt, p = p, self.cola.tryGet()
        if pkt is None:
            return False
        img = pkt.getCvFrame()
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.T_ct, self.txt_tab, corners, ids = self._pose_tablero(self.det, self.board, gris, self.K, self.dist)
        if corners is not None and ids is not None and len(ids):
            try:                                          # solo es el dibujo: que nunca pare la calibración
                cv2.aruco.drawDetectedCornersCharuco(img, corners, ids, (0, 255, 0))
            except cv2.error:
                pass
        if self.T_ct is not None:
            cv2.drawFrameAxes(img, self.K, self.dist, cv2.Rodrigues(self.T_ct[:3, :3])[0], self.T_ct[:3, 3],
                              TABLERO["lado"] * 2)
        lineas = list(hud) + [
            "robot: " + ("sin lectura" if self.q is None else ("quieto" if self.robot.quieto else "en movimiento"))
            + ((("  |  COLISION: reiniciando..." if self.fr5.en_error or time.monotonic() - self._t_reset < 2.0
                  else "  |  PEDAL PISADO" if self.armado else "  |  pedal suelto (P)")) if self.mover else ""),
            "tablero: " + self.txt_tab]
        if self.mensaje and time.monotonic() - self.t_mensaje < 4:
            lineas.append(self.mensaje)
        y = 34
        for i, l in enumerate(lineas):
            color = (0, 200, 255) if (self.mensaje and l == self.mensaje) else (255, 255, 255)
            cv2.putText(img, l, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, l, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            y += 32
        cv2.imshow(self.ventana, img)
        return True

    @property
    def listo(self):
        return self.q is not None and self.robot.quieto and self.T_ct is not None

    def capturar(self, hud=(), objetivo=None):
        """Promedia FOTOS_POR_MUESTRA poses del tablero con el robot quieto. Muestra o None.
        objetivo: la postura en la que está (se sigue publicando para que la mantenga)."""
        fotos, q = [], None
        t0 = time.monotonic()
        while len(fotos) < FOTOS_POR_MUESTRA:
            _, nueva = self.vuelta(objetivo, hud=list(hud) + ["capturando..."])
            if not nueva:
                continue
            if not self.listo:
                if time.monotonic() - t0 > 3.0:
                    return None
                continue
            fotos.append(self.T_ct)
            q = self.q
        return dict(q=[float(v) for v in q], T_cam_tablero=C.promediar(fotos).tolist(),
                    tablero=self.txt_tab, t=time.time())

    def ir_a(self, q_obj, hud=()):
        """Lleva el robot a q_obj (solo con el pedal) y espera a que esté quieto allí.
        False (y el porqué en motivo_salto) si esa postura hay que saltarla: el trayecto desde
        donde está el robot no vale, la colisión salta 3 veces o el robot se para sin llegar.
        Lanza Detenido si, en marcha, la postura real del robot se sale de lo comprobado."""
        q_obj = np.asarray(q_obj, float)
        self.motivo_salto = ""
        seg = self.seguridad
        if seg is not None:
            # Última barrera: el trayecto se comprueba ENTERO (mano, tablero, brazo, soporte…) desde
            # donde está el robot AHORA, que no siempre es donde el plan creía (postura saltada,
            # colisión a medio camino, pedal soltado…). Si no vale, el robot no se mueve.
            while self.q is None:
                self.vuelta(None, hud=hud)
            if not self._comprobar_trayecto(q_obj):
                return False
        try:
            return self._seguir_trayecto(q_obj, hud)
        finally:
            self._trayecto = None

    def _comprobar_trayecto(self, q_obj):
        """Simula el trayecto desde la postura REAL hasta q_obj y lo comprueba entero. Si vale, lo
        deja en _trayecto (la vigilancia exige que el robot no se salga de él) y devuelve True."""
        camino = np.array(C.simular_camino(np.array(self.q), q_obj, VEL_MAX))
        motivo = self.seguridad.motivo_posturas(camino, MARGEN_TRAYECTO)
        if motivo is not None:
            self._trayecto = None
            self.motivo_salto = "por el camino, " + motivo
            self.avisar("NO voy a esa postura: " + self.motivo_salto)
            return False
        self._trayecto = camino
        return True

    def _seguir_trayecto(self, q_obj, hud):
        t_llegada = None
        n_desvios = 0                                    # veces que se ha salido del trayecto con el pedal pisado
        n0 = self.n_colisiones
        q_ref, t_ref = None, time.monotonic()           # para notar que el robot, armado, no avanza
        while True:
            if self.n_colisiones - n0 >= 3:
                self.motivo_salto = "3 colisiones seguidas yendo a ella"
                return False
            try:
                self.vuelta(q_obj, hud=hud)              # (vigila la postura real: puede lanzar Detenido)
            except FueraDeTrayecto as e:
                # Con el pedal suelto lo han movido a mano: se comprueba desde donde está y, si no
                # vale, se salta la postura. Con el pedal pisado se ha desviado él: se le deja
                # corregir un par de veces, pero un robot que no sigue lo que se le pide se para.
                n_desvios += e.armado
                self.avisar("el robot no está en el trayecto comprobado: lo compruebo desde donde está")
                vale = n_desvios < 3 and self._comprobar_trayecto(q_obj)
                if not vale and e.armado:
                    self._parar()
                    raise Detenido("el robot no sigue el trayecto que se le pide"
                                   + (f" (se ha desviado más de {TUBO_DEG:.0f}° tres veces yendo a la misma postura)"
                                      if n_desvios >= 3 else f" y desde donde está ya no vale: {self.motivo_salto}")
                                   + ". Se queda parado: revisa el robot antes de volver a empezar.")
                if not vale:
                    return False
                continue
            ahora = time.monotonic()
            cerca = self.q is not None and float(np.degrees(np.max(np.abs(self.q - q_obj)))) < LLEGADA_DEG
            t_llegada = (t_llegada or ahora) if cerca else None
            if t_llegada and ahora - t_llegada >= ESPERA_S and self.robot.quieto:
                return True
            # Con el pedal pisado el robot tiene que avanzar. Si se queda parado lejos del objetivo
            # es que alguien ha recortado lo pedido (límites blandos de la controladora más estrechos
            # que los de config…): esa postura no se alcanza nunca → se deja, no se espera más.
            if not self.armado or self.q is None or cerca:
                q_ref, t_ref = None, ahora
            elif q_ref is None or float(np.degrees(np.max(np.abs(self.q - q_ref)))) > 0.2:
                q_ref, t_ref = np.array(self.q), ahora
            elif ahora - t_ref > SIN_AVANCE_S:
                self.motivo_salto = (f"el robot se ha parado antes de llegar y lleva {SIN_AVANCE_S:.0f} s sin "
                                     "avanzar (¿límites blandos de la controladora?)")
                return False

    def recorrer(self, posturas, titulo, cada=None):
        """Recorre las posturas ({q, capturar}) y captura donde toque. Devuelve las muestras."""
        muestras = []
        n_cap = sum(p["capturar"] for p in posturas)
        for k, p in enumerate(posturas):
            hud = [titulo, f"postura {sum(x['capturar'] for x in posturas[:k + 1])}/{n_cap}"
                   + ("" if p["capturar"] else " (paso intermedio)") + f" · muestras {len(muestras)}"]
            m, llegado = None, True
            for _ in range(2):                         # si la colisión salta mientras captura, se repite
                n_col = self.n_colisiones
                if not self.ir_a(p["q"], hud=hud):
                    llegado = False
                    break
                if not p["capturar"]:
                    break
                m = self.capturar(hud=hud, objetivo=np.asarray(p["q"], float))
                if m is not None or self.n_colisiones == n_col:
                    break
            if not llegado:
                self.avisar(f"postura {k + 1}: {self.motivo_salto} → la salto")
                continue
            if not p["capturar"]:
                continue
            if m is None:
                self.avisar(f"postura {k + 1}: no se ve bien el tablero → la salto")
                continue
            muestras.append(m)
            if cada is not None:
                cada(muestras)
        self.fr5.habilitar(False)
        self.armado = False
        return muestras

    def esperar_tablero(self, hud, segundos=None):
        """Espera a tener pose del robot y el tablero a la vista (robot quieto)."""
        t0 = time.monotonic()
        while not self.listo:
            self.vuelta(None, hud=hud)
            if segundos is not None and time.monotonic() - t0 > segundos:
                return False
        return True


# ═════════════ modos ═════════════

def _guardar_muestras(s, muestras):
    _escribir_json(MUESTRAS, dict(fecha=datetime.datetime.now().isoformat(timespec="seconds"), tablero=TABLERO,
                                  intrinsecos=dict(K=s.K.tolist(), dist=s.dist.tolist(), ancho=s.info["ancho"],
                                                   alto=s.info["alto"]), mxid=s.info["mxid"], muestras=muestras))


def _texto_revision(rev):
    """Qué decirle al operario cuando la exploración no deja planificar (C.revisar_exploracion)."""
    lineas = ["✗ No sigo: con lo medido al explorar no se puede saber dónde están la cámara y el tablero."]
    for codigo, texto in rev["problemas"]:
        lineas.append("   · " + texto[0].upper() + texto[1:] + ".")
        if codigo == "escala" and np.isfinite(rev["escala"]):
            n = int(TABLERO["cuadros"][0])
            lado_mm = float(TABLERO["lado"]) * 1000
            lineas.append(f"     Mide con regla una FILA ENTERA de {n} cuadros (de borde a borde del ajedrezado, no un "
                          f"cuadro suelto): con lo que pone config.py mediría {n * lado_mm:.1f} mm; según el robot, "
                          f"≈ {n * lado_mm * rev['escala']:.0f} mm.")
            lineas.append("     Pon en CALIB_ROBOT_TABLERO (config.py) lado = lo medido / "
                          f"{n} y marcador = el lado del cuadrado NEGRO de un ArUco, medido igual (en metros).")
            lineas.append("     Si la regla da justo lo de config.py, avisa: con solo la exploración (movimientos "
                          "pequeños) esta estimación tiene ±2 % de incertidumbre.")
        elif codigo == "error":
            lineas.append("     Comprueba que el tablero está rígido y bien sujeto a la mano (que no se mueva "
                          "respecto a ella) y que la cámara lo ve entero y nítido.")
    lineas.append("   El robot solo ha hecho los movimientos pequeños de la exploración. Corrígelo y vuelve a "
                  "lanzar --auto.")
    return "\n".join(lineas)


def modo_auto(s=None):
    s = s or Sesion(mover=True)
    try:
        print("\n① Pon el robot (arrastre en la WebApp) donde la cámara vea el tablero entero y de frente.")
        s.esperar_tablero(["AUTO: coloca el robot donde la camara vea el tablero", "Q = salir"])
        q0 = np.array(s.q)
        print("   Tablero a la vista. Midiendo los huecos de la mano en esta postura (unos segundos)…")
        col = crear_seguridad(s.cadena, q0)              # sin tablero todavía (no se sabe dónde está)
        if col.mano.motivo(q0):
            sys.exit("✗ En la postura de inicio " + col.mano.motivo(q0) + ". Sepáralo (arrastre en la "
                     "WebApp: muñeca más recta, más lejos del soporte) y vuelve a probar.")
        s.seguridad = col
        if col.base:
            print(f"⚠️  El modelo de colisión ya ve choques en la postura de inicio {list(col.base)}: o el modelo "
                  "del soporte no cuadra con el real o el robot está muy cerca. Se tolera, pero no pueden empeorar.")
        explorar = []                                    # ida a cada una y vuelta a la de inicio
        for q in posturas_exploracion(s.cadena, q0, col):
            explorar += [dict(q=q.tolist(), capturar=True), dict(q=q0.tolist(), capturar=False)]
        if len(explorar) < 10:
            sys.exit("✗ Desde esta postura no hay sitio para explorar sin chocar: aléjala del soporte y la mesa.")
        print(f"\n② Explorando ({len(explorar) // 2} movimientos pequeños). MANTÉN EL PEDAL (P).")
        m0 = s.capturar(hud=["AUTO 1/3: explorando"])
        m1 = ([m0] if m0 else []) + s.recorrer(explorar, "AUTO 1/3: explorando")
        if len(m1) < 5:
            sys.exit(f"✗ Solo {len(m1)} muestras al explorar: ¿se ve bien el tablero? Acércalo o ponlo más de frente.")
        Tb = [s.cadena.fk_marcos(np.array(m["q"]))[5] for m in m1]
        Tc = [np.array(m["T_cam_tablero"]) for m in m1]
        rev = C.revisar_exploracion(Tb, Tc, s.P, float(TABLERO["lado"]))
        aprox = rev["aprox"]
        print(f"   calibración aproximada: error medio {aprox['error_medio_mm']:.1f} mm "
              f"(cámara en {np.round(aprox['T_mundo_cam'][:3, 3], 3).tolist()} m) · cuadrado del tablero "
              f"≈ {rev['lado_estimado'] * 1000:.1f} mm")
        if not rev["ok"]:                                # con esto no se puede planificar nada seguro
            sys.exit(_texto_revision(rev))

        print("\n③ Planificando posturas (sin moverse)…")
        plan, info, col = planificar_medida(s.cadena, q0, aprox["T_mundo_cam"], aprox["T_brida_tablero"],
                                            s.K, s.info["ancho"], s.info["alto"])
        s.seguridad = col
        print(f"   {info}")
        if info.get("elegidas", 0) < 8:
            sys.exit("✗ No salen suficientes posturas seguras y con el tablero a la vista. Prueba otra postura de "
                     "inicio (tablero más centrado en la imagen) o revisa CALIB_AUTO_AMPLITUD_DEG.")
        print(f"\n④ Midiendo {info['elegidas']} posturas. MANTÉN EL PEDAL (P).")
        m2 = s.recorrer(plan, "AUTO 3/3: midiendo", cada=lambda m: _guardar_muestras(s, m1 + m))
        muestras = m1 + m2
        _guardar_muestras(s, muestras)
        _escribir_json(POSTURAS, [dict(q=q0.tolist(), capturar=True)] + explorar + plan)
    except Cancelado:
        print("⏹  Cancelado.")
        return None
    except Detenido as e:
        sys.exit(f"⛔ Calibración detenida: {e}")
    finally:
        s.cerrar()
    return calcular_y_guardar(muestras, s.cadena, s.info)[1]


def modo_repetir_o_verificar(verificar, aqui=False, s=None):
    posturas = []
    if not aqui:
        posturas = _normalizar_posturas(_leer_json(POSTURAS, []))
        if not posturas:
            sys.exit(f"✗ No hay posturas en {POSTURAS}: calibra primero con --auto (o --manual).")
        if verificar:                                    # unas pocas de captura, con sus pasos intermedios
            idx_cap = [i for i, p in enumerate(posturas) if p["capturar"]]
            sel = set(np.array(idx_cap)[np.unique(np.linspace(0, len(idx_cap) - 1,
                                                              min(N_VERIFICAR, len(idx_cap))).round().astype(int))])
            posturas = [dict(p, capturar=(i in sel)) for i, p in enumerate(posturas) if i <= max(sel)]
    calib = C.cargar(ARCHIVO) if verificar else None
    s = s or Sesion(mover=not aqui)
    muestras = []
    try:
        if aqui:
            try:
                while True:
                    key, _ = s.vuelta(None, hud=["VERIFICAR (postura actual): ESPACIO = medir", "Q = terminar"])
                    if key != 32:
                        continue
                    m = s.capturar() if s.listo else None
                    if m is None:
                        s.avisar("no se puede medir: robot moviéndose o tablero no visible")
                        continue
                    muestras.append(m)
                    e = C.verificar(calib, [s.cadena.fk_marcos(np.array(m["q"]))[5]],
                                    [np.array(m["T_cam_tablero"])], s.P)[0]
                    s.avisar(f"medida {len(muestras)}: error {e:.1f} mm")
            except Cancelado:
                pass                                     # Q = terminar
        else:
            while s.q is None:
                s.vuelta(None, hud=["leyendo la postura del robot..."])
            q_act = np.array(s.q)
            try:                                         # para contar con el tablero y con la cámara
                guardada = C.cargar(ARCHIVO)
                T_bt, T_mc = guardada["T_brida_tablero"], guardada["T_mundo_cam"]
            except Exception:
                T_bt = T_mc = None
            # Mismas reglas que en --auto, con la primera postura guardada (la de inicio de aquella
            # calibración) como referencia: se revisan las posturas, los trayectos entre ellas y el
            # trayecto desde donde está el robot ahora. Si algo no vale, no se mueve.
            col = crear_seguridad(s.cadena, posturas[0]["q"], T_bt, T_mc)
            malas = [f"{k + 1} ({col.motivo(p['q'])})" for k, p in enumerate(posturas) if not col.libre(p["q"])]
            if malas:
                sys.exit("✗ Hay posturas guardadas que ya no valen: " + "; ".join(malas[:4])
                         + (f"… y {len(malas) - 4} más" if len(malas) > 4 else "")
                         + ". Vuelve a calibrar con --auto (elige posturas nuevas y seguras).")
            malos = [k + 1 for k in range(len(posturas) - 1)
                     if col.motivo_camino(posturas[k]["q"], posturas[k + 1]["q"], VEL_MAX) is not None]
            if malos:
                sys.exit(f"✗ Los trayectos que salen de las posturas guardadas {malos} no valen. "
                         "Vuelve a calibrar con --auto (planifica trayectos seguros).")
            motivo = col.motivo_camino(q_act, posturas[0]["q"], VEL_MAX, margen=MARGEN_TRAYECTO)
            if motivo is not None:
                sys.exit(f"✗ Desde donde está ahora el robot no puedo ir a la primera postura guardada: {motivo}. "
                         "Llévalo (arrastre en la WebApp) cerca de la postura de inicio de la calibración.")
            s.seguridad = col                            # ir_a: trayecto desde la postura real + vigilancia
            muestras = s.recorrer(posturas, "VERIFICAR" if verificar else "REPETIR",
                                  cada=None if verificar else (lambda m: _guardar_muestras(s, m)))
    except Cancelado:
        print("⏹  Cancelado.")
        return None
    except Detenido as e:
        sys.exit(f"⛔ Calibración detenida: {e}")
    finally:
        s.cerrar()
    if not verificar:
        if len(muestras) < 3:
            sys.exit("✗ Muy pocas muestras para recalibrar.")
        _guardar_muestras(s, muestras)
        return calcular_y_guardar(muestras, s.cadena, s.info)[1]
    if not muestras:
        print("Sin medidas.")
        return None
    Tb = [s.cadena.fk_marcos(np.array(m["q"]))[5] for m in muestras]
    Tc = [np.array(m["T_cam_tablero"]) for m in muestras]
    e = C.verificar(calib, Tb, Tc, s.P)
    ok = float(np.max(e)) <= ERROR_MAX
    _escribir_json(os.path.splitext(ARCHIVO)[0] + "_verificacion.json",
                   dict(fecha=datetime.datetime.now().isoformat(timespec="seconds"),
                        errores_mm=[round(float(v), 2) for v in e], ok=ok, umbral_mm=ERROR_MAX))
    print(f"\n{'✅' if ok else '❌'} Verificación: error medio {np.mean(e):.1f} mm, máx {np.max(e):.1f} mm "
          f"(umbral {ERROR_MAX:.0f} mm)" + ("" if ok else
          " → la cámara o el tablero se han movido: recalibra con  python calibrar_camara_robot.py --repetir"))
    return ok


def modo_manual(s=None):
    s = s or Sesion(mover=False)
    muestras = []
    s.mano = choque_mano(s.cadena)
    t_aviso = 0.0
    if os.path.exists(MUESTRAS):
        os.replace(MUESTRAS, MUESTRAS + ".bak")
    try:
        while True:
            key, _ = s.vuelta(None, hud=["MANUAL: mueve el robot (arrastre en la WebApp), ESPACIO = capturar",
                                         f"muestras {len(muestras)} (min. {MIN_MUESTRAS})",
                                         "D = quitar ultima   C = calcular   Q = terminar"])
            if s.q is not None and time.monotonic() - t_aviso > 3.0 and not s.mano.ok(s.q):
                s.avisar("¡CUIDADO! " + (s.mano.motivo(s.q) or "la mano está demasiado cerca de algo"))
                t_aviso = time.monotonic()
            if key == 32:
                m = s.capturar() if s.listo else None
                if m is None:
                    s.avisar("no capturo: " + ("sin pose del robot" if s.q is None else
                                               "el robot se mueve" if not s.robot.quieto else "no se ve el tablero"))
                    continue
                muestras.append(m)
                _escribir_json(POSTURAS, [mm["q"] for mm in muestras])
                _guardar_muestras(s, muestras)
                giro, rec = C.diversidad([s.cadena.fk_marcos(np.array(mm["q"]))[5] for mm in muestras])
                s.avisar(f"muestra {len(muestras)} guardada · giro {giro:.0f}° · recorrido {rec:.0f} mm")
            elif key in (8, 65288, ord("d"), ord("D")) and muestras:
                muestras.pop()
                _escribir_json(POSTURAS, [mm["q"] for mm in muestras])
                _guardar_muestras(s, muestras)
                s.avisar(f"quitada la última: quedan {len(muestras)}")
            elif key in (ord("c"), ord("C")):
                if len(muestras) < 3:
                    s.avisar("hacen falta al menos 3 muestras")
                else:
                    calcular_y_guardar(muestras, s.cadena, s.info)
    except Cancelado:
        pass
    finally:
        s.cerrar()
    if len(muestras) < 3:
        print("Menos de 3 muestras: no se calcula nada.")
        return None
    if len(muestras) < MIN_MUESTRAS:
        print(f"⚠️  Solo {len(muestras)} muestras (recomendadas ≥ {MIN_MUESTRAS}).")
    return calcular_y_guardar(muestras, s.cadena, s.info)[1]


def main():
    ap = argparse.ArgumentParser(description="Calibración cámara de cabeza → robot (paso 4)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--generar-tablero", action="store_true")
    g.add_argument("--auto", action="store_true", help="calibración automática (el robot elige y recorre las posturas)")
    g.add_argument("--manual", action="store_true", help="mueves tú el robot y pulsas ESPACIO en cada postura")
    g.add_argument("--repetir", action="store_true", help="el robot repite solo las posturas guardadas (con pedal)")
    g.add_argument("--verificar", action="store_true", help="comprueba la calibración guardada")
    g.add_argument("--recalcular", action="store_true", help="recalcula con calib_muestras.json")
    ap.add_argument("--aqui", action="store_true", help="con --verificar: en la postura actual, sin mover el robot")
    args = ap.parse_args()

    if args.generar_tablero:
        return generar_tablero()
    if args.recalcular:
        from skill_shield.teleop_fr5.cinematica import cargar_cadena
        d = _leer_json(MUESTRAS)
        if not d or len(d.get("muestras", [])) < 3:
            sys.exit(f"✗ No hay suficientes muestras en {MUESTRAS}")
        i = d["intrinsecos"]
        _, ok = calcular_y_guardar(d["muestras"], cargar_cadena(getattr(cfg, "RUTA_XML", None)),
                                   dict(mxid=d.get("mxid"), K=np.array(i["K"]), dist=np.array(i["dist"]),
                                        ancho=i["ancho"], alto=i["alto"]))
        return 0 if ok else 1
    if args.verificar and not os.path.exists(ARCHIVO):
        sys.exit(f"✗ No hay calibración ({ARCHIVO}): calibra primero con --auto.")
    if args.repetir or args.verificar:
        ok = modo_repetir_o_verificar(args.verificar, aqui=args.aqui)
    elif args.manual:
        ok = modo_manual()
    else:
        if not args.auto:
            print("(Sin opciones = calibración automática. --manual para hacerlo a mano.)")
        ok = modo_auto()
    return 0 if ok in (True, None) else 1


if __name__ == "__main__":
    sys.exit(main())
