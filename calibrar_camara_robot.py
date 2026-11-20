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
  python calibrar_camara_robot.py --verificar --sin   # (19-nov) igual pero SIN PEDAL: entra por la postura
                                                      #   guardada más cercana y va sola a ~4 posturas
  python calibrar_camara_robot.py --verificar --aqui  # comprueba en la postura actual, sin moverlo
  python calibrar_camara_robot.py --manual --sliders  # (20-nov) MANUAL CON SLIDERS: mueves el robot con la
                                                      #   ventana de sliders (ARMAR + pedal P mantenido);
                                                      #   cada movimiento se comprueba antes (soporte, mesa…).
                                                      #   + --sin: sin pedal, basta con ARMAR los sliders
  python calibrar_camara_robot.py --manual            # MANUAL: mueves tú el robot (arrastre en la
                                                      #   WebApp) y pulsas ESPACIO en cada postura
  python calibrar_camara_robot.py --recalcular        # recalcula con las muestras guardadas
  python calibrar_camara_robot.py --auto --sin        # (17-nov) cualquier modo + --sin: SIN PEDAL, el robot se
                                                      #   mueve solo (P = pausa/seguir, Q/Esc = parar y salir)
  python calibrar_camara_robot.py --girar-posturas 180  # la mano se ha vuelto a montar girada 180° en la
                                                      #   brida: adapta UNA VEZ las posturas guardadas
                                                      #   (J6 ± 180°) y el tablero de camara_robot.json
  python calibrar_camara_robot.py --camara a --auto   # (19-nov) la CÁMARA A (OAK-D fija de la mesa): igual,
                                                      #   con sus archivos (AUTONOMO_CAM_A_CALIB_ARCHIVO,
                                                      #   _POSTURAS, _MUESTRAS); vale con cualquier modo

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
TABLERO = dict(cfg.CALIB_ROBOT_TABLERO)


def _ruta(nombre_cfg, defecto):
    r = getattr(cfg, nombre_cfg, defecto)
    return r if os.path.isabs(r) else os.path.join(RAIZ, r)


ARCHIVO = _ruta("CALIB_ROBOT_ARCHIVO", "camara_robot.json")
POSTURAS = _ruta("CALIB_ROBOT_POSTURAS", "calib_posturas.json")
MUESTRAS = _ruta("CALIB_ROBOT_MUESTRAS", "calib_muestras.json")
# (19-nov) qué cámara se calibra: la de cabeza (por defecto) o la cámara A de la mesa (--camara a). Cada una con su
# archivo, sus posturas y sus muestras; las dos con el mismo tablero
CAMARA = "cabeza"
CAM_MXID = getattr(cfg, "AUTONOMO_CAM_MXID", None)
CAM_NOMBRE = "Cámara de cabeza"


def elegir_camara(cual):
    """(19-nov) Cambia archivos y MxId a los de la cámara 'cabeza' o 'a'."""
    global CAMARA, CAM_MXID, CAM_NOMBRE, ARCHIVO, POSTURAS, MUESTRAS
    CAMARA = cual
    if cual == "a":
        CAM_MXID = getattr(cfg, "AUTONOMO_CAM_A_MXID", None)
        CAM_NOMBRE = "Cámara A (mesa)"
        ARCHIVO = _ruta("AUTONOMO_CAM_A_CALIB_ARCHIVO", "camara_a_robot.json")
        POSTURAS = _ruta("AUTONOMO_CAM_A_POSTURAS", "calib_posturas_a.json")
        MUESTRAS = _ruta("AUTONOMO_CAM_A_MUESTRAS", "calib_muestras_a.json")
        if not CAM_MXID:
            sys.exit("✗ Falta AUTONOMO_CAM_A_MXID en config.py (el MxId de la cámara A).")
        print(f"📷 Calibrando la CÁMARA A (mesa) {CAM_MXID} → {os.path.basename(ARCHIVO)} "
              f"(posturas en {os.path.basename(POSTURAS)})")
MIN_MUESTRAS = int(getattr(cfg, "CALIB_ROBOT_MIN_MUESTRAS", 15))
ERROR_MAX = float(getattr(cfg, "CALIB_ROBOT_ERROR_MAX_MM", 5.0))
VEL_MAX = float(cfg.CALIB_ROBOT_VEL_MAX_DEG_S)
ESPERA_S = float(getattr(cfg, "CALIB_ROBOT_ESPERA_S", 1.0))
N_VERIFICAR = int(getattr(cfg, "CALIB_ROBOT_VERIFICAR_N", 4))
N_AUTO = int(getattr(cfg, "CALIB_AUTO_POSTURAS", 20))
RADIO_MANO = float(cfg.CALIB_AUTO_RADIO_MANO)
AMPLITUD = tuple(cfg.CALIB_AUTO_AMPLITUD_DEG)
AUTO_RESET = bool(getattr(cfg, "CALIB_AUTO_RESET", True))   # colisión → Clear, disable, enable solo
SIN_AVANCE_S = float(getattr(cfg, "CALIB_SIN_AVANCE_S", 5.0))   # armado y parado sin llegar → se deja esa postura
# (Los parámetros de seguridad de arriba se leen SIN valor por defecto: si faltan en config.py,
#  error al arrancar en vez de calibrar con otros más permisivos.)
# Lo PLANIFICADO se comprueba sin manga. Al robot de verdad, que nunca queda exactamente donde se
# le pide, se le dan unos mm: al comprobar el trayecto desde donde está y, algo más, al vigilarlo.
MARGEN_TRAYECTO = 0.006         # m de manga al comprobar un trayecto desde la postura REAL, antes de moverse
MARGEN_VIGILANCIA = 0.012       # m de manga a la postura REAL mientras se mueve (va algo por detrás de la pedida)
TUBO_DEG = 3.0                  # ° que la postura REAL puede separarse del trayecto comprobado sin volver a comprobarlo
FOTOS_POR_MUESTRA = int(getattr(cfg, "CALIB_ROBOT_FOTOS", 5))   # se promedia la pose del tablero en N imágenes seguidas
DISTANCIA = tuple(getattr(cfg, "CALIB_AUTO_DIST_M", (0.25, 1.6)))  # tablero a esta distancia de la cámara (m) al planificar
# (16-nov) el robot tiembla al pararse y la mano + el tablero siguen oscilando un rato aunque los joints
# ya digan «quieto»: se captura solo cuando el TABLERO está quieto EN LA IMAGEN (sus esquinas se mueven
# menos de TEMBLOR_PX de una imagen a la siguiente durante QUIETO_IMAGENES seguidas)
TEMBLOR_PX = float(getattr(cfg, "CALIB_ROBOT_TEMBLOR_PX", 0.35))
QUIETO_IMAGENES = int(getattr(cfg, "CALIB_ROBOT_QUIETO_IMAGENES", 5))
TEMBLOR_ESPERA_MAX_S = float(getattr(cfg, "CALIB_ROBOT_TEMBLOR_ESPERA_S", 8.0))
PERDIDAS_MAX = int(getattr(cfg, "CALIB_ROBOT_PERDIDAS_MAX", 2))   # imágenes seguidas sin tablero que se toleran al capturar
QUIETO_DEG = 0.05               # el robot cuenta como quieto si en 0,6 s se mueve menos que esto
LLEGADA_DEG = 0.3               # postura alcanzada
# (19-nov) ninguna espera sin fin: antes, si el robot llegaba pero no bajaba de QUIETO_DEG (tiembla con
# la mano + tablero), si la cámara dejaba de mandar imágenes o si la controladora se quedaba en error,
# el programa se quedaba esperando para siempre en esa postura («se queda pillado»)
CASI_QUIETO_DEG = float(getattr(cfg, "CALIB_ROBOT_CASI_QUIETO_DEG", 0.3))   # tras ESPERA_S + LLEGADA_TOLERANCIA_S, basta con esto
LLEGADA_TOLERANCIA_S = float(getattr(cfg, "CALIB_ROBOT_LLEGADA_TOLERANCIA_S", 4.0))
LLEGADA_MAX_S = float(getattr(cfg, "CALIB_ROBOT_LLEGADA_MAX_S", 15.0))   # llegado pero sin quedarse quieto → se salta
POSTURA_MARGEN_S = float(getattr(cfg, "CALIB_ROBOT_POSTURA_MARGEN_S", 25.0))   # sobre lo que dura el trayecto
ERROR_PERSISTENTE_S = float(getattr(cfg, "CALIB_ROBOT_ERROR_MAX_S", 30.0))      # controladora en error tanto → se para
SIN_IMAGEN_S = float(getattr(cfg, "CALIB_ROBOT_SIN_IMAGEN_S", 3.0))            # cámara sin imágenes tanto → no se captura
SIN_LECTURA_S = 10.0            # sin pose real del robot tanto → se para


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
    try:                                   # (16-nov) con los 300 ppp dentro del PNG: al 100 % sale a su tamaño
        from PIL import Image
        Image.fromarray(img).save(salida, dpi=(300, 300))
    except Exception:
        cv2.imwrite(salida, img)
    mm = lambda n: n * 25.4 / 300
    print(f"Guardado {salida} ({cuadros[0]}x{cuadros[1]} cuadros de {lado * 1000:.0f} mm, "
          f"diccionario {TABLERO['diccionario']}): dibujo de {cuadros[0] * lado * 1000:.0f} x "
          f"{cuadros[1] * lado * 1000:.0f} mm, hoja de {mm(img.shape[1]):.0f} x {mm(img.shape[0]):.0f} mm con el borde blanco."
          f"\nImprímelo a 300 ppp al 100 % (sin «ajustar a página»), "
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


def girar_posturas(posturas, grados, limites):
    """(12-nov) Posturas guardadas con la mano montada de otra forma: si ahora la mano (y el tablero)
    está girada 'grados' alrededor del eje de la brida respecto a entonces, girando el J6 al revés
    el tablero vuelve a quedar donde estaba (mismo sitio, misma cara hacia la cámara). De las dos
    opciones (J6 − grados, J6 − grados ± 360°) se coge la que cabe en los límites y queda más cerca
    de la anterior (sin vueltas de los cables). Las que no caben se quitan.
    Devuelve (posturas nuevas, cuántas se han quitado)."""
    g = math.radians(float(grados))
    lo, hi = float(limites[5][0]), float(limites[5][1])
    out, quitadas, ant = [], 0, None
    for p in posturas:
        q = list(map(float, p["q"]))
        cands = [q[5] - g + k * 2 * math.pi for k in (-1, 0, 1)]
        cands = [c for c in cands if lo <= c <= hi]
        if not cands:
            quitadas += 1
            continue
        q[5] = min(cands, key=lambda c: abs(c - (ant if ant is not None else q[5] - g)))
        ant = q[5]
        out.append(dict(p, q=q))
    return out, quitadas


def T_brida_tablero_girado(T_bt, grados):
    """El tablero, en el marco de la brida, con la mano girada 'grados' alrededor de su eje (z)."""
    a = math.radians(float(grados))
    R = np.eye(4)
    R[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return R @ np.asarray(T_bt, float)


def modo_girar_posturas(grados):
    """Adapta UNA VEZ calib_posturas.json y el tablero de camara_robot.json a la mano girada (con
    copia de seguridad *.antes_giro). La cámara (T_mundo_cam) no cambia: no depende de la mano."""
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    import shutil
    cadena = cargar_cadena(cfg.RUTA_XML, verbose=False)
    posturas = _normalizar_posturas(_leer_json(POSTURAS, []))
    if not posturas:
        sys.exit(f"✗ No hay posturas en {POSTURAS}.")
    nuevas, quitadas = girar_posturas(posturas, grados, limites_articulares(cadena))
    shutil.copy(POSTURAS, POSTURAS + ".antes_giro")
    _escribir_json(POSTURAS, nuevas)
    print(f"✓ {POSTURAS}: {len(nuevas)} posturas con el J6 girado {-float(grados):+.0f}°"
          + (f" ({quitadas} no cabían en los límites del J6 y se han quitado)" if quitadas else "")
          + f". Copia de las de antes: {POSTURAS}.antes_giro")
    if os.path.exists(ARCHIVO):
        with open(ARCHIVO, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("T_brida_tablero") is not None:
            shutil.copy(ARCHIVO, ARCHIVO + ".antes_giro")
            d["T_brida_tablero"] = T_brida_tablero_girado(d["T_brida_tablero"], grados).round(6).tolist()
            d["mano_girada_deg"] = float(d.get("mano_girada_deg", 0.0)) + float(grados)
            _escribir_json(ARCHIVO, d)
            print(f"✓ {ARCHIVO}: tablero girado {float(grados):.0f}° con la mano (la cámara no cambia). "
                  f"Copia: {ARCHIVO}.antes_giro")
    print("Ahora:  python calibrar_camara_robot.py --verificar   (y si sale mal, --repetir)")
    return True


def _escalar_traslacion(T, k):
    """T (4x4) con la traslación multiplicada por k (un tablero k veces mayor se ve k veces más lejos)."""
    T = np.array(T, float)
    T[:3, 3] *= k
    return T.tolist()


def calcular_y_guardar(muestras, cadena, info_cam):
    """Calibra con las muestras (FK recalculada desde q), enseña el resultado y lo guarda."""
    P = C.puntos_tablero(TABLERO["cuadros"], TABLERO["lado"])
    Tb = [cadena.fk_marcos(np.array(m["q"]))[5] for m in muestras]
    Tc = [np.array(m["T_cam_tablero"]) for m in muestras]
    giro, recorrido = C.diversidad(Tb)
    res, usadas = C.descartar_peores(Tb, Tc, P, umbral_mm=max(8.0, 1.6 * ERROR_MAX))
    print("\n── Resultado ──────────────────────────────────────────")
    for k, i in enumerate(usadas):
        m = muestras[i]
        extra = (f"   (dispersión {m['dispersion_mm']:.1f} mm{', TEMBLABA' if m.get('temblor') else ''})"
                 if "dispersion_mm" in m else "")
        print(f"   muestra {i + 1:2d}: {res['errores_mm'][k]:5.2f} mm{extra}")
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
    mxid = CAM_MXID                                     # (19-nov) la de cabeza o la A (--camara)
    print(f"📷 OAK-D disponibles: {listar_camaras()}")
    # (16-nov) más resolución (tablero pequeño), enfoque fijo y, si se quiere, exposición fija
    pipeline, w, h = _pipeline_charuco_hd(fps=int(getattr(cfg, "CALIB_CAM_FPS", 15)),
                                          escala=getattr(cfg, "CALIB_CAM_ESCALA", None), control=True)
    dev = abrir_dispositivo(pipeline, mxid)
    comprobar_usb(dev, CAMARA)
    K, dist = intrinsecos_color(dev, w, h)
    q = dev.getOutputQueue("color_hd", maxSize=2, blocking=False)
    ajustar_camara(dev)
    info = dict(mxid=mxid_de(dev), K=K, dist=np.asarray(dist).ravel(), ancho=w, alto=h)
    print(f"📷 {CAM_NOMBRE} {info['mxid']}: {w}x{h}, fx={K[0, 0]:.0f} fy={K[1, 1]:.0f} px")
    return dev, q, info


def ajustar_camara(dev):
    """Enfoque, exposición, nitidez y ruido de la cámara para calibrar (CALIB_CAM_* de config.py).
    ENFOQUE = "calibrado": la posición de la lente con la que se midieron en fábrica los intrínsecos
    que se usan (con el autoenfoque la lente se mueve, la focal cambia un poco y el tablero «mide»
    otra cosa); un número 0-255 = esa posición; None = autoenfoque. Nunca para la calibración: si la
    cámara no lo admite (foco fijo), lo dice y sigue."""
    import depthai as dai
    enfoque = getattr(cfg, "CALIB_CAM_ENFOQUE", "calibrado")
    expo = getattr(cfg, "CALIB_CAM_EXPOSICION", None)
    nitidez, ruido = getattr(cfg, "CALIB_CAM_NITIDEZ", None), getattr(cfg, "CALIB_CAM_RUIDO", None)
    hecho = []
    try:
        ctrl = dai.CameraControl()
        if enfoque is not None:
            lp = (dev.readCalibration().getLensPosition(dai.CameraBoardSocket.CAM_A)
                  if enfoque == "calibrado" else int(enfoque))
            if enfoque == "calibrado" and not lp:
                print("ℹ️  La cámara no guarda posición de lente (¿foco fijo?): se deja el enfoque como está.")
            else:
                ctrl.setManualFocus(int(lp))
                hecho.append(f"enfoque fijo {int(lp)}")
        if expo is not None:
            ctrl.setManualExposure(int(expo[0]), int(expo[1]))
            hecho.append(f"exposición {int(expo[0])} µs ISO {int(expo[1])}")
        if nitidez is not None:
            ctrl.setSharpness(int(nitidez))
            hecho.append(f"nitidez {int(nitidez)}")
        if ruido is not None:
            ctrl.setLumaDenoise(int(ruido))
            hecho.append(f"reducción de ruido {int(ruido)}")
        if hecho:
            dev.getInputQueue("control_hd").send(ctrl)
            print("📷 Cámara para calibrar: " + ", ".join(hecho))
    except Exception as e:
        print(f"⚠️  No se pudieron ajustar enfoque/exposición de la cámara ({e}): sigo con los automáticos.")


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


SIN_PEDAL = False   # --sin (17-nov): lo pone main()


class SinPedal:
    """--sin (17-nov): hace de pedal SIEMPRE PISADO, así que el robot recorre las posturas solo.
    Siguen todas las demás barreras (trayectos comprobados, vigilancia de la postura real,
    velocidad CALIB_ROBOT_VEL_MAX_DEG_S, reinicio tras colisión). La P alterna pausa / seguir
    (con la ventana de la cámara delante) y Q/Esc para y sale. Tras una colisión o un reinicio
    sigue solo, sin volver a pulsar nada (la Sesion deja 2 s para que acabe el reinicio)."""
    modo = "sin"
    hombre_muerto = True
    descripcion = "SIN PEDAL (--sin): el robot se mueve SOLO; P = pausa/seguir, Q/Esc = parar y salir"

    def __init__(self, teclas=(ord("p"), ord("P"))):
        self.teclas = tuple(teclas)
        self.pausa = False
        self.armado = True

    def tecla(self, key):
        if key is not None and key >= 0 and key in self.teclas:
            self.pausa = not self.pausa
            print("⏸  PAUSA: el robot se queda quieto (P para seguir)" if self.pausa
                  else "▶  Sigo (P para pausar)")
        self.armado = not self.pausa

    def actualizar(self):
        pass

    def desarmar(self):
        pass                     # sin pedal no hay que «soltar y volver a pisar»

    def cerrar(self):
        pass


def crear_pedal():
    if SIN_PEDAL:
        print("⚠️  --sin: SIN PEDAL DE HOMBRE MUERTO. El robot se moverá solo en cuanto el programa lo pida.\n"
              "    Nadie dentro de su alcance y la seta de emergencia a mano. P = pausa, Q/Esc = parar y salir.")
        return SinPedal()
    from skill_shield.teleop_fr5.pedal import Pedal
    p = Pedal.desde_config(cfg)
    if p.modo is None or not p.hombre_muerto:
        sys.exit("✗ Para que el robot se mueva solo hace falta un pedal de HOMBRE MUERTO: "
                 "TELEOP_FR5_PEDAL = \"teclado\" (tecla P mantenida) o \"evdev\" con "
                 "TELEOP_FR5_PEDAL_HOMBRE_MUERTO = True.")
    print(f"🦶 Pedal: {p.descripcion}. El robot solo se mueve MIENTRAS lo mantienes.")
    return p


def modelo_colision(cadena, T_mundo_cam=None):
    """El de la teleoperación (soporte, suelo, mesa/obstáculos de config) con la mano más gorda
    (lleva el tablero) y, si ya se sabe, la cámara de cabeza como obstáculo."""
    from skill_shield.autonomo.mapa import modelo_colision as _mc
    if CAMARA == "a":
        # (19-nov) calibrando la A: ella es «la cámara» del modelo y la de cabeza, si está calibrada, otra caja más
        from skill_shield.autonomo.mapa import _centro_calibrada
        c_cab = _centro_calibrada({k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
                                  "CALIB_ROBOT_ARCHIVO", "camara_robot.json")
        return _mc(cadena, cfg, T_mundo_cam, radio_mano=RADIO_MANO, nombre_camara="camara_a",
                   otras_camaras=[("camara_cabeza", c_cab)] if c_cab is not None else [])
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
                      float(cfg.CALIB_MANO_GIRO_DEG))


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
                                       limites=limites_articulares(cadena), distancia=DISTANCIA)
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
    def movimiento_deg(self):
        """Lo máximo que se ha movido un joint en los últimos 0,6 s (°), o None si no hay historial."""
        if len(self._hist) < 5 or self._hist[-1][0] - self._hist[0][0] < 0.45:
            return None
        Q = np.array([q for _, q in self._hist])
        return float(np.degrees(np.max(Q.max(0) - Q.min(0))))

    @property
    def quieto(self):
        m = self.movimiento_deg
        return m is not None and m < QUIETO_DEG


# ═════════════ sesión: ventana + cámara + robot ═════════════

def abrir_mano_recta(fabrica=None, esperar_s=None):
    """(12-nov) Sin par, la OrcaHand se queda floja y el tablero pegado a ella se mueve (y se ve de
    lado). Con CALIB_MANO_RECTA, se conecta con par (como en la tarea), se pone recta (muñeca a 0 y
    dedos estirados: CALIB_MANO_POSTURA, por defecto la mano abierta de la tarea, que es como la
    tienen las comprobaciones de choque) y se queda así hasta el final de la calibración. Si no se
    puede (otro programa usa el puerto, mano apagada…), se avisa y se sigue sin ella.
    Devuelve la ManoActiva (o None)."""
    if not getattr(cfg, "CALIB_MANO_RECTA", True):
        return None
    from skill_shield.autonomo.tarea import MANO_ABIERTA, ManoActiva
    m = ManoActiva(getattr(cfg, "PUERTO_HARDWARE", "/dev/ttyUSB0"), fabrica=fabrica)
    print("✋ Mano: conectando con par para dejarla recta (cierra antes el modo autónomo y la teleoperación)…")
    if not m.conectar():
        print(f"⚠️  Mano: {m.error}. Sigo SIN par: sujeta bien el tablero (no debe moverse respecto a la mano).")
        return None
    # None = la mano abierta de la TAREA tal como esté en config.py (antes: la copia fija de tarea.py)
    postura = {int(k): float(v) for k, v in (getattr(cfg, "CALIB_MANO_POSTURA", None)
                                             or getattr(cfg, "AUTONOMO_TAREA_MANO_ABIERTA", None)
                                             or MANO_ABIERTA).items()}
    m.enviar(postura)
    espera = float(getattr(cfg, "CALIB_MANO_ESPERA_S", 3.0) if esperar_s is None else esperar_s)
    if espera > 0:
        print(f"✋ Mano: poniéndose recta ({espera:.0f} s)…")
        time.sleep(espera)
    print("✋ Mano recta y con par mientras dure la calibración.")
    return m


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
        # (19-nov) la mano ANTES de abrir la ventana: su espera de 3 s (time.sleep) dejaba la ventana
        # recién creada congelada («no responde»)
        self.mano_recta = abrir_mano_recta()             # (12-nov) la mano con par, recta, sujetando el tablero
        self.ventana = "Calibracion camara-robot"
        cv2.namedWindow(self.ventana, cv2.WINDOW_NORMAL)
        _f = min(0.75, 1400.0 / float(self.info["ancho"]))   # (16-nov) a 2028x1520 no cabe en pantalla
        cv2.resizeWindow(self.ventana, int(self.info["ancho"] * _f), int(self.info["alto"] * _f))

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
        self.mov_px, self._esq_ant = None, None      # (16-nov) cuánto se ha movido el tablero en la imagen
        self.mensaje, self.t_mensaje = "", 0.0
        self.mano_recta = None

    def cerrar(self):
        self.fr5.habilitar(False)
        cv2.destroyAllWindows()
        if getattr(self, "mano_recta", None) is not None:
            self.mano_recta.cerrar()                     # vuelve a su postura neutra y se suelta
        if self.pedal is not None:
            self.pedal.cerrar()
        self.fr5.parar()
        self.dev.close()

    def avisar(self, txt):
        self.mensaje, self.t_mensaje = txt, time.monotonic()
        print("   " + txt)

    def mostrar_error(self, texto, segundos=None):
        """(19-nov) Antes, un sys.exit con la ventana abierta la cerraba sin más y el porqué solo
        salía en la terminal. Ahora el robot se queda quieto y el motivo se ve EN LA VENTANA hasta
        pulsar Q/Esc/ESPACIO o pasar CALIB_ERROR_VISIBLE_S (para que con --sin no se quede esperando)."""
        import textwrap
        print("✗ " + texto)
        self.fr5.habilitar(False)
        self.armado = False
        seg = float(getattr(cfg, "CALIB_ERROR_VISIBLE_S", 20.0) if segundos is None else segundos)
        lineas = ["NO SE PUEDE SEGUIR (Q / ESPACIO = cerrar)"] + textwrap.wrap(texto, 80)
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < seg:
                key, _ = self.vuelta(None, hud=lineas)
                if key in (13, 32):
                    break
        except Cancelado:
            pass

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
        self.avisar(f"{motivo}: borrando el error (Clear, disable, enable)... "
                    + ("sigo solo en 2 s" if isinstance(self.pedal, SinPedal) else "suelta la P y vuelve a pulsarla"))

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
            if not isinstance(self.pedal, SinPedal):
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
        self.mov_px = self._movimiento(corners, ids)
        if self.mov_px is not None:
            self.txt_tab += f"  |  mov {self.mov_px:.2f}px"
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
                  else self._txt_pedal())) if self.mover else ""),
            "tablero: " + self.txt_tab]
        if self.mensaje and time.monotonic() - self.t_mensaje < 4:
            lineas.append(self.mensaje)
        k = max(1.0, img.shape[1] / 1352.0)              # (16-nov) letra legible también a 2028x1520
        y = int(34 * k)
        for i, l in enumerate(lineas):
            color = (0, 200, 255) if (self.mensaje and l == self.mensaje) else (255, 255, 255)
            cv2.putText(img, l, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, (0, 0, 0), int(4 * k), cv2.LINE_AA)
            cv2.putText(img, l, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, color, int(2 * k), cv2.LINE_AA)
            y += int(32 * k)
        cv2.imshow(self.ventana, img)
        return True

    def _txt_pedal(self):
        if isinstance(self.pedal, SinPedal):
            return "  |  SIN PEDAL: EN PAUSA (P)" if self.pedal.pausa else "  |  SIN PEDAL (P = pausa)"
        return "  |  PEDAL PISADO" if self.armado else "  |  pedal suelto (P)"

    def _movimiento(self, corners, ids):
        """Mediana (px) de lo que se han movido las esquinas del tablero desde la imagen anterior
        (solo las que salen en las dos), o None si no hay con qué comparar."""
        if corners is None or ids is None or len(ids) == 0:
            return None          # (17-nov) se guarda la última vista: un fotograma perdido no reinicia
        act = {int(i): c for i, c in zip(np.asarray(ids).ravel(), np.asarray(corners).reshape(-1, 2))}
        ant, self._esq_ant = self._esq_ant, act
        if not ant:
            return None
        comunes = [k for k in act if k in ant]
        if len(comunes) < 4:
            return None
        return float(np.median([np.linalg.norm(act[k] - ant[k]) for k in comunes]))

    @property
    def listo(self):
        return self.q is not None and self.robot.quieto and self.T_ct is not None

    def capturar(self, hud=(), objetivo=None):
        """Promedia FOTOS_POR_MUESTRA poses del tablero con el robot quieto. Muestra o None.
        objetivo: la postura en la que está (se sigue publicando para que la mantenga)."""
        fotos, q = [], None
        t0 = time.monotonic()
        self.motivo_captura = ""
        quietas, temblor, movs, perdidas = 0, False, [], 0
        t_img = t0
        while len(fotos) < FOTOS_POR_MUESTRA:
            _, nueva = self.vuelta(objetivo, hud=list(hud) + ["capturando..." if quietas >= QUIETO_IMAGENES
                                                              else "esperando a que el tablero deje de temblar..."])
            if not nueva:
                # (19-nov) antes, sin imágenes nuevas se esperaba para siempre
                if time.monotonic() - t_img > SIN_IMAGEN_S:
                    self.motivo_captura = f"la cámara lleva {SIN_IMAGEN_S:.0f} s sin mandar imágenes"
                    return None
                continue
            t_img = time.monotonic()
            # (19-nov) como en ir_a: pasado LLEGADA_TOLERANCIA_S basta con «casi quieto» (la mano con el
            # tablero tiembla algo más que QUIETO_DEG); lo que manda es que el TABLERO esté quieto en la imagen
            mov = self.robot.movimiento_deg
            robot_ok = self.robot.quieto or (mov is not None and mov < CASI_QUIETO_DEG
                                              and time.monotonic() - t0 > LLEGADA_TOLERANCIA_S)
            if not (self.q is not None and self.T_ct is not None and robot_ok):
                # (17-nov) con el robot quieto, 1-2 imágenes sueltas sin tablero no reinician la espera
                if self.q is not None and robot_ok and self.T_ct is None and perdidas < PERDIDAS_MAX:
                    perdidas += 1
                    continue
                if time.monotonic() - t0 > max(3.0, LLEGADA_TOLERANCIA_S + 1.0) + (TEMBLOR_ESPERA_MAX_S if movs else 0.0):
                    # (12-nov) por qué: para saber qué arreglar (tablero, luz, robot que no se para…)
                    self.motivo_captura = (f"el robot no se queda quieto (se mueve {mov or 0:.2f}°)"
                                           if self.q is not None and not robot_ok
                                           else "tablero: " + (self.txt_tab or "no se ve"))
                    return None
                quietas, fotos = 0, []
                continue
            perdidas = 0
            # (16-nov) el tablero tiene que estar quieto EN LA IMAGEN; si se mueve, se empieza otra vez.
            # Si tras TEMBLOR_ESPERA_MAX_S sigue temblando, se captura igual y se avisa (no se pierde la postura).
            m = self.mov_px
            if m is not None:
                movs.append(m)
            esperando = time.monotonic() - t0 < TEMBLOR_ESPERA_MAX_S
            if (m is None or m > TEMBLOR_PX) and esperando:
                quietas, fotos = 0, []
                continue
            if m is not None and m > TEMBLOR_PX:
                temblor = True
            quietas += 1
            if quietas <= QUIETO_IMAGENES and esperando:
                continue
            fotos.append(self.T_ct)
            q = self.q
        if temblor:
            print(f"   ⚠️  el tablero seguía temblando ({np.median(movs[-FOTOS_POR_MUESTRA:]):.2f} px entre imágenes): "
                  "esta muestra será peor")
        Ts = np.array([T[:3, 3] for T in fotos])
        dispersion_mm = float(np.max(np.linalg.norm(Ts - Ts.mean(0), axis=1)) * 1000)
        return dict(q=[float(v) for v in q], T_cam_tablero=C.promediar(fotos).tolist(),
                    tablero=self.txt_tab, t=time.time(), temblor=bool(temblor),
                    dispersion_mm=round(dispersion_mm, 2), espera_s=round(time.monotonic() - t0, 1))

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
            t0 = time.monotonic()
            while self.q is None:
                self.vuelta(None, hud=list(hud) + ["esperando la postura real del robot..."])
                if time.monotonic() - t0 > SIN_LECTURA_S:
                    self._parar()
                    raise Detenido(f"lleva {SIN_LECTURA_S:.0f} s sin leer la postura del robot (¿conexión con el FR5?)")
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
        # (19-nov) tiempo máximo para llegar: lo que dura el trayecto comprobado + margen, sin contar
        # el rato en pausa / con el pedal suelto (eso es cosa del operario)
        t0 = t_prev = time.monotonic()
        t_parado = 0.0
        dura = 0.0 if self._trayecto is None else len(self._trayecto) * 0.04
        limite = 1.5 * dura + POSTURA_MARGEN_S
        t_error, estado = None, ""
        sin_pedal = isinstance(self.pedal, SinPedal)
        while True:
            if self.n_colisiones - n0 >= 3:
                self.motivo_salto = "3 colisiones seguidas yendo a ella"
                return False
            try:
                self.vuelta(q_obj, hud=list(hud) + ([estado] if estado else []))   # (puede lanzar Detenido)
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
            dt, t_prev = ahora - t_prev, ahora
            en_pausa = (sin_pedal and self.pedal.pausa) or (not sin_pedal and not self.armado
                                                             and not self.fr5.en_error)
            if en_pausa:
                t_parado += dt
            # controladora en error que no se borra: con --sin nadie pulsa X → se reintenta solo
            if self.fr5.en_error:
                t_error = t_error or ahora
                if sin_pedal and ahora - self._t_reset > 8.0:
                    self._reiniciar("Sigue en error")
                if ahora - t_error > ERROR_PERSISTENTE_S:
                    self._parar()
                    raise Detenido(f"la controladora lleva {ERROR_PERSISTENTE_S:.0f} s en error y no se borra: "
                                   "bórralo en la WebApp (Clear, Disable, Enable) y vuelve a empezar")
            else:
                t_error = None
            dist = None if self.q is None else float(np.degrees(np.max(np.abs(self.q - q_obj))))
            mov = self.robot.movimiento_deg
            cerca = dist is not None and dist < LLEGADA_DEG
            t_llegada = (t_llegada or ahora) if cerca else None
            estado = ("sin lectura del robot" if dist is None else
                      f"llegado, esperando a que se quede quieto (se mueve {mov:.2f} deg)" if cerca and mov is not None
                      else f"a {dist:.1f} deg del objetivo")
            if t_llegada and ahora - t_llegada >= ESPERA_S and self.robot.quieto:
                return True
            if t_llegada and ahora - t_llegada >= ESPERA_S + LLEGADA_TOLERANCIA_S \
                    and mov is not None and mov < CASI_QUIETO_DEG:
                # tiembla algo más que QUIETO_DEG (mano + tablero), pero está ahí: la captura ya
                # espera a que el TABLERO esté quieto en la imagen
                return True
            if t_llegada and ahora - t_llegada > LLEGADA_MAX_S:
                self.motivo_salto = (f"llega pero no se queda quieto (se mueve {mov or 0:.2f}° en 0,6 s; "
                                     f"CALIB_ROBOT_CASI_QUIETO_DEG = {CASI_QUIETO_DEG})")
                return False
            if ahora - t0 - t_parado > limite:
                self.motivo_salto = (f"lleva {limite:.0f} s sin llegar"
                                     + ("" if dist is None else f" (se ha quedado a {dist:.1f}° del objetivo)"))
                return False
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
                self.avisar(f"postura {k + 1}: no se ve bien el tablero ({getattr(self, 'motivo_captura', '')}) "
                            f"→ la salto")
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
        print(f"\n② Explorando ({len(explorar) // 2} movimientos pequeños). " + ("SIN PEDAL: va solo (P = pausa)." if SIN_PEDAL else "MANTÉN EL PEDAL (P)."))
        m0 = s.capturar(hud=["AUTO 1/3: explorando"])
        m1 = ([m0] if m0 else []) + s.recorrer(explorar, "AUTO 1/3: explorando")
        if len(m1) < 5:
            sys.exit(f"✗ Solo {len(m1)} muestras al explorar (último motivo: {getattr(s, 'motivo_captura', '') or '?'}): "
                     "¿se ve bien el tablero? Acércalo o ponlo más de frente. Si dice «reproy … px», el tablero no "
                     "está plano (cartón doblado, plástico encima que brilla).")
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
        print(f"\n④ Midiendo {info['elegidas']} posturas. " + ("SIN PEDAL: va solo (P = pausa)." if SIN_PEDAL else "MANTÉN EL PEDAL (P)."))
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


def _ruta_verificacion(posturas, col, i0, n, latido=None):
    """(19-nov) Ruta corta para --verificar desde la postura guardada i0 (a la que se entra desde
    donde esté el robot). Antes se recorrían TODAS las posturas hasta la última elegida (con 4
    capturas, las ~50 de la calibración). Ahora: se va hacia el lado con más posturas de captura,
    se eligen n de ellas repartidas y entre una y otra se ataja por las posturas guardadas: de cada
    una se salta a la más lejana (sin pasar la siguiente captura) cuyo trayecto DIRECTO vale con
    el mismo comprobador que --auto (col.motivo_camino); si ninguno vale, al paso siguiente. Así
    los atajos nunca son menos seguros que el plan, e ir_a lo vuelve a comprobar desde la postura
    real antes de moverse. latido(): se llama antes de cada comprobación (mantiene viva la ventana).
    Devuelve la lista de {q, capturar} o None si no se puede trazar."""
    q = [np.asarray(p["q"], float) for p in posturas]
    cap = [i for i, p in enumerate(posturas) if p["capturar"]]
    adelante, atras = [i for i in cap if i >= i0], [i for i in cap if i <= i0]
    if len(adelante) >= len(atras):
        orden, caps = list(range(i0, len(posturas))), adelante
    else:
        orden, caps = list(range(i0, -1, -1)), atras[::-1]
    if not caps:
        return None
    sel = {caps[k] for k in np.unique(np.linspace(0, len(caps) - 1, min(n, len(caps))).round().astype(int))}
    pos_de = {idx: k for k, idx in enumerate(orden)}
    ruta = [dict(q=q[orden[0]].tolist(), capturar=orden[0] in sel)]
    pos = 0
    for t in sorted(pos_de[i] for i in sel):
        while pos < t:
            for j in range(t, pos, -1):
                if latido is not None:
                    latido()
                if col.motivo_camino(q[orden[pos]], q[orden[j]], VEL_MAX) is None:
                    break
            else:
                return None                               # ni el paso a la siguiente postura vale
            ruta.append(dict(q=q[orden[j]].tolist(), capturar=orden[j] in sel))
            pos = j
    return ruta


def _entrada(posturas, col, q_act, latido=None, candidatas=12):
    """(19-nov) Índice de la postura guardada más cercana (en joints) a la que se puede ir desde donde
    está el robot AHORA (trayecto comprobado con la manga de la postura real), o None. Antes solo
    valía la primera postura guardada: tras un --auto el robot se queda en la última, y si lo habían
    movido después (modo autónomo, WebApp…) el programa se cerraba al poco de abrir la ventana."""
    q_act = np.asarray(q_act, float)
    dist = [float(np.max(np.abs(np.asarray(p["q"], float) - q_act))) for p in posturas]
    for k in np.argsort(dist)[:candidatas]:
        if latido is not None:
            latido()
        if col.motivo_camino(q_act, posturas[int(k)]["q"], VEL_MAX, margen=MARGEN_TRAYECTO) is None:
            return int(k)
    return None


def modo_repetir_o_verificar(verificar, aqui=False, s=None):
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    titulo = "VERIFICAR" if verificar else "REPETIR"
    if SIN_PEDAL and not aqui:
        titulo += " (SIN PEDAL: P = pausa, Q = salir)"
    elif not aqui:
        titulo += " (MANTEN EL PEDAL: P)"
    if aqui and SIN_PEDAL:
        print("ℹ️  --sin no hace nada con --aqui: el robot no se mueve.")
    calib = C.cargar(ARCHIVO) if verificar else None
    posturas, col = [], None
    if not aqui:
        posturas = _normalizar_posturas(_leer_json(POSTURAS, []))
        if not posturas:
            sys.exit(f"✗ No hay posturas en {POSTURAS}: calibra primero con --auto (o --manual).")
        # (19-nov) Todo lo que no necesita el robot se comprueba ANTES de abrir la cámara y la
        # ventana: antes se hacía con la ventana ya abierta y sin atenderla (≈10 s congelada) y,
        # si algo no valía, sys.exit la cerraba sin decir nada en ella.
        print("🔎 Comprobando las posturas guardadas con el modelo de choques (unos segundos)…")
        cadena = s.cadena if s is not None else cargar_cadena(getattr(cfg, "RUTA_XML", None))
        try:                                             # para contar con el tablero y con la cámara
            guardada = calib if calib is not None else C.cargar(ARCHIVO)
            T_bt, T_mc = guardada["T_brida_tablero"], guardada["T_mundo_cam"]
        except Exception:
            T_bt = T_mc = None
        # Mismas reglas que en --auto, con la primera postura guardada (la de inicio de aquella
        # calibración) como referencia de los huecos.
        col = crear_seguridad(cadena, posturas[0]["q"], T_bt, T_mc)
        malas = [f"{k + 1} ({col.motivo(p['q'])})" for k, p in enumerate(posturas) if not col.libre(p["q"])]
        if malas:
            sys.exit("✗ Hay posturas guardadas que ya no valen: " + "; ".join(malas[:4])
                     + (f"… y {len(malas) - 4} más" if len(malas) > 4 else "")
                     + ". Vuelve a calibrar con --auto (elige posturas nuevas y seguras).")
        if not verificar:                                # --repetir recorre TODOS los trayectos guardados
            malos = [k + 1 for k in range(len(posturas) - 1)
                     if col.motivo_camino(posturas[k]["q"], posturas[k + 1]["q"], VEL_MAX) is not None]
            if malos:
                sys.exit(f"✗ Los trayectos que salen de las posturas guardadas {malos} no valen. "
                         "Vuelve a calibrar con --auto (planifica trayectos seguros).")
    s = s or Sesion(mover=not aqui)
    muestras = []
    error = None
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
                s.vuelta(None, hud=[titulo, "leyendo la postura del robot..."])
            hud_prep = [titulo, "preparando la ruta (el robot no se mueve)..."]
            latido = lambda: s.vuelta(None, hud=hud_prep)    # la ventana sigue viva mientras se comprueba
            if verificar:
                i0 = _entrada(posturas, col, s.q, latido)
                ruta = None if i0 is None else _ruta_verificacion(posturas, col, i0, N_VERIFICAR, latido)
                if i0 is None:
                    error = ("Desde donde está ahora el robot no hay un trayecto seguro a ninguna de las posturas "
                             "guardadas cercanas. Llévalo (arrastre en la WebApp) cerca de la postura de inicio "
                             "de la calibración y vuelve a lanzar --verificar.")
                elif ruta is None:
                    error = "No se puede trazar una ruta segura por las posturas guardadas. Recalibra con --auto."
                else:
                    print(f"   Ruta de verificación: entra por la postura guardada {i0 + 1}, "
                          f"{len(ruta)} posturas, {sum(p['capturar'] for p in ruta)} medidas.")
            else:
                ruta = posturas
                motivo = col.motivo_camino(s.q, posturas[0]["q"], VEL_MAX, margen=MARGEN_TRAYECTO)
                if motivo is not None:
                    error = (f"Desde donde está ahora el robot no puedo ir a la primera postura guardada: {motivo}. "
                             "Llévalo (arrastre en la WebApp) cerca de la postura de inicio de la calibración.")
            if error is None:
                print("   " + ("SIN PEDAL: el robot va solo (P = pausa, Q/Esc = parar)." if SIN_PEDAL
                               else "MANTÉN EL PEDAL (P) con la ventana delante."))
                s.seguridad = col                        # ir_a: trayecto desde la postura real + vigilancia
                muestras = s.recorrer(ruta, titulo,
                                      cada=None if verificar else (lambda m: _guardar_muestras(s, m)))
            else:
                s.mostrar_error(error)
    except Cancelado:
        print("⏹  Cancelado.")
        return None
    except Detenido as e:
        s.mostrar_error(f"Calibración detenida: {e}")
        sys.exit(f"⛔ Calibración detenida: {e}")
    finally:
        s.cerrar()
    if error is not None:
        sys.exit("✗ " + error)
    if not verificar:
        if len(muestras) < 3:
            sys.exit("✗ Muy pocas muestras para recalibrar.")
        _guardar_muestras(s, muestras)
        return calcular_y_guardar(muestras, s.cadena, s.info)[1]
    if not muestras:
        print("❌ Verificación sin medidas: no se pudo capturar en ninguna postura (mira arriba el motivo de cada una).")
        return False
    Tb = [s.cadena.fk_marcos(np.array(m["q"]))[5] for m in muestras]
    Tc = [np.array(m["T_cam_tablero"]) for m in muestras]
    e_guardado = C.verificar(calib, Tb, Tc, s.P)
    e = e_guardado
    if len(muestras) >= 3:
        # (12-nov) Si el tablero se ha vuelto a pegar en la mano (o la mano se ha montado de otra
        # forma), el T_brida_tablero guardado ya no vale aunque la cámara esté perfecta. Se vuelve
        # a medir dónde está el tablero en la mano con estas mismas muestras y la cámara guardada:
        # si con eso cuadran, la CÁMARA está bien (lo que se movió fue el tablero).
        T_bt = C.promediar([C.inv_T(b) @ calib["T_mundo_cam"] @ c for b, c in zip(Tb, Tc)])
        e = C.verificar(dict(calib, T_brida_tablero=T_bt), Tb, Tc, s.P)
        print(f"   con el tablero como estaba guardado: error medio {np.mean(e_guardado):.1f} mm · "
              f"con el tablero medido ahora en la mano: {np.mean(e):.1f} mm (esto es lo que mide la cámara)")
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


# ═════════════ (20-nov) manual con los sliders del brazo ═════════════

def abrir_sliders():
    """La ventana de sliders del modo «gemelo con sliders» (skill_shield/interfaz/sliders_brazo.py).
    Se abre ANTES que la Sesion: es un proceso aparte (fork) y tiene que nacer antes de la cámara."""
    from skill_shield.interfaz.sliders_brazo import SlidersBrazo
    sl = SlidersBrazo.desde_config({k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    if not sl.iniciar():
        sys.exit("✗ No se puede abrir la ventana de sliders (¿sudo apt install python3.12-tk?).")
    return sl


def modo_manual_sliders(s=None, sliders=None):
    """--manual --sliders: como --manual, pero el robot se mueve con la ventana de sliders en vez de
    con el arrastre de la WebApp. El robot solo se mueve si:
      · los sliders están ARMADOS (botón ARMAR; STOP / Espacio en su ventana / cerrarla = desarmar),
      · el pedal de hombre muerto está pisado (TELEOP_FR5_PEDAL; con "teclado", la P mantenida, que
        se pulsa con la ventana de la CÁMARA delante; luego se puede soltar el foco e ir a los
        sliders con el ratón sin soltar la P),
      · y el trayecto desde donde está el robot DE VERDAD hasta lo que piden los sliders pasa las
        mismas comprobaciones que --auto (brazo, mano + tablero, soporte, cabeza, suelo, cajas de
        TELEOP_FR5_OBSTACULOS —¡la mesa!—, y ningún hueco se cierra más de lo permitido respecto
        a la postura de inicio). Si no las pasa, el robot no va y los sliders se desarman.
    Con --sin no hace falta el pedal: el robot se mueve en cuanto los sliders están ARMADOS (STOP,
    Espacio en su ventana o cerrarla lo para; la P en la ventana de la cámara hace de pausa).
    Velocidad máxima CALIB_ROBOT_VEL_MAX_DEG_S. Con el robot quieto, ESPACIO (en la ventana de la
    cámara) captura; D quita la última; C calcula; Q termina."""
    from skill_shield.interfaz.sliders_brazo import FR5_ENVIANDO, FR5_ERROR, FR5_ESPERA
    sl = sliders or abrir_sliders()
    s = s or Sesion(mover=True)
    muestras = []
    lim = limites_articulares(s.cadena)
    objetivo = None                 # lo que se le manda al robot: lo último de los sliders que ha pasado
    pedido_ant, t_comprobado = None, -1e9
    if os.path.exists(MUESTRAS):
        os.replace(MUESTRAS, MUESTRAS + ".bak")

    def rechazar(motivo):
        nonlocal objetivo, pedido_ant
        objetivo, pedido_ant = None, None
        s._trayecto = None
        s.fr5.habilitar(False)
        s.armado = False
        sl.desarmar("NO voy ahí: " + motivo)
        s.avisar("NO voy ahí: " + motivo + " → sliders desarmados (vuelve a ARMAR)")

    try:
        t0 = time.monotonic()
        while s.q is None:                               # sin postura real no se puede comprobar nada
            s.vuelta(None, hud=["esperando la postura real del robot..."])
            if time.monotonic() - t0 > SIN_LECTURA_S:
                raise Detenido(f"lleva {SIN_LECTURA_S:.0f} s sin leer la postura del robot (¿conexión con el FR5?)")
        # La postura de inicio es la referencia de los huecos, como en --auto: empieza con el robot
        # en una postura cómoda, sin nada apretado y con el tablero a la vista de la cámara.
        s.seguridad = crear_seguridad(s.cadena, np.array(s.q))
        if not s.seguridad.libre(np.array(s.q)):
            raise Detenido("la postura de inicio ya está demasiado cerca de algo ("
                           + (s.seguridad.motivo(np.array(s.q)) or "?") + "): sepáralo antes de empezar")
        while True:
            if s.q is not None:
                sl.publicar_real(sl.a_grados(s.q), True)
            sl.estado_fr5(FR5_ERROR if s.fr5.en_error else FR5_ENVIANDO if s.armado else FR5_ESPERA)
            if sl.pide_reset():
                s._reiniciar("RESET desde los sliders")
            if not sl.armado:
                objetivo, pedido_ant, s._trayecto = None, None, None
            elif objetivo is None:                       # recién armado: parte de la postura real
                if s.q is not None:
                    objetivo = np.array(s.q, float)
                    pedido_ant = objetivo.copy()
                    s._trayecto = objetivo[None, :].copy()
            else:
                pedido = np.clip(np.asarray(sl.objetivo_rad(), float), lim[:, 0], lim[:, 1])
                ahora = time.monotonic()
                if (s.q is not None and ahora - t_comprobado >= 0.1
                        and float(np.degrees(np.max(np.abs(pedido - pedido_ant)))) > 0.2):
                    t_comprobado, pedido_ant = ahora, pedido
                    camino = np.array(C.simular_camino(np.array(s.q), pedido, VEL_MAX))
                    motivo = s.seguridad.motivo_posturas(camino, MARGEN_TRAYECTO)
                    if motivo is None:
                        objetivo, s._trayecto = pedido, camino
                    else:
                        rechazar("por el camino, " + motivo)
            sin_pedal = isinstance(s.pedal, SinPedal)
            hud = ["MANUAL CON SLIDERS: ARMAR en los sliders"
                   + (" (SIN PEDAL: P = pausa)" if sin_pedal else " + P mantenida") + "; ESPACIO aqui = capturar",
                   f"muestras {len(muestras)} (min. {MIN_MUESTRAS}) · sliders "
                   + ("ARMADOS" if sl.armado else "desarmados") + " · robot " + ("en marcha" if s.armado else "quieto"),
                   "D = quitar ultima   C = calcular   Q = terminar"]
            try:
                key, _ = s.vuelta(objetivo, hud=hud)
            except FueraDeTrayecto:
                # el robot no está en el trayecto comprobado: se comprueba desde donde está
                if objetivo is not None and s.q is not None:
                    camino = np.array(C.simular_camino(np.array(s.q), objetivo, VEL_MAX))
                    motivo = s.seguridad.motivo_posturas(camino, MARGEN_TRAYECTO)
                    if motivo is None:
                        s._trayecto = camino
                    else:
                        rechazar("desde donde está el robot, " + motivo)
                continue
            if key == 32:
                m = s.capturar(objetivo=objetivo) if s.listo else None
                if m is None:
                    s.avisar("no capturo: " + ("sin pose del robot" if s.q is None else
                                               "el robot se mueve" if not s.robot.quieto else
                                               getattr(s, "motivo_captura", "") or "no se ve el tablero"))
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
    except Detenido as e:
        s.mostrar_error(str(e))
    finally:
        sl.desarmar()
        s.fr5.habilitar(False)
        s.cerrar()
        sl.parar()
    if len(muestras) < 3:
        print("Menos de 3 muestras: no se calcula nada.")
        return None
    if len(muestras) < MIN_MUESTRAS:
        print(f"⚠️  Solo {len(muestras)} muestras (recomendadas ≥ {MIN_MUESTRAS}).")
    return calcular_y_guardar(muestras, s.cadena, s.info)[1]


def main():
    ap = argparse.ArgumentParser(description="Calibración cámara de cabeza (o cámara A) → robot (paso 4)")
    ap.add_argument("--camara", choices=("cabeza", "a"), default="cabeza",
                    help="(19-nov) cabeza = la de siempre · a = la OAK-D fija de la mesa (AUTONOMO_CAM_A_*)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--generar-tablero", action="store_true")
    g.add_argument("--auto", action="store_true", help="calibración automática (el robot elige y recorre las posturas)")
    g.add_argument("--manual", action="store_true", help="mueves tú el robot y pulsas ESPACIO en cada postura")
    g.add_argument("--repetir", action="store_true", help="el robot repite solo las posturas guardadas (con pedal)")
    g.add_argument("--verificar", action="store_true", help="comprueba la calibración guardada")
    g.add_argument("--recalcular", action="store_true", help="recalcula con calib_muestras.json")
    ap.add_argument("--ajustar-lado", action="store_true",
                    help="con --recalcular: usa el tamaño del cuadro que mide el robot (no cambia config.py)")
    g.add_argument("--girar-posturas", type=float, metavar="GRADOS", default=None,
                   help="la mano se ha montado girada GRADOS en la brida: adapta las posturas guardadas")
    ap.add_argument("--aqui", action="store_true", help="con --verificar: en la postura actual, sin mover el robot")
    ap.add_argument("--sliders", action="store_true",
                    help="(20-nov) con --manual: mueves el robot con la ventana de sliders (ARMAR + pedal)")
    ap.add_argument("--sin", action="store_true",
                    help="SIN PEDAL: el robot se mueve solo (P = pausa/seguir, Q/Esc = parar). Vale con cualquier modo")
    args = ap.parse_args()
    global SIN_PEDAL
    SIN_PEDAL = bool(args.sin)
    if args.sliders and not args.manual:
        sys.exit("✗ --sliders va con --manual:  python calibrar_camara_robot.py [--camara a] --manual --sliders")
    if args.camara != "cabeza":
        elegir_camara(args.camara)

    if args.generar_tablero:
        return generar_tablero()
    if args.recalcular:
        from skill_shield.teleop_fr5.cinematica import cargar_cadena
        d = _leer_json(MUESTRAS)
        if not d or len(d.get("muestras", [])) < 3:
            sys.exit(f"✗ No hay suficientes muestras en {MUESTRAS}")
        i = d["intrinsecos"]
        cadena = cargar_cadena(getattr(cfg, "RUTA_XML", None))
        muestras = d["muestras"]
        # (16-nov) las poses del tablero se midieron con el lado de ENTONCES: si en config pone otro, se
        # reescalan (con solvePnP, la traslación es proporcional al tamaño del tablero; el giro no cambia)
        lado_usado = float((d.get("tablero") or {}).get("lado", TABLERO["lado"]))
        if args.ajustar_lado:
            Tb = [cadena.fk_marcos(np.array(m["q"]))[5] for m in muestras]
            Tc = [np.array(m["T_cam_tablero"]) for m in muestras]
            escala = C.estimar_escala(Tb, Tc)
            TABLERO["lado"] = lado_usado * escala
            TABLERO["marcador"] = float(TABLERO["marcador"]) * TABLERO["lado"] / float(cfg.CALIB_ROBOT_TABLERO["lado"])
            print(f"📏 Según el robot, el cuadro mide {TABLERO['lado'] * 1000:.2f} mm (se midió con "
                  f"{lado_usado * 1000:.2f} mm). Recalculo con ese tamaño.")
        if abs(float(TABLERO["lado"]) - lado_usado) > 1e-7:
            k = float(TABLERO["lado"]) / lado_usado
            muestras = [dict(m, T_cam_tablero=_escalar_traslacion(m["T_cam_tablero"], k)) for m in muestras]
            if not args.ajustar_lado:
                print(f"📏 Las muestras se tomaron con lado = {lado_usado * 1000:.2f} mm y config dice "
                      f"{TABLERO['lado'] * 1000:.2f} mm: reescaladas.")
        _, ok = calcular_y_guardar(muestras, cadena,
                                   dict(mxid=d.get("mxid"), K=np.array(i["K"]), dist=np.array(i["dist"]),
                                        ancho=i["ancho"], alto=i["alto"]))
        if args.ajustar_lado:
            print(f"👉 Si con el calibre el cuadro mide de verdad ≈ {TABLERO['lado'] * 1000:.1f} mm, pon en config.py "
                  f"lado={TABLERO['lado']:.4f}, marcador={TABLERO['marcador']:.4f} (si no, la diferencia es de la "
                  "cámara —enfoque/intrínsecos— y no del tablero: no lo cambies).")
        return 0 if ok else 1
    if args.girar_posturas is not None:
        return 0 if modo_girar_posturas(args.girar_posturas) else 1
    if args.verificar and not os.path.exists(ARCHIVO):
        sys.exit(f"✗ No hay calibración ({ARCHIVO}): calibra primero con --auto.")
    if args.repetir or args.verificar:
        ok = modo_repetir_o_verificar(args.verificar, aqui=args.aqui)
    elif args.manual:
        ok = modo_manual_sliders() if args.sliders else modo_manual()
    else:
        if not args.auto:
            print("(Sin opciones = calibración automática. --manual para hacerlo a mano.)")
        ok = modo_auto()
    return 0 if ok in (True, None) else 1


if __name__ == "__main__":
    sys.exit(main())
