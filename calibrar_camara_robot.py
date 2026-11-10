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

Calibración automática, por dentro:
  1. EXPLORAR (~1 min): pequeños giros de la muñeca y del brazo desde la postura de inicio →
     calibración aproximada (ya se sabe más o menos dónde está la cámara).
  2. PLANIFICAR (un par de segundos, sin moverse): miles de posturas candidatas; se quedan las
     que no chocan (modelo de colisión de la teleoperación, con el tablero incluido en la mano y
     la cámara como obstáculo) y dejan el tablero bien visible. Se eligen ~20 muy distintas, se
     ordenan y se comprueba cada trayecto tal como lo moverá fr5_servo.
  3. MEDIR (~3-5 min): las recorre, captura en cada una y calcula la calibración definitiva.
     Las posturas se guardan para --repetir y --verificar.
"""
import argparse
import collections
import datetime
import json
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
FOTOS_POR_MUESTRA = 5           # se promedia la pose del tablero en 5 imágenes seguidas
QUIETO_DEG = 0.05               # el robot cuenta como quieto si en 0,6 s se mueve menos que esto
LLEGADA_DEG = 0.3               # postura alcanzada


class Cancelado(Exception):
    pass


# ═════════════ utilidades ═════════════

def generar_tablero():
    from calibrar_extrinseca import crear_tablero
    cuadros, lado = tuple(TABLERO["cuadros"]), float(TABLERO["lado"])
    board, _ = crear_tablero(cuadros, lado, TABLERO["marcador"], TABLERO["diccionario"])
    px = int(round(lado * 1000 * 300 / 25.4))            # 300 ppp
    img = board.generateImage((cuadros[0] * px, cuadros[1] * px), marginSize=int(px * 0.4))
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
    from skill_shield.teleop_fr5.ik_teleop import Caja, Entorno, ModeloColision, entorno_soporte_hombros
    from skill_shield.teleop_fr5.obstaculos import cajas_desde_config
    g = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    ent = entorno_soporte_hombros(cfg.SOPORTE_SUELO_Z) if getattr(cfg, "SOPORTE_HOMBROS", False) \
        else Entorno(z_suelo=0.0)
    ent.cajas = cajas_desde_config(g, ent.z_suelo, verbose=False)
    if T_mundo_cam is not None:
        ent.cajas.append(Caja("camara_cabeza", T_mundo_cam[:3, 3], (0.07, 0.07, 0.07)))
    m = ModeloColision(cadena, ent)
    m.R_MANO = RADIO_MANO
    return m


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
        self.armado = False
        self.q = None
        self.T_ct, self.txt_tab = None, ""
        self.mensaje, self.t_mensaje = "", 0.0
        self.ventana = "Calibracion camara-robot"
        cv2.namedWindow(self.ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.ventana, int(self.info["ancho"] * 0.75), int(self.info["alto"] * 0.75))

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

    def _pedal(self, key, puede):
        self.pedal.tecla(key)
        self.pedal.actualizar()
        nuevo = bool(self.pedal.armado) and not self.fr5.en_error and puede
        if self.fr5.en_error and self.armado:
            self.pedal.desarmar()
            print("✗ El FR5 marca un error: desarmado. Bórralo (WebApp o web) y vuelve a pisar.")
        if nuevo != self.armado:
            self.armado = nuevo
            self.fr5.habilitar(nuevo)
            print("🦶 " + ("pedal PISADO → el robot se mueve" if nuevo else "pedal suelto → quieto"))

    def vuelta(self, objetivo=None, hud=()):
        """Una vuelta: lee robot y cámara, publica el objetivo y atiende el pedal. Devuelve
        (tecla, hay imagen nueva). Lanza Cancelado con Q/Esc."""
        self.q = self.robot.leer()
        if objetivo is not None:
            self.fr5.publicar(objetivo)                  # cada vuelta (watchdog de fr5_servo)
        p, pkt = self.cola.tryGet(), None
        while p is not None:
            pkt, p = p, self.cola.tryGet()
        if pkt is None:
            key = cv2.waitKeyEx(3)
            if self.mover:
                self._pedal(key, objetivo is not None)
            if key in (27, ord("q"), ord("Q")):
                raise Cancelado()
            return key, False
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
            + (("  |  PEDAL PISADO" if self.armado else "  |  pedal suelto (P)") if self.mover else ""),
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
        key = cv2.waitKeyEx(1)
        if self.mover:
            self._pedal(key, objetivo is not None)
        if key in (27, ord("q"), ord("Q")):
            raise Cancelado()
        return key, True

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
        """Lleva el robot a q_obj (solo con el pedal) y espera a que esté quieto allí."""
        q_obj = np.asarray(q_obj, float)
        t_llegada = None
        while True:
            self.vuelta(q_obj, hud=hud)
            cerca = self.q is not None and float(np.degrees(np.max(np.abs(self.q - q_obj)))) < LLEGADA_DEG
            t_llegada = (t_llegada or time.monotonic()) if cerca else None
            if t_llegada and time.monotonic() - t_llegada >= ESPERA_S and self.robot.quieto:
                return

    def recorrer(self, posturas, titulo, cada=None):
        """Recorre las posturas ({q, capturar}) y captura donde toque. Devuelve las muestras."""
        muestras = []
        n_cap = sum(p["capturar"] for p in posturas)
        for k, p in enumerate(posturas):
            hud = [titulo, f"postura {sum(x['capturar'] for x in posturas[:k + 1])}/{n_cap}"
                   + ("" if p["capturar"] else " (paso intermedio)") + f" · muestras {len(muestras)}"]
            self.ir_a(p["q"], hud=hud)
            if not p["capturar"]:
                continue
            m = self.capturar(hud=hud, objetivo=np.asarray(p["q"], float))
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


def modo_auto():
    s = Sesion(mover=True)
    try:
        print("\n① Pon el robot (arrastre en la WebApp) donde la cámara vea el tablero entero y de frente.")
        s.esperar_tablero(["AUTO: coloca el robot donde la camara vea el tablero", "Q = salir"])
        q0 = np.array(s.q)
        col = C.Colisiones(modelo_colision(s.cadena), q0)
        if col.base:
            print(f"⚠️  El modelo de colisión ya ve choques en la postura de inicio {list(col.base)}: o el modelo "
                  "del soporte no cuadra con el real o el robot está muy cerca. Se tolera, pero no pueden empeorar.")
        explorar = []                                    # ida a cada una y vuelta a la de inicio
        for q in C.posturas_explorar(q0, col, s.cadena.limites, VEL_MAX):
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
        aprox = C.calibrar(Tb, Tc, s.P)
        print(f"   calibración aproximada: error medio {aprox['error_medio_mm']:.1f} mm "
              f"(cámara en {np.round(aprox['T_mundo_cam'][:3, 3], 3).tolist()} m)")

        print("\n③ Planificando posturas (sin moverse)…")
        col = C.Colisiones(modelo_colision(s.cadena, aprox["T_mundo_cam"]), q0)
        plan, info = C.planificar_posturas(q0, s.cadena, col, aprox["T_mundo_cam"], aprox["T_brida_tablero"],
                                           s.K, s.info["ancho"], s.info["alto"], s.P, n=N_AUTO,
                                           amplitud_deg=AMPLITUD, v_max_deg=VEL_MAX)
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
    finally:
        s.cerrar()
    return calcular_y_guardar(muestras, s.cadena, s.info)[1]


def modo_repetir_o_verificar(verificar, aqui=False):
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
    s = Sesion(mover=not aqui)
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
            # el primer trayecto sale de donde esté el robot ahora: comprobarlo antes
            while s.q is None:
                s.vuelta(None, hud=["leyendo la postura del robot..."])
            q_act = np.array(s.q)
            modelo = modelo_colision(s.cadena)
            if not C.Colisiones(modelo, q_act).camino_libre(q_act, posturas[0]["q"], VEL_MAX):
                sys.exit("✗ El trayecto desde la postura actual hasta la primera postura guardada podría chocar. "
                         "Lleva el robot (arrastre) cerca de esa postura de inicio y vuelve a probar.")
            col = C.Colisiones(modelo, posturas[0]["q"])  # los demás trayectos, también
            malos = [k + 1 for k in range(len(posturas) - 1)
                     if not col.camino_libre(posturas[k]["q"], posturas[k + 1]["q"], VEL_MAX)]
            if malos:
                sys.exit(f"✗ Los trayectos que salen de las posturas {malos} podrían chocar según el modelo de "
                         "colisión. Vuelve a calibrar con --auto (planifica trayectos seguros).")
            muestras = s.recorrer(posturas, "VERIFICAR" if verificar else "REPETIR",
                                  cada=None if verificar else (lambda m: _guardar_muestras(s, m)))
    except Cancelado:
        print("⏹  Cancelado.")
        return None
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


def modo_manual():
    s = Sesion(mover=False)
    muestras = []
    if os.path.exists(MUESTRAS):
        os.replace(MUESTRAS, MUESTRAS + ".bak")
    try:
        while True:
            key, _ = s.vuelta(None, hud=["MANUAL: mueve el robot (arrastre en la WebApp), ESPACIO = capturar",
                                         f"muestras {len(muestras)} (min. {MIN_MUESTRAS})",
                                         "D = quitar ultima   C = calcular   Q = terminar"])
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
