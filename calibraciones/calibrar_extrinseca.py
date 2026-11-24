"""
Calibración EXTRÍNSECA entre las dos OAK-D: dónde está la cámara B respecto a la A.
Genera calibraciones/datos/camaras_extrinsecas.json (p_A = R_ab · p_B + t_ab), que lee Skill_Shield_Nivel_6.py.

  python calibraciones/calibrar_extrinseca.py --generar-tablero       # guarda calibraciones/tableros/tablero_charuco.png
  python calibraciones/calibrar_extrinseca.py                         # calibra con el tablero ChArUco (recomendado)
  python calibraciones/calibrar_extrinseca.py --metodo cuerpo         # sin tablero: mueve la muñeca delante de las dos

ChArUco: imprime el tablero a ESCALA REAL (comprueba con una regla el lado del cuadrado),
pégalo en una superficie plana y rígida y colócalo donde LAS DOS cámaras lo vean a la vez.

  ¡OJO con la orientación! Si las cámaras están a ~90° una de otra y pones el tablero de
  frente a una, la otra lo ve "de canto" y no detecta nada. Oriéntalo hacia la BISECTRIZ
  (a medio camino entre las dos: ~45° a cada una) y mira el HUD: muestra cuántos marcadores
  ve cada cámara y a qué ángulo ve el tablero. Cuando las dos salgan en verde, ESPACIO.

Pulsa ESPACIO con varias posiciones/inclinaciones distintas (≥ 12), C para calcular y
guardar, Q para salir. Con las cámaras a más de ~110° no hay orientación que valga para
las dos: usa --metodo cuerpo.

Por defecto la detección se hace sobre la imagen del ISP (1352x1014 con CAM_FOV_COMPLETO),
no sobre el preview de 640x480: el doble de píxeles por marcador = el doble de alcance.
--sin-hd vuelve al preview (si el USB no da para las dos cámaras en HD).
"""
import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np

# Este script vive en calibraciones/: la raíz del repo (config.py, skill_shield/) tiene que estar en sys.path
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)
DATOS = os.path.join(RAIZ, "calibraciones", "datos")          # resultados (camaras_extrinsecas*.json)
TABLEROS = os.path.join(RAIZ, "calibraciones", "tableros")    # PNG de los tableros para imprimir

# ═════════════ funciones puras (probadas con datos sintéticos) ═════════════

def crear_tablero(cuadros=(7, 5), lado=0.04, marcador=0.03, diccionario="DICT_4X4_50"):
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, diccionario))
    board = cv2.aruco.CharucoBoard(tuple(cuadros), float(lado), float(marcador), dic)
    # Parámetros del detector de marcadores pensados para marcadores PEQUEÑOS e INCLINADOS
    # (el tablero lejos y visto de lado): por defecto OpenCV descarta todo marcador cuyo
    # perímetro sea < 3 % del lado mayor de la imagen, y eso a 2-3 m es casi todo.
    dp = cv2.aruco.DetectorParameters()
    dp.minMarkerPerimeterRate = 0.01
    dp.adaptiveThreshWinSizeMax = 53
    dp.adaptiveThreshWinSizeStep = 4
    dp.perspectiveRemovePixelPerCell = 8
    dp.polygonalApproxAccuracyRate = 0.05
    cp = cv2.aruco.CharucoParameters()
    # (16-nov) Con la disposición del tablero se buscan otra vez los marcadores que no salieron a la
    # primera (refineDetectedMarkers): con marcadores pequeños o de canto recupera bastantes y, con
    # ellos, más esquinas. (OpenCV ≥ 4.8; en versiones anteriores no existe y no pasa nada.)
    if hasattr(cp, "tryRefineMarkers"):
        cp.tryRefineMarkers = True
    return board, cv2.aruco.CharucoDetector(board, cp, dp)


def angulo_vision(T):
    """Ángulo (deg) entre la normal del tablero y la línea cámara→tablero.
    0° = de frente; 90° = de canto. Por encima de ~60° la detección se vuelve muy difícil."""
    n = T[:3, 2]
    v = T[:3, 3] / max(np.linalg.norm(T[:3, 3]), 1e-9)
    return float(np.degrees(np.arccos(np.clip(abs(n @ v), 0, 1))))


def filtrar_marcadores(m_corners, m_ids, board, holgura=2.0):
    """(17-nov) Quita los FALSOS POSITIVOS antes de buscar las esquinas ChArUco.

    Con marcadores pequeños (minMarkerPerimeterRate bajo) y DICT_4X4 (pocos bits), en una escena
    con cajoneras, listones, pantallas... salen 1-3 «marcadores» falsos por imagen, con un ID al azar
    entre 0 y 49. Si el ID no es del tablero da igual, pero si coincide con uno del tablero (≈20 %
    de las veces con el 5x4) ese ID sale REPETIDO y el CharucoDetector de OpenCV ≥ 4.9
    (CharucoParameters.checkMarkers, activo por defecto) da el tablero entero por falso:
    0 esquinas aunque se vean todos los marcadores → la detección «parpadea».

    Se queda con los IDs del tablero; de cada ID repetido, el más cercano al centro del grupo; y
    descarta los que estén más lejos del centro de lo que permite el tamaño del tablero.
    Devuelve (corners, ids, n_descartados)."""
    if m_ids is None or len(m_ids) == 0:
        return m_corners, m_ids, 0
    ids_tab = set(int(i) for i in np.asarray(board.getIds()).ravel())
    cs = [np.asarray(c, np.float32).reshape(4, 2) for c in m_corners]
    ids = [int(i) for i in np.asarray(m_ids).ravel()]
    sel = [k for k, i in enumerate(ids) if i in ids_tab]
    if not sel:
        return (), None, len(ids)
    centros = np.array([cs[k].mean(0) for k in sel])
    lados = np.array([np.mean(np.linalg.norm(cs[k] - np.roll(cs[k], 1, axis=0), axis=1)) for k in sel])
    centro, lado = np.median(centros, axis=0), float(np.median(lados))
    d = np.linalg.norm(centros - centro, axis=1)
    # radio del tablero medido en «lados de marcador» (sale de su propia geometría)
    obj = [np.asarray(o).reshape(-1, 3)[:, :2] for o in board.getObjPoints()]
    oc = np.array([o.mean(0) for o in obj])
    lado_obj = float(np.linalg.norm(obj[0][1] - obj[0][0]))
    radio = float(np.max(np.linalg.norm(oc - oc.mean(0), axis=1))) / max(lado_obj, 1e-9)
    mejor = {}
    for j, k in enumerate(sel):
        if d[j] > holgura * radio * lado + 2 * lado:
            continue
        if ids[k] not in mejor or d[j] < mejor[ids[k]][0]:
            mejor[ids[k]] = (d[j], k)
    quedan = sorted(k for _, k in mejor.values())
    if not quedan:
        return (), None, len(ids)
    return (tuple(cs[k].reshape(1, 4, 2) for k in quedan),
            np.array([[ids[k]] for k in quedan], np.int32), len(ids) - len(quedan))


_DETECTORES_ARUCO = {}


def detectar_tablero(detector, board, gris):
    """detectBoard, pero detectando y filtrando antes los marcadores (filtrar_marcadores).
    Devuelve (corners, ids, m_corners, m_ids, n_descartados)."""
    ad = _DETECTORES_ARUCO.get(id(detector))
    if ad is None:
        ad = _DETECTORES_ARUCO[id(detector)] = cv2.aruco.ArucoDetector(
            board.getDictionary(), detector.getDetectorParameters(), detector.getRefineParameters())
    mc, mi, rech = ad.detectMarkers(gris)
    mc, mi, n_desc = filtrar_marcadores(mc, mi, board)
    if mi is None or len(mi) == 0:
        return None, None, mc, mi, n_desc
    if getattr(detector.getCharucoParameters(), "tryRefineMarkers", False):
        try:                                   # recupera marcadores que no salieron a la primera
            mc, mi, rech, _ = ad.refineDetectedMarkers(gris, board, mc, mi, rech)
            mc, mi, n2 = filtrar_marcadores(mc, mi, board)
            n_desc += n2
        except cv2.error:
            pass
    corners, ids, mc, mi, n2 = tablero_de_marcadores(detector, gris, mc, mi)
    return corners, ids, mc, mi, n_desc + n2


def tablero_de_marcadores(detector, gris, mc, mi):
    """Esquinas ChArUco a partir de marcadores ya filtrados. Devuelve (corners, ids, mc, mi, n_quitados)."""
    n_desc = 0
    corners, ids, _, _ = detector.detectBoard(gris, markerCorners=mc, markerIds=mi)
    # Si aun así OpenCV rechaza el tablero (un falso con ID del tablero justo encima o al lado),
    # se prueba quitando el marcador más alejado del centro, hasta 2 veces.
    for _ in range(2):
        if ids is not None and len(ids) or len(mi) < 5:
            break
        cen = np.array([np.asarray(c).reshape(4, 2).mean(0) for c in mc])
        quita = int(np.argmax(np.linalg.norm(cen - np.median(cen, axis=0), axis=1)))
        mc = tuple(c for k, c in enumerate(mc) if k != quita)
        mi = np.delete(mi, quita, axis=0)
        n_desc += 1
        corners, ids, _, _ = detector.detectBoard(gris, markerCorners=mc, markerIds=mi)
    return corners, ids, mc, mi, n_desc


def pose_tablero(detector, board, gris, K, dist, min_esquinas=6, max_reproj_px=1.5):
    """T (4x4) cámara←tablero con solvePnP sobre las esquinas ChArUco, o (None, motivo)."""
    corners, ids, m_corners, m_ids, n_desc = detectar_tablero(detector, board, gris)
    n_marc = 0 if m_ids is None else len(m_ids)
    n_esq = 0 if ids is None else len(ids)
    if corners is None or ids is None or n_esq < min_esquinas:
        desc = f" (+{n_desc} falsos)" if n_desc else ""
        return None, f"marcadores {n_marc}{desc} | esquinas {n_esq}/{min_esquinas}", corners, ids
    obj = board.getChessboardCorners()[ids.flatten()].astype(np.float64)
    img = corners.reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, "solvePnP falló", corners, ids
    proy, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proy.reshape(-1, 2) - img) ** 2, axis=1))))
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rvec)[0]; T[:3, 3] = tvec.ravel()
    info = f"{n_esq} esq, {np.linalg.norm(tvec):.1f} m, {angulo_vision(T):.0f}°"
    if err > max_reproj_px:
        return None, f"{info}, reproy {err:.2f}px > {max_reproj_px}", corners, ids
    return T, f"{info}, err {err:.2f}px", corners, ids


def proyectar_a_so3(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R


def promediar_transformaciones(Ts):
    """Media de transformaciones 4x4. Devuelve (T, std_giro_deg, std_traslacion_mm)."""
    R = proyectar_a_so3(np.mean([T[:3, :3] for T in Ts], axis=0))
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    ang = [np.degrees(np.arccos(np.clip((np.trace(R.T @ T[:3, :3]) - 1) / 2, -1, 1))) for T in Ts]
    dt  = [np.linalg.norm(T[:3, 3] - t) * 1000 for T in Ts]
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T, float(np.std(ang)), float(np.std(dt))


def kabsch(PB, PA):
    """R, t que minimizan |R·PB + t − PA|² (puntos Nx3 correspondientes)."""
    cb, ca = PB.mean(0), PA.mean(0)
    H = (PB - cb).T @ (PA - ca)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, ca - R @ cb


def kabsch_recortado(PB, PA, descartar=0.2, iteraciones=3):
    """Kabsch descartando en cada vuelta el 'descartar' % de puntos con más residuo."""
    idx = np.arange(len(PB))
    for _ in range(iteraciones):
        R, t = kabsch(PB[idx], PA[idx])
        res = np.linalg.norm((PB @ R.T + t) - PA, axis=1)
        corte = np.quantile(res[idx], 1 - descartar)
        idx = np.where(res <= corte)[0]
    R, t = kabsch(PB[idx], PA[idx])
    res = np.linalg.norm((PB[idx] @ R.T + t) - PA[idx], axis=1)
    return R, t, float(np.sqrt(np.mean(res ** 2))), len(idx)


def ruta_salida(nombre):
    """--salida: un nombre suelto (camaras_extrinsecas_ac.json) va a calibraciones/datos/; con carpeta, tal cual."""
    if os.path.isabs(nombre) or os.path.dirname(nombre):
        return os.path.abspath(nombre)
    return os.path.join(DATOS, nombre)


def guardar(ruta, R, t, mxid_a, mxid_b, metodo, n, extra):
    d = {"mxid_a": mxid_a, "mxid_b": mxid_b, "R_ab": R.tolist(), "t_ab": np.asarray(t).tolist(),
         "convencion": "p_A = R_ab @ p_B + t_ab", "metodo": metodo, "n_muestras": n,
         "fecha": datetime.datetime.now().isoformat(timespec="seconds"), **extra}
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


class AcumuladorParadas:
    """
    Calibración por CUERPO, versión 2: solo cuentan los instantes en que la muñeca está QUIETA.

    Con la muñeca en movimiento, las dos cámaras (que no están sincronizadas por hardware)
    ven la mano en sitios distintos aunque los fotogramas difieran solo 30-60 ms: a 1,5 m/s
    son 5-9 cm de discrepancia, y eso arruina la transformación entre cámaras.
    Aquí se pide "colócate, quédate quieto un instante, muévete a otro sitio":
      · la quietud se mide en PÍXELES de las dos imágenes (la profundidad es demasiado
        ruidosa para decidir si te mueves): la muñeca debe quedarse dentro de r_pix píxeles
        de su posición media durante t_quieto segundos;
      · durante la parada se acumulan varios fotogramas y se guarda la MEDIANA 3D de cada
        cámara (promediar reduce el ruido del estéreo);
      · una parada nueva solo se guarda si está a ≥ dist_min de las ya guardadas (puntos bien
        repartidos por el volumen, que es lo que necesita el ajuste).
    """

    def __init__(self, r_pix=8.0, t_quieto=0.5, n_min=8, dist_min=0.15, dispersion_max=0.08):
        self.r_pix, self.t_quieto, self.n_min = r_pix, t_quieto, n_min
        self.dist_min, self.dispersion_max = dist_min, dispersion_max
        self.reset()

    def reset(self):
        self.PA, self.PB = [], []        # paradas guardadas (mediana 3D en cada cámara)
        self._reiniciar_parada()
        self.estado = "esperando"

    def _reiniciar_parada(self):
        self._buf = []                   # (t, pA, pB)
        self._pix = []                   # posiciones de la muñeca en píxeles (A y B)

    def añadir(self, t, pix_a, pix_b, p3_a, p3_b):
        """Un par de fotogramas SINCRONIZADOS. Cualquiera de los argumentos puede ser None
        (muñeca no vista). Devuelve True si en este fotograma se guardó una parada nueva."""
        if pix_a is None or pix_b is None or p3_a is None or p3_b is None:
            self._reiniciar_parada(); self.estado = "no se ve la muñeca en las dos"
            return False
        pix = np.concatenate([np.asarray(pix_a, float), np.asarray(pix_b, float)])
        if self._pix:
            ref = np.mean(self._pix, axis=0)
            if max(np.linalg.norm(pix[:2] - ref[:2]), np.linalg.norm(pix[2:] - ref[2:])) > self.r_pix:
                self._buf, self._pix = [], []                      # te has movido: empieza otra parada
                self.estado = "en movimiento"
        self._pix.append(pix)
        self._buf.append((t, np.asarray(p3_a, float), np.asarray(p3_b, float)))
        dur = self._buf[-1][0] - self._buf[0][0]
        if dur < self.t_quieto or len(self._buf) < self.n_min:
            self.estado = f"quieto… {max(dur, 0):.1f}/{self.t_quieto:.1f} s"
            return False
        A = np.array([b[1] for b in self._buf]); B = np.array([b[2] for b in self._buf])
        mA, mB = np.median(A, axis=0), np.median(B, axis=0)
        if max(np.linalg.norm(A - mA, axis=1).max(), np.linalg.norm(B - mB, axis=1).max()) > 3 * self.dispersion_max:
            h = len(self._buf) // 2                                  # medidas muy dispersas: reintenta
            self._buf, self._pix = self._buf[h:], self._pix[h:]
            self.estado = "medida inestable"
            return False
        if self.PA and min(np.linalg.norm(mA - q) for q in self.PA) < self.dist_min:
            self.estado = "ya guardado aquí: muévete a OTRO sitio"
            self._buf, self._pix = self._buf[-1:], self._pix[-1:]
            return False
        self.PA.append(mA); self.PB.append(mB)
        self._reiniciar_parada()
        self.estado = f"GUARDADO ({len(self.PA)})"
        return True

    def cobertura(self):
        """Extensión (m) de las paradas en x, y, z (marco de la cámara A)."""
        if len(self.PA) < 2:
            return (0.0, 0.0, 0.0)
        P = np.array(self.PA)
        return tuple(float(v) for v in (P.max(0) - P.min(0)))


def emparejar_por_tiempo(buf_a, buf_b, max_dt=0.02):
    """Del fotograma más reciente de A, el de B más cercano en el tiempo (≤ max_dt s)."""
    if not buf_a or not buf_b:
        return None
    pa = buf_a[-1]
    pb = min(buf_b, key=lambda p: abs((p.getTimestamp() - pa.getTimestamp()).total_seconds()))
    return (pa, pb) if abs((pb.getTimestamp() - pa.getTimestamp()).total_seconds()) <= max_dt else None


# ═════════════ parte con cámaras ═════════════

def _pipeline_charuco_hd(fps=15, escala=None, control=False):
    """
    Pipeline mínimo SOLO de color para el ChArUco, sacando la imagen completa del ISP
    (1352x1014 con CAM_FOV_COMPLETO, 1920x1080 si no). Mismo sensor y mismo modo que
    camara_oak.py → mismo sistema de coordenadas de cámara, así que la extrínseca vale
    igual para el preview de 640x480 que usa Skill_Shield_Nivel_6.py.
    Sin estéreo ni profundidad (no hacen falta aquí) para dejar el USB a la imagen HD.
    escala (16-nov, solo con CAM_FOV_COMPLETO): reducción del ISP, (1, 3) por defecto → 1352x1014;
    (1, 2) → 2028x1520: tablero pequeño más lejos (necesita USB3; con fps bajos sobra).
    control: añade la entrada "control_hd" (enfoque, exposición… con dai.CameraControl).
    """
    import depthai as dai
    import config as _cfg
    p = dai.Pipeline()
    cam = p.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setInterleaved(False)
    if getattr(_cfg, "CAM_FOV_COMPLETO", False):
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
        cam.setIspScale(*(tuple(escala) if escala else (1, 3)))
    else:
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(fps)
    xo = p.create(dai.node.XLinkOut)
    xo.setStreamName("color_hd")
    cam.isp.link(xo.input)
    if control:
        xi = p.create(dai.node.XLinkIn)
        xi.setStreamName("control_hd")
        xi.out.link(cam.inputControl)
    w, h = cam.getIspSize()
    return p, int(w), int(h)


def _abrir_las_dos(args):
    from config import ANCHO_CAM, ALTO_CAM
    from skill_shield.vision.camara_oak import crear_pipeline_oak
    from skill_shield.vision.multicamara import (abrir_dispositivo, comprobar_usb, activar_ir,
                                                 intrinsecos_color, listar_camaras, mxid_de)
    libres = listar_camaras()
    print("OAK-D disponibles:", libres)
    if len(libres) < 2 and not (args.mxid_a and args.mxid_b):
        sys.exit("Hacen falta 2 OAK-D conectadas (y libres: cierra Skill_Shield_Nivel_6.py).")
    hd = args.metodo == "charuco" and not args.sin_hd
    if hd:
        (pa, w, h), (pb, _, _) = _pipeline_charuco_hd(), _pipeline_charuco_hd()
        cola = "color_hd"
        print(f"🔎 Detección del tablero sobre la imagen del ISP: {w}x{h}")
    else:
        pa = crear_pipeline_oak(con_hd=False); pb = crear_pipeline_oak(con_hd=False)
        w, h, cola = ANCHO_CAM, ALTO_CAM, "color"
    dev_a = abrir_dispositivo(pa, args.mxid_a, excluir=[args.mxid_b])
    dev_b = abrir_dispositivo(pb, args.mxid_b, excluir=[mxid_de(dev_a)])
    cams = []
    for nombre, dev in (("A", dev_a), ("B", dev_b)):
        comprobar_usb(dev, nombre)
        if args.metodo == "cuerpo":
            activar_ir(dev, nombre)
        K, dist = intrinsecos_color(dev, w, h)
        c = {"nombre": nombre, "dev": dev, "mxid": mxid_de(dev), "K": K, "dist": dist, "w": w, "h": h,
             "qc": dev.getOutputQueue(cola, maxSize=2, blocking=False)}
        if not hd:
            c["qd"] = dev.getOutputQueue("depth", maxSize=2, blocking=False)
        cams.append(c)
    print(f"A = {cams[0]['mxid']}   B = {cams[1]['mxid']}")
    return cams


def _ultimo(q, previo):
    p = q.tryGet()
    while p is not None:
        previo = p
        p = q.tryGet()
    return previo


def calibrar_charuco(args, cams):
    board, det = crear_tablero(args.cuadros, args.lado, args.marcador, args.diccionario)
    ultimo = [None, None]
    pares, ahora_ok = [], False
    ANCHO_VIS = 640
    print("\nESPACIO = capturar pareja | C = calcular y guardar | Q = salir")
    print("Orienta el tablero a medio camino entre las dos cámaras (~45° a cada una).\n")
    while True:
        vis, poses, msgs = [], [], []
        for i, c in enumerate(cams):
            ultimo[i] = _ultimo(c["qc"], ultimo[i])
            if ultimo[i] is None:
                continue
            img = ultimo[i].getCvFrame()
            T, msg, corners, ids = pose_tablero(det, board, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                                c["K"], c["dist"], args.min_esquinas, args.max_reproj)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(img, corners, ids)
            esc = ANCHO_VIS / img.shape[1]
            if esc != 1.0:
                img = cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
            col = (0, 255, 0) if T is not None else (0, 0, 255)
            cv2.putText(img, f"{c['nombre']}: {msg}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            vis.append(img); poses.append(T); msgs.append(msg)
        if len(vis) == 2:
            ahora_ok = poses[0] is not None and poses[1] is not None
            if ahora_ok:
                pista = "LISTO: ESPACIO"
            elif poses[0] is not None:
                pista = "gira el tablero un poco hacia B"
            elif poses[1] is not None:
                pista = "gira el tablero un poco hacia A"
            else:
                pista = "acerca el tablero / oriéntalo entre las dos"
            cv2.putText(vis[0], f"parejas: {len(pares)}  {pista}",
                        (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            if vis[0].shape[0] != vis[1].shape[0]:
                vis[1] = cv2.resize(vis[1], (vis[1].shape[1], vis[0].shape[0]))
            cv2.imshow("Calibracion extrinseca  (A | B)", np.hstack(vis))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            return None
        if k == ord(' ') and len(vis) == 2 and ahora_ok:
            TA, TB = poses
            pares.append(TA @ np.linalg.inv(TB))          # B → A
            print(f"  pareja {len(pares)} guardada   A: {msgs[0]}   B: {msgs[1]}")
        if k == ord('c'):
            if len(pares) < 5:
                print("  Faltan parejas (mínimo 5, ideal ≥ 12)."); continue
            T, std_g, std_t = promediar_transformaciones(pares)
            print(f"\n  {len(pares)} parejas → dispersión: giro {std_g:.2f}°  traslación {std_t:.1f} mm")
            if std_g > 2.0 or std_t > 30:
                print("  ⚠️ Dispersión alta: repite con el tablero más plano/rígido, mejor iluminado y más inclinaciones.")
            return T[:3, :3], T[:3, 3], len(pares), {"dispersion_giro_deg": std_g, "dispersion_t_mm": std_t}


def calibrar_cuerpo(args, cams):
    from collections import deque
    import mediapipe as mp
    from config import ANCHO_CAM, ALTO_CAM, PS_MUNECA_DER, PS_MUNECA_IZQ, PROF_Z_MIN, PROF_Z_MAX
    from skill_shield.brazo.angulos_humano import profundidad_robusta
    from skill_shield.vision.profundidad_3d import pixel_a_3d
    idx = PS_MUNECA_DER if args.brazo == "derecho" else PS_MUNECA_IZQ
    poses = [mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.6,
                                    min_tracking_confidence=0.6) for _ in cams]
    bc = [deque(maxlen=6) for _ in cams]            # fotogramas de color recientes (con su marca de tiempo)
    bd = [deque(maxlen=6) for _ in cams]            # y de profundidad
    acu = AcumuladorParadas(dist_min=args.dist_min)
    ultimo_ts = None
    grabando = False
    print("\nESPACIO = empezar/parar | C = calcular y guardar | Q = salir")
    print("Cómo: con el brazo estirado, COLOCA la muñeca en un punto, QUÉDATE QUIETO ~1 s hasta que ponga")
    print("GUARDADO, y muévela a OTRO sitio (≥ 15 cm). Recorre todo el volumen de trabajo: arriba/abajo,")
    print("cerca/lejos, izquierda/derecha. Hacen falta ≥ 20 puntos (ideal 30).\n")

    def medir(i, img, depth):
        r = poses[i].process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not r.pose_landmarks:
            return None, None, None
        mp.solutions.drawing_utils.draw_landmarks(img, r.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        l = r.pose_landmarks.landmark[idx]
        if l.visibility < 0.8 or not (0 <= l.x < 1 and 0 <= l.y < 1):
            return r, None, None
        z = profundidad_robusta(depth, int(l.x * depth.shape[1]), int(l.y * depth.shape[0]))
        if z is None or not (PROF_Z_MIN < z < PROF_Z_MAX):
            return r, np.array([l.x * ANCHO_CAM, l.y * ALTO_CAM]), None
        p3 = np.array(pixel_a_3d(l.x * ANCHO_CAM, l.y * ALTO_CAM, z, ANCHO_CAM, ALTO_CAM, float(cams[i]["K"][0][0])))
        return r, np.array([l.x * ANCHO_CAM, l.y * ALTO_CAM]), p3

    while True:
        for i, c in enumerate(cams):
            for q, buf in ((c["qc"], bc[i]), (c["qd"], bd[i])):
                p = q.tryGet()
                while p is not None:
                    buf.append(p); p = q.tryGet()
        par = emparejar_por_tiempo(bc[0], bc[1])
        if par is not None and par[0].getTimestamp() != ultimo_ts:
            ultimo_ts = par[0].getTimestamp()
            salida = []
            for i, pc in enumerate(par):
                dep = min(bd[i], key=lambda p: abs((p.getTimestamp() - pc.getTimestamp()).total_seconds()), default=None)
                img = pc.getCvFrame()
                if dep is None or abs((dep.getTimestamp() - pc.getTimestamp()).total_seconds()) > 0.03:
                    salida.append((img, None, None)); continue
                _, pix, p3 = medir(i, img, dep.getFrame())
                salida.append((img, pix, p3))
            if grabando:
                t = ultimo_ts.total_seconds()
                if acu.añadir(t, salida[0][1], salida[1][1], salida[0][2], salida[1][2]) is True:
                    print(f"  parada {len(acu.PA)} guardada   cobertura x/y/z = "
                          + "/".join(f"{v:.2f}" for v in acu.cobertura()) + " m")
            imgs = [s[0] for s in salida]
            cv2.putText(imgs[0], f"{'GRABANDO' if grabando else 'pausa'} | paradas: {len(acu.PA)} | {acu.estado}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("Calibracion extrinseca  (A | B)", np.hstack(imgs))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            return None
        if k == ord(' '):
            grabando = not grabando
        if k == ord('c'):
            if len(acu.PA) < 12:
                print(f"  Faltan paradas ({len(acu.PA)}; mínimo 12, ideal ≥ 20)."); continue
            cx, cy, cz = acu.cobertura()
            if min(cx, cy) < 0.4 or cz < 0.3:
                print(f"  ⚠️ Cobertura pobre (x {cx:.2f} m, y {cy:.2f} m, z {cz:.2f} m). Necesito ≥ 0.4 en x e y y "
                      f"≥ 0.3 en profundidad: recorre más volumen.")
                continue
            R, t, rms, n = kabsch_recortado(np.array(acu.PB), np.array(acu.PA), descartar=0.15)
            print(f"\n  {n}/{len(acu.PA)} paradas usadas → error RMS {rms * 1000:.1f} mm")
            return R, t, n, {"error_rms_m": rms, "cobertura_m": [cx, cy, cz]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metodo", choices=("charuco", "cuerpo"), default="charuco")
    ap.add_argument("--generar-tablero", action="store_true")
    ap.add_argument("--cuadros", type=int, nargs=2, default=(7, 5), metavar=("X", "Y"))
    ap.add_argument("--lado", type=float, default=0.04, help="lado del cuadrado IMPRESO (m)")
    ap.add_argument("--marcador", type=float, default=0.03, help="lado del marcador ArUco impreso (m)")
    ap.add_argument("--diccionario", default="DICT_4X4_50")
    ap.add_argument("--min-esquinas", type=int, default=6)
    ap.add_argument("--max-reproj", type=float, default=1.5)
    ap.add_argument("--sin-hd", action="store_true",
                    help="detectar el tablero en el preview (640x480) en vez de en la imagen del ISP")
    ap.add_argument("--brazo", choices=("derecho", "izquierdo"), default="derecho")
    ap.add_argument("--mxid-a"); ap.add_argument("--mxid-b")
    ap.add_argument("--salida", default=None,
                    help="archivo de salida (por defecto EXTRINSECAS_ARCHIVO de config.py); "
                         "un nombre suelto se guarda en calibraciones/datos/")
    ap.add_argument("--dist-min", type=float, default=0.15, help="separación mínima entre paradas (m), método cuerpo")
    ap.add_argument("--rms-max", type=float, default=0.06, help="error RMS máximo (m) para guardar; --forzar lo ignora")
    ap.add_argument("--forzar", action="store_true", help="guardar aunque el error sea alto")
    args = ap.parse_args()

    if args.generar_tablero:
        board, _ = crear_tablero(args.cuadros, args.lado, args.marcador, args.diccionario)
        px = 400                                            # píxeles por cuadrado: nítido hasta A2
        os.makedirs(TABLEROS, exist_ok=True)
        salida = os.path.join(TABLEROS, "tablero_charuco.png")
        cv2.imwrite(salida, board.generateImage((args.cuadros[0] * px, args.cuadros[1] * px),
                                                marginSize=px // 5))
        ancho_cm = args.cuadros[0] * args.lado * 100
        alto_cm = args.cuadros[1] * args.lado * 100
        print(f"Guardado {salida} ({args.cuadros[0]}x{args.cuadros[1]}). Imprímelo para que cada "
              f"cuadrado mida {args.lado * 1000:.0f} mm (tablero {ancho_cm:.0f}x{alto_cm:.0f} cm sin margen) "
              f"y mídelo con una regla. Al calibrar pasa los MISMOS --cuadros/--lado/--marcador.")
        return

    from skill_shield.vision.multicamara import _ruta_extrinsecas
    cams = _abrir_las_dos(args)
    try:
        res = (calibrar_charuco if args.metodo == "charuco" else calibrar_cuerpo)(args, cams)
    finally:
        cv2.destroyAllWindows()
        for c in cams:
            c["dev"].close()
    if res is None:
        print("Cancelado: no se ha guardado nada."); return
    R, t, n, extra = res
    rms = extra.get("error_rms_m")
    if rms is not None and rms > args.rms_max and not args.forzar:
        print(f"\n❌ Error RMS {rms * 1000:.0f} mm > {args.rms_max * 1000:.0f} mm: NO se guarda (una calibración mala "
              f"empeora la fusión). Repite con más paradas bien repartidas, o usa --metodo charuco.")
        sys.exit(1)
    ruta = ruta_salida(args.salida) if args.salida else _ruta_extrinsecas()
    guardar(ruta, R, t, cams[0]["mxid"], cams[1]["mxid"], args.metodo, n, extra)
    sep = np.linalg.norm(t)
    giro = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"\n✅ Guardado {ruta}\n   B respecto a A: giro {giro:.1f}°, distancia entre cámaras {sep:.2f} m")


if __name__ == "__main__":
    main()