"""
Escala de la profundidad del estéreo de la cámara de CABEZA (18-nov) → AUTONOMO_CAM_ESCALA_PROF.

La profundidad de la OAK-D puede salir un % larga o corta (calibración de fábrica, golpes, temperatura) y,
a ~1 m, eso son 1-2 cm: la mano iba a coger la pelota «más lejos» de donde estaba. Antes se medía con un
cartón y una cinta métrica; aquí se usa un tablero ChArUco, cuya distancia sale de la imagen en COLOR con
solvePnP (escala real: la del tamaño de sus cuadros, medido con calibre), y se compara, píxel a píxel del
tablero, con la profundidad del estéreo (la misma tubería que el modo autónomo, sin escalar).

    python calibrar_escala_profundidad.py                       # tablero de CALIB_ROBOT_TABLERO
    python calibrar_escala_profundidad.py --tablero 7,5,0.040,0.030      # cuadros x, y, lado, marcador (m)

Mueve el tablero (plano y rígido) por la zona de trabajo, de ~0,5 a ~1,3 m, por el centro y por los lados de
la imagen, despacio. Q o Esc para terminar. Al final dice el valor de AUTONOMO_CAM_ESCALA_PROF y si la escala
cambia con la distancia (entonces mejor recalibrar la OAK con la herramienta de Luxonis).

Con el tablero pequeño de la calibración (11,5 x 9,2 cm) solo se detecta hasta ~0,7 m a 640x480: para llegar
a la distancia de trabajo (~1 m) conviene imprimir uno grande (A4: --tablero 7,5,0.035,0.026, mídelo).
Cierra antes el modo autónomo: la cámara tiene que estar libre.
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)


def rayos_pixeles(us, vs, K, dist):
    """Rayos (x/z, y/z) de los píxeles, sin la distorsión de la lente."""
    p = np.column_stack([us, vs]).astype(np.float64).reshape(-1, 1, 2)
    if dist is not None and np.any(dist):
        p = cv2.undistortPoints(p, K, np.asarray(dist, float).ravel())
        return p.reshape(-1, 2)
    p = p.reshape(-1, 2)
    return np.column_stack([(p[:, 0] - K[0, 2]) / K[0, 0], (p[:, 1] - K[1, 2]) / K[1, 1]])


def medir_imagen(T_cam_tab, cuadros, lado, prof_mm, K, dist, encoger_px=4, n_min=200):
    """(Z mediana del estéreo m, cociente mediano Z_tablero/Z_estéreo, n píxeles) o None.
    Z_tablero: la del plano del tablero (solvePnP) por el rayo de cada píxel."""
    h, w = prof_mm.shape[:2]
    esq = np.array([[0, 0, 0], [cuadros[0] * lado, 0, 0], [cuadros[0] * lado, cuadros[1] * lado, 0],
                    [0, cuadros[1] * lado, 0]], np.float64)
    rvec, _ = cv2.Rodrigues(T_cam_tab[:3, :3])
    px, _ = cv2.projectPoints(esq, rvec, T_cam_tab[:3, 3], K, dist)
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, np.round(px.reshape(-1, 2)).astype(np.int32), 255)
    if encoger_px:
        m = cv2.erode(m, np.ones((2 * encoger_px + 1, 2 * encoger_px + 1), np.uint8))
    vs, us = np.nonzero((m > 0) & (prof_mm > 0))
    if len(us) < n_min:
        return None
    z_est = prof_mm[vs, us].astype(np.float64) * 0.001
    r = rayos_pixeles(us, vs, K, dist)
    ray = np.column_stack([r, np.ones(len(r))])
    n = T_cam_tab[:3, 2]
    t = T_cam_tab[:3, 3]
    den = ray @ n
    ok = np.abs(den) > 1e-6
    z_tab = (n @ t) / den[ok]                       # Z del plano por cada rayo (el rayo tiene z = 1)
    q = z_tab / z_est[ok]
    q = q[np.isfinite(q) & (q > 0.8) & (q < 1.2)]
    if len(q) < n_min:
        return None
    return float(np.median(z_est)), float(np.median(q)), int(len(q))


def resumen(medidas):
    """medidas: [(Z estéreo, cociente, n)] → texto y escala recomendada."""
    M = np.array([(z, c) for z, c, _ in medidas])
    esc = float(np.median(M[:, 1]))
    pend = float(np.polyfit(M[:, 0], M[:, 1], 1)[0]) if len(M) >= 5 and np.ptp(M[:, 0]) > 0.15 else 0.0
    lineas = [f"{len(M)} imágenes, de {M[:, 0].min():.2f} a {M[:, 0].max():.2f} m",
              f"cociente (tablero / estéreo): mediana {esc:.4f}, dispersión {np.std(M[:, 1]) * 100:.2f} %"]
    for a, b in ((0.0, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 9.0)):
        sel = (M[:, 0] >= a) & (M[:, 0] < b)
        if sel.any():
            c = float(np.median(M[sel, 1]))
            lineas.append(f"   {a:.1f}-{min(b, 2.0):.1f} m: {c:.4f}  (a 1 m serían {abs(c - 1) * 100:.1f} cm "
                          f"{'de más' if c < 1 else 'de menos'})")
    if abs(pend) > 0.01:
        lineas.append(f"⚠️  La escala cambia con la distancia ({pend * 100:+.1f} % por metro): no es solo una escala; "
                      "mejor recalibrar la OAK (herramienta de calibración de Luxonis).")
    return esc, lineas


def main():
    import config as cfg
    ap = argparse.ArgumentParser(description="Escala de la profundidad del estéreo (cámara de cabeza)")
    ap.add_argument("--tablero", default=None, help="cuadros_x,cuadros_y,lado_m,marcador_m (por defecto CALIB_ROBOT_TABLERO)")
    ap.add_argument("--diccionario", default=None)
    ap.add_argument("--mxid", default=getattr(cfg, "AUTONOMO_CAM_MXID", None))
    ap.add_argument("--max", type=int, default=150, help="imágenes como mucho")
    a = ap.parse_args()
    tb = dict(getattr(cfg, "CALIB_ROBOT_TABLERO", dict(cuadros=(5, 4), lado=0.023, marcador=0.0165,
                                                        diccionario="DICT_4X4_50")))
    if a.tablero:
        cx, cy, lado, marc = a.tablero.split(",")
        tb.update(cuadros=(int(cx), int(cy)), lado=float(lado), marcador=float(marc))
    if a.diccionario:
        tb["diccionario"] = a.diccionario
    from calibrar_extrinseca import crear_tablero, pose_tablero
    from config import ANCHO_CAM, ALTO_CAM
    from skill_shield.autonomo.camara import _texto_modo, abrir_como_autonomo, rol_de_mxid
    from skill_shield.vision.camara_oak import fijar_enfoque
    from skill_shield.vision.multicamara import activar_ir, intrinsecos_color
    board, det = crear_tablero(tuple(tb["cuadros"]), float(tb["lado"]), float(tb["marcador"]), tb["diccionario"])
    # la MISMA profundidad que el modo autónomo (sin rellenar huecos ni filtro temporal), sin escalar.
    # (20-nov) Y con el MISMO estéreo que usa el modo autónomo para ESTA cámara (cabeza o A: 800P, extendida…) y la
    # lente fija: la escala que se mide con otro modo del estéreo no vale para el que usa el robot
    rol = rol_de_mxid(cfg, a.mxid)
    clave = "AUTONOMO_CAM_A_ESCALA_PROF" if rol == "a" else "AUTONOMO_CAM_ESCALA_PROF"
    dev, modo, error = abrir_como_autonomo(
        cfg, rol, a.mxid, "cámara A" if rol == "a" else "cámara de cabeza",
        con_hd=False, rellenar_huecos=getattr(cfg, "AUTONOMO_CAM_RELLENAR_HUECOS", 0),
        temporal=getattr(cfg, "AUTONOMO_CAM_FILTRO_TEMPORAL", False),
        confianza=getattr(cfg, "AUTONOMO_CAM_CONFIANZA", 200))
    if dev is None:
        sys.exit(f"✗ No se pudo abrir la cámara (MxId {a.mxid}): {error}")
    print(f"📏 {'Cámara A' if rol == 'a' else 'Cámara de cabeza'}: estéreo {_texto_modo(modo)} (el del modo autónomo)")
    medidas, ultima = [], 0.0
    with dev:
        activar_ir(dev, rol, int(getattr(cfg, "AUTONOMO_CAM_A_IR_MA" if rol == "a" else "AUTONOMO_CAM_IR_MA", 800)))
        fijar_enfoque(dev, getattr(cfg, "AUTONOMO_CAM_ENFOQUE", "calibrado"), "cámara")
        K, dist = intrinsecos_color(dev, ANCHO_CAM, ALTO_CAM)
        qc = dev.getOutputQueue("color", maxSize=1, blocking=False)
        qd = dev.getOutputQueue("depth", maxSize=1, blocking=False)
        prof = None
        print(f"Tablero {tb['cuadros'][0]}x{tb['cuadros'][1]} de {tb['lado'] * 1000:.1f} mm. Muévelo despacio; "
              "Q o Esc para terminar.")
        while len(medidas) < a.max:
            pd = qd.tryGet()
            if pd is not None:
                prof = pd.getFrame()
            pc = qc.tryGet()
            if pc is None:
                time.sleep(0.003)
                continue
            img = pc.getCvFrame()
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            T, txt, corners, _ = pose_tablero(det, board, gris, K, dist, min_esquinas=6, max_reproj_px=1.0)
            if T is not None and prof is not None and time.monotonic() - ultima > 0.3:
                r = medir_imagen(T, tb["cuadros"], float(tb["lado"]), prof, K, dist)
                if r is not None:
                    ultima = time.monotonic()
                    medidas.append(r)
                    txt = f"{r[0]:.2f} m  cociente {r[1]:.4f}  ({len(medidas)})"
            vis = img.copy()
            if corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis, corners)
            cv2.putText(vis, str(txt), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Escala de la profundidad", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    cv2.destroyAllWindows()
    if len(medidas) < 5:
        sys.exit("Muy pocas medidas (hace falta ver el tablero entero, plano, con profundidad).")
    esc, lineas = resumen(medidas)
    print("\n" + "\n".join(lineas))
    print(f"\n→ config.py:  {clave} = {esc:.4f}"
          + ("   (≈ 1: no hace falta corregir)" if abs(esc - 1.0) < 0.003 else ""))


if __name__ == "__main__":
    main()
