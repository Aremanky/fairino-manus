#!/usr/bin/env python3
"""
Fotos para el dataset con la OAK-D de la CABEZA del robot (las mismas condiciones que verá el YOLO).

    ./venv_nivel6/bin/python yolo_herramientas/capturar_oak.py --sesion sala_martillos
    ./venv_nivel6/bin/python yolo_herramientas/capturar_oak.py --sesion mesa --mxid 1944301081B5FA4800

Teclas (con la ventana delante): ESPACIO = una foto · R = ráfaga on/off (una foto cada --cada s, para
ir moviendo el robot o las herramientas) · Q / Esc = salir.
Se guardan en <DATOS>/propias/<sesion>/images/ (2028x1520, el sensor entero). Cierra antes el programa
principal y el modo autónomo: la cámara tiene que estar libre.

Para que el modelo generalice, en cada sesión cambia: fondo (mesa, suelo, bandeja), luz, distancia
(0,3-1,3 m), ángulo, herramientas juntas y tapándose, la mano del robot cerca… y haz también fotos SIN
herramientas (fondo) — también enseñan.
"""
import argparse
import os
import sys
import time

import cv2

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
from comun import dir_datos  # noqa: E402


def mxid_por_defecto():
    try:
        sys.path.insert(0, RAIZ)
        import config
        return getattr(config, "AUTONOMO_CAM_MXID", None) or getattr(config, "CAM_B_MXID", None)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sesion", required=True, help="nombre de la sesión (carpeta)")
    ap.add_argument("--mxid", default=mxid_por_defecto(), help="por defecto, la cámara de cabeza de config.py")
    ap.add_argument("--cada", type=float, default=0.7, help="segundos entre fotos en ráfaga")
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    import depthai as dai
    salida = os.path.join(dir_datos(), "propias", a.sesion, "images")
    os.makedirs(salida, exist_ok=True)

    p = dai.Pipeline()
    cam = p.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam.setIspScale(1, 2)                                # 4056x3040 → 2028x1520 (4:3, campo de visión entero)
    cam.setFps(a.fps)
    xout = p.create(dai.node.XLinkOut)
    xout.setStreamName("isp")
    cam.isp.link(xout.input)

    if a.mxid:
        print(f"📷 Abriendo la OAK {a.mxid}…")
        dev = dai.Device(p, dai.DeviceInfo(a.mxid))
    else:
        print("📷 Abriendo la primera OAK que haya…")
        dev = dai.Device(p)
    n = len(os.listdir(salida))
    rafaga, t_ult = False, 0.0
    with dev:
        q = dev.getOutputQueue("isp", maxSize=2, blocking=False)
        while True:
            frame = q.get().getCvFrame()
            guardar = False
            vista = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            txt = f"{a.sesion}: {n} fotos  |  ESPACIO foto  R rafaga {'ON' if rafaga else 'off'}  Q salir"
            cv2.putText(vista, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(vista, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if rafaga else (255, 255, 255), 2)
            cv2.imshow("capturar_oak", vista)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                guardar = True
            if k == ord("r"):
                rafaga = not rafaga
            if rafaga and time.time() - t_ult >= a.cada:
                guardar = True
            if guardar:
                t_ult = time.time()
                ruta = os.path.join(salida, f"{a.sesion}_{time.strftime('%Y%m%d_%H%M%S')}_{n:05d}.jpg")
                cv2.imwrite(ruta, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n += 1
    cv2.destroyAllWindows()
    print(f"✅ {n} fotos en {salida}")


if __name__ == "__main__":
    main()
