#!/usr/bin/env python3
"""
Prueba en vivo del modelo con la OAK-D de la cabeza: herramientas y mangos pintados encima de la imagen.

    ./venv_yolo/bin/python yolo_herramientas/probar_oak.py --modelo runs/segment/herramientas_v1/weights/best.pt

Teclas: Q / Esc = salir · ESPACIO = guardar la imagen con lo detectado en <DATOS>/pruebas/ (las que
salgan mal son justo las que hay que etiquetar y añadir al dataset).
"""
import argparse
import os
import sys
import time

import cv2

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from capturar_oak import mxid_por_defecto  # noqa: E402
from comun import dir_datos  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--mxid", default=mxid_por_defecto())
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=960)
    a = ap.parse_args()

    import depthai as dai
    from ultralytics import YOLO
    modelo = YOLO(a.modelo)

    p = dai.Pipeline()
    cam = p.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam.setIspScale(1, 3)                                # 1352x1014
    cam.setFps(15)
    xout = p.create(dai.node.XLinkOut)
    xout.setStreamName("isp")
    cam.isp.link(xout.input)
    dev = dai.Device(p, dai.DeviceInfo(a.mxid)) if a.mxid else dai.Device(p)
    salida = os.path.join(dir_datos(), "pruebas")
    os.makedirs(salida, exist_ok=True)
    with dev:
        q = dev.getOutputQueue("isp", maxSize=2, blocking=False)
        t0, n = time.time(), 0
        while True:
            frame = q.get().getCvFrame()
            r = modelo.predict(frame, conf=a.conf, imgsz=a.imgsz, retina_masks=True, verbose=False)[0]
            vista = r.plot()
            n += 1
            fps = n / max(time.time() - t0, 1e-6)
            cv2.putText(vista, f"{fps:.1f} fps · {len(r.boxes)} detecciones", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("probar_oak", vista)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                base = os.path.join(salida, time.strftime("%Y%m%d_%H%M%S"))
                cv2.imwrite(base + ".jpg", frame)
                cv2.imwrite(base + "_detecciones.jpg", vista)
                print(f"💾 {base}.jpg")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
