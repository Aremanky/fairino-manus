"""
Proceso de CÁMARA del modo autónomo (10-nov) — cámara de cabeza (la antigua cámara B).

Abre la OAK-D Pro por su MxId con el mismo pipeline que la teleoperación (color 640x480 +
profundidad alineada, crear_pipeline_oak) y deja en memoria compartida:
  · canal de imagen: el último fotograma de color en JPEG (lo sirve la web)
  · canal de info:   JSON con MxId, USB, FPS, matriz K de fábrica y la profundidad del centro

También calcula el MAPA DE OBSTÁCULOS (mapa.py) con la profundidad, si hay calibración
cámara→robot, y detecta las HERRAMIENTAS con marcadores ArUco (herramientas.py), dibujándolas
sobre la imagen; cada cosa a su canal. (YOLO irá aquí mismo en el paso 12.) Va en su propio proceso para que, si hiciera falta, se
pueda lanzar con OTRO entorno virtual (AUTONOMO_PYTHON_PERCEPCION en config.py).

Prueba aislada, sin robot ni MuJoCo (se ve en una ventana; Q o Esc para salir):
    python -m skill_shield.autonomo.camara --ventana
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


def _parsear():
    import config as cfg
    ap = argparse.ArgumentParser(description="Cámara de cabeza del modo autónomo")
    ap.add_argument("--mxid", default=getattr(cfg, "AUTONOMO_CAM_MXID", None))
    ap.add_argument("--canal-img", default=None, help="nombre del canal de imagen (None = no publicar)")
    ap.add_argument("--canal-info", default=None, help="nombre del canal de info (None = no publicar)")
    ap.add_argument("--ventana", action="store_true", help="enseñar la imagen en una ventana")
    ap.add_argument("--calidad", type=int, default=int(getattr(cfg, "AUTONOMO_JPEG_CALIDAD", 80)))
    ap.add_argument("--padre", type=int, default=0, help="PID del modo autónomo (si muere, salir)")
    ap.add_argument("--ir-ma", type=int, default=int(getattr(cfg, "AUTONOMO_CAM_IR_MA", 800)))
    ap.add_argument("--canal-qbrazo", default=None, help="pose real del brazo (para quitarlo del mapa)")
    ap.add_argument("--canal-mapa", default=None, help="mapa de obstáculos (None = no calcularlo)")
    ap.add_argument("--canal-mando", default=None, help="mandos de la web (borrar mapa…)")
    ap.add_argument("--canal-herramientas", default=None, help="herramientas detectadas (None = no detectar)")
    return ap.parse_args()


def _ultimo(cola):
    p, q = None, cola.tryGet()
    while q is not None:
        p, q = q, cola.tryGet()
    return p


def _prof_centro_mm(prof, r=6):
    """Mediana de la profundidad válida en un cuadrado de 2r px en el centro (mm) o None."""
    if prof is None:
        return None
    h, w = prof.shape[:2]
    trozo = prof[h // 2 - r:h // 2 + r, w // 2 - r:w // 2 + r]
    v = trozo[trozo > 0]
    return float(np.median(v)) if v.size else None


def main():
    args = _parsear()
    import depthai as dai
    from config import ANCHO_CAM, ALTO_CAM
    from skill_shield.vision.camara_oak import crear_pipeline_oak
    from skill_shield.vision.multicamara import (abrir_dispositivo, activar_ir, comprobar_usb,
                                                 intrinsecos_color, listar_camaras, mxid_de)
    from skill_shield.autonomo.canal import Canal

    canal_img = Canal.esperar(args.canal_img) if args.canal_img else None
    canal_info = Canal.esperar(args.canal_info) if args.canal_info else None
    constructor = None
    if args.canal_mapa:                                  # mapa de obstáculos (paso 7)
        import config as cfg
        from skill_shield.autonomo.mapa import ConstructorMapa
        constructor = ConstructorMapa(cfg, os.path.dirname(os.path.abspath(cfg.__file__)),
                                      Canal.esperar(args.canal_qbrazo) if args.canal_qbrazo else None,
                                      Canal.esperar(args.canal_mapa),
                                      Canal.esperar(args.canal_mando) if args.canal_mando else None)
        print(f"🧱 [autónomo] Mapa de obstáculos: rejilla {tuple(int(v) for v in constructor.mapa.n)} de "
              f"{constructor.mapa.res * 100:.0f} cm")
    detector, canal_herr, canal_mando_h = None, None, None
    if args.canal_herramientas:                          # herramientas con ArUco (paso 5)
        import config as cfg
        from skill_shield.autonomo.herramientas import DetectorHerramientas
        detector = DetectorHerramientas(cfg, os.path.dirname(os.path.abspath(cfg.__file__)))
        canal_herr = Canal.esperar(args.canal_herramientas)
        canal_mando_h = Canal.esperar(args.canal_mando) if args.canal_mando else None
        print(f"🔧 [autónomo] Herramientas en el catálogo: "
              + ", ".join(f"{e['nombre']} (id {i})" for i, e in sorted(detector.cat.items())))
    t_herr, t_pub = 0.0, 0.0

    def _info(**kw):
        if canal_info is not None:
            canal_info.escribir(json.dumps(kw).encode())

    print(f"📷 [autónomo] OAK-D disponibles: {listar_camaras()}")
    try:
        dev = abrir_dispositivo(crear_pipeline_oak(con_hd=False), args.mxid)
    except Exception as e:
        print(f"✗ [autónomo] No se pudo abrir la cámara de cabeza (MxId {args.mxid}): {e}")
        print("   ¿Está conectada? ¿La tiene abierta otro programa (teleoperación, calibración)?")
        _info(ok=False, error=str(e), mxid=args.mxid, t=time.time())
        return 1

    with dev:
        usb = comprobar_usb(dev, "cabeza")
        activar_ir(dev, "cabeza", args.ir_ma)
        K, dist = intrinsecos_color(dev, ANCHO_CAM, ALTO_CAM)
        mxid = mxid_de(dev)
        print(f"📷 [autónomo] Cámara de cabeza {mxid}: {ANCHO_CAM}x{ALTO_CAM}, "
              f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f} px")
        qc = dev.getOutputQueue("color", maxSize=1, blocking=False)
        qd = dev.getOutputQueue("depth", maxSize=1, blocking=False)
        prof, n, t_fps, fps, t_info = None, 0, time.monotonic(), 0.0, 0.0
        while True:
            pd = _ultimo(qd)
            if pd is not None:
                prof = pd.getFrame()
                if constructor is not None:
                    try:
                        constructor.paso(prof, K)
                    except Exception as e:                # el mapa nunca debe tirar la cámara
                        print(f"⚠️  [autónomo] Mapa de obstáculos: {e}")
            pc = _ultimo(qc)
            if pc is None:
                time.sleep(0.002)
                if args.padre and os.getppid() != args.padre:      # el modo autónomo murió
                    print("📷 [autónomo] El proceso principal ya no existe → cierro la cámara.")
                    break
            else:
                img = pc.getCvFrame()
                n += 1
                ahora = time.monotonic()
                if detector is not None:
                    try:
                        if ahora - t_herr >= 1.0 / 15:          # detección a ~15 Hz
                            t_herr = ahora
                            detector.paso(img, K, dist, prof_mm=prof)
                        detector.dibujar(img, K, dist)
                        if ahora - t_pub >= 0.1:
                            t_pub = ahora
                            objetivo = None
                            if canal_mando_h is not None:
                                b_m, _ = canal_mando_h.leer()
                                objetivo = json.loads(b_m).get("objetivo_id") if b_m else None
                            canal_herr.escribir(json.dumps(dict(lista=detector.lista(), objetivo=objetivo,
                                                                calibrada=detector.T_mundo_cam is not None)).encode())
                            if constructor is not None:          # la herramienta a coger no es obstáculo
                                caja = detector.caja_mundo(objetivo) if objetivo is not None else None
                                constructor.excluir_extra = [caja] if caja else []
                    except Exception as e:                        # nunca debe tirar la cámara
                        print(f"⚠️  [autónomo] Herramientas: {e}")
                if ahora - t_fps >= 1.0:
                    fps, n, t_fps = n / (ahora - t_fps), 0, ahora
                z = _prof_centro_mm(prof)
                h, w = img.shape[:2]
                cv2.drawMarker(img, (w // 2, h // 2), (0, 255, 255), cv2.MARKER_CROSS, 18, 1)
                cv2.putText(img, f"{fps:4.1f} fps   centro: " + (f"{z:.0f} mm" if z else "sin profundidad"),
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                if canal_img is not None:
                    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, args.calidad])
                    if ok and len(jpg) <= canal_img.capacidad:
                        canal_img.escribir(jpg.tobytes())
                if ahora - t_info >= 0.5:
                    t_info = ahora
                    _info(ok=True, mxid=mxid, usb=getattr(usb, "name", str(usb)), fps=round(fps, 1),
                          ancho=w, alto=h, K=K.tolist(), dist=np.asarray(dist).ravel().tolist(),
                          prof_centro_mm=z, t=time.time(),
                          mapa=constructor.estado() if constructor is not None else None)
                if args.ventana:
                    cv2.imshow("Autonomo - camara de cabeza", img)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
