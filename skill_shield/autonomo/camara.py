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
Guardar la escena (color + profundidad + calibración) para revisar el mapa sin el robot:
    python -m skill_shield.autonomo.camara --grabar 10
"""
import argparse
import json
import os
import sys
import threading
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
    ap.add_argument("--canal-visto", default=None, help="qué ha visto la cámara (para el planificador)")
    ap.add_argument("--canal-herramientas", default=None, help="herramientas detectadas (None = no detectar)")
    ap.add_argument("--canal-congelar", default=None,
                    help="(18-nov) aviso del modo autónomo: hay una tarea → no aplicar la autocorrección")
    ap.add_argument("--grabar", type=int, default=0, metavar="N",
                    help="guardar N pares color+profundidad en diagnosticos/escena_*.npz y salir "
                         "(con el modo autónomo cerrado), para revisar el mapa fuera del robot")
    return ap.parse_args()


def _ultimo(cola):
    p, q = None, cola.tryGet()
    while q is not None:
        p, q = q, cola.tryGet()
    return p


class _HiloMapa:
    """(18-nov) El mapa de obstáculos en su PROPIO hilo. Antes iba en el mismo bucle que el vídeo y cada
    actualización (TSDF, quitar el robot, autocorrección con el brazo…: 100-250 ms, AUTONOMO_MAPA_HZ veces por
    segundo) dejaba la imagen congelada: la web se veía a tirones (~5 fps). Ahora el bucle de la cámara solo le
    pasa la última profundidad y sigue; el hilo coge siempre la más reciente (las intermedias se saltan, como
    antes con el límite de Hz). La mayor parte del trabajo es numpy, que suelta el GIL."""

    def __init__(self, constructor):
        self.c = constructor
        self._lock = threading.Lock()
        self._hay = threading.Event()
        self._datos = None
        self.ms = None                                    # lo que tarda una actualización (media)
        self.vivo = True
        threading.Thread(target=self._bucle, daemon=True, name="mapa").start()

    def dar(self, prof, K, img, halos):
        with self._lock:
            self._datos = (prof, K, img, halos)
        self._hay.set()

    def _bucle(self):
        while self.vivo:
            if not self._hay.wait(0.5):
                continue
            self._hay.clear()
            with self._lock:
                d, self._datos = self._datos, None
            if d is None:
                continue
            t0 = time.perf_counter()
            try:
                hecho = self.c.paso(d[0], d[1], d[2], halos=d[3])
            except Exception as e:                        # el mapa nunca debe tirar la cámara
                print(f"⚠️  [autónomo] Mapa de obstáculos: {e}")
                continue
            if hecho:
                ms = 1000.0 * (time.perf_counter() - t0)
                self.ms = ms if self.ms is None else 0.8 * self.ms + 0.2 * ms


def _prof_centro_mm(prof, r=6):
    """Mediana de la profundidad válida en un cuadrado de 2r px en el centro (mm) o None."""
    if prof is None:
        return None
    h, w = prof.shape[:2]
    trozo = prof[h // 2 - r:h // 2 + r, w // 2 - r:w // 2 + r]
    v = trozo[trozo > 0]
    return float(np.median(v)) if v.size else None


def _grabar(dev, K, n):
    import config as cfg
    raiz = os.path.dirname(os.path.abspath(cfg.__file__))
    qc = dev.getOutputQueue("color", maxSize=1, blocking=False)
    qd = dev.getOutputQueue("depth", maxSize=1, blocking=False)
    colores, profs, prof = [], [], None
    time.sleep(1.0)                                       # que se asiente la exposición
    while len(colores) < n:
        pd = _ultimo(qd)
        prof = pd.getFrame() if pd is not None else prof
        pc = _ultimo(qc)
        if pc is None or prof is None:
            time.sleep(0.01)
            continue
        colores.append(pc.getCvFrame())
        profs.append(prof.copy())
        time.sleep(0.2)
    T = None
    ruta_c = os.path.join(raiz, getattr(cfg, "CALIB_ROBOT_ARCHIVO", "camara_robot.json"))
    if os.path.exists(ruta_c):
        from skill_shield.autonomo.calibracion import cargar
        T = cargar(ruta_c)["T_mundo_cam"]
    os.makedirs(os.path.join(raiz, "diagnosticos"), exist_ok=True)
    ruta = os.path.join(raiz, "diagnosticos", time.strftime("escena_%Y%m%d_%H%M%S.npz"))
    corr = np.zeros(3)
    if T is not None:                                     # la autocorrección con el brazo (12-nov)
        from skill_shield.autonomo.autocalib import Correccion
        c = Correccion(ruta_c)
        c.nueva_calibracion(os.path.getmtime(ruta_c))
        corr = c.t
    np.savez_compressed(ruta, color=np.array(colores), profundidad=np.array(profs), K=K,
                        T_mundo_cam=T if T is not None else np.full((4, 4), np.nan), correccion=corr)
    print(f"💾 [autónomo] {n} imágenes guardadas en {ruta}")
    return 0


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
                                      Canal.esperar(args.canal_mando) if args.canal_mando else None,
                                      Canal.esperar(args.canal_visto) if args.canal_visto else None,
                                      canal_congelar=Canal.esperar(args.canal_congelar) if args.canal_congelar
                                      else None)
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
    pelotas = None
    if args.canal_herramientas:                          # pelotas por color + profundidad (11-nov)
        import config as cfg
        if getattr(cfg, "AUTONOMO_PELOTAS", None):
            from skill_shield.autonomo.pelotas import DetectorPelotas
            pelotas = DetectorPelotas(cfg, os.path.dirname(os.path.abspath(cfg.__file__)))
            print("⚽ [autónomo] Pelotas: " + ", ".join(e["nombre"] for e in pelotas.cat))
    t_herr, t_pub = 0.0, 0.0

    def _info(**kw):
        if canal_info is not None:
            canal_info.escribir(json.dumps(kw).encode())

    print(f"📷 [autónomo] OAK-D disponibles: {listar_camaras()}")
    try:
        import config as cfg_
        dev = abrir_dispositivo(crear_pipeline_oak(
            con_hd=False,                                 # profundidad «honrada» para el mapa (11-nov)
            rellenar_huecos=getattr(cfg_, "AUTONOMO_CAM_RELLENAR_HUECOS", 0),
            temporal=getattr(cfg_, "AUTONOMO_CAM_FILTRO_TEMPORAL", False),
            confianza=getattr(cfg_, "AUTONOMO_CAM_CONFIANZA", 200)), args.mxid)
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
        if args.grabar:
            return _grabar(dev, K, args.grabar)
        prof, n, t_fps, fps, t_info = None, 0, time.monotonic(), 0.0, 0.0
        # (18-nov) corrección de escala de la profundidad del estéreo (1.0 = tal cual). Con una cinta: un cartón
        # plano en el centro de la imagen a D mm medidos; si «centro:» marca M, ESCALA = D / M
        escala_prof = float(getattr(cfg_, "AUTONOMO_CAM_ESCALA_PROF", 1.0) or 1.0)
        if abs(escala_prof - 1.0) > 1e-6:
            print(f"📏 [autónomo] Profundidad del estéreo × {escala_prof:.4f} (AUTONOMO_CAM_ESCALA_PROF)")
        img_ult = None                                    # último color (el mapa descarta lo saturado)
        hilo_mapa = _HiloMapa(constructor) if constructor is not None else None      # (18-nov) aparte
        ms_det, ms_jpg, t_tiempos = None, None, time.monotonic()

        def _media(m, v):
            return v if m is None else 0.9 * m + 0.1 * v
        while True:
            pd = _ultimo(qd)
            if pd is not None:
                prof = pd.getFrame()
                if abs(escala_prof - 1.0) > 1e-6:
                    prof = np.clip(prof.astype(np.float32) * escala_prof, 0, 65535).astype(np.uint16)
                if hilo_mapa is not None:
                    try:
                        # los halos se calculan aquí (las pelotas cambian en este hilo) y el mapa, en el suyo
                        hilo_mapa.dar(prof, K, img_ult, (
                            pelotas.pixeles_halo() if pelotas is not None and getattr(cfg_, "AUTONOMO_MAPA_SIN_HALO_PELOTAS", True)
                            else None))
                        # lo que corrige el brazo vale para todo lo que ve la cámara (12-nov)
                        T_corr = constructor.T_corr
                        if detector is not None:
                            detector.correccion = T_corr
                        if pelotas is not None:
                            pelotas.correccion = T_corr
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
                img_ult = img.copy()
                n += 1
                ahora = time.monotonic()
                if detector is not None:
                    try:
                        if ahora - t_herr >= 1.0 / 15:          # detección a ~15 Hz
                            t_herr = ahora
                            t_d = time.perf_counter()
                            detector.paso(img, K, dist, prof_mm=prof)
                            if pelotas is not None:
                                pelotas.paso(img, K, prof, dist=dist if getattr(
                                    cfg_, "AUTONOMO_PELOTAS_QUITAR_DISTORSION", True) else None)
                            ms_det = _media(ms_det, 1000.0 * (time.perf_counter() - t_d))
                        detector.dibujar(img, K, dist)
                        if pelotas is not None:
                            pelotas.dibujar(img)
                        if ahora - t_pub >= 0.1:
                            t_pub = ahora
                            objetivo = None
                            if canal_mando_h is not None:
                                b_m, _ = canal_mando_h.leer()
                                objetivo = json.loads(b_m).get("objetivo_id") if b_m else None
                            lista = detector.lista() + (pelotas.lista() if pelotas is not None else [])
                            T_mc = detector.T_mundo_cam          # (18-nov) con la autocorrección: para afinar
                            canal_herr.escribir(json.dumps(dict(lista=lista, objetivo=objetivo,
                                                                calibrada=T_mc is not None,
                                                                T_mundo_cam=None if T_mc is None else
                                                                np.round(np.asarray(T_mc, float), 6).tolist())).encode())
                            if constructor is not None:          # la herramienta a coger no es obstáculo
                                caja = None
                                if objetivo is not None:
                                    caja = detector.caja_mundo(objetivo) or (
                                        pelotas.caja_mundo(objetivo) if pelotas is not None else None)
                                constructor.excluir_extra = ([caja] if caja else []) + (
                                    pelotas.cajas_mundo() if pelotas is not None and getattr(cfg, "AUTONOMO_MAPA_SIN_HALO_PELOTAS", True)
                                    else []) + (
                                    detector.cajas_hojas(hoja=tuple(getattr(cfg, "AUTONOMO_MAPA_HOJA", (0.6, 0.6, 0.6))),
                                                         margen=tuple(getattr(cfg, "AUTONOMO_MAPA_HOJA_MARGEN",
                                                                              (0.005, 0.005, 0.03))))
                                    if getattr(cfg, "AUTONOMO_MAPA_QUITAR_HOJAS", True) else [])
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
                    t_j = time.perf_counter()
                    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, args.calidad])
                    if ok and len(jpg) <= canal_img.capacidad:
                        canal_img.escribir(jpg.tobytes())
                    ms_jpg = _media(ms_jpg, 1000.0 * (time.perf_counter() - t_j))
                if ahora - t_info >= 0.5:
                    t_info = ahora
                    try:
                        est_mapa = constructor.estado() if constructor is not None else None
                    except Exception:                     # (lo está cambiando el hilo del mapa: a la próxima)
                        est_mapa = None
                    tiempos = dict(mapa_ms=None if hilo_mapa is None or hilo_mapa.ms is None else round(hilo_mapa.ms),
                                   deteccion_ms=None if ms_det is None else round(ms_det),
                                   jpeg_ms=None if ms_jpg is None else round(ms_jpg, 1))
                    _info(ok=True, mxid=mxid, usb=getattr(usb, "name", str(usb)), fps=round(fps, 1),
                          ancho=w, alto=h, K=K.tolist(), dist=np.asarray(dist).ravel().tolist(),
                          prof_centro_mm=z, t=time.time(), mapa=est_mapa, tiempos=tiempos)
                    if ahora - t_tiempos >= 10.0:         # (18-nov) para ver qué frena el vídeo, si algo
                        t_tiempos = ahora
                        print(f"📷 [autónomo] Cámara: {fps:.1f} fps · mapa {tiempos['mapa_ms']} ms (en su hilo) · "
                              f"detección {tiempos['deteccion_ms']} ms · jpeg {tiempos['jpeg_ms']} ms")
                if args.ventana:
                    cv2.imshow("Autonomo - camara de cabeza", img)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
        if hilo_mapa is not None:
            hilo_mapa.vivo = False
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
