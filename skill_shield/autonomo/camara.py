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

(19-nov) El mismo proceso sirve para la CÁMARA A (OAK-D fija mirando la mesa de cerca, camara_a.py): con
--rol a, su MxId y su calibración (--calibracion camara_a_robot.json), sin mapa de obstáculos propio (el mapa lo hace
la cabeza): imagen, pelotas y herramientas. (20-nov) Con --canal-prof-a, además manda su profundidad preparada a la
cabeza, que la funde en su mapa (AUTONOMO_CAM_A_AL_MAPA).

(20-nov) Profundidad más fina: mono a 800P, disparidad extendida y más bits de subpíxel (AUTONOMO_CAM_MONO_800P…,
AUTONOMO_CAM_A_MONO_800P…). Si la OAK no acepta la combinación, se reintenta sin la extendida y luego como antes.

Prueba aislada, sin robot ni MuJoCo (se ve en una ventana; Q o Esc para salir):
    python -m skill_shield.autonomo.camara --ventana
    python -m skill_shield.autonomo.camara --ventana --rol a        (la cámara A, con lo de config.py)
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
    ap = argparse.ArgumentParser(description="Cámara de cabeza (o cámara A de la mesa) del modo autónomo")
    ap.add_argument("--rol", choices=("cabeza", "a"), default="cabeza",
                    help="(19-nov) cabeza = la de siempre · a = la OAK-D fija que mira la mesa de cerca")
    ap.add_argument("--mxid", default=None, help="por defecto, AUTONOMO_CAM_MXID (cabeza) o AUTONOMO_CAM_A_MXID (a)")
    ap.add_argument("--calibracion", default=None,
                    help="(19-nov) calibración cámara→robot de ESTA cámara; por defecto, CALIB_ROBOT_ARCHIVO "
                         "(cabeza) o AUTONOMO_CAM_A_CALIB_ARCHIVO (a)")
    ap.add_argument("--escala-prof", type=float, default=None,
                    help="corrección de escala del estéreo; por defecto AUTONOMO_CAM_ESCALA_PROF (cabeza) o "
                         "AUTONOMO_CAM_A_ESCALA_PROF (a)")
    ap.add_argument("--canal-img", default=None, help="nombre del canal de imagen (None = no publicar)")
    ap.add_argument("--canal-info", default=None, help="nombre del canal de info (None = no publicar)")
    ap.add_argument("--ventana", action="store_true", help="enseñar la imagen en una ventana")
    ap.add_argument("--calidad", type=int, default=int(getattr(cfg, "AUTONOMO_JPEG_CALIDAD", 80)))
    ap.add_argument("--padre", type=int, default=0, help="PID del modo autónomo (si muere, salir)")
    ap.add_argument("--ir-ma", type=int, default=None)
    ap.add_argument("--canal-qbrazo", default=None, help="pose real del brazo (para quitarlo del mapa)")
    ap.add_argument("--canal-mapa", default=None, help="mapa de obstáculos (None = no calcularlo)")
    ap.add_argument("--canal-mando", default=None, help="mandos de la web (borrar mapa…)")
    ap.add_argument("--canal-visto", default=None, help="qué ha visto la cámara (para el planificador)")
    ap.add_argument("--canal-herramientas", default=None, help="herramientas detectadas (None = no detectar)")
    ap.add_argument("--canal-prof-a", default=None,
                    help="(20-nov) profundidad de la cámara A para el mapa: la A la escribe, la cabeza la funde")
    ap.add_argument("--canal-congelar", default=None,
                    help="(18-nov) aviso del modo autónomo: hay una tarea → no aplicar la autocorrección")
    ap.add_argument("--grabar", type=int, default=0, metavar="N",
                    help="guardar N pares color+profundidad en diagnosticos/escena_*.npz y salir "
                         "(con el modo autónomo cerrado), para revisar el mapa fuera del robot")
    args = ap.parse_args()
    a = args.rol == "a"
    if args.mxid is None:
        args.mxid = getattr(cfg, "AUTONOMO_CAM_A_MXID" if a else "AUTONOMO_CAM_MXID", None)
    if args.calibracion is None:
        args.calibracion = (getattr(cfg, "AUTONOMO_CAM_A_CALIB_ARCHIVO", "calibraciones/datos/camara_a_robot.json") if a
                            else getattr(cfg, "CALIB_ROBOT_ARCHIVO", "calibraciones/datos/camara_robot.json"))
    if args.escala_prof is None:
        args.escala_prof = float(getattr(cfg, "AUTONOMO_CAM_A_ESCALA_PROF" if a else "AUTONOMO_CAM_ESCALA_PROF",
                                         1.0) or 1.0)
    if args.ir_ma is None:
        args.ir_ma = int(getattr(cfg, "AUTONOMO_CAM_A_IR_MA" if a else "AUTONOMO_CAM_IR_MA", 800))
    args.nombre = "cámara A (mesa)" if a else "cámara de cabeza"
    return args


def _modos_estereo(cfg, rol):
    """(20-nov) Modos del estéreo a probar al abrir la OAK, de mejor a «como siempre»: [(mono_800p, extendida, bits)].
    (None, None, None) = el pipeline de siempre (400P, lo de config.py)."""
    pre = "AUTONOMO_CAM_A_" if rol == "a" else "AUTONOMO_CAM_"
    m800 = bool(getattr(cfg, pre + "MONO_800P", False))
    ext = bool(getattr(cfg, pre + "EXTENDIDA", False))
    bits = getattr(cfg, pre + "SUBPIXEL_BITS", None)
    bits = int(bits) if bits else None
    modos = []
    for m in ((m800, ext, bits), (m800, False, bits), (None, None, None)):
        if m not in modos and (m[0] or m[1] or m[2] or m == (None, None, None)):
            modos.append(m)
    return modos


def _texto_modo(m):
    if m == (None, None, None):
        return "400P (como siempre)"
    return (("800P" if m[0] else "400P") + (" + extendida" if m[1] else "")
            + (f" + subpíxel {m[2]} bits" if m[2] else ""))


def rol_de_mxid(cfg, mxid):
    """(20-nov) "a" si ese MxId es el de la cámara A del modo autónomo; si no, "cabeza"."""
    mx_a = getattr(cfg, "AUTONOMO_CAM_A_MXID", None)
    return "a" if mxid and mx_a and str(mxid) == str(mx_a) else "cabeza"


def abrir_como_autonomo(cfg, rol, mxid, nombre="cámara", **kw):
    """(20-nov) La OAK abierta con el MISMO estéreo que usa el modo autónomo para ese rol (del mejor modo a «como
    siempre», _modos_estereo) y con la entrada de control para fijar el enfoque (camara_oak.fijar_enfoque). La usan
    este proceso y lo que tiene que medir lo mismo que ve el robot (calibraciones/calibrar_escala_profundidad.py,
    diagnosticos/medir_pelota.py): la escala o el error medidos con otro modo del estéreo no valen para este.
    kw: el resto de crear_pipeline_oak (con_hd, rellenar_huecos…). Devuelve (dev o None, modo, último error)."""
    from skill_shield.vision.camara_oak import crear_pipeline_oak
    from skill_shield.vision.multicamara import abrir_dispositivo
    dev, modo, error = None, (None, None, None), None
    for modo in _modos_estereo(cfg, rol):
        try:
            dev = abrir_dispositivo(crear_pipeline_oak(mono_800p=modo[0], extendida=modo[1], subpixel_bits=modo[2],
                                                       control=True, **kw), mxid)
            break
        except Exception as e:
            error = e
            print(f"⚠️  [autónomo] {nombre.capitalize()}: no se abre con estéreo {_texto_modo(modo)} ({e})")
    return dev, modo, error


class _CalibArchivo:
    """(20-nov) T_mundo_cam de un archivo de calibración, recargada si cambia (para la profundidad de la A)."""

    def __init__(self, ruta):
        self.ruta, self.T, self._mtime, self._t = ruta, None, None, 0.0

    def actual(self):
        ahora = time.monotonic()
        if ahora - self._t >= 1.0:
            self._t = ahora
            try:
                m = os.path.getmtime(self.ruta)
            except OSError:
                m = None
            if m != self._mtime:
                self._mtime = m
                self.T = None
                if m is not None:
                    try:
                        from skill_shield.autonomo.calibracion import cargar
                        self.T = cargar(self.ruta)["T_mundo_cam"]
                    except Exception as e:
                        print(f"⚠️  [autónomo] No se pudo leer {self.ruta}: {e}")
        return self.T


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


def _grabar(dev, K, n, ruta_calib=None):
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
    ruta_c = ruta_calib or getattr(cfg, "CALIB_ROBOT_ARCHIVO", "calibraciones/datos/camara_robot.json")
    ruta_c = ruta_c if os.path.isabs(ruta_c) else os.path.join(raiz, ruta_c)
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
    from skill_shield.vision.camara_oak import fijar_enfoque
    from skill_shield.vision.multicamara import (activar_ir, comprobar_usb, intrinsecos_color, listar_camaras,
                                                 mxid_de)
    from skill_shield.autonomo.canal import Canal

    canal_img = Canal.esperar(args.canal_img) if args.canal_img else None
    canal_info = Canal.esperar(args.canal_info) if args.canal_info else None
    import config as cfg_g
    # (19-nov) los detectores leen la calibración de CALIB_ROBOT_ARCHIVO: aquí, la de ESTA cámara
    g_det = {k: getattr(cfg_g, k) for k in dir(cfg_g) if k.isupper()}
    g_det["CALIB_ROBOT_ARCHIVO"] = args.calibracion
    constructor = None
    if args.canal_mapa and args.rol == "a":
        print("ℹ️  [autónomo] Cámara A: sin mapa de obstáculos (lo hace la cámara de cabeza).")
        args.canal_mapa = None
    if args.canal_mapa:                                  # mapa de obstáculos (paso 7)
        import config as cfg
        from skill_shield.autonomo.mapa import ConstructorMapa
        constructor = ConstructorMapa(cfg, os.path.dirname(os.path.abspath(cfg.__file__)),
                                      Canal.esperar(args.canal_qbrazo) if args.canal_qbrazo else None,
                                      Canal.esperar(args.canal_mapa),
                                      Canal.esperar(args.canal_mando) if args.canal_mando else None,
                                      Canal.esperar(args.canal_visto) if args.canal_visto else None,
                                      canal_congelar=Canal.esperar(args.canal_congelar) if args.canal_congelar
                                      else None,
                                      canal_prof_a=Canal.esperar(args.canal_prof_a) if args.canal_prof_a else None)
        if constructor.canal_prof_a is not None:
            print("🧱 [autónomo] Mapa de obstáculos: también con la profundidad de la cámara A")
        print(f"🧱 [autónomo] Mapa de obstáculos: rejilla {tuple(int(v) for v in constructor.mapa.n)} de "
              f"{constructor.mapa.res * 100:.0f} cm")
    detector, canal_herr, canal_mando_h = None, None, None
    if args.canal_herramientas:                          # herramientas con ArUco (paso 5)
        import config as cfg
        from skill_shield.autonomo.herramientas import DetectorHerramientas
        detector = DetectorHerramientas(g_det, os.path.dirname(os.path.abspath(cfg.__file__)))
        canal_herr = Canal.esperar(args.canal_herramientas)
        canal_mando_h = Canal.esperar(args.canal_mando) if args.canal_mando else None
        print(f"🔧 [autónomo] Herramientas en el catálogo: "
              + ", ".join(f"{e['nombre']} (id {i})" for i, e in sorted(detector.cat.items())))
    pelotas = None
    if args.canal_herramientas:                          # pelotas por color + profundidad (11-nov)
        import config as cfg
        if getattr(cfg, "AUTONOMO_PELOTAS", None):
            from skill_shield.autonomo.pelotas import DetectorPelotas
            pelotas = DetectorPelotas(g_det, os.path.dirname(os.path.abspath(cfg.__file__)))
            print(f"⚽ [autónomo] Pelotas ({args.nombre}): " + ", ".join(e["nombre"] for e in pelotas.cat))
    t_herr, t_pub = 0.0, 0.0

    def _info(**kw):
        if canal_info is not None:
            canal_info.escribir(json.dumps(kw).encode())

    print(f"📷 [autónomo] OAK-D disponibles: {listar_camaras()}")
    import config as cfg_
    from skill_shield.autonomo.camara_a import distancia_minima
    # (20-nov) del mejor modo del estéreo a «como siempre», con la entrada de control (enfoque fijo)
    dev, modo, error = abrir_como_autonomo(
        cfg_, args.rol, args.mxid, args.nombre,
        con_hd=False,                                     # profundidad «honrada» para el mapa (11-nov)
        rellenar_huecos=getattr(cfg_, "AUTONOMO_CAM_RELLENAR_HUECOS", 0),
        temporal=getattr(cfg_, "AUTONOMO_CAM_FILTRO_TEMPORAL", False),
        confianza=getattr(cfg_, "AUTONOMO_CAM_CONFIANZA", 200))
    if dev is None:
        print(f"✗ [autónomo] No se pudo abrir la {args.nombre} (MxId {args.mxid}): {error}")
        print("   ¿Está conectada? ¿La tiene abierta otro programa (teleoperación, calibración)?")
        _info(ok=False, error=str(error), mxid=args.mxid, t=time.time())
        return 1
    # (20-nov) más cerca de esto el estéreo no mide con el modo abierto (y AUTONOMO_MAPA_Z_MIN como mínimo)
    ext = bool(getattr(cfg_, "STEREO_EXTENDED", False)) if modo[1] is None else bool(modo[1])
    z_min_cam = max(float(getattr(cfg_, "AUTONOMO_MAPA_Z_MIN", 0.35)), distancia_minima(bool(modo[0]), ext))
    print(f"📏 [autónomo] {args.nombre.capitalize()}: estéreo {_texto_modo(modo)} · el mapa no usa lo que esté a "
          f"menos de {100 * z_min_cam:.0f} cm")
    if constructor is not None:
        constructor.mapa.z_min = z_min_cam
    # (20-nov) cámara A → mapa de la cabeza: su profundidad preparada (sin sus pelotas ni lo poco fiable)
    canal_prof_a, calib_a, t_prof_a = None, None, 0.0
    if args.rol == "a" and args.canal_prof_a:
        canal_prof_a = Canal.esperar(args.canal_prof_a)
        calib_a = _CalibArchivo(args.calibracion if os.path.isabs(args.calibracion) else os.path.join(
            os.path.dirname(os.path.abspath(cfg_.__file__)), args.calibracion))
        periodo_a = 1.0 / max(float(getattr(cfg_, "AUTONOMO_CAM_A_MAPA_HZ", 4.0)), 0.5)
        dist_max_a = getattr(cfg_, "AUTONOMO_CAM_A_MAPA_DIST_MAX", 1.2)
        libre_a = bool(getattr(cfg_, "AUTONOMO_MAPA_PELOTAS_LIBRE_DELANTE", True))
        print(f"🧱 [autónomo] Cámara A: manda su profundidad al mapa de la cabeza ({1 / periodo_a:.0f}/s)")

    with dev:
        usb = comprobar_usb(dev, args.rol)
        activar_ir(dev, args.rol, args.ir_ma)
        # (20-nov) lente fija donde se calibró (la calibración cámara→robot también la fija): con el autoenfoque,
        # K y la profundidad alineada dejan de valer
        fijar_enfoque(dev, getattr(cfg_, "AUTONOMO_CAM_ENFOQUE", "calibrado"), args.nombre)
        K, dist = intrinsecos_color(dev, ANCHO_CAM, ALTO_CAM)
        mxid = mxid_de(dev)
        print(f"📷 [autónomo] {args.nombre.capitalize()} {mxid}: {ANCHO_CAM}x{ALTO_CAM}, "
              f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f} px")
        qc = dev.getOutputQueue("color", maxSize=1, blocking=False)
        qd = dev.getOutputQueue("depth", maxSize=1, blocking=False)
        if args.grabar:
            return _grabar(dev, K, args.grabar, args.calibracion)
        prof, n, t_fps, fps, t_info = None, 0, time.monotonic(), 0.0, 0.0
        t_prof, t_arranque, avisado_prof = time.monotonic(), time.monotonic(), False   # (20-nov) ¿llega profundidad?
        # (18-nov) corrección de escala de la profundidad del estéreo (1.0 = tal cual). Con una cinta: un cartón
        # plano en el centro de la imagen a D mm medidos; si «centro:» marca M, ESCALA = D / M
        escala_prof = float(args.escala_prof or 1.0)
        if abs(escala_prof - 1.0) > 1e-6:
            print(f"📏 [autónomo] {args.nombre}: profundidad del estéreo × {escala_prof:.4f}")
        img_ult = None                                    # último color (el mapa descarta lo saturado)
        hilo_mapa = _HiloMapa(constructor) if constructor is not None else None      # (18-nov) aparte
        ms_det, ms_jpg, t_tiempos = None, None, time.monotonic()

        def _media(m, v):
            return v if m is None else 0.9 * m + 0.1 * v
        while True:
            pd = _ultimo(qd)
            if pd is not None:
                prof = pd.getFrame()
                t_prof = time.monotonic()
                if abs(escala_prof - 1.0) > 1e-6:
                    prof = np.clip(prof.astype(np.float32) * escala_prof, 0, 65535).astype(np.uint16)
                if canal_prof_a is not None and time.monotonic() - t_prof_a >= periodo_a:
                    t_prof_a = time.monotonic()
                    try:
                        T_a = calib_a.actual()
                        if T_a is not None:
                            from skill_shield.autonomo.camara_a import empaquetar_prof, preparar_prof
                            halos_a = (pelotas.pixeles_halo() if pelotas is not None and getattr(
                                cfg_, "AUTONOMO_MAPA_SIN_HALO_PELOTAS", True) else None)
                            sub, Ks, msub = preparar_prof(prof, K, img_ult, halos_a, T_a, paso=2, z_min=z_min_cam,
                                                          libre_halos=libre_a)
                            b_a = empaquetar_prof(sub, Ks, T_a, msub, dist_max=dist_max_a)
                            if len(b_a) <= canal_prof_a.capacidad:
                                canal_prof_a.escribir(b_a)
                    except Exception as e:                # el mapa nunca debe tirar la cámara
                        print(f"⚠️  [autónomo] Profundidad de la cámara A para el mapa: {e}")
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
                    # (20-nov) con un modo del estéreo nuevo, la OAK podría abrir y no dar profundidad: avisarlo
                    if not avisado_prof and ahora - t_prof > 5.0 and ahora - t_arranque > 8.0:
                        avisado_prof = True
                        print(f"⚠️  [autónomo] {args.nombre.capitalize()}: llega imagen pero NO profundidad desde hace "
                              f"{ahora - t_prof:.0f} s (estéreo {_texto_modo(modo)}). Si sigue así, poned "
                              f"AUTONOMO_CAM_{'A_' if args.rol == 'a' else ''}EXTENDIDA = False (o MONO_800P = False).")
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
                        print(f"📷 [autónomo] {args.nombre}: {fps:.1f} fps · mapa {tiempos['mapa_ms']} ms (en su hilo) · "
                              f"detección {tiempos['deteccion_ms']} ms · jpeg {tiempos['jpeg_ms']} ms")
                if args.ventana:
                    cv2.imshow("Autonomo - " + ("camara A (mesa)" if args.rol == "a" else "camara de cabeza"), img)
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
