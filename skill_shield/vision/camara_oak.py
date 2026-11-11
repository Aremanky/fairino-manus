"""
Pipeline DepthAI de la OAK-D (color + estéreo para profundidad real).

(Extraído de Skill_Shield_Nivel_6.py — código sin cambios, solo reubicado.)

7-oct: las salidas XLinkOut ya NO acumulan fotogramas en la cámara (cola de 1, sin
bloquear). Antes, por defecto, cada salida guardaba hasta 8 fotogramas en la propia
OAK-D: si el USB no daba abasto (dos cámaras, imagen HD, USB2 o un hub compartido) esa
cola se llenaba y TODO lo que veía el programa llegaba con 1-2 s de retraso.
"""
import depthai as dai

from config import *


def _salida_sin_retraso(xout):
    """XLinkOut que, si el USB va justo, TIRA fotogramas viejos en vez de encolarlos."""
    try:
        xout.input.setBlocking(False)
        xout.input.setQueueSize(1)
    except Exception as e:
        print(f"⚠️  [OAK] No se pudo ajustar la cola de '{xout.getStreamName()}': {e}")


def latencia_s(paquete) -> float:
    """Segundos desde que el sensor capturó el fotograma hasta AHORA (en el PC).
    getTimestamp() va en el reloj del PC (DepthAI lo sincroniza), igual que dai.Clock.now()."""
    try:
        return (dai.Clock.now() - paquete.getTimestamp()).total_seconds()
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE DEPTHAI  (OAK-D: color + estéreo para profundidad real)
# ══════════════════════════════════════════════════════════════════════════════

def crear_pipeline_oak(ancho: int = ANCHO_CAM, alto: int = ALTO_CAM,
                       con_hd: bool = True, rellenar_huecos=None, temporal=None,
                       confianza=None) -> dai.Pipeline:
    """con_hd=False: no se crea la salida 'color_hd' (recorte de mano). Lo usa la cámara
    SECUNDARIA para ahorrar ~2/3 del ancho de banda USB (ver multicamara.py).
    rellenar_huecos / temporal / confianza (11-nov): None = los de la teleoperación
    (STEREO_HOLE_FILLING, STEREO_TEMPORAL, la del preset). El mapa de obstáculos pide 0 / False /
    más estricta: rellenar huecos y el filtro temporal se INVENTAN profundidad (paredes blancas,
    ventanas, estelas) y eso sale como obstáculos flotando."""
    huecos = STEREO_HOLE_FILLING if rellenar_huecos is None else int(rellenar_huecos)
    con_temporal = STEREO_TEMPORAL if temporal is None else bool(temporal)
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setInterleaved(False)

    if CAM_FOV_COMPLETO:
        # ── SENSOR COMPLETO (4:3) ────────────────────────────────────────────
        # El IMX378 es 4056x3040 (4:3). El modo 1080p es 16:9, y el firmware lo
        # obtiene RECORTANDO franjas arriba y abajo → se pierde ~24 % de altura.
        # Pidiendo 12 MP y reduciendo en el ISP (1/3 → 1352x1014) se conserva
        # TODO el campo de visión vertical: VFOV ~42° → ~55°.
        # 1352x1014 y 640x480 tienen exactamente la misma relación 4:3, así que
        # el preview es un reescalado limpio, SIN recorte adicional.
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
        cam_rgb.setIspScale(1, 3)
    else:
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    cam_rgb.setPreviewSize(ancho, alto)
    cam_rgb.setPreviewKeepAspectRatio(True)
    cam_rgb.setFps(CAM_FPS)

    mono_izq = pipeline.create(dai.node.MonoCamera)
    mono_der = pipeline.create(dai.node.MonoCamera)
    for cam, socket in [
        (mono_izq, dai.CameraBoardSocket.CAM_B),
        (mono_der, dai.CameraBoardSocket.CAM_C),
    ]:
        # THE_400_P es el modo "binned" del OV9282: usa el sensor ENTERO (1280x800)
        # → mantiene todo el FOV vertical. NO cambiar a 720p: eso sí recorta.
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam.setBoardSocket(socket)
        cam.setFps(CAM_FPS)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(ancho, alto)   # misma medida que el color → mismos píxeles
    stereo.setLeftRightCheck(True)      # descarta píxeles donde las dos cámaras no coinciden
    stereo.setSubpixel(STEREO_SUBPIXEL) # profundidad más fina (menos "escalones")
    if STEREO_EXTENDED:
        stereo.setExtendedDisparity(True)

    # ── Post-procesado EN LA CÁMARA (0 % de CPU del PC) ──────────────────────
    # Los agujeros del mapa de profundidad sobre el brazo (superficie lisa, poco
    # textura, movimiento) son la causa principal de que el brazo "se pierda":
    # profundidad_robusta() devuelve None y la articulación se descarta entera.
    if STEREO_FILTROS:
        try:
            _cfg = stereo.initialConfig.get()
            _cfg.postProcessing.speckleFilter.enable      = True
            _cfg.postProcessing.speckleFilter.speckleRange = 50
            _cfg.postProcessing.spatialFilter.enable      = True
            _cfg.postProcessing.spatialFilter.holeFillingRadius = huecos
            _cfg.postProcessing.spatialFilter.numIterations     = 1
            _cfg.postProcessing.temporalFilter.enable    = con_temporal
            if confianza is not None:           # más bajo = más estricto (0-255)
                try:
                    _cfg.costMatching.confidenceThreshold = int(confianza)
                except Exception as _e_c:
                    print(f"⚠️  [OAK] No se pudo poner la confianza del estéreo: {_e_c}")
            _cfg.postProcessing.thresholdFilter.minRange = int(PROF_Z_MIN * 1000)
            _cfg.postProcessing.thresholdFilter.maxRange = int(PROF_Z_MAX * 1000)
            stereo.initialConfig.set(_cfg)
            print("🧹 [OAK] Filtros de profundidad activados "
                  f"(hole-filling={huecos}, temporal={con_temporal}"
                  + (f", confianza={confianza}" if confianza is not None else "") + ")")
        except Exception as _e_pp:
            print(f"⚠️  [OAK] No se pudieron aplicar los filtros de profundidad: {_e_pp}")

    mono_izq.out.link(stereo.left)
    mono_der.out.link(stereo.right)

    xout_c = pipeline.create(dai.node.XLinkOut)
    xout_c.setStreamName("color")
    cam_rgb.preview.link(xout_c.input)
    _salida_sin_retraso(xout_c)

    # Imagen del ISP (1352x1014 con CAM_FOV_COMPLETO): mismo campo de visión que el
    # preview pero 2,1x más píxeles. Solo se usa para recortar la mano (MANO_RECORTE_HD).
    if con_hd and MANO_RECORTE and MANO_RECORTE_HD and CAM_FOV_COMPLETO:
        xout_hd = pipeline.create(dai.node.XLinkOut)
        xout_hd.setStreamName("color_hd")
        cam_rgb.isp.link(xout_hd.input)
        _salida_sin_retraso(xout_hd)

    xout_d = pipeline.create(dai.node.XLinkOut)
    xout_d.setStreamName("depth")
    stereo.depth.link(xout_d.input)
    _salida_sin_retraso(xout_d)

    return pipeline