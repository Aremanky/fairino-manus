"""
Segunda cámara OAK-D (cámara B) para la teleoperación con dos puntos de vista.

· abrir_dispositivo()   abre una OAK-D concreta por MxId (o la primera libre).
· CamaraSecundaria      context manager: abre la cámara B, activa su IR, saca el fotograma
                        de color + profundidad más cercano en el tiempo al de la cámara A
                        y ejecuta SU PROPIA MediaPipe Pose en un hilo aparte.
· cargar_extrinsecas()  lee camaras_extrinsecas.json (lo genera calibraciones/calibrar_extrinseca.py).

La cámara A sigue funcionando EXACTAMENTE como antes en Skill_Shield_Nivel_6.py; la B
solo añade información. Convención de extrinsecos:  p_A = R_ab · p_B + t_ab.
"""
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import depthai as dai

import config as _cfg
from config import ANCHO_CAM, ALTO_CAM, F_PX
from skill_shield.vision.camara_oak import crear_pipeline_oak

CAM_A_MXID         = getattr(_cfg, "CAM_A_MXID", None)
CAM_B_MXID         = getattr(_cfg, "CAM_B_MXID", None)
SYNC_MAX_S         = float(getattr(_cfg, "MULTICAM_SYNC_MAX_S", 0.04))
EXTRINSECAS_ARCHIVO = getattr(_cfg, "EXTRINSECAS_ARCHIVO", "calibraciones/datos/camaras_extrinsecas.json")
IR_B_MA            = int(getattr(_cfg, "CAM_B_IR_MA", 800))


def _ruta_extrinsecas(ruta=None):
    ruta = ruta or EXTRINSECAS_ARCHIVO
    if os.path.isabs(ruta):
        return ruta
    return os.path.join(os.path.dirname(os.path.abspath(_cfg.__file__)), ruta)


def mxid_de(obj):
    """MxId de un dai.Device o dai.DeviceInfo. El nombre de la API cambia según la versión de
    DepthAI (getMxId() / mxid / getDeviceId() / deviceId), así que se prueban todos."""
    for nombre in ("getMxId", "getDeviceId"):
        f = getattr(obj, nombre, None)
        if callable(f):
            try:
                return str(f())
            except Exception:
                pass
    for nombre in ("mxid", "deviceId"):
        v = getattr(obj, nombre, None)
        if v:
            return str(v)
    return None


def listar_camaras():
    """[(mxid, estado)] de las OAK conectadas y libres."""
    return [(mxid_de(d), getattr(getattr(d, "state", None), "name", "?"))
            for d in dai.Device.getAllAvailableDevices()]


def abrir_dispositivo(pipeline, mxid=None, excluir=()):
    """
    dai.Device(pipeline) sobre una OAK concreta. Si mxid es None se coge la primera libre
    que NO esté en 'excluir' (para que la A no se quede con la que está reservada a la B).
    """
    if mxid:
        return dai.Device(pipeline, dai.DeviceInfo(mxid))
    excluir = {m for m in excluir if m}
    if not excluir:
        return dai.Device(pipeline)
    for d in dai.Device.getAllAvailableDevices():
        if mxid_de(d) not in excluir:
            return dai.Device(pipeline, d)
    raise RuntimeError("No hay ninguna OAK libre aparte de la reservada para la cámara B")


def intrinsecos_color(device, ancho=ANCHO_CAM, alto=ALTO_CAM):
    """(K 3x3, distorsión) de la cámara de color para ese tamaño de fotograma."""
    cal = device.readCalibration()
    K = np.array(cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, ancho, alto), dtype=float)
    dist = np.array(cal.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A), dtype=float)
    return K, dist


def activar_ir(device, nombre, ma=IR_B_MA):
    try:
        device.setIrLaserDotProjectorBrightness(int(ma))
        print(f"💡 [{nombre}] Proyector IR de puntos activado ({ma} mA)")
    except Exception as e1:
        try:
            device.setIrLaserDotProjectorIntensity(min(1.0, ma / 1200.0))
            print(f"💡 [{nombre}] Proyector IR de puntos activado")
        except Exception as e2:
            print(f"⚠️  [{nombre}] No se pudo activar el proyector IR: {e1} / {e2}")


def comprobar_usb(device, nombre):
    try:
        usb = device.getUsbSpeed()
        print(f"🔌 [{nombre}] Conectada por USB: {usb.name}")
        if usb not in (dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS):
            print(f"⚠️  [{nombre}] NO va por USB3: usa un puerto/cable USB3, mejor en un controlador "
                  f"distinto al de la otra cámara y sin hub.")
        return usb
    except Exception as e:
        print(f"⚠️  [{nombre}] No se pudo leer la velocidad USB: {e}")
        return None


def cargar_extrinsecas(mxid_a=None, mxid_b=None, ruta=None):
    """
    Devuelve (R_ab, t_ab, info) con  p_A = R_ab · p_B + t_ab, o None si no hay archivo o no
    corresponde a estas cámaras. Si se abrieron en orden contrario al de la calibración
    (A↔B), se invierte la transformación automáticamente.
    """
    ruta = _ruta_extrinsecas(ruta)
    if not os.path.exists(ruta):
        print(f"⚠️  No existe {os.path.basename(ruta)}: ejecuta  python calibraciones/calibrar_extrinseca.py")
        return None
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    R = np.array(d["R_ab"], dtype=float).reshape(3, 3)
    t = np.array(d["t_ab"], dtype=float).reshape(3)
    fa, fb = d.get("mxid_a"), d.get("mxid_b")
    if mxid_a and mxid_b and fa and fb:
        if (mxid_a, mxid_b) == (fa, fb):
            pass
        elif (mxid_a, mxid_b) == (fb, fa):
            print("↔️  Las cámaras se abrieron en orden inverso al de la calibración: "
                  "se invierte la transformación (fija CAM_A_MXID/CAM_B_MXID en config.py).")
            R, t = R.T, -R.T @ t
        else:
            print("⚠️  Las cámaras abiertas NO coinciden con las de camaras_extrinsecas.json "
                  f"({fa} / {fb}). Recalibra con calibraciones/calibrar_extrinseca.py.")
            return None
    return R, t, d


class CamaraSecundaria:
    """
    Cámara B. Uso (ver el parche en Skill_Shield_Nivel_6.py):

        with abrir_dispositivo(pipeline, ...) as device, CamaraSecundaria(device_a=...) as cam_b:
            fut = cam_b.lanzar(t_color_A)         # antes de la Pose de A (corre en otro hilo)
            ...pose de A...
            ent_b = cam_b.recoger(fut)            # (pose_lm, pose_world_lm, depth, f_px) o None
    """

    def __init__(self, mxid=None, excluir=(), ancho=ANCHO_CAM, alto=ALTO_CAM):
        self.mxid_pedido = mxid or CAM_B_MXID
        self.excluir     = tuple(excluir)
        self.ancho, self.alto = ancho, alto
        self.device = None
        self.mxid   = None
        self.f_px   = F_PX
        self.K = self.dist = None
        self.frame_bgr = None          # último fotograma de B (para la ventana de depuración)
        self.n_perdidos = 0
        self._buf_color = deque(maxlen=6)
        self._buf_depth = deque(maxlen=6)
        self._pool = None
        self._pose = None

    # — ciclo de vida —
    def __enter__(self):
        import mediapipe as mp
        print("⚡ Conectando OAK-D secundaria (cámara B)...")
        pipeline = crear_pipeline_oak(self.ancho, self.alto, con_hd=False)
        self.device = abrir_dispositivo(pipeline, self.mxid_pedido, self.excluir)
        self.mxid = mxid_de(self.device)
        print(f"📷 [B] OAK-D abierta: {self.mxid}")
        comprobar_usb(self.device, "B")
        activar_ir(self.device, "B")
        self._q_color = self.device.getOutputQueue("color", maxSize=4, blocking=False)
        self._q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
        try:
            self.K, self.dist = intrinsecos_color(self.device, self.ancho, self.alto)
            self.f_px = float(self.K[0][0])
            print(f"📷 [B] Focal calibrada: {self.f_px:.1f} px")
        except Exception as e:
            print(f"⚠️  [B] No se pudo leer la calibración ({e}); uso F_PX={F_PX}")
        # Instancia PROPIA de Pose: MediaPipe hace seguimiento entre fotogramas y no se
        # puede compartir entre dos cámaras.
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=_cfg.POSE_COMPLEJIDAD,
            smooth_landmarks=True,
            min_detection_confidence=_cfg.POSE_CONF_DETECCION,
            min_tracking_confidence=_cfg.POSE_CONF_SEGUIMIENTO)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="PoseB")
        return self

    def __exit__(self, *exc):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        if self._pose is not None:
            self._pose.close()
        if self.device is not None:
            self.device.close()
        return False

    # — captura sincronizada —
    def _fotograma_cercano(self, t_ref):
        for q, buf in ((self._q_color, self._buf_color), (self._q_depth, self._buf_depth)):
            while True:
                p = q.tryGet()
                if p is None:
                    break
                buf.append(p)
        if not self._buf_color or not self._buf_depth:
            return None
        dt = lambda p: abs((p.getTimestamp() - t_ref).total_seconds())
        pc = min(self._buf_color, key=dt)
        if dt(pc) > SYNC_MAX_S:
            return None
        pd = min(self._buf_depth, key=lambda p: abs((p.getTimestamp() - pc.getTimestamp()).total_seconds()))
        return pc.getCvFrame(), pd.getFrame()

    def _trabajo(self, t_ref):
        par = self._fotograma_cercano(t_ref)
        if par is None:
            self.n_perdidos += 1
            return None
        frame_bgr, depth = par
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._pose.process(rgb)
        self.frame_bgr = frame_bgr
        if not (res.pose_landmarks and res.pose_world_landmarks):
            return (None, None, depth, self.f_px)
        return (res.pose_landmarks, res.pose_world_landmarks, depth, self.f_px)

    def lanzar(self, t_ref):
        """Pide a B su fotograma más cercano a t_ref y le pasa la Pose (hilo aparte)."""
        return self._pool.submit(self._trabajo, t_ref)

    def recoger(self, futuro):
        """(pose_lm, pose_world_lm, depth, f_px) listo para la fusión, o None si B no ve el cuerpo."""
        try:
            r = futuro.result()
        except Exception as e:
            print(f"⚠️  [B] error en el hilo de Pose: {e}")
            return None
        if r is None or r[0] is None:
            return None
        return r

    def dibujar_y_mostrar(self, nombre="Camara B", escala=0.6, pose_lm=None, texto=""):
        """Ventana de depuración de la cámara B (esqueleto + texto)."""
        if self.frame_bgr is None:
            return
        import mediapipe as mp
        img = self.frame_bgr.copy()
        if pose_lm is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                img, pose_lm, mp.solutions.pose.POSE_CONNECTIONS)
        if texto:
            cv2.putText(img, texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow(nombre, cv2.resize(img, None, fx=escala, fy=escala))
