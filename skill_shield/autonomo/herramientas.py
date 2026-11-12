"""
Detección de herramientas con marcadores ArUco (paso 5 de la hoja de ruta v2, 10-nov).

Cada herramienta (y el punto de uso y el de dejada) lleva un marcador ArUco impreso. El
catálogo está en config.py (AUTONOMO_HERRAMIENTAS): id del marcador, nombre, tamaño impreso,
dónde está el marcador respecto a la herramienta y el tamaño de la herramienta (caja).

Por cada imagen de la cámara de cabeza:
  marcadores → pose 6D de cada uno (solvePnP IPPE_SQUARE, se descartan las ambiguas y las de
  mucho error) → la distancia se corrige con la profundidad estéreo del propio marcador (con
  marcadores pequeños es lo que más falla) → pose de la herramienta (marcador · desplazamiento del catálogo) → filtro en
  el tiempo (suaviza y exige varias imágenes antes de darla por «estable») → con la
  calibración cámara→robot, pose en el MUNDO (marco del MJCF, el del gemelo).

Diccionario DICT_4X4_50 (el tablero de calibración usa 5x5, para no confundirlos).

Herramientas de línea de comandos:
    python -m skill_shield.autonomo.herramientas --generar   # PNG de cada marcador para imprimir
    python -m skill_shield.autonomo.herramientas --probar    # cámara de cabeza en vivo: poses y
                                                             #   cuánto tiemblan (cierra antes el modo)
"""
import math
import os
import time

import cv2
import numpy as np

from skill_shield.autonomo.autocalib import ConCorreccion

DICCIONARIO = "DICT_4X4_50"
TIPOS = ("herramienta", "punto_uso", "punto_dejada")


def _rot_xyz(g):
    a, b, c = np.radians(g)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def catalogo(cfg):
    """{id: entrada} con valores por defecto. cfg: módulo config o dict."""
    lista = (cfg.get("AUTONOMO_HERRAMIENTAS") if isinstance(cfg, dict)
             else getattr(cfg, "AUTONOMO_HERRAMIENTAS", None)) or []
    out = {}
    for h in lista:
        e = dict(tipo="herramienta", lado=0.04, desplazamiento=(0.0, 0.0, 0.0), giro_deg=(0.0, 0.0, 0.0),
                 caja=(0.02, 0.02, 0.02))
        e.update(h)
        e["T_marcador_herramienta"] = _T(_rot_xyz(e["giro_deg"]), e["desplazamiento"])
        out[int(e["id"])] = e
    return out


def ids_del_tablero(g, diccionario):
    """Ids de los marcadores del tablero ChArUco de calibración si usa el MISMO diccionario que
    las herramientas (un ChArUco de nx × ny cuadros lleva nx·ny // 2 marcadores: ids 0, 1, …)."""
    t = g.get("CALIB_ROBOT_TABLERO") or {}
    if t.get("diccionario") != diccionario:
        return set()
    nx, ny = t.get("cuadros", (0, 0))
    return set(range(int(nx) * int(ny) // 2))


def crear_detector(diccionario=DICCIONARIO):
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, diccionario))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX    # esquinas al subpíxel: pose más fina
    p.minMarkerPerimeterRate = 0.02                              # marcadores pequeños / lejos
    p.adaptiveThreshWinSizeMax = 43
    return cv2.aruco.ArucoDetector(dic, p)


def pose_marcador(esquinas, lado, K, dist, max_error_px=2.0, min_ratio=1.5):
    """(T cámara←marcador, estado, error_px, T alternativa). Marco del marcador: x a la derecha,
    y arriba, z saliendo hacia la cámara. IPPE da dos soluciones: si sus errores se parecen mucho
    la pose es «ambigua» (marcador pequeño y de frente) y se devuelve también la otra."""
    h = lado / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    img = np.asarray(esquinas, np.float64).reshape(4, 2)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error as e:
        return None, f"solvePnP: {e}", None, None
    if not n:
        return None, "sin solución", None, None
    errs = np.asarray(errs, float).ravel()
    orden = np.argsort(errs)
    k = int(orden[0])
    err = float(errs[k])
    if err > max_error_px:
        return None, f"error {err:.1f} px", err, None
    ambigua = len(errs) > 1 and errs[orden[1]] < min_ratio * max(err, 1e-6) and errs[orden[1]] < max_error_px
    T = _T(cv2.Rodrigues(rvecs[k])[0], np.asarray(tvecs[k]).ravel())
    if T[2, 3] <= 0:
        return None, "detrás de la cámara", err, None
    T_alt = _T(cv2.Rodrigues(rvecs[int(orden[1])])[0], np.asarray(tvecs[int(orden[1])]).ravel()) if ambigua else None
    return T, ("ambigua" if ambigua else "ok"), err, T_alt


def corregir_con_profundidad(T, esquinas, prof_mm, max_dif_m=0.08, min_px=12):
    """Corrige la DISTANCIA de la pose con la profundidad estéreo de la OAK-D (alineada con el
    color). Con un marcador pequeño (2-3 cm a 0,8 m son ~20 px) el error de solvePnP está casi
    todo en la distancia (1-3 cm); la profundidad del propio marcador (blanco y negro: el estéreo
    lo mide muy bien) la deja en unos mm. La dirección (el rayo del píxel) y el giro siguen
    saliendo de la imagen. Devuelve (T corregida, True) o (T, False) si no hay profundidad fiable."""
    if prof_mm is None:
        return T, False
    h, w = prof_mm.shape[:2]
    c = np.asarray(esquinas, np.float32).reshape(4, 2)
    centro = c.mean(0)
    poli = (centro + 0.7 * (c - centro)).astype(np.int32)       # 70 % interior: sin los bordes
    x0, y0 = np.maximum(poli.min(0), 0)
    x1, y1 = np.minimum(poli.max(0) + 1, [w, h])
    if x1 <= x0 or y1 <= y0:
        return T, False
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, poli - [x0, y0], 1)
    z = prof_mm[y0:y1, x0:x1][mask > 0]
    z = z[z > 0]
    if z.size < min_px:
        return T, False
    z_est = float(np.median(z)) * 0.001
    z_pnp = float(T[2, 3])
    if abs(z_est - z_pnp) > max_dif_m:
        return T, False
    T2 = T.copy()
    T2[:3, 3] = T[:3, 3] * (z_est / z_pnp)                       # mismo rayo, otra distancia
    return T2, True


def normal_por_profundidad(esquinas, prof_mm, K, escala=1.6, min_px=30):
    """Normal (marco de la cámara, apuntando hacia la cámara) del plano en el que está pegado el
    marcador, ajustando un plano a la profundidad estéreo alrededor de él. Sirve para deshacer
    la ambigüedad de IPPE con marcadores casi de frente. None si no hay datos suficientes."""
    if prof_mm is None:
        return None
    h, w = prof_mm.shape[:2]
    c = np.asarray(esquinas, np.float32).reshape(4, 2)
    centro = c.mean(0)
    poli = (centro + escala * (c - centro)).astype(np.int32)    # el marcador con su margen blanco
    x0, y0 = np.maximum(poli.min(0), 0)
    x1, y1 = np.minimum(poli.max(0) + 1, [w, h])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, poli - [x0, y0], 1)
    vv, uu = np.nonzero(mask)
    z = prof_mm[y0 + vv, x0 + uu].astype(np.float64) * 0.001
    ok = z > 0
    if ok.sum() < min_px:
        return None
    z, u, v = z[ok], uu[ok] + x0, vv[ok] + y0
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    med = np.median(P[:, 2])
    P = P[np.abs(P[:, 2] - med) < 0.03]                          # fuera lo que no es el plano
    if len(P) < min_px:
        return None
    _, sv, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    if sv[2] > 0.5 * sv[1]:                                      # no parece un plano
        return None
    n = Vt[2]
    return -n if n[2] > 0 else n                                 # hacia la cámara (z negativa)


class _Seguimiento:
    """Filtro de una herramienta: posición con media exponencial, giro con media de matrices
    proyectada (simple y suficiente a 15-30 Hz), cuenta de imágenes y hora de la última."""

    def __init__(self, alfa=0.4):
        self.alfa = float(alfa)
        self.T = None
        self.n = 0
        self.t = 0.0
        self.err = None
        self.hist = []                         # últimas posiciones (para el temblor)

    def nueva(self, T, err, ahora):
        if self.T is None or ahora - self.t > 1.0 or np.linalg.norm(T[:3, 3] - self.T[:3, 3]) > 0.10:
            self.T, self.n, self.hist = T.copy(), 0, []          # nueva o se ha movido mucho: reinicia
        else:
            a = self.alfa
            p = (1 - a) * self.T[:3, 3] + a * T[:3, 3]
            M = (1 - a) * self.T[:3, :3] + a * T[:3, :3]
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            self.T = _T(R, p)
        self.n += 1
        self.t = ahora
        self.err = err
        self.hist = (self.hist + [T[:3, 3].copy()])[-30:]

    def temblor_mm(self):
        if len(self.hist) < 5:
            return None
        return float(np.linalg.norm(np.std(np.array(self.hist), axis=0)) * 1000)


class DetectorHerramientas(ConCorreccion):
    """Detecta, sigue y publica las herramientas del catálogo. paso(img, K, dist) por imagen."""

    def __init__(self, cfg, raiz, n_estable=5, olvido_s=2.0, max_error_px=2.0):
        g = cfg if isinstance(cfg, dict) else {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
        self.cat = catalogo(g)
        self.det = crear_detector(g.get("AUTONOMO_HERRAMIENTAS_DICCIONARIO", DICCIONARIO))
        ruta = g.get("CALIB_ROBOT_ARCHIVO", "camara_robot.json")
        self.ruta_calib = ruta if os.path.isabs(ruta) else os.path.join(raiz, ruta)
        self.n_estable = int(g.get("AUTONOMO_HERRAMIENTAS_N_ESTABLE", n_estable))
        self.olvido_s = float(g.get("AUTONOMO_HERRAMIENTAS_OLVIDO_S", olvido_s))
        self.max_error_px = float(max_error_px)
        self.ignorar = ids_del_tablero(g, g.get("AUTONOMO_HERRAMIENTAS_DICCIONARIO", DICCIONARIO))
        self.seg = {}                                    # id → _Seguimiento
        self.desconocidos = {}                           # id → hora (marcadores fuera del catálogo)
        self.T_mundo_cam, self._mtime, self._t_calib = None, None, 0.0
        self.ultimas = []                                # detecciones de la última imagen (para dibujar)

    def _recargar(self, ahora):
        if ahora - self._t_calib < 1.0:
            return
        self._t_calib = ahora
        try:
            m = os.path.getmtime(self.ruta_calib)
        except OSError:
            m = None
        if m != self._mtime:
            self._mtime = m
            if m is None:
                self.T_mundo_cam = None
            else:
                from skill_shield.autonomo.calibracion import cargar
                self.T_mundo_cam = cargar(self.ruta_calib)["T_mundo_cam"]

    def paso(self, img_bgr, K, dist, ahora=None, prof_mm=None):
        """prof_mm: profundidad alineada con la imagen (mm). Si se da, corrige la distancia."""
        ahora = time.time() if ahora is None else ahora
        self._recargar(time.monotonic())
        gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        esquinas, ids, _ = self.det.detectMarkers(gris)
        self.ultimas = []
        if ids is None:
            return
        for c, i in zip(esquinas, ids.ravel()):
            i = int(i)
            if i in self.ignorar and i not in self.cat:  # marcadores del tablero de calibración
                continue
            e = self.cat.get(i)
            if e is None:
                self.desconocidos[i] = ahora
                self.ultimas.append(dict(id=i, esquinas=c.reshape(4, 2), T=None, estado="no está en el catálogo"))
                continue
            T, estado, err, alt = pose_marcador(c, float(e["lado"]), K, dist, self.max_error_px)
            self.ultimas.append(dict(id=i, esquinas=c.reshape(4, 2), T=T, estado=estado))
            if T is None:
                continue
            s_ = self.seg.setdefault(i, _Seguimiento())
            if estado == "ambigua":
                # Marcador pequeño y casi de frente: dos giros posibles. Se decide con el plano que
                # mide el estéreo alrededor del marcador; si no hay, con la pose anterior; si
                # tampoco, se espera a una imagen sin dudas.
                n = normal_por_profundidad(c, prof_mm, K)
                if n is not None and alt is not None:
                    if abs(alt[:3, 2] @ n) > abs(T[:3, 2] @ n):
                        T = alt
                elif s_.T is not None and ahora - s_.t <= 0.5:
                    T_ant = s_.T @ np.linalg.inv(e["T_marcador_herramienta"])
                    if alt is not None and np.trace(alt[:3, :3].T @ T_ant[:3, :3]) > np.trace(T[:3, :3].T @ T_ant[:3, :3]):
                        T = alt
                else:
                    continue
            T, con_prof = corregir_con_profundidad(T, c, prof_mm)
            s_.nueva(T @ e["T_marcador_herramienta"], err, ahora)
            s_.estereo = con_prof

    def lista(self, ahora=None):
        """Lo que se publica: una entrada por elemento del catálogo visto recientemente."""
        ahora = time.time() if ahora is None else ahora
        out = []
        for i, s in sorted(self.seg.items()):
            edad = ahora - s.t
            if edad > self.olvido_s:
                continue
            e = self.cat[i]
            T_m = self.T_mundo_cam @ s.T if self.T_mundo_cam is not None else None
            out.append(dict(id=i, nombre=e["nombre"], tipo=e["tipo"], estable=s.n >= self.n_estable,
                            edad_s=round(edad, 2), n=s.n, error_px=round(float(s.err), 2),
                            temblor_mm=None if s.temblor_mm() is None else round(s.temblor_mm(), 1),
                            estereo=bool(getattr(s, "estereo", False)),
                            distancia_m=round(float(np.linalg.norm(s.T[:3, 3])), 3),
                            T_cam=np.round(s.T, 5).tolist(),
                            T_mundo=None if T_m is None else np.round(T_m, 5).tolist(),
                            caja=list(map(float, e["caja"]))))
        for i, t in sorted(self.desconocidos.items()):
            if ahora - t <= self.olvido_s:
                out.append(dict(id=i, nombre=f"Marcador {i}", tipo="desconocido", estable=False,
                                edad_s=round(ahora - t, 2)))
        return out

    def caja_mundo(self, id_):
        """Caja alineada con el mundo (centro, semi) que envuelve la herramienta, con 2 cm de
        margen: para quitarla del mapa de obstáculos cuando es la que se va a coger."""
        s, e = self.seg.get(int(id_)), self.cat.get(int(id_))
        if s is None or e is None or self.T_mundo_cam is None or time.time() - s.t > self.olvido_s:
            return None
        T = self.T_mundo_cam @ s.T
        semi = np.abs(T[:3, :3]) @ np.asarray(e["caja"], float) + 0.02
        return T[:3, 3].tolist(), semi.tolist()

    def cajas_hojas(self, ahora=None, margen=(0.02, 0.02, 0.035), hoja=(1.2, 1.3, 1.3)):
        """(12-nov) Las hojas de los marcadores NO son obstáculos: una caja orientada por marcador
        visto (centro, semiejes, R en el mundo) del tamaño de la hoja y con un grosor que cubre el
        ruido de la profundidad. Lo que haya debajo (la mesa, una caja) sigue contando.
        hoja = (medio ancho, hasta arriba, hasta abajo) desde el centro del marcador, en lados del
        marcador; cada marcador del catálogo puede llevar la suya (clave «hoja»): p. ej. el del
        destornillador, con la hoja colgando de una pinza, llega más abajo."""
        ahora = time.time() if ahora is None else ahora
        if self.T_mundo_cam is None:
            return []
        out = []
        for i, s in self.seg.items():
            e = self.cat.get(i)
            if e is None or s.T is None or ahora - s.t > self.olvido_s:
                continue
            T = self.T_mundo_cam @ s.T @ np.linalg.inv(e["T_marcador_herramienta"])     # el marcador
            lado = float(e["lado"])
            ancho, arriba, abajo = (float(v) for v in (e.get("hoja") or hoja))
            # marcador de OpenCV: x a la derecha, y hacia ARRIBA, z saliendo de la hoja
            semi = np.array([lado * ancho + margen[0], lado * (arriba + abajo) / 2 + margen[1], margen[2]])
            centro = T[:3, :3] @ np.array([0.0, lado * (arriba - abajo) / 2, 0.0]) + T[:3, 3]
            out.append((centro, semi, T[:3, :3].copy()))
        return out

    def dibujar(self, img, K, dist):
        """Marcadores, ejes y nombres sobre la imagen de la cámara (para la web)."""
        for d in self.ultimas:
            c = d["esquinas"].astype(int)
            e = self.cat.get(d["id"])
            ok = d["T"] is not None and d["estado"] == "ok"
            color = (0, 200, 0) if ok else (0, 165, 255)
            cv2.polylines(img, [c.reshape(-1, 1, 2)], True, color, 2)
            if ok and e is not None:
                cv2.drawFrameAxes(img, K, dist, cv2.Rodrigues(d["T"][:3, :3])[0], d["T"][:3, 3], float(e["lado"]) * 0.75)
            nombre = e["nombre"] if e is not None else f"id {d['id']} (?)"
            txt = nombre + ("" if ok else f" ({d['estado']})")
            x, y = int(c[:, 0].min()), int(c[:, 1].min()) - 6
            cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def dibujar_herramientas(scn, lista, objetivo=None):
    """Cajas de las herramientas detectadas (con pose en el mundo) en una escena de MuJoCo:
    verdes, la elegida en naranja; los puntos de uso/dejada como discos azules."""
    import mujoco
    n = 0
    for h in lista:
        T = h.get("T_mundo")
        if T is None or scn.ngeom >= scn.maxgeom:
            continue
        T = np.asarray(T, float)
        if h["tipo"] == "pelota":
            c = list(h.get("color") or (1.0, 0.8, 0.1))
            rgba = np.array([1.0, 0.45, 0.0, 0.9] if h["id"] == objetivo else c[:3] + [0.85], np.float32)
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([h["caja"][0], 0, 0]),
                                T[:3, 3], np.eye(3).flatten(), rgba)
        elif h["tipo"] == "herramienta":
            rgba = np.array([1.0, 0.45, 0.0, 0.85] if h["id"] == objetivo else [0.1, 0.8, 0.25, 0.75], np.float32)
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.asarray(h["caja"], float),
                                T[:3, 3], T[:3, :3].flatten(), rgba)
        else:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.04, 0.003, 0.0]),
                                T[:3, 3], T[:3, :3].flatten(), np.array([0.1, 0.45, 1.0, 0.7], np.float32))
        scn.ngeom += 1
        n += 1
    return n


# ═════════════ línea de comandos ═════════════

def generar_marcadores(cfg, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICCIONARIO))
    for i, e in sorted(catalogo(cfg).items()):
        px = int(round(float(e["lado"]) * 1000 * 300 / 25.4))   # 300 ppp
        m = cv2.aruco.generateImageMarker(dic, i, px)
        borde = px // 4                                           # margen blanco: hace falta para detectarlo
        m = cv2.copyMakeBorder(m, borde, borde + px // 3, borde, borde, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(m, f"{e['nombre']} · id {i} · {float(e['lado']) * 1000:.0f} mm", (borde, m.shape[0] - px // 8),
                    cv2.FONT_HERSHEY_SIMPLEX, px / 700.0, 0, max(1, px // 250), cv2.LINE_AA)
        ruta = os.path.join(carpeta, f"marcador_{i:02d}_{e['nombre'].lower().replace(' ', '_')}.png")
        cv2.imwrite(ruta, m)
        print(f"   {ruta}  ({float(e['lado']) * 1000:.0f} mm de lado negro)")
    print("Imprímelos a 300 ppp al 100 %, comprueba con una regla el lado del cuadrado negro y pégalos "
          "planos (sin recortar el margen blanco).")


def probar(cfg):
    """Cámara de cabeza en vivo: poses y temblor de cada herramienta."""
    from config import ANCHO_CAM, ALTO_CAM
    from skill_shield.vision.camara_oak import crear_pipeline_oak
    from skill_shield.vision.multicamara import abrir_dispositivo, intrinsecos_color
    det = DetectorHerramientas(cfg, os.path.dirname(os.path.abspath(cfg.__file__)))
    dev = abrir_dispositivo(crear_pipeline_oak(con_hd=False), getattr(cfg, "AUTONOMO_CAM_MXID", None))
    K, dist = intrinsecos_color(dev, ANCHO_CAM, ALTO_CAM)
    q = dev.getOutputQueue("color", maxSize=1, blocking=False)
    qd = dev.getOutputQueue("depth", maxSize=1, blocking=False)
    prof = None
    print("Q o Esc para salir.", "Con calibración: posición en el mundo." if det.T_mundo_cam is not None
          else "Sin calibración: posición respecto a la cámara.")
    t_imp = 0.0
    with dev:
        while True:
            p = q.tryGet()
            if p is None:
                time.sleep(0.003)
                continue
            img = p.getCvFrame()
            pd = qd.tryGet()
            prof = pd.getFrame() if pd is not None else prof
            det.paso(img, K, dist, prof_mm=prof)
            det.dibujar(img, K, dist)
            cv2.imshow("Herramientas", img)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            if time.monotonic() - t_imp > 1.0:
                t_imp = time.monotonic()
                for h in det.lista():
                    if h["tipo"] == "desconocido":
                        print(f"   {h['nombre']}: no está en AUTONOMO_HERRAMIENTAS")
                        continue
                    T = np.array(h["T_mundo"] if h["T_mundo"] is not None else h["T_cam"])
                    print(f"   {h['nombre']:<18} {'estable' if h['estable'] else 'buscando':<8} "
                          f"xyz=({T[0, 3]:+.3f}, {T[1, 3]:+.3f}, {T[2, 3]:+.3f}) m  dist {h['distancia_m']:.2f} m  "
                          f"temblor {h['temblor_mm']} mm  err {h['error_px']} px"
                          + ("  (distancia del estéreo)" if h.get("estereo") else "  (sin estéreo)"))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    import config as _cfg
    ap = argparse.ArgumentParser(description="Herramientas con marcadores ArUco")
    ap.add_argument("--generar", action="store_true", help="PNG de los marcadores del catálogo")
    ap.add_argument("--probar", action="store_true", help="detección en vivo con la cámara de cabeza")
    a = ap.parse_args()
    if a.generar:
        generar_marcadores(_cfg, os.path.join(os.path.dirname(os.path.abspath(_cfg.__file__)), "marcadores"))
    elif a.probar:
        probar(_cfg)
    else:
        ap.print_help()
