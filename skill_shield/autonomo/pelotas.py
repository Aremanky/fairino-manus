"""
Detección de PELOTAS con la cámara de cabeza (11-nov), sin marcadores: por color y con la
profundidad de la OAK-D.

Por cada imagen (y su profundidad alineada):
  color (HSV) de cada pelota del catálogo AUTONOMO_PELOTAS → manchas → se quedan las REDONDAS
  (círculo mínimo bien relleno) → distancia con la profundidad estéreo del centro de la mancha →
  TAMAÑO REAL = diámetro en píxeles × distancia / focal: si no es el de esa pelota (±40 %), no
  es una pelota (un estuche rojo, la cinta amarilla…) → centro 3D = punto de la superficie + el
  radio hacia dentro → seguimiento en el tiempo (estable tras varias imágenes) → con la
  calibración cámara→robot, posición en el MUNDO (marco del MJCF, el del gemelo).

Salen en la misma lista que las herramientas (tipo "pelota"), así que aparecen en la web, en
el gemelo (esferas) y se pueden elegir como objetivo de la tarea. Su marco: centro de la pelota
y z apuntando hacia la cámara (por donde se acerca la mano).

Ajustar los colores (con el modo autónomo cerrado):
    python -m skill_shield.autonomo.pelotas --calibrar
      Clic en una pelota → imprime su color (HSV), un rango para config.py y su tamaño medido.
      La ventana «máscara» enseña lo que se está detectando con el catálogo actual.
"""
import math
import os
import time

import cv2
import numpy as np

from skill_shield.autonomo.autocalib import ConCorreccion

ID_BASE = 100              # ids de las pelotas en la lista (las herramientas ArUco van del 20 al 49)
DIAM_MIN, DIAM_MAX = 0.03, 0.20
# (18-nov) La mediana de la profundidad dentro de 0,6·r cae, de media, a ρ ≈ 0,6/√2·r del centro del disco, donde
# la superficie de la esfera está r·(1 − √(1 − 0,18)) ≈ 0,095·r más lejos que su punto más cercano: sin
# corregirlo, el centro salía ~3 mm más lejos (pelota de 7 cm)
SESGO_MEDIANA = 1.0 - math.sqrt(1.0 - 0.18)


def catalogo(cfg):
    g = cfg if isinstance(cfg, dict) else {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    out = []
    for i, p in enumerate(g.get("AUTONOMO_PELOTAS") or []):
        e = dict(diametro=None, tolerancia=0.4, color_gemelo=(1.0, 0.8, 0.1))
        e.update(p)
        e["colores"] = [(np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in e["colores"]]
        e["tipo_n"] = i
        out.append(e)
    return out


def mascara(hsv, colores):
    m = None
    for lo, hi in colores:
        x = cv2.inRange(hsv, lo, hi)
        m = x if m is None else cv2.bitwise_or(m, x)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def profundidad_en(prof_mm, u, v, r):
    """Mediana de la profundidad (m) dentro del círculo (u, v, r·0,6), o None."""
    if prof_mm is None:
        return None
    h, w = prof_mm.shape[:2]
    rr = max(2, int(r * 0.6))
    x0, x1 = max(int(u) - rr, 0), min(int(u) + rr + 1, w)
    y0, y1 = max(int(v) - rr, 0), min(int(v) + rr + 1, h)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dentro = (xx - u) ** 2 + (yy - v) ** 2 <= rr * rr
    z = prof_mm[y0:y1, x0:x1][dentro]
    z = z[z > 0]
    return float(np.median(z)) * 0.001 if z.size >= 5 else None


def relleno(m, c):
    """(18-nov) Parte del contorno c que es de verdad del color (píxeles de la máscara m dentro de c / su área).
    Una pelota sale casi entera (≈1); un ARO (el anillo rojo de las tapas de las articulaciones del FR5) tiene el
    mismo contorno exterior redondo pero está hueco por dentro (≈0,2-0,4)."""
    x, y, w, h = cv2.boundingRect(c)
    dentro = np.zeros((h, w), np.uint8)
    cv2.drawContours(dentro, [c - np.array([x, y])], -1, 255, -1)
    n = int(np.count_nonzero(dentro))
    return float(np.count_nonzero(m[y:y + h, x:x + w][dentro > 0])) / max(n, 1)


def ajustar_circulo(pts, r_ref, tol_px=1.5, iteraciones=80, min_frac=0.45, rng=None):
    """(18-nov) Círculo (u, v, r) que pasa por la mayor parte del contorno (RANSAC con 3 puntos + ajuste por
    mínimos cuadrados de los que caen en él), o None. Con la pelota iluminada de lado, la máscara de color
    pierde la parte en sombra: el borde de ese lado NO es la silueta (es la línea de sombra, más curvada o más
    recta) y minEnclosingCircle, que se apoya en los extremos, ponía el centro desplazado hacia la luz. Así
    solo cuenta el arco que de verdad es la silueta. r_ref: radio del círculo mínimo (para acotar)."""
    P = np.asarray(pts, float).reshape(-1, 2)
    if len(P) < 12:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    mejor, n_mejor = None, 0
    for _ in range(int(iteraciones)):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-9:
            continue
        aa, bb, cc = a @ a, b @ b, c @ c
        ux = (aa * (b[1] - c[1]) + bb * (c[1] - a[1]) + cc * (a[1] - b[1])) / d
        uy = (aa * (c[0] - b[0]) + bb * (a[0] - c[0]) + cc * (b[0] - a[0])) / d
        r = math.hypot(a[0] - ux, a[1] - uy)
        if not 0.8 * r_ref <= r <= 1.35 * r_ref:
            continue
        inl = np.abs(np.hypot(P[:, 0] - ux, P[:, 1] - uy) - r) < tol_px
        if inl.sum() > n_mejor:
            mejor, n_mejor = inl, int(inl.sum())
    if mejor is None or n_mejor < min_frac * len(P):
        return None
    Q = P[mejor]                                          # Kåsa: x² + y² + D x + E y + F = 0
    A = np.column_stack([Q[:, 0], Q[:, 1], np.ones(len(Q))])
    D, E, F = np.linalg.lstsq(A, -(Q[:, 0] ** 2 + Q[:, 1] ** 2), rcond=None)[0]
    u, v = -D / 2.0, -E / 2.0
    r2 = u * u + v * v - F
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    return (u, v, r) if 0.8 * r_ref <= r <= 1.35 * r_ref else None


def rayo(K, u, v, dist=None):
    """(18-nov) Rayo (x/z, y/z, 1) del píxel (u, v), quitando la distorsión de la lente si se da."""
    if dist is not None and np.any(np.asarray(dist, float)):
        p = cv2.undistortPoints(np.array([[[float(u), float(v)]]], np.float64), np.asarray(K, float),
                                np.asarray(dist, float).ravel())
        return np.array([p[0, 0, 0], p[0, 0, 1], 1.0])
    return np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])


def candidatas(img_bgr, prof_mm, K, cat, min_area=60, min_redondez=0.6, min_relleno=0.7, dist=None, ajuste=True):
    """[(entrada del catálogo, centro 3D en la cámara, diámetro m, (u, v, r_px))] de una imagen.
    (18-nov) dist: distorsión de la lente (se quita al calcular el rayo). ajuste: centro con ajustar_circulo
    (silueta) en vez del círculo mínimo."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    f = 0.5 * (K[0, 0] + K[1, 1])
    out = []
    for e in cat:
        m = mascara(hsv, e["colores"])
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            (u, v), r = cv2.minEnclosingCircle(c)
            if r < 3 or area / (math.pi * r * r) < min_redondez:
                continue
            if min_relleno and relleno(m, c) < min_relleno:
                continue                                        # (18-nov) hueco por dentro: un aro, no una pelota
            if ajuste:
                h, w = m.shape[:2]
                pc = c.reshape(-1, 2)
                borde = (pc[:, 0] <= 0) | (pc[:, 1] <= 0) | (pc[:, 0] >= w - 1) | (pc[:, 1] >= h - 1)
                ajustado = ajustar_circulo(pc[~borde], r)        # (lo cortado por el borde de la imagen, fuera)
                if ajustado is not None:
                    u, v, r = ajustado
            z = profundidad_en(prof_mm, u, v, r)
            if z is None:
                continue
            diam = 2.0 * r * z / f
            if e["diametro"]:
                if abs(diam - e["diametro"]) > e["tolerancia"] * e["diametro"]:
                    continue
                diam = float(e["diametro"])                     # el de verdad: más preciso
            elif not DIAM_MIN <= diam <= DIAM_MAX:
                continue
            ray = rayo(K, u, v, dist)
            sup = ray * z                                       # punto de la superficie (lo que mide el estéreo)
            centro = sup + sup / np.linalg.norm(sup) * diam / 2.0 * (1.0 - SESGO_MEDIANA)
            out.append((e, centro, diam, (u, v, r)))
    return out


def apoyo_bajo(prof_mm, K, T_mundo_cam, u, v, r_px, centro_cam, diam, rng=None,
               anillo=(1.6, 4.5), banda=0.06, tol=0.008, n_min=40, inclinacion_max_deg=15.0,
               mesa_m=None, mesa_tol=0.012):
    """(17-nov) Altura (z del mundo) de la superficie en la que está APOYADA la pelota: plano casi
    horizontal ajustado (RANSAC) a la profundidad de un anillo alrededor de la pelota en la imagen,
    a la altura de su parte de abajo (± banda). None si no hay bastante mesa a la vista (la tapa la
    mano, la pelota está en el aire…). Sirve igual para cualquier objeto apoyado (herramientas).
    mesa_m (17-nov): además, HASTA DÓNDE llega la mesa: los píxeles de una ventana de ese radio (m)
    alrededor de la pelota que caen en el plano → su envolvente en planta. Devuelve entonces
    (z, [[x, y], …] vértices de la envolvente o None)."""
    if prof_mm is None or T_mundo_cam is None:
        return None
    h, w = prof_mm.shape[:2]
    r0, r1 = anillo[0] * r_px, anillo[1] * r_px
    x0, x1 = max(int(u - r1), 0), min(int(u + r1) + 1, w)
    y0, y1 = max(int(v - r1), 0), min(int(v + r1) + 1, h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - u) ** 2 + (yy - v) ** 2
    z = prof_mm[y0:y1, x0:x1].astype(np.float32) * 0.001
    ok = (d2 >= r0 * r0) & (d2 <= r1 * r1) & (z > 0)
    if ok.sum() < n_min:
        return None
    zz = z[ok]
    P = np.column_stack([(xx[ok] - K[0, 2]) / K[0, 0] * zz, (yy[ok] - K[1, 2]) / K[1, 1] * zz, zz])
    P = P @ T_mundo_cam[:3, :3].T + T_mundo_cam[:3, 3]
    c = T_mundo_cam[:3, :3] @ np.asarray(centro_cam, float) + T_mundo_cam[:3, 3]
    zb = c[2] - diam / 2.0
    P = P[np.abs(P[:, 2] - zb) < banda]
    if len(P) < n_min:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    if len(P) > 3000:
        P = P[rng.choice(len(P), 3000, replace=False)]
    cos_max = math.cos(math.radians(inclinacion_max_deg))
    mejor = None
    for _ in range(60):
        A, B, C = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(B - A, C - A)
        nn = np.linalg.norm(n)
        if nn < 1e-9 or abs(n[2]) / nn < cos_max:
            continue
        n /= nn
        inl = np.abs((P - A) @ n) < tol
        if mejor is None or inl.sum() > mejor.sum():
            mejor = inl
    if mejor is None or mejor.sum() < n_min:
        return None
    Q = P[mejor]                                          # ajuste fino: z = a·x + b·y + c
    M = np.column_stack([Q[:, 0], Q[:, 1], np.ones(len(Q))])
    a, b, c0 = np.linalg.lstsq(M, Q[:, 2], rcond=None)[0]
    z_apoyo = float(a * c[0] + b * c[1] + c0)
    if mesa_m is None:
        return z_apoyo
    return z_apoyo, _mesa_en_plano(prof_mm, K, T_mundo_cam, u, v, centro_cam, (a, b, c0), mesa_m, mesa_tol)


def _mesa_en_plano(prof_mm, K, T_mundo_cam, u, v, centro_cam, plano, radio_m, tol, paso=3, n_min=60):
    """Envolvente en planta (vértices [[x, y], …]) de lo que cae en el plano z = a·x + b·y + c dentro
    de una ventana de radio_m alrededor de la pelota, o None."""
    h, w = prof_mm.shape[:2]
    zc = max(float(np.asarray(centro_cam, float)[2]), 0.2)
    r_px = radio_m * 0.5 * (K[0, 0] + K[1, 1]) / zc
    x0, x1 = max(int(u - r_px), 0), min(int(u + r_px) + 1, w)
    y0, y1 = max(int(v - r_px), 0), min(int(v + r_px) + 1, h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    yy, xx = np.mgrid[y0:y1:paso, x0:x1:paso]
    z = prof_mm[y0:y1:paso, x0:x1:paso].astype(np.float32) * 0.001
    ok = z > 0
    if ok.sum() < n_min:
        return None
    zz = z[ok]
    P = np.column_stack([(xx[ok] - K[0, 2]) / K[0, 0] * zz, (yy[ok] - K[1, 2]) / K[1, 1] * zz, zz])
    P = P @ T_mundo_cam[:3, :3].T + T_mundo_cam[:3, 3]
    a, b, c0 = plano
    c = T_mundo_cam[:3, :3] @ np.asarray(centro_cam, float) + T_mundo_cam[:3, 3]
    en = (np.abs(P[:, 2] - (a * P[:, 0] + b * P[:, 1] + c0)) < tol) & \
         (np.linalg.norm(P[:, :2] - c[:2], axis=1) < radio_m)
    if en.sum() < n_min:
        return None
    try:
        from scipy.spatial import ConvexHull
        Q = P[en, :2]
        return Q[ConvexHull(Q).vertices].round(4).tolist()
    except Exception:
        return None


def zona_propia(c, r, T_mundo_cam=None, res=0.02, k_ruido=0.02, k_trunc=0.02, banda=0.35, margen=0.0):
    """(18-nov, tarde) Dónde pueden caer los cubos de la PROPIA pelota (centro c y radio r, en el mundo): una
    cápsula a lo largo del rayo de la cámara → (a, b, radio). Es lo que se quita del mapa y del planificador.
      · radio = r + medio cubo en diagonal (√3/2 · res: un cubo que toca la pelota tiene su centro hasta ahí fuera
        de ella) + margen. De LADO no se quita nada más: lo que haya al lado de la pelota sigue contando;
      · a lo largo del rayo, el ruido de la profundidad a esa distancia (k_ruido · d²) hacia los dos lados: el
        centro estimado y la superficie medida se desvían a lo largo del rayo, no de lado;
      · hacia la cámara, además, lo que la TSDF engorda cada superficie (banda · truncado: ~2 cm a 1 m).
    Antes era una esfera o una caja de r + 1 cm alrededor del centro estimado: con el centro 2 cm desviado por el
    estéreo (a ~1 m es lo normal), la cara de la pelota que mira a la cámara quedaba fuera y salía como obstáculo
    justo donde tiene que bajar la mano. Sin cámara: una esfera (a = b = c) con el ruido a 1 m."""
    c = np.asarray(c, float).reshape(3)
    radio = float(r) + 0.5 * math.sqrt(3.0) * float(res) + float(margen)
    if T_mundo_cam is None:
        return c.copy(), c.copy(), radio + float(k_ruido)
    hacia = np.asarray(T_mundo_cam, float)[:3, 3] - c
    d = float(np.linalg.norm(hacia))
    if d < 1e-6:
        return c.copy(), c.copy(), radio
    u = hacia / d
    ruido = float(k_ruido) * d * d
    delante = ruido + float(banda) * (2.0 * float(res) + float(k_trunc) * d * d)
    return c + u * delante, c - u * ruido, radio


def esferas_capsula(a, b, radio, paso=0.01):
    """La cápsula a-b como esferas [(centro, radio)] seguidas (cada paso m): el formato de «excluir» del
    planificador. Con paso 1 cm y radio ≥ 4 cm, entre dos esferas falta menos de 0,3 mm de la cápsula."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(int(math.ceil(float(np.linalg.norm(b - a)) / float(paso))), 0) + 1
    return [(a + (b - a) * t, float(radio)) for t in (np.linspace(0.0, 1.0, n) if n > 1 else [0.0])]


def params_zona(g):
    """Parámetros de zona_propia sacados de la configuración (los mismos para el mapa y el planificador)."""
    from skill_shield.autonomo.mapa import MapaOcupacion
    tsdf = str(g.get("AUTONOMO_MAPA_METODO", "tsdf")) == "tsdf"
    return dict(res=float(g.get("AUTONOMO_MAPA_RES", 0.02)), k_ruido=float(g.get("AUTONOMO_TAREA_RUIDO_PROF_K", 0.02)),
                k_trunc=float(g.get("AUTONOMO_MAPA_TSDF_K", 0.02)), banda=MapaOcupacion.D_OCUPADO if tsdf else 0.0)


class _Pista:
    def __init__(self, id_, e, p, diam, ahora):
        self.id, self.e, self.p, self.diam = id_, e, p.copy(), diam
        self.n, self.t = 1, ahora
        self.hist = [p.copy()]
        self.apoyos = []                     # (17-nov) alturas de la mesa medidas (las últimas)
        self.mesa = None                     # (17-nov) envolvente en planta de la mesa (la última)

    def apoyo(self):
        return float(np.median(self.apoyos)) if len(self.apoyos) >= 3 else None

    def nueva(self, p, diam, ahora, alfa=0.4):
        self.p = (1 - alfa) * self.p + alfa * p
        self.diam = (1 - alfa) * self.diam + alfa * diam
        self.n += 1
        self.t = ahora
        self.hist = (self.hist + [p.copy()])[-30:]

    def temblor_mm(self):
        return None if len(self.hist) < 5 else float(np.linalg.norm(np.std(np.array(self.hist), 0)) * 1000)


class DetectorPelotas(ConCorreccion):
    """paso(img, K, prof_mm) por imagen; lista() para publicar (mismo formato que las herramientas)."""

    def __init__(self, cfg, raiz, n_estable=5, olvido_s=2.0, salto_m=0.08):
        g = cfg if isinstance(cfg, dict) else {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
        self.cat = catalogo(g)
        ruta = g.get("CALIB_ROBOT_ARCHIVO", "camara_robot.json")
        self.ruta_calib = ruta if os.path.isabs(ruta) else os.path.join(raiz, ruta)
        self.n_estable = int(g.get("AUTONOMO_PELOTAS_N_ESTABLE", n_estable))
        self.olvido_s = float(g.get("AUTONOMO_PELOTAS_OLVIDO_S", olvido_s))
        self.salto_m = float(salto_m)
        self.medir_apoyo = bool(g.get("AUTONOMO_PELOTAS_MEDIR_APOYO", True))
        self.mesa_m = float(g.get("AUTONOMO_PELOTAS_MESA_RADIO", 0.35))   # m alrededor donde se mide la mesa
        self.min_relleno = float(g.get("AUTONOMO_PELOTAS_RELLENO_MIN", 0.7))   # (18-nov) aros fuera
        self.ajuste = bool(g.get("AUTONOMO_PELOTAS_AJUSTE_SILUETA", True))      # (18-nov) centro por la silueta
        # (18-nov) lo que se quita del mapa alrededor de cada pelota: solo ella (zona_propia, + este margen EXTRA);
        # lo de alrededor cuenta
        self.margen_mapa = float(g.get("AUTONOMO_MAPA_MARGEN_PELOTA", 0.0))
        self.zona = params_zona(g)
        self.halo = dict(factor=1.1, extra_px=2, delante=0.02, detras=0.02)
        self.halo.update(g.get("AUTONOMO_MAPA_HALO_PELOTA") or {})
        self.pistas = {}
        self._sig = ID_BASE
        self.T_mundo_cam, self._mtime, self._t_calib = None, None, 0.0
        self.ultimas = []

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

    def paso(self, img_bgr, K, prof_mm, ahora=None, dist=None):
        ahora = time.time() if ahora is None else ahora
        self._recargar(time.monotonic())
        self.ultimas = candidatas(img_bgr, prof_mm, K, self.cat, min_relleno=self.min_relleno, dist=dist,
                                  ajuste=self.ajuste)
        usadas = set()
        for e, p, diam, uvr in sorted(self.ultimas, key=lambda c: c[1][2]):
            libres = [s for i, s in self.pistas.items() if s.e is e and i not in usadas
                      and ahora - s.t <= self.olvido_s]
            s = min(libres, key=lambda s: np.linalg.norm(s.p - p), default=None)
            if s is not None and np.linalg.norm(s.p - p) <= self.salto_m:
                s.nueva(p, diam, ahora)
            else:
                s = _Pista(self._sig, e, p, diam, ahora)
                self.pistas[self._sig] = s
                self._sig += 1
            usadas.add(s.id)
            if self.medir_apoyo:
                try:
                    r_ = apoyo_bajo(prof_mm, K, self.T_mundo_cam, *uvr, p, diam, mesa_m=self.mesa_m)
                except Exception:
                    r_ = None
                if r_ is not None:
                    z, poli = r_
                    s.apoyos = (s.apoyos + [z])[-15:]
                    if poli is not None:
                        s.mesa = poli
        for i in [i for i, s in self.pistas.items() if ahora - s.t > 3 * self.olvido_s]:
            del self.pistas[i]

    def _T(self, p):
        """Marco de la pelota: en su centro, z hacia la cámara (por donde se acerca la mano)."""
        z = -p / max(np.linalg.norm(p), 1e-9)
        u = np.cross([0.0, -1.0, 0.0], z)            # x «a la derecha» en la imagen
        if np.linalg.norm(u) < 1e-6:
            u = np.array([1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([u, np.cross(z, u), z])
        T[:3, 3] = p
        return T

    def lista(self, ahora=None):
        ahora = time.time() if ahora is None else ahora
        vivas = sorted((s for s in self.pistas.values() if ahora - s.t <= self.olvido_s), key=lambda s: s.id)
        cuenta = {}
        for s in vivas:
            cuenta[s.e["nombre"]] = cuenta.get(s.e["nombre"], 0) + 1
        vistos, out = {}, []
        for s in vivas:
            nombre = s.e["nombre"]
            if cuenta[nombre] > 1:
                vistos[nombre] = vistos.get(nombre, 0) + 1
                nombre = f"{nombre} {vistos[nombre]}"
            T = self._T(s.p)
            Tm = self.T_mundo_cam @ T if self.T_mundo_cam is not None else None
            r = s.diam / 2.0
            out.append(dict(id=s.id, nombre=nombre, tipo="pelota", estable=s.n >= self.n_estable,
                            edad_s=round(ahora - s.t, 2), n=s.n, estereo=True,
                            distancia_m=round(float(np.linalg.norm(s.p)), 3), diametro_m=round(s.diam, 3),
                            temblor_mm=None if s.temblor_mm() is None else round(s.temblor_mm(), 1),
                            T_cam=np.round(T, 5).tolist(), T_mundo=None if Tm is None else np.round(Tm, 5).tolist(),
                            caja=[r, r, r], color=list(s.e["color_gemelo"]),
                            apoyo_z=None if s.apoyo() is None or self.T_mundo_cam is None else round(s.apoyo(), 4),
                            apoyo_poligono=s.mesa if s.apoyo() is not None and self.T_mundo_cam is not None
                            else None))
        return out

    def _zona_mundo(self, p, diam):
        """Lo que se quita del mapa por una pelota (centro p en la cámara): su zona_propia (cápsula a lo largo
        del rayo) desde medio cubo por encima de su parte de abajo: la MESA de debajo no se quita (17-nov: antes
        quedaba un agujero en la mesa justo donde hay que cogerla). dict(a, b, r, z_min) para MapaOcupacion."""
        T = self.T_mundo_cam
        c = T[:3, :3] @ np.asarray(p, float) + T[:3, 3]
        r = float(diam) / 2.0
        a, b, radio = zona_propia(c, r, T, margen=self.margen_mapa, **self.zona)
        return dict(a=a, b=b, r=radio, z_min=float(c[2] - r + 0.5 * self.zona["res"]))

    def caja_mundo(self, id_):
        """La zona de la pelota id_ (la elegida) que se quita del mapa, o None (ver _zona_mundo)."""
        s = self.pistas.get(id_) if id_ is not None else None
        if s is None or self.T_mundo_cam is None or time.time() - s.t > self.olvido_s:
            return None
        return self._zona_mundo(s.p, s.diam)

    def pixeles_halo(self, factor=None, extra_px=None, delante=None, detras=None):
        """(12-nov) Píxeles de la profundidad que NO van al mapa: alrededor de cada pelota vista en
        la última imagen (su círculo × factor) y desde un poco delante de ella hasta bastante
        detrás. En el borde de una pelota el estéreo mezcla la pelota con el fondo y salen puntos
        sueltos detrás y alrededor de ella: en el mapa la «envolvían» de cubos que no existen.
        [(u, v, radio px, z mínima m, z máxima m)].
        (18-nov) Por defecto ya solo la pelota: su círculo × 1,1 y de 2 cm delante de su cara a 2 cm detrás de su
        fondo (AUTONOMO_MAPA_HALO_PELOTA); antes ×1,4 y hasta 30 cm detrás, y se comía lo que hubiera junto a ella.
        (18-nov, tarde) Más el ruido de la profundidad a esa distancia hacia los dos lados (la cara estimada se
        desvía eso): solo cambia qué píxeles DENTRO de su círculo se quitan."""
        h = self.halo
        factor = h["factor"] if factor is None else factor
        extra_px = h["extra_px"] if extra_px is None else extra_px
        delante = h["delante"] if delante is None else delante
        detras = h["detras"] if detras is None else detras
        out = []
        for e, p, diam, (u, v, r) in self.ultimas:
            d = float(np.linalg.norm(p))
            z_sup = float(p[2]) * max(d - diam / 2.0, 0.05) / max(d, 1e-6)       # profundidad (Z) de su cara
            # (18-nov, tarde) + el ruido de la profundidad a esa distancia: la cara estimada se desvía tanto como eso
            # y, con solo ±2 cm, a ~1 m los píxeles de la propia pelota se colaban en el mapa
            ruido = self.zona["k_ruido"] * d * d
            # (17-nov) 7º: altura (mundo) por debajo de la cual NO se quita nada: la mesa en la que está apoyada
            # (antes el halo se llevaba también la mesa de detrás de la pelota)
            z_min = None
            if self.T_mundo_cam is not None:
                c = self.T_mundo_cam[:3, :3] @ p + self.T_mundo_cam[:3, 3]
                z_min = float(c[2] - diam / 2.0 + 0.01)
            out.append((float(u), float(v), float(r) * factor + extra_px, z_sup - delante - ruido,
                        z_sup + diam + detras + ruido,
                        float(v) + 0.5 * float(r), z_min))       # por debajo de esta fila, la mesa: no se toca
        return out

    def cajas_mundo(self):
        """La zona (_zona_mundo) de cada pelota vista en la última imagen: lo que se hubiera marcado antes en el
        mapa ahí se borra. Las pelotas no son obstáculo (la mesa de debajo, sí)."""
        if self.T_mundo_cam is None:
            return []
        return [self._zona_mundo(p, diam) for e, p, diam, _ in self.ultimas]

    def dibujar(self, img):
        for e, p, diam, (u, v, r) in self.ultimas:
            c = tuple(int(255 * x) for x in reversed(e["color_gemelo"]))
            cv2.circle(img, (int(u), int(v)), int(r), c, 2)
            txt = f"{e['nombre']} {diam * 100:.0f} cm"
            for grosor, col in ((3, (0, 0, 0)), (1, c)):
                cv2.putText(img, txt, (int(u - r), int(v - r) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col,
                            grosor, cv2.LINE_AA)


# ═════════════ calibrar los colores ═════════════

def rango_sugerido(hsv_patch):
    """Rango HSV para config a partir de los píxeles de una pelota (mediana ± margen)."""
    h, s, v = [np.median(hsv_patch[..., i]) for i in range(3)]
    dh = 8
    lo_s, lo_v = max(int(s - 80), 50), max(int(v - 90), 40)
    if h - dh < 0 or h + dh > 179:                 # el rojo da la vuelta (0 = 180)
        return [((0, lo_s, lo_v), (int((h + dh) % 180), 255, 255)), ((int((h - dh) % 180), lo_s, lo_v), (179, 255, 255))]
    return [((int(h - dh), lo_s, lo_v), (int(h + dh), 255, 255))]


def calibrar(cfg):
    from config import ANCHO_CAM, ALTO_CAM
    from skill_shield.vision.camara_oak import crear_pipeline_oak
    from skill_shield.vision.multicamara import abrir_dispositivo, intrinsecos_color
    det = DetectorPelotas(cfg, os.path.dirname(os.path.abspath(cfg.__file__)))
    dev = abrir_dispositivo(crear_pipeline_oak(con_hd=False), getattr(cfg, "AUTONOMO_CAM_MXID", None))
    K, _ = intrinsecos_color(dev, ANCHO_CAM, ALTO_CAM)
    f = 0.5 * (K[0, 0] + K[1, 1])
    estado = dict(img=None, prof=None)

    def clic(ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN or estado["img"] is None:
            return
        hsv = cv2.cvtColor(cv2.GaussianBlur(estado["img"], (5, 5), 0), cv2.COLOR_BGR2HSV)
        parche = hsv[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5]
        rango = rango_sugerido(parche)
        m = mascara(hsv, [(np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in rango])
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(m)
        diam_txt = "?"
        if etiquetas[y, x] > 0:
            ys, xs = np.nonzero(etiquetas == etiquetas[y, x])
            (u, v), r = cv2.minEnclosingCircle(np.column_stack([xs, ys]).astype(np.float32))
            z = profundidad_en(estado["prof"], u, v, r)
            if z:
                diam_txt = f"{2 * r * z / f:.3f}"
        h, s_, v_ = [int(np.median(parche[..., i])) for i in range(3)]
        print(f"\n🎯 Color en ({x}, {y}): H={h} S={s_} V={v_} · diámetro medido ≈ {diam_txt} m")
        print(f"   dict(nombre=\"Pelota …\", colores={[(list(lo), list(hi)) for lo, hi in rango]}, "
              f"diametro={diam_txt if diam_txt != '?' else 'None'}, color_gemelo=(1.0, 0.8, 0.1)),")

    cv2.namedWindow("Pelotas")
    cv2.setMouseCallback("Pelotas", clic)
    print("Clic en una pelota para ver su color y tamaño. Q o Esc para salir.")
    q = dev.getOutputQueue("color", maxSize=1, blocking=False)
    qd = dev.getOutputQueue("depth", maxSize=1, blocking=False)
    with dev:
        while True:
            p = q.tryGet()
            if p is None:
                time.sleep(0.003)
                continue
            img = p.getCvFrame()
            pd = qd.tryGet()
            estado["prof"] = pd.getFrame() if pd is not None else estado["prof"]
            estado["img"] = img.copy()
            det.paso(img, K, estado["prof"])
            vis = img.copy()
            det.dibujar(vis)
            hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
            todo = np.zeros(img.shape[:2], np.uint8)
            for e in det.cat:
                todo = cv2.bitwise_or(todo, mascara(hsv, e["colores"]))
            cv2.imshow("Pelotas", vis)
            cv2.imshow("mascara", todo)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    import config as _cfg
    ap = argparse.ArgumentParser(description="Pelotas por color + profundidad")
    ap.add_argument("--calibrar", action="store_true", help="ventana: clic en una pelota → su color y tamaño")
    a = ap.parse_args()
    if a.calibrar:
        calibrar(_cfg)
    else:
        ap.print_help()
