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


def candidatas(img_bgr, prof_mm, K, cat, min_area=60, min_redondez=0.6):
    """[(entrada del catálogo, centro 3D en la cámara, diámetro m, (u, v, r_px))] de una imagen."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    f = 0.5 * (K[0, 0] + K[1, 1])
    out = []
    for e in cat:
        cs, _ = cv2.findContours(mascara(hsv, e["colores"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            (u, v), r = cv2.minEnclosingCircle(c)
            if r < 3 or area / (math.pi * r * r) < min_redondez:
                continue
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
            ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
            sup = ray * z                                       # punto de la superficie (lo que mide el estéreo)
            centro = sup + sup / np.linalg.norm(sup) * diam / 2.0
            out.append((e, centro, diam, (u, v, r)))
    return out


class _Pista:
    def __init__(self, id_, e, p, diam, ahora):
        self.id, self.e, self.p, self.diam = id_, e, p.copy(), diam
        self.n, self.t = 1, ahora
        self.hist = [p.copy()]

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

    def paso(self, img_bgr, K, prof_mm, ahora=None):
        ahora = time.time() if ahora is None else ahora
        self._recargar(time.monotonic())
        self.ultimas = candidatas(img_bgr, prof_mm, K, self.cat)
        usadas = set()
        for e, p, diam, _ in sorted(self.ultimas, key=lambda c: c[1][2]):
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
                            caja=[r, r, r], color=list(s.e["color_gemelo"])))
        return out

    def caja_mundo(self, id_):
        s = self.pistas.get(id_) if id_ is not None else None
        if s is None or self.T_mundo_cam is None or time.time() - s.t > self.olvido_s:
            return None
        c = self.T_mundo_cam[:3, :3] @ s.p + self.T_mundo_cam[:3, 3]
        return c.tolist(), [s.diam / 2.0 + 0.02] * 3

    def pixeles_halo(self, factor=1.4, extra_px=3, delante=0.05, detras=0.30):
        """(12-nov) Píxeles de la profundidad que NO van al mapa: alrededor de cada pelota vista en
        la última imagen (su círculo × factor) y desde un poco delante de ella hasta bastante
        detrás. En el borde de una pelota el estéreo mezcla la pelota con el fondo y salen puntos
        sueltos detrás y alrededor de ella: en el mapa la «envolvían» de cubos que no existen.
        [(u, v, radio px, z mínima m, z máxima m)]."""
        out = []
        for e, p, diam, (u, v, r) in self.ultimas:
            d = float(np.linalg.norm(p))
            z_sup = float(p[2]) * max(d - diam / 2.0, 0.05) / max(d, 1e-6)       # profundidad (Z) de su cara
            out.append((float(u), float(v), float(r) * factor + extra_px, z_sup - delante, z_sup + diam + detras,
                        float(v) + 0.5 * float(r)))       # por debajo de esta fila, la mesa: no se toca
        return out

    def cajas_mundo(self, margen=0.04):
        """Caja (centro, semi) en el mundo alrededor de cada pelota vista: lo que hubiera marcado
        antes en el mapa ahí (el halo) se borra."""
        if self.T_mundo_cam is None:
            return []
        out = []
        for e, p, diam, _ in self.ultimas:
            c = self.T_mundo_cam[:3, :3] @ p + self.T_mundo_cam[:3, 3]
            r = diam / 2.0
            # de 1 cm por encima de su parte de abajo hacia arriba: lo que la sujeta (la mesa) se queda
            z0, z1 = c[2] - r + 0.01, c[2] + r + margen
            out.append((np.array([c[0], c[1], (z0 + z1) / 2]), np.array([r + margen, r + margen, (z1 - z0) / 2])))
        return out

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
