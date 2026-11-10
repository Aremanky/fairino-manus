"""
Choque de la OrcaHand con el PROPIO brazo (antebrazo y giro del J4), con las piezas reales.

El modelo de cápsulas de la teleoperación (ik_teleop.ModeloColision) NO comprueba la mano contra
el antebrazo: con cápsulas, mano y antebrazo "chocan" casi siempre porque están pegados por la
muñeca. Pero si la muñeca se dobla mucho (J4/J5), la mano vuelve hacia atrás y golpea el giro
del J4 — y eso rompe una pieza de la mano.

Aquí se usan las MISMAS mallas que dibuja el gemelo de MuJoCo (fairino_mjcf.xml + la OrcaHand
que incluye), muestreadas como puntos cada ~6 mm: la mano (+ el tablero, si se le pasa) en el
marco de la brida (wrist3) y el brazo en el marco de cada eslabón. Para una postura q se mueven
los puntos de la mano al marco de cada eslabón y se busca la distancia mínima (árbol k-d).
No hace falta MuJoCo (solo numpy + scipy), así que vale en cualquier proceso.

La distancia solo depende de J4, J5 y J6 (lo que hay entre el antebrazo y la mano), no de dónde
esté el robot.

Uso desde línea de comandos (comprobar posturas guardadas, sin mover nada):
    python -m skill_shield.autonomo.choque_mano                    # calib_posturas.json
    python -m skill_shield.autonomo.choque_mano --mapa             # tabla J4×J5 de distancias
"""
import functools
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

CUERPOS_BRAZO = ("upperarm_Link", "forearm_Link", "wrist1_Link")   # wrist1 = el giro del J4
MARCO_CADENA = {"upperarm_Link": 1, "forearm_Link": 2, "wrist1_Link": 3, "wrist3_Link": 5}
CUERPO_MANO = "mano_mount"


# ── geometría básica ──
def _quat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rot(eje, a):
    e = np.asarray(eje, float) / np.linalg.norm(eje)
    K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _T_local(el, en_grados=False):
    """pos + quat / euler / axisangle de un <body> o <geom> de MJCF."""
    T = np.eye(4)
    T[:3, 3] = [float(v) for v in el.get("pos", "0 0 0").split()]
    k = math.pi / 180.0 if en_grados else 1.0
    if el.get("quat"):
        T[:3, :3] = _quat([float(v) for v in el.get("quat").split()])
    elif el.get("euler"):                       # convenio por defecto de MuJoCo: "xyz" (ejes que giran)
        R = np.eye(3)
        for eje, a in zip(np.eye(3), [float(v) * k for v in el.get("euler").split()]):
            R = R @ _rot(eje, a)
        T[:3, :3] = R
    elif el.get("axisangle"):
        x, y, z, a = [float(v) for v in el.get("axisangle").split()]
        T[:3, :3] = _rot([x, y, z], a * k)
    return T


def leer_stl(ruta):
    """Triángulos (n, 3, 3) de un STL binario o ASCII."""
    with open(ruta, "rb") as f:
        b = f.read()
    if len(b) >= 84:
        n = int(np.frombuffer(b[80:84], "<u4")[0])
        if 84 + 50 * n == len(b):
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(b, dt, count=n, offset=84)["v"].astype(float)
    vs = [list(map(float, l.split()[1:4])) for l in b.decode("latin-1").splitlines()
          if l.strip().startswith("vertex")]
    return np.asarray(vs, float).reshape(-1, 3, 3)


def muestrear(tri, paso, rng):
    """Puntos sobre la superficie (vértices + muestreo por área) reducidos a uno por celda de 'paso'."""
    if len(tri) == 0:
        return np.zeros((0, 3))
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = int(min(max(4 * a.sum() / paso ** 2, 1), 400000))
    idx = rng.choice(len(tri), n, p=a / a.sum()) if a.sum() > 0 else np.zeros(0, int)
    u, v = rng.random((2, n))
    fuera = u + v > 1
    u[fuera], v[fuera] = 1 - u[fuera], 1 - v[fuera]
    t = tri[idx]
    p = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    return reducir(np.vstack([tri.reshape(-1, 3), p]), paso)


def reducir(P, paso):
    if len(P) == 0:
        return P
    _, i = np.unique(np.floor(P / paso).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(i)]


def puntos_tablero(T_brida_tablero, cuadros=(7, 5), lado=0.020, borde=0.012, paso=0.008):
    """Puntos de la placa del tablero (con un borde alrededor) en el marco de la brida."""
    nx, ny = cuadros
    xs = np.arange(-borde, nx * lado + borde + 1e-9, paso)
    ys = np.arange(-borde, ny * lado + borde + 1e-9, paso)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size), np.ones(X.size)])
    return (np.asarray(T_brida_tablero, float) @ P.T).T[:, :3]


# ── lectura del MJCF ──
def _expandir_includes(el, base):
    for i, h in enumerate(list(el)):
        if h.tag == "include":
            sub = ET.parse(os.path.join(base, h.get("file"))).getroot()
            el.remove(h)
            for k, x in enumerate(list(sub)):
                el.insert(i + k, x)
                _expandir_includes(x, os.path.dirname(os.path.join(base, h.get("file"))))
        else:
            _expandir_includes(h, base)


@functools.lru_cache(maxsize=4)
def nubes_mjcf(ruta_xml, paso=0.006, semilla=0):
    """{cuerpo: puntos (n,3) en su marco} para los eslabones de CUERPOS_BRAZO y la mano entera
    (subárbol de mano_mount, dedos en q = 0) expresada en el marco de wrist3_Link."""
    base = os.path.dirname(os.path.abspath(ruta_xml))
    root = ET.parse(ruta_xml).getroot()
    _expandir_includes(root, base)
    comp = root.find("compiler")
    grados = comp is not None and comp.get("angle", "degree") == "degree"
    meshdir = os.path.join(base, comp.get("meshdir", "")) if comp is not None else base
    mallas = {}
    for m in root.iter("mesh"):
        if m.get("file"):
            esc = np.array([float(v) for v in m.get("scale", "1 1 1").split()])
            mallas[m.get("name") or os.path.splitext(os.path.basename(m.get("file")))[0]] = \
                (os.path.join(meshdir, m.get("file")), esc)
    cuerpos = {b.get("name"): b for b in root.iter("body")}
    rng = np.random.default_rng(semilla)
    cache = {}

    def nube_body(b, T, solo_este=False):
        out = []
        for g in b.findall("geom"):
            nm = g.get("mesh")
            if not nm or nm not in mallas:
                continue
            if nm not in cache:
                ruta, esc = mallas[nm]
                cache[nm] = muestrear(leer_stl(ruta) * esc, paso, rng) if os.path.exists(ruta) else np.zeros((0, 3))
            P = cache[nm]
            Tg = T @ _T_local(g, grados)
            out.append(P @ Tg[:3, :3].T + Tg[:3, 3])
        if not solo_este:
            for h in b.findall("body"):
                out.extend(nube_body(h, T @ _T_local(h, grados)))
        return out

    nubes = {}
    for nombre in CUERPOS_BRAZO:
        if nombre in cuerpos:
            P = nube_body(cuerpos[nombre], np.eye(4), solo_este=True)
            nubes[nombre] = reducir(np.vstack(P), paso) if P else np.zeros((0, 3))
    if CUERPO_MANO in cuerpos:
        P = nube_body(cuerpos[CUERPO_MANO], _T_local(cuerpos[CUERPO_MANO], grados))
        nubes["mano"] = reducir(np.vstack(P), paso) if P else np.zeros((0, 3))
    return nubes


class ChoqueManoBrazo:
    """Hueco entre la mano y el antebrazo / giro del J4 (y, aparte, entre el tablero y el brazo).
      · mano: SIEMPRE ≥ holgura (es lo que se rompe).
      · tablero (si se da T_brida_tablero): ≥ holgura_tablero; como va pegado a la mano puede
        estar ya más cerca en la postura de inicio → fijar_inicio(q0) rebaja su mínimo a lo que
        haya en q0 (no puede acercarse MÁS).
    ok(q) / camino_ok(lista de q) / distancia(q)."""

    def __init__(self, cadena, ruta_xml, holgura=0.03, paso=0.006, T_brida_tablero=None, tablero=None,
                 holgura_tablero=0.015):
        from scipy.spatial import cKDTree
        self.c = cadena
        self.holgura = float(holgura)
        self.holgura_tablero = float(holgura_tablero)
        nubes = nubes_mjcf(ruta_xml, paso)
        if "mano" not in nubes or len(nubes["mano"]) == 0:
            raise RuntimeError(f"{os.path.basename(ruta_xml)}: no encuentro las mallas de la mano")
        self.mano = nubes["mano"].copy()
        self.tablero = None
        if T_brida_tablero is not None:
            t = tablero or {}
            self.tablero = puntos_tablero(T_brida_tablero, tuple(t.get("cuadros", (7, 5))), float(t.get("lado", 0.020)))
        self.arboles = {k: (MARCO_CADENA[k], cKDTree(v)) for k, v in nubes.items()
                        if k in MARCO_CADENA and len(v)}
        if not self.arboles:
            raise RuntimeError(f"{os.path.basename(ruta_xml)}: no encuentro las mallas del brazo")
        self._grueso = {"mano": reducir(self.mano, 0.02)}     # para descartar rápido
        if self.tablero is not None:
            self._grueso["tablero"] = reducir(self.tablero, 0.02)

    def _dist(self, q, cota, que="mano"):
        P = self.mano if que == "mano" else self.tablero
        if P is None:
            return math.inf, ""
        m = self.c.fk_marcos(np.asarray(q, float))
        Tb = m[MARCO_CADENA["wrist3_Link"]]
        mejor, donde = math.inf, ""
        for nombre, (i, arbol) in self.arboles.items():
            T = np.linalg.inv(m[i]) @ Tb
            # primero con pocos puntos (si incluso así está lejos, no hace falta más)
            dg, _ = arbol.query(self._grueso[que] @ T[:3, :3].T + T[:3, 3], distance_upper_bound=cota + 0.03)
            if np.all(np.isinf(dg)):
                continue
            d, _ = arbol.query(P @ T[:3, :3].T + T[:3, 3], distance_upper_bound=cota)
            dm = float(d.min())
            if dm < mejor:
                mejor, donde = dm, nombre
        return mejor, donde

    def distancia(self, q, hasta=0.30, que="mano"):
        """(metros, eslabón más cercano) de la mano (o del tablero). inf si está a más de 'hasta'."""
        return self._dist(q, hasta, que)

    def fijar_inicio(self, q0):
        """El tablero no puede acercarse al brazo más de lo que ya esté en q0."""
        if self.tablero is not None:
            d0 = self._dist(q0, self.holgura_tablero, "tablero")[0]
            if d0 < self.holgura_tablero:
                self.holgura_tablero = max(d0 - 0.002, 0.0)
        return self

    def ok(self, q):
        if self._dist(q, self.holgura, "mano")[0] < self.holgura:
            return False
        return self.tablero is None or self.holgura_tablero <= 0.0 or \
            self._dist(q, self.holgura_tablero, "tablero")[0] >= self.holgura_tablero

    def camino_ok(self, qs):
        return all(self.ok(q) for q in qs)


def desde_config(cadena, cfg, T_brida_tablero=None):
    """Con los valores de config (CALIB_MANO_HOLGURA, RUTA_XML, CALIB_ROBOT_TABLERO)."""
    g = cfg if isinstance(cfg, dict) else {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    return ChoqueManoBrazo(cadena, g["RUTA_XML"], float(g.get("CALIB_MANO_HOLGURA", 0.03)),
                           T_brida_tablero=T_brida_tablero, tablero=g.get("CALIB_ROBOT_TABLERO"))


def _main():
    import argparse
    import json
    import sys
    sys.path.insert(0, os.getcwd())
    import config as cfg
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    ap = argparse.ArgumentParser(description="¿La mano se acerca demasiado al antebrazo / giro del J4?")
    ap.add_argument("posturas", nargs="?", default=getattr(cfg, "CALIB_ROBOT_POSTURAS", "calib_posturas.json"))
    ap.add_argument("--mapa", action="store_true", help="tabla de distancias para J4 × J5 (J6 = 0)")
    a = ap.parse_args()
    cad = cargar_cadena(cfg.RUTA_XML, verbose=False)
    ch = desde_config(cad, cfg)
    print(f"mano: {len(ch.mano)} puntos · holgura mínima {ch.holgura * 100:.0f} cm")
    if a.mapa:
        q5s = list(range(-180, 181, 20))
        print("J4\\J5 " + " ".join(f"{v:4d}" for v in q5s) + "   (cm hasta el brazo; ** = demasiado cerca)")
        for q4 in range(-180, 181, 20):
            fila = []
            for q5 in q5s:
                d = ch.distancia(np.radians([0, -90, 90, q4, q5, 0]))[0]
                fila.append("  **" if d < ch.holgura else f"{min(d * 100, 99):4.0f}")
            print(f"{q4:5d} " + " ".join(fila))
        return
    if not os.path.exists(a.posturas):
        sys.exit(f"No existe {a.posturas}")
    datos = json.load(open(a.posturas))
    malas = 0
    for k, p in enumerate(datos):
        q = np.asarray(p["q"] if isinstance(p, dict) else p, float)
        d, donde = ch.distancia(q)
        mal = d < ch.holgura
        malas += mal
        print(f"{'❌' if mal else '✓ '} postura {k + 1:2d}: {d * 100:5.1f} cm hasta {donde}   "
              f"(J4 {math.degrees(q[3]):6.1f}°  J5 {math.degrees(q[4]):6.1f}°  J6 {math.degrees(q[5]):6.1f}°)")
    print(f"\n{malas} posturas demasiado cerca." if malas else "\nTodas bien.")


if __name__ == "__main__":
    _main()
