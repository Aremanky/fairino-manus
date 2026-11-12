"""
Choques de la OrcaHand —y de lo que lleva encima: el tablero de calibración, las clavijas y los
cables— con el PROPIO brazo y con el soporte, usando las piezas reales del gemelo.

El modelo de cápsulas de la teleoperación (ik_teleop.ModeloColision) NO comprueba la mano contra
el antebrazo: con cápsulas, mano y antebrazo "chocan" casi siempre porque están pegados por la
muñeca. Pero si la muñeca se dobla mucho (J4/J5), la mano vuelve hacia atrás y golpea el giro
del J4 — y eso rompe una pieza de la mano. Y antes de romper nada, aplasta lo que SOBRESALE de
la mano por detrás (clavijas, cables). Tampoco conoce el tablero: un tablero tamaño folio
sobresale de la mano mucho más que la cápsula con la que se la representa.

Aquí se usan las MISMAS mallas que dibuja el gemelo de MuJoCo (fairino_mjcf.xml + la OrcaHand
que incluye + el soporte de decoracion/soporte_hombros.xml), muestreadas como puntos:
  · la mano (dedos en q = 0) y la placa del tablero, en el marco de la brida (wrist3);
  · el hombro, el brazo, el antebrazo y el giro del J4, cada uno en su marco;
  · el soporte (columna, cabezal, placas de los hombros, bastidor), fijo en el mundo;
  · y, si se le pasa el entorno de la teleoperación (ik_teleop.Entorno), sus cápsulas y cajas
    (cabeza con la cámara, mesa, zonas prohibidas…) y el suelo.
Para una postura q se llevan los puntos de la mano y del tablero a cada marco y se busca la
distancia mínima (árbol k-d). No hace falta MuJoCo (solo numpy + scipy).

Huecos que se exigen (ChoqueManoBrazo.ok):
  · nunca menos de un MÍNIMO (mano-brazo: CALIB_MANO_HOLGURA);
  · tras fijar_inicio(q0), ningún hueco puede cerrarse más de CALIB_MANO_ACERCAR_MAX respecto al
    que había en la postura de inicio —donde el operario ve que nada queda pillado ni tirante—,
    salvo que siga sobrando el hueco OBJETIVO (mano-brazo: holgura + CALIB_MANO_SALIENTES, lo
    que pueden sobresalir clavijas y cables), a partir del cual ya no importa.

Uso desde línea de comandos (comprobar posturas guardadas, sin mover nada):
    python -m skill_shield.autonomo.choque_mano                    # calib_posturas.json
    python -m skill_shield.autonomo.choque_mano --mapa             # tabla J4×J5 de distancias
"""
import functools
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

# wrist1 = el giro del J4. (wrist2, el del J6, no se mira: la mano va atornillada justo delante
# y gira con él alrededor de su propio eje.)
CUERPOS_BRAZO = ("shoulder_Link", "upperarm_Link", "forearm_Link", "wrist1_Link")
MARCO_CADENA = {"shoulder_Link": 0, "upperarm_Link": 1, "forearm_Link": 2, "wrist1_Link": 3, "wrist3_Link": 5}
CUERPO_MANO = "mano_mount"
PREFIJO_SOPORTE = "sop_"            # geoms fijos del soporte del robot (decoracion/soporte_hombros.xml)
NOMBRES = {"shoulder_Link": "del hombro del robot", "upperarm_Link": "del brazo", "forearm_Link": "del antebrazo",
           "wrist1_Link": "del giro del J4", "soporte": "del soporte", "columna": "de la columna del soporte",
           "cabezal": "de la cabeza del soporte", "bastidor": "del bastidor del soporte", "suelo": "del suelo",
           "camara_cabeza": "de la cámara de cabeza"}          # «queda a X cm …»


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
    """Puntos de la placa del tablero (el dibujo y un borde de papel/cartón alrededor) en el
    marco de la brida."""
    nx, ny = cuadros
    xs = _rejilla(-borde, nx * lado + borde, paso)
    ys = _rejilla(-borde, ny * lado + borde, paso)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size), np.ones(X.size)])
    return (np.asarray(T_brida_tablero, float) @ P.T).T[:, :3]


# ── puntos sobre cajas, cilindros y envolventes (para el soporte) ──
def _rejilla(a, b, paso):
    """De a a b (los dos incluidos) con separación ≤ paso."""
    return np.linspace(a, b, max(int(math.ceil((b - a) / paso)), 1) + 1)


def _puntos_caja(semi, paso):
    """Superficie de una caja centrada en el origen (semi-lados): una rejilla por cara."""
    out = []
    for eje in range(3):
        a, b = [k for k in range(3) if k != eje]
        A, B = np.meshgrid(_rejilla(-semi[a], semi[a], paso), _rejilla(-semi[b], semi[b], paso))
        for s in (-1.0, 1.0):
            P = np.zeros((A.size, 3))
            P[:, a], P[:, b], P[:, eje] = A.ravel(), B.ravel(), s * semi[eje]
            out.append(P)
    return np.vstack(out)


def _puntos_cilindro(a, b, r, paso):
    """Superficie de un cilindro de radio r entre a y b (lateral y las dos tapas)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    z = (b - a) / L if L > 1e-9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    def aro(radio):
        ang = np.linspace(0.0, 2.0 * math.pi, max(int(math.ceil(2.0 * math.pi * radio / paso)), 8), endpoint=False)
        return radio * (np.outer(np.cos(ang), x) + np.outer(np.sin(ang), y))
    out = [a + t * z + aro(r) for t in _rejilla(0.0, L, paso)]
    for radio in _rejilla(0.0, r, paso)[1:-1]:
        out += [a + aro(radio), b + aro(radio)]
    out += [a[None, :], b[None, :]]
    return np.vstack(out)


def _puntos_envolvente(V, paso, rng):
    """Superficie de la envolvente convexa de unos vértices (mallas escritas en el propio XML)."""
    try:
        from scipy.spatial import ConvexHull
        return muestrear(V[ConvexHull(V).simplices], paso, rng)
    except Exception:                               # degenerada (plana…): los vértices tal cual
        return np.asarray(V, float)


def _puntos_geom(g, en_grados, paso, rng, vertices, mallas):
    """Puntos (en el marco del padre) de un <geom> box / cylinder / mesh. None si es de otro tipo."""
    tipo = g.get("type", "sphere")
    tam = [float(v) for v in g.get("size", "").split()]
    T = _T_local(g, en_grados)
    if tipo == "box" and len(tam) == 3:
        P = _puntos_caja(tam, paso)
    elif tipo == "cylinder" and tam:
        if g.get("fromto"):
            ft = [float(v) for v in g.get("fromto").split()]
            return _puntos_cilindro(ft[:3], ft[3:], tam[0], paso)
        if len(tam) < 2:
            return None
        P = _puntos_cilindro([0.0, 0.0, -tam[1]], [0.0, 0.0, tam[1]], tam[0], paso)
    elif tipo == "mesh" and g.get("mesh") in vertices:
        P = _puntos_envolvente(vertices[g.get("mesh")], paso, rng)
    elif tipo == "mesh" and g.get("mesh") in mallas and os.path.exists(mallas[g.get("mesh")][0]):
        ruta, esc = mallas[g.get("mesh")]
        P = muestrear(leer_stl(ruta) * esc, paso, rng)
    else:
        return None
    return P @ T[:3, :3].T + T[:3, 3]


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
def nubes_mjcf(ruta_xml, paso=0.006, semilla=0, paso_soporte=0.010):
    """{cuerpo: puntos (n,3)}:
      · los eslabones de CUERPOS_BRAZO, cada uno en su marco;
      · "mano": la mano entera (subárbol de mano_mount, dedos en q = 0) en el marco de wrist3_Link;
      · "soporte": los geoms fijos cuyo nombre empieza por PREFIJO_SOPORTE, en el mundo."""
    base = os.path.dirname(os.path.abspath(ruta_xml))
    root = ET.parse(ruta_xml).getroot()
    _expandir_includes(root, base)
    comp = root.find("compiler")
    grados = comp is not None and comp.get("angle", "degree") == "degree"
    meshdir = os.path.join(base, comp.get("meshdir", "")) if comp is not None else base
    mallas, vertices = {}, {}
    for m in root.iter("mesh"):
        if m.get("file"):
            esc = np.array([float(v) for v in m.get("scale", "1 1 1").split()])
            mallas[m.get("name") or os.path.splitext(os.path.basename(m.get("file")))[0]] = \
                (os.path.join(meshdir, m.get("file")), esc)
        elif m.get("vertex") and m.get("name"):
            vertices[m.get("name")] = np.array([float(v) for v in m.get("vertex").split()]).reshape(-1, 3)
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
    P = []
    for wb in root.iter("worldbody"):
        for g in wb.findall("geom"):
            if (g.get("name") or "").startswith(PREFIJO_SOPORTE):
                Pg = _puntos_geom(g, grados, paso_soporte, rng, vertices, mallas)
                if Pg is not None and len(Pg):
                    P.append(Pg)
    nubes["soporte"] = reducir(np.vstack(P), paso_soporte) if P else np.zeros((0, 3))
    return nubes


# Joints de la mano en el orden de sus actuadores (índice de ctrl 0..16 de la mano, el de
# LIMITES_CTRL y OrcaHardwareWrapper.JOINT_A_CTRL; ctrl 16 «thumb_dip» = right_t-pip).
JOINTS_MANO = ("right_wrist", "right_p-abd", "right_p-mcp", "right_p-pip", "right_r-abd", "right_r-mcp",
               "right_r-pip", "right_m-abd", "right_m-mcp", "right_m-pip", "right_i-abd", "right_i-mcp",
               "right_i-pip", "right_t-cmc", "right_t-abd", "right_t-mcp", "right_t-pip")


@functools.lru_cache(maxsize=16)
def nube_mano(ruta_xml, postura=(), paso=0.006, semilla=0):
    """La mano (subárbol de mano_mount) en el marco de wrist3_Link con los dedos en una POSTURA
    (12-nov): postura = ((índice de ctrl 0..16, rad), …); lo que no se indica, a 0. Los puntos
    salen en el mismo orden de geoms que nubes_mjcf()["mano"] (con postura vacía son los mismos)."""
    angulos = {JOINTS_MANO[int(k)]: float(v) for k, v in postura}
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
    if CUERPO_MANO not in cuerpos:
        return np.zeros((0, 3))
    rng = np.random.default_rng(semilla)
    cache = {}

    def giro_joint(b):
        T = np.eye(4)
        for j in b.findall("joint"):
            a = angulos.get(j.get("name"), 0.0)
            if a == 0.0 or j.get("type", "hinge") != "hinge":
                continue
            p = np.array([float(v) for v in j.get("pos", "0 0 0").split()])
            G = np.eye(4)
            G[:3, :3] = _rot([float(v) for v in j.get("axis", "0 0 1").split()], a)
            G[:3, 3] = p - G[:3, :3] @ p                 # giro alrededor del eje que pasa por pos
            T = T @ G
        return T

    def nube_body(b, T):
        T = T @ giro_joint(b)
        out = []
        for g in b.findall("geom"):
            nm = g.get("mesh")
            if not nm or nm not in mallas:
                continue
            if nm not in cache:
                ruta, esc = mallas[nm]
                cache[nm] = muestrear(leer_stl(ruta) * esc, paso, rng) if os.path.exists(ruta) else np.zeros((0, 3))
            Tg = T @ _T_local(g, grados)
            out.append(cache[nm] @ Tg[:3, :3].T + Tg[:3, 3])
        for h in b.findall("body"):
            out.extend(nube_body(h, T @ _T_local(h, grados)))
        return out
    P = nube_body(cuerpos[CUERPO_MANO], _T_local(cuerpos[CUERPO_MANO], grados))
    return reducir(np.vstack(P), paso) if P else np.zeros((0, 3))


# ── distancias de una nube de puntos a las formas del entorno ──
def _dist_arbol(arbol, grueso, fino, T, cota):
    """Distancia mínima de los puntos 'fino' (llevados con T al marco del árbol) a la nube del
    árbol; inf si están a más de 'cota'. Primero con pocos puntos ('grueso', uno por celda de
    2 cm: cualquier punto fino está a < 3,5 cm de uno grueso), para no buscar si está lejos."""
    dg, _ = arbol.query(grueso @ T[:3, :3].T + T[:3, 3], distance_upper_bound=cota + 0.035)
    if np.all(np.isinf(dg)):
        return math.inf
    d, _ = arbol.query(fino @ T[:3, :3].T + T[:3, 3], distance_upper_bound=cota)
    return float(d.min())


def _dist_segmento(P, a, b):
    """Distancia (n,) de cada punto de P al segmento a-b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    t = np.clip((P - a) @ ab / L2, 0.0, 1.0) if L2 > 1e-12 else np.zeros(len(P))
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _dist_caja(P, caja):
    """Distancia con signo (n,) de cada punto de P a una ik_teleop.Caja (negativa dentro)."""
    l = (P - caja.centro) @ caja.R                      # al marco de la caja (R⁻¹ = Rᵀ)
    q = np.abs(l) - caja.semi
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


class ChoqueManoBrazo:
    """Huecos entre lo que va en la brida (la mano y, si se da T_brida_tablero, la placa del
    tablero) y todo lo demás: hombro, brazo, antebrazo y giro del J4 del robot, el soporte y, si
    se da 'entorno' (ik_teleop.Entorno), sus cápsulas, sus cajas y el suelo.

    Cada pareja (objeto, cosa) tiene un hueco OBJETIVO y un MÍNIMO:
        mano-brazo:     holgura + salientes   /  holgura
        tablero-brazo:  holgura_tablero       /  holgura_tablero_min
        con el entorno: holgura_entorno       /  holgura_entorno_min
    ok(q) exige el mínimo. Tras fijar_inicio(q0) exige, además, que ningún hueco se cierre más
    de 'acercar_max' respecto al de q0, hasta su objetivo (más allá ya da igual): lo que
    sobresale de la mano y no está en las mallas —clavijas, cables, cinta— cabía en el hueco que
    había al empezar, así que no se deja estrechar.

    ok(q) / incumple(q) / motivo(q) / camino_ok(lista de q) / distancia(q)."""

    def __init__(self, cadena, ruta_xml, holgura=0.03, paso=0.006, T_brida_tablero=None, tablero=None,
                 holgura_tablero=0.04, salientes=0.0, acercar_max=0.01, holgura_tablero_min=0.015,
                 margen_tablero=0.012, entorno=None, holgura_entorno=0.05, holgura_entorno_min=0.03):
        from scipy.spatial import cKDTree
        self.c = cadena
        self.holgura = float(holgura)
        self.salientes = max(float(salientes), 0.0)
        self.acercar_max = max(float(acercar_max), 0.0)
        self.holgura_tablero = float(holgura_tablero)
        self.holgura_tablero_min = min(float(holgura_tablero_min), self.holgura_tablero)
        self.holgura_entorno = float(holgura_entorno)
        self.holgura_entorno_min = min(float(holgura_entorno_min), self.holgura_entorno)
        nubes = nubes_mjcf(ruta_xml, paso)
        if "mano" not in nubes or len(nubes["mano"]) == 0:
            raise RuntimeError(f"{os.path.basename(ruta_xml)}: no encuentro las mallas de la mano")
        self.mano = nubes["mano"].copy()
        self.tablero = None
        if T_brida_tablero is not None:
            t = tablero or {}
            self.tablero = puntos_tablero(T_brida_tablero, tuple(t.get("cuadros", (7, 5))),
                                          float(t.get("lado", 0.020)), borde=float(margen_tablero))
        self.arboles = {k: (MARCO_CADENA[k], cKDTree(nubes[k])) for k in CUERPOS_BRAZO
                        if k in nubes and len(nubes[k])}
        if not self.arboles:
            raise RuntimeError(f"{os.path.basename(ruta_xml)}: no encuentro las mallas del brazo")
        self.soporte = cKDTree(nubes["soporte"]) if len(nubes.get("soporte", ())) else None
        self.entorno = entorno
        self._objetos = {"mano": self.mano}
        if self.tablero is not None:
            self._objetos["tablero"] = self.tablero
        self._grueso = {k: reducir(P, 0.02) for k, P in self._objetos.items()}     # para descartar rápido
        self.exigido = {}                    # (objeto, cosa) → hueco que se exige (lo pone fijar_inicio)

    # ── qué se compara con qué ──
    def _cosas(self):
        cosas = list(self.arboles)
        if self.soporte is not None:
            cosas.append("soporte")
        if self.entorno is not None:
            cosas += [c.nombre for c in self.entorno.capsulas] + [c.nombre for c in self.entorno.cajas]
            if self.entorno.z_suelo is not None:
                cosas.append("suelo")
        return cosas

    def _rango(self, objeto, cosa):
        """(hueco objetivo, hueco mínimo) de una pareja."""
        if cosa in self.arboles:
            if objeto == "mano":
                return self.holgura + self.salientes, self.holgura
            return self.holgura_tablero, self.holgura_tablero_min
        return self.holgura_entorno, self.holgura_entorno_min

    def _hueco_exigido(self, objeto):
        return {c: self.exigido.get((objeto, c), self._rango(objeto, c)[1]) for c in self._cosas()}

    # ── distancias ──
    def _dists(self, q, objeto="mano", cota=0.30, solo_brazo=False):
        """{cosa: metros} del objeto ("mano" o "tablero") a cada cosa. Con las nubes de puntos
        (eslabones y soporte), inf si está a más de su cota. cota: un número o {cosa: número}."""
        P = self._objetos.get(objeto)
        if P is None:
            return {}
        m = self.c.fk_marcos(np.asarray(q, float))
        Tb = m[MARCO_CADENA["wrist3_Link"]]
        G = self._grueso[objeto]

        def c_(cosa):
            return float(cota[cosa]) if isinstance(cota, dict) else float(cota)
        out = {n: _dist_arbol(arbol, G, P, np.linalg.inv(m[i]) @ Tb, c_(n)) for n, (i, arbol) in self.arboles.items()}
        if solo_brazo:
            return out
        if self.soporte is not None:
            out["soporte"] = _dist_arbol(self.soporte, G, P, Tb, c_("soporte"))
        ent = self.entorno
        if ent is not None:
            Pw = P @ Tb[:3, :3].T + Tb[:3, 3]
            for cap in ent.capsulas:
                out[cap.nombre] = float(_dist_segmento(Pw, cap.a, cap.b).min() - cap.r)
            for caja in ent.cajas:
                out[caja.nombre] = float(_dist_caja(Pw, caja).min())
            if ent.z_suelo is not None:
                out["suelo"] = float(Pw[:, 2].min() - ent.z_suelo)
        return out

    def distancia(self, q, hasta=0.30, que="mano"):
        """(metros, eslabón más cercano) de la mano (o del tablero) al BRAZO. inf si está a más
        de 'hasta'."""
        d = self._dists(q, que, hasta, solo_brazo=True)
        donde = min(d, key=d.get) if d else ""
        return (d[donde], donde) if d and math.isfinite(d[donde]) else (math.inf, "")

    # ── referencia: la postura de inicio ──
    def fijar_inicio(self, q0):
        """Toma la postura de inicio como referencia: desde ahora cada hueco (mano o tablero con
        cada eslabón, con el soporte, con cada obstáculo) tiene que valer al menos lo que valía
        en q0 menos 'acercar_max' — sin bajar nunca de su mínimo ni exigir más que su objetivo."""
        self.exigido = {}
        for objeto in self._objetos:
            cota = {c: self._rango(objeto, c)[0] + self.acercar_max for c in self._cosas()}
            for cosa, d0 in self._dists(q0, objeto, cota).items():
                objetivo, minimo = self._rango(objeto, cosa)
                self.exigido[(objeto, cosa)] = min(max(d0 - self.acercar_max, minimo), objetivo)
        return self

    # ── comprobaciones ──
    def incumple(self, q, margen=0.0):
        """Lista de (objeto, cosa, hueco en metros, hueco exigido) de lo que queda demasiado cerca
        en la postura q. Vacía si todo vale. margen (m): cuánto menos de lo exigido se tolera."""
        out = []
        for objeto in self._objetos:
            exigido = self._hueco_exigido(objeto)
            for cosa, d in self._dists(q, objeto, exigido).items():
                if d < exigido[cosa] - margen:
                    out.append((objeto, cosa, d, exigido[cosa]))
        return out

    def ok(self, q, margen=0.0):
        for objeto in self._objetos:
            exigido = self._hueco_exigido(objeto)
            if any(d < exigido[cosa] - margen for cosa, d in self._dists(q, objeto, exigido).items()):
                return False
        return True

    def motivo(self, q, margen=0.0):
        """Texto con lo que más cerca queda de lo permitido en q, o None si todo vale."""
        malos = self.incumple(q, margen)
        if not malos:
            return None
        objeto, cosa, d, exigido = min(malos, key=lambda x: x[2] - x[3])
        return (f"{'la mano' if objeto == 'mano' else 'el tablero'} queda a {max(d, 0.0) * 100:.1f} cm "
                f"{NOMBRES.get(cosa, f'de «{cosa}»')} (mínimo {exigido * 100:.1f} cm)")

    def camino_ok(self, qs):
        return all(self.ok(q) for q in qs)


def desde_config(cadena, cfg, T_brida_tablero=None, entorno=None):
    """Con los valores de config (RUTA_XML, CALIB_ROBOT_TABLERO y los CALIB_MANO_*, CALIB_TABLERO_*
    y CALIB_ENTORNO_HOLGURA). entorno: ik_teleop.Entorno (el de mapa.modelo_colision) para comprobar
    también la cabeza con la cámara, las cajas de config y el suelo."""
    g = cfg if isinstance(cfg, dict) else {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    return ChoqueManoBrazo(cadena, g["RUTA_XML"], float(g.get("CALIB_MANO_HOLGURA", 0.03)),
                           T_brida_tablero=T_brida_tablero, tablero=g.get("CALIB_ROBOT_TABLERO"),
                           holgura_tablero=float(g.get("CALIB_TABLERO_HOLGURA", 0.04)),
                           margen_tablero=float(g.get("CALIB_TABLERO_MARGEN", 0.025)),
                           salientes=float(g.get("CALIB_MANO_SALIENTES", 0.04)),
                           acercar_max=float(g.get("CALIB_MANO_ACERCAR_MAX", 0.01)),
                           entorno=entorno, holgura_entorno=float(g.get("CALIB_ENTORNO_HOLGURA", 0.05)))


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
