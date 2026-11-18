"""
Escena FIJA (17-nov): separar lo que NO se mueve (mesas, cajonera, muebles) de lo que cambia
(pelotas, herramientas, manos). Funciones puras (numpy + scipy), sin cámara ni robot.

Por qué: con una sola cámara, la tapa de la mesa sale como una capa gruesa de 2-6 cm con ruido y,
en cuanto la mano la tapa (justo al bajar sobre la pelota), desaparece o queda a trozos. Lo fijo no
hace falta medirlo en cada imagen: se ESCANEA UNA VEZ con calma y se guarda como geometría limpia.

  1. ESCANEO (botón «Escanear escena fija» de la web, con la mesa despejada y el brazo apartado):
     se borra el mapa, la TSDF promedia AUTONOMO_ESCENA_FIJA_ESCANEO_S segundos y se ajustan CAJAS:
       · cada mesa encontrada como plano → caja con su rectángulo mínimo en planta, de la tapa
         hasta el suelo (o hasta la tapa de lo que tenga debajo);
       · lo demás, por grupos de cubos → caja orientada (girada en z) si la llena bien; si no
         (formas raras), se guardan sus cubos tal cual.
     Se guarda en escena_fija.json (editable a mano). También se pueden poner cajas medidas a mano
     en config (AUTONOMO_ESCENA_FIJA_CAJAS).
  2. EN VIVO: la cámara no pone obstáculos dentro de las cajas fijas (+ un margen): solo añade lo
     que cambia. El planificador usa la superficie de las cajas como cubos del mapa y sus tapas como
     «suelo» (mano y brazo no bajan de ellas), y cuRobo las recibe en su ESDF. Lo fijo sigue ahí
     aunque la mano o el brazo lo tapen.

Caja: dict(nombre, centro [x, y, z], semi [sx, sy, sz], yaw_deg, origen) en el marco del MJCF.

Línea de comandos:
    python -m skill_shield.autonomo.escena_fija            # qué hay guardado
"""
import json
import math
import os
import time

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHIVO = "escena_fija.json"


# ═════════════════════════════ cajas ═════════════════════════════

def R_yaw(yaw_deg):
    """Rotación en z (columnas = ejes de la caja en el mundo)."""
    a = math.radians(float(yaw_deg or 0.0))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def caja(nombre, centro, semi, yaw_deg=0.0, origen="config"):
    return dict(nombre=str(nombre), centro=[round(float(v), 4) for v in centro],
                semi=[round(abs(float(v)), 4) for v in semi], yaw_deg=round(float(yaw_deg), 2), origen=origen)


def normalizar(c, z_suelo=None):
    """Caja de config: dict(nombre, centro, semi, yaw_deg) o, como TELEOP_FR5_OBSTACULOS,
    dict(nombre, xy, semi_xy, z_sup, desde_suelo=True[, yaw_deg])."""
    if c.get("centro") is not None:
        return caja(c.get("nombre", "fija"), c["centro"], c["semi"], c.get("yaw_deg", 0.0), c.get("origen", "config"))
    z0 = float(z_suelo if z_suelo is not None else c.get("z_inf", 0.0))
    z1 = float(c["z_sup"])
    return caja(c.get("nombre", "fija"), (c["xy"][0], c["xy"][1], 0.5 * (z0 + z1)),
                (c["semi_xy"][0], c["semi_xy"][1], 0.5 * abs(z1 - z0)), c.get("yaw_deg", 0.0), "config")


def en_cajas(P, cajas, margen=0.0):
    """Máscara de los puntos P (N x 3) dentro de alguna caja (agrandada 'margen' m por cada lado)."""
    P = np.asarray(P, float).reshape(-1, 3)
    out = np.zeros(len(P), bool)
    for c in cajas or ():
        ctr, semi = np.asarray(c["centro"], float), np.asarray(c["semi"], float) + float(margen)
        r = float(np.linalg.norm(semi))
        sel = np.nonzero(np.all(np.abs(P - ctr) <= r, axis=1) & ~out)[0]   # descarte rápido
        if not len(sel):
            continue
        D = (P[sel] - ctr) @ R_yaw(c.get("yaw_deg", 0.0))
        out[sel[np.all(np.abs(D) <= semi, axis=1)]] = True
    return out


def tapa(c):
    """(z de la tapa, 4 vértices xy) de una caja."""
    ctr, semi = np.asarray(c["centro"], float), np.asarray(c["semi"], float)
    R = R_yaw(c.get("yaw_deg", 0.0))[:2, :2]
    esq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * semi[:2]
    return float(ctr[2] + semi[2]), esq @ R.T + ctr[:2]


def superficie(cajas, paso=0.02):
    """Puntos (mundo) de la TAPA y los 4 LADOS de cada caja, cada 'paso' m (la base va en el suelo).
    Para el planificador: hacen de cubos ocupados limpios."""
    out = []
    for c in cajas or ():
        ctr, s = np.asarray(c["centro"], float), np.asarray(c["semi"], float)
        R = R_yaw(c.get("yaw_deg", 0.0))

        def ejes(L):
            n = max(int(math.ceil(2 * L / paso)), 1)
            return np.linspace(-L, L, n + 1)
        gx, gy, gz = ejes(s[0]), ejes(s[1]), ejes(s[2])
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        caras = [np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, s[2])])]
        for sgn in (-1.0, 1.0):
            A, Z = np.meshgrid(gy, gz, indexing="ij")
            caras.append(np.column_stack([np.full(A.size, sgn * s[0]), A.ravel(), Z.ravel()]))
            A, Z = np.meshgrid(gx, gz, indexing="ij")
            caras.append(np.column_stack([A.ravel(), np.full(A.size, sgn * s[1]), Z.ravel()]))
        L = np.unique(np.round(np.vstack(caras), 5), axis=0)
        out.append(L @ R.T + ctr)
    return np.vstack(out) if out else np.zeros((0, 3))


# ═════════════════════════════ ajuste ═════════════════════════════

def rect_minimo(P2):
    """Rectángulo de área mínima que contiene los puntos 2D: (centro xy, semi (a, b), yaw en grados).
    Se prueba la orientación de cada lado de la envolvente convexa (el óptimo tiene uno apoyado)."""
    P2 = np.unique(np.asarray(P2, float).reshape(-1, 2), axis=0)
    if len(P2) < 3:
        lo, hi = P2.min(0), P2.max(0)
        return 0.5 * (lo + hi), 0.5 * (hi - lo), 0.0
    try:
        from scipy.spatial import ConvexHull
        H = P2[ConvexHull(P2).vertices]
    except Exception:                                    # todos en línea
        H = P2
    mejor = None
    E = np.roll(H, -1, axis=0) - H
    for ang in np.unique(np.round(np.arctan2(E[:, 1], E[:, 0]) % (math.pi / 2), 6)):
        c, s = math.cos(ang), math.sin(ang)
        Q = H @ np.array([[c, -s], [s, c]])               # coordenadas en los ejes girados 'ang'
        lo, hi = Q.min(0), Q.max(0)
        area = float(np.prod(hi - lo))
        if mejor is None or area < mejor[0] - 1e-12:
            mejor = (area, ang, lo, hi)
    _, ang, lo, hi = mejor
    c, s = math.cos(ang), math.sin(ang)
    centro = np.array([[c, -s], [s, c]]) @ (0.5 * (lo + hi))
    semi = 0.5 * (hi - lo)
    yaw = math.degrees(ang)
    if yaw > 45.0:                                       # el mismo rectángulo con yaw en (-45, 45]
        yaw -= 90.0
        semi = semi[::-1]
    return centro, semi, yaw


def ajustar(centros, res, superficies=(), z_suelo=None, quitar=None, grupo_min=20, llenado_min=0.45,
            area_min=0.02, suelo_cerca=0.15, banda_tapa=0.06, alto_min=0.03):
    """Escena fija a partir de un mapa escaneado.
    centros: cubos ocupados (N x 3) · superficies: mesas como planos [(z, vértices xy)] ·
    quitar: máscara (N,) de cubos que no cuentan (p. ej. pegados al soporte, que ya está en el modelo).
    Devuelve dict(cajas=[...], voxeles=[[x, y, z], ...], res)."""
    C = np.asarray(centros, float).reshape(-1, 3)
    if quitar is not None and len(C):
        C = C[~np.asarray(quitar, bool)]
    z0 = float(z_suelo) if z_suelo is not None else (float(C[:, 2].min()) - 0.5 * res if len(C) else 0.0)
    cajas = []
    # ── 1. mesas (planos): de la tapa hasta el suelo o hasta lo que tengan debajo ──
    planos = []
    for z, poli in sorted(superficies or (), key=lambda s: float(s[0])):
        poli = np.asarray(poli, float).reshape(-1, 2)
        if len(poli) < 3:
            continue
        ctr, semi, yaw = rect_minimo(poli)
        if 4.0 * semi[0] * semi[1] < area_min:
            continue
        planos.append((float(z), ctr, semi, yaw))
    for k, (z, ctr, semi, yaw) in enumerate(planos):
        base = z0
        for zb, cb, sb, yb in planos[:k]:                # otra tapa más baja debajo de su centro
            d = R_yaw(yb)[:2, :2].T @ (ctr - cb)
            if zb < z - alto_min and np.all(np.abs(d) <= sb):
                base = max(base, zb)
        cajas.append(caja(f"mesa_{k + 1}", (ctr[0], ctr[1], 0.5 * (z + base)), (semi[0], semi[1], 0.5 * (z - base)),
                          yaw, "plano"))
    # lo que ya explican las mesas (también la capa de ruido de encima de la tapa): fuera
    if cajas and len(C):
        usado = np.zeros(len(C), bool)
        for c in cajas:
            alta = dict(c, centro=list(np.asarray(c["centro"]) + [0, 0, 0.5 * banda_tapa]),
                        semi=list(np.asarray(c["semi"]) + [0, 0, 0.5 * banda_tapa]))
            usado |= en_cajas(C, [alta], margen=res)
        C = C[~usado]
    # ── 2. lo demás, por grupos de cubos ──
    sueltos = []
    if len(C):
        from scipy.ndimage import label
        idx = np.round((C - C.min(0)) / res).astype(int)
        g = np.zeros(tuple(idx.max(0) + 1), bool)
        g[tuple(idx.T)] = True
        etq, n = label(g, structure=np.ones((3, 3, 3), bool))
        e = etq[tuple(idx.T)]
        tapas = [tapa(c) for c in cajas]
        for k in range(1, n + 1):
            G = C[e == k]
            if len(G) < grupo_min:
                continue                                 # ruido
            ctr, semi, yaw = rect_minimo(G[:, :2])
            semi = semi + 0.5 * res                      # los cubos ocupan medio cubo más
            zl, zh = float(G[:, 2].min()) - 0.5 * res, float(G[:, 2].max()) + 0.5 * res
            if zl - z0 < suelo_cerca:                    # apoyado en el suelo (la cámara ve el frente y la tapa)
                zl = z0
            else:                                        # apoyado en una mesa de las de antes
                from skill_shield.autonomo.tsdf import dentro_poligono
                for zt, poli in tapas:
                    if zl - 0.08 <= zt <= zl + res and dentro_poligono(poli, ctr[None])[0]:
                        zl = zt
                        break
            # ¿la caja está bien llena? (en planta: columnas con algún cubo / área del rectángulo)
            cols = len(np.unique(np.round(G[:, :2] / res).astype(int), axis=0))
            llenado = cols * res * res / max(4.0 * semi[0] * semi[1], 1e-9)
            if llenado >= llenado_min and zh - zl >= alto_min:
                cajas.append(caja(f"bloque_{len(cajas) + 1}", (ctr[0], ctr[1], 0.5 * (zl + zh)),
                                  (semi[0], semi[1], 0.5 * (zh - zl)), yaw, "grupo"))
            else:
                sueltos.append(G)
    vox = np.vstack(sueltos) if sueltos else np.zeros((0, 3))
    return dict(cajas=cajas, voxeles=np.round(vox, 4).tolist(), res=float(res))


# ═════════════════════════════ archivo ═════════════════════════════

def ruta(g=None):
    nombre = (g or {}).get("AUTONOMO_ESCENA_FIJA_ARCHIVO", ARCHIVO) if isinstance(g, dict) else ARCHIVO
    return nombre if os.path.isabs(nombre) else os.path.join(RAIZ, nombre)


def guardar(ruta_archivo, escena, **info):
    datos = dict(version=1, fecha=time.strftime("%Y-%m-%d %H:%M:%S"), **info, **escena)
    tmp = ruta_archivo + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=1)
    os.replace(tmp, ruta_archivo)                        # el otro proceso nunca lee uno a medias
    return ruta_archivo


def cargar(ruta_archivo):
    try:
        with open(ruta_archivo, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    d["cajas"] = [normalizar(c) for c in d.get("cajas") or []]
    d["voxeles"] = np.asarray(d.get("voxeles") or [], float).reshape(-1, 3)
    return d


def olvidar(ruta_archivo):
    """Aparta el archivo (queda como .bak por si acaso)."""
    if os.path.exists(ruta_archivo):
        os.replace(ruta_archivo, ruta_archivo + ".bak")


class EscenaFija:
    """La escena fija de un proceso: lo del archivo (se recarga si cambia) + lo medido a mano en config.
    .cajas, .voxeles, .version (cambia cada vez que cambia algo)."""

    def __init__(self, g):
        g = g if isinstance(g, dict) else {k: getattr(g, k) for k in dir(g) if k.isupper()}
        self.activa = bool(g.get("AUTONOMO_ESCENA_FIJA", True))
        forzar = os.environ.get("SKILL_SHIELD_ESCENA_FIJA")      # "0" = nunca (las pruebas) · "1" = siempre
        if forzar in ("0", "1"):
            self.activa = forzar == "1"
        self.ruta = ruta(g)
        z_suelo = g.get("SOPORTE_SUELO_Z") if g.get("SOPORTE_HOMBROS", False) else 0.0
        self.de_config = [normalizar(c, z_suelo) for c in (g.get("AUTONOMO_ESCENA_FIJA_CAJAS") or [])
                          if c.get("activo", True)]
        self._mtime = -1.0
        self.cajas, self.voxeles, self.fecha, self.version = [], np.zeros((0, 3)), None, 0
        self.actualizar()

    def actualizar(self):
        """Recarga si el archivo ha cambiado. True si ha cambiado algo."""
        if not self.activa:
            if self.version == 0:
                self.version = 1
                return True
            return False
        try:
            m = os.path.getmtime(self.ruta)
        except OSError:
            m = None
        if m == self._mtime:
            return False
        self._mtime = m
        d = cargar(self.ruta) if m is not None else None
        self.cajas = list(self.de_config) + (d["cajas"] if d else [])
        self.voxeles = d["voxeles"] if d else np.zeros((0, 3))
        self.fecha = d.get("fecha") if d else None
        self.version += 1
        if d:
            print(f"🧱 Escena fija: {len(d['cajas'])} cajas y {len(self.voxeles)} cubos del escaneo del {self.fecha}"
                  + (f" + {len(self.de_config)} de config" if self.de_config else ""))
        return True

    def tapas(self):
        return [tapa(c) for c in self.cajas]

    def texto(self):
        if not self.activa:
            return "escena fija desactivada"
        if not self.cajas and not len(self.voxeles):
            return "sin escena fija"
        return f"escena fija: {len(self.cajas)} cajas" + (f" + {len(self.voxeles)} cubos" if len(self.voxeles) else "")


def dibujar(scn, cajas, rgba=(0.25, 0.55, 1.0, 0.25)):
    """Añade las cajas fijas a una escena de MuJoCo (visor.user_scn o Renderer.scene)."""
    import mujoco
    n = 0
    for c in cajas or ():
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.asarray(c["semi"], float),
                            np.asarray(c["centro"], float), R_yaw(c.get("yaw_deg", 0.0)).flatten(),
                            np.asarray(rgba, np.float32))
        scn.ngeom += 1
        n += 1
    return n


if __name__ == "__main__":
    import sys
    sys.path.insert(0, RAIZ)
    import config
    e = EscenaFija(config)
    print(f"{e.ruta}: {e.texto()}" + (f" (escaneada el {e.fecha})" if e.fecha else ""))
    for c in e.cajas:
        zt, _ = tapa(c)
        print(f"  · {c['nombre']:<12} [{c['origen']}] centro ({', '.join(f'{v:+.3f}' for v in c['centro'])}) "
              f"medidas {2 * c['semi'][0]:.2f} × {2 * c['semi'][1]:.2f} × {2 * c['semi'][2]:.2f} m · "
              f"giro {c['yaw_deg']:+.0f}° · tapa a z = {zt:+.3f} m")
