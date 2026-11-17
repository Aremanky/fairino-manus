"""
MJCF del proyecto → URDF para cuRobo (17-nov).

    python -m skill_shield.autonomo.curobo_fr5.urdf               # escribe curobo_fr5/fr5_orca.urdf

  · Eslabones: world → base_link (fijo, donde está en el MJCF: base girada, soporte…) → shoulder_Link
    … wrist3_Link (joint1..joint6 revolutos, con sus límites) → mano_mount (fijo). El marco del
    URDF es el MUNDO del MJCF: el mismo del gemelo, de la IK y del mapa (así el mapa no hay que
    transformarlo).
  · Colisión del brazo: las mallas de colisión del MJCF (fr5_col_*); si un cuerpo no tiene, las
    visuales.
  · La MANO va fija en mano_mount con TODAS sus mallas en varias posturas a la vez (por defecto la
    de reposo, la pinza abierta y la de la pelota abierta): su envolvente. Para ir de un sitio a otro
    es conservador; el planificador del proyecto comprueba luego cada postura con la mano tal como
    va de verdad.
  · Las rutas de las mallas son relativas a la raíz del proyecto (asset_path de cuRobo).
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

from skill_shield.autonomo.choque_mano import JOINTS_MANO, _T_local, _expandir_includes, _rot

CADENA = ("base_link", "shoulder_Link", "upperarm_Link", "forearm_Link", "wrist1_Link", "wrist2_Link",
          "wrist3_Link")
JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
HERRAMIENTA = "mano_mount"


def rpy_de(R):
    """(roll, pitch, yaw) del URDF (R = Rz(yaw)·Ry(pitch)·Rx(roll))."""
    R = np.asarray(R, float)
    sy = -R[2, 0]
    if abs(sy) < 1 - 1e-9:
        p = math.asin(max(-1.0, min(1.0, sy)))
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:                                            # cardán bloqueado
        p = math.copysign(math.pi / 2, sy)
        r = math.atan2(-R[1, 2], R[1, 1])
        y = 0.0
    return r, p, y


def R_de_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origen(T):
    r, p, y = rpy_de(T[:3, :3])
    return dict(xyz=" ".join(f"{v:.9g}" for v in T[:3, 3]), rpy=f"{r:.9g} {p:.9g} {y:.9g}")


def _cargar(ruta_xml):
    base = os.path.dirname(os.path.abspath(ruta_xml))
    root = ET.parse(ruta_xml).getroot()
    _expandir_includes(root, base)
    comp = root.find("compiler")
    grados = comp is not None and comp.get("angle", "degree") == "degree"
    meshdir = os.path.join(base, comp.get("meshdir", "")) if comp is not None else base
    mallas = {}
    for m in root.iter("mesh"):
        if m.get("file"):
            esc = [float(v) for v in m.get("scale", "1 1 1").split()]
            mallas[m.get("name") or os.path.splitext(os.path.basename(m.get("file")))[0]] = \
                (os.path.normpath(os.path.join(meshdir, m.get("file"))), esc)
    cuerpos = {b.get("name"): b for b in root.iter("body")}
    return root, grados, mallas, cuerpos


def _mallas_cuerpo(b, mallas, grados, solo_colision=True):
    """[(ruta, escala, T del geom en el cuerpo)] de los geoms con malla de un cuerpo (sin hijos)."""
    geoms = [g for g in b.findall("geom") if g.get("mesh") in mallas]
    col = [g for g in geoms if "col" in g.get("mesh", "") or g.get("group") == "3"]
    usar = col if (solo_colision and col) else geoms
    return [(mallas[g.get("mesh")][0], mallas[g.get("mesh")][1], _T_local(g, grados)) for g in usar]


def _mallas_mano(b, mallas, grados, angulos, T):
    """Todas las mallas del subárbol de la mano con los dedos en los ángulos dados, en el marco de
    mano_mount (T = acumulado hasta este cuerpo)."""
    G = np.eye(4)
    for j in b.findall("joint"):
        a = angulos.get(j.get("name"), 0.0)
        if a and j.get("type", "hinge") == "hinge":
            p = np.array([float(v) for v in j.get("pos", "0 0 0").split()])
            J = np.eye(4)
            J[:3, :3] = _rot([float(v) for v in j.get("axis", "0 0 1").split()], a)
            J[:3, 3] = p - J[:3, :3] @ p
            G = G @ J
    T = T @ G
    out = [(ruta, esc, T @ Tg) for ruta, esc, Tg in _mallas_cuerpo(b, mallas, grados, solo_colision=False)]
    for h in b.findall("body"):
        out += _mallas_mano(h, mallas, grados, angulos, T @ _T_local(h, grados))
    return out


def posturas_por_defecto(cfg=None):
    """Posturas de la mano (índice de ctrl 0..16 → rad) cuya envolvente lleva el URDF."""
    from skill_shield.autonomo import tarea as TA
    g = {} if cfg is None else (cfg if isinstance(cfg, dict) else
                                {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    return [dict(g.get("AUTONOMO_TAREA_MANO_ABIERTA") or TA.MANO_ABIERTA),
            dict(g.get("AUTONOMO_TAREA_MANO_PINZA_ABIERTA") or TA.MANO_PINZA_ABIERTA),
            dict(g.get("AUTONOMO_TAREA_MANO_PELOTA_ABIERTA") or TA.MANO_PELOTA_ABIERTA)]


def generar_urdf(ruta_xml, salida, raiz=None, posturas=None, nombre="fr5_orca", vel_rad_s=3.0, esfuerzo=150.0):
    """Escribe el URDF y devuelve su ruta. raiz: carpeta a la que se hacen relativas las mallas
    (la que se le da a cuRobo como asset_path; por defecto, la del URDF)."""
    root, grados, mallas, cuerpos = _cargar(ruta_xml)
    faltan = [c for c in CADENA + (HERRAMIENTA,) if c not in cuerpos]
    if faltan:
        raise ValueError(f"el MJCF no tiene los cuerpos {faltan}")
    salida = os.path.abspath(salida)
    raiz = os.path.abspath(raiz or os.path.dirname(salida))
    posturas = posturas if posturas is not None else posturas_por_defecto()

    def rel(ruta):
        return os.path.relpath(ruta, raiz)

    robot = ET.Element("robot", name=nombre)

    def eslabon(nombre_l, geoms):
        L = ET.SubElement(robot, "link", name=nombre_l)
        for ruta, esc, Tg in geoms:
            for tipo in ("visual", "collision"):
                e = ET.SubElement(L, tipo)
                ET.SubElement(e, "origin", **_origen(Tg))
                g = ET.SubElement(e, "geometry")
                ET.SubElement(g, "mesh", filename=rel(ruta), scale=" ".join(f"{v:.9g}" for v in esc))
        return L

    def junta(nombre_j, tipo, padre, hijo, T, eje=None, lim=None):
        J = ET.SubElement(robot, "joint", name=nombre_j, type=tipo)
        ET.SubElement(J, "origin", **_origen(T))
        ET.SubElement(J, "parent", link=padre)
        ET.SubElement(J, "child", link=hijo)
        if eje is not None:
            ET.SubElement(J, "axis", xyz=" ".join(f"{v:.9g}" for v in eje))
        if lim is not None:
            ET.SubElement(J, "limit", lower=f"{lim[0]:.9g}", upper=f"{lim[1]:.9g}",
                          velocity=f"{vel_rad_s:.9g}", effort=f"{esfuerzo:.9g}")

    # marco de base_link en el mundo (todo lo que haya por encima en el MJCF)
    padres = {h: p for p in root.iter("body") for h in p.findall("body")}
    T_base = np.eye(4)
    b = cuerpos["base_link"]
    cadena_arriba = []
    while b is not None:
        cadena_arriba.append(b)
        b = padres.get(b)
    for b in reversed(cadena_arriba):
        T_base = T_base @ _T_local(b, grados)

    ET.SubElement(robot, "link", name="world")
    eslabon("base_link", _mallas_cuerpo(cuerpos["base_link"], mallas, grados))
    junta("world_a_base", "fixed", "world", "base_link", T_base)
    padre = "base_link"
    for nombre_c, nombre_j in zip(CADENA[1:], JOINTS):
        b = cuerpos[nombre_c]
        js = [j for j in b.findall("joint") if j.get("type", "hinge") == "hinge"]
        if len(js) != 1 or js[0].get("name") != nombre_j:
            raise ValueError(f"{nombre_c}: se esperaba un único hinge «{nombre_j}»")
        j = js[0]
        p = np.array([float(v) for v in j.get("pos", "0 0 0").split()])
        if np.linalg.norm(p) > 1e-9:
            raise ValueError(f"{nombre_j}: el joint no está en el origen de su cuerpo (pos={p})")
        rango = [float(v) for v in j.get("range").split()]
        k = math.pi / 180.0 if grados else 1.0
        eslabon(nombre_c, _mallas_cuerpo(b, mallas, grados))
        junta(nombre_j, "revolute", padre, nombre_c, _T_local(b, grados),
              eje=[float(v) for v in j.get("axis", "0 0 1").split()], lim=(rango[0] * k, rango[1] * k))
        padre = nombre_c
    # la mano: envolvente de las posturas
    mano = cuerpos[HERRAMIENTA]
    geoms = []
    for post in posturas:
        angulos = {JOINTS_MANO[int(i)]: float(a) for i, a in post.items()}
        geoms += _mallas_mano(mano, mallas, grados, angulos, np.eye(4))
    eslabon(HERRAMIENTA, geoms)
    junta("brida_a_mano", "fixed", padre, HERRAMIENTA, _T_local(mano, grados))

    ET.indent(robot) if hasattr(ET, "indent") else None
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    ET.ElementTree(robot).write(salida, encoding="utf-8", xml_declaration=True)
    return salida


# ── FK del URDF (para comprobarlo contra la cadena del proyecto, sin cuRobo) ──
def fk_urdf(ruta_urdf, q):
    """{eslabón: T 4x4 en el mundo} del URDF con los joints revolutos en q (orden joint1..6)."""
    root = ET.parse(ruta_urdf).getroot()
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    qd = dict(zip(JOINTS, q))
    T = {"world": np.eye(4)}

    def marco(link):
        if link in T:
            return T[link]
        j = joints[link]
        o = j.find("origin")
        M = np.eye(4)
        M[:3, :3] = R_de_rpy(*[float(v) for v in o.get("rpy").split()])
        M[:3, 3] = [float(v) for v in o.get("xyz").split()]
        if j.get("type") == "revolute":
            G = np.eye(4)
            G[:3, :3] = _rot([float(v) for v in j.find("axis").get("xyz").split()], qd[j.get("name")])
            M = M @ G
        T[link] = marco(j.find("parent").get("link")) @ M
        return T[link]
    for link in joints:
        marco(link)
    return T


def main():
    import argparse
    import config as cfg
    raiz = os.path.dirname(os.path.abspath(cfg.__file__))
    ap = argparse.ArgumentParser(description="MJCF del proyecto → URDF para cuRobo")
    ap.add_argument("--salida", default=os.path.join(raiz, "curobo_fr5", "fr5_orca.urdf"))
    args = ap.parse_args()
    ruta = generar_urdf(cfg.RUTA_XML, args.salida, raiz=raiz, posturas=posturas_por_defecto(cfg))
    n = len(ET.parse(ruta).getroot().findall(".//collision"))
    print(f"✅ URDF escrito en {ruta} ({n} mallas de colisión; mallas relativas a {raiz})")


if __name__ == "__main__":
    main()
