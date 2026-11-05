"""
Obstáculos y zonas prohibidas de la IK del FR5 (5-nov).

Antes las colisiones de la IK solo conocían el SUELO y el SOPORTE de hombros. Ahora el
entorno también tiene, como cajas en el marco del MUNDO del simulador (fairino_mjcf.xml):
  · TELEOP_FR5_OBSTACULOS: objetos fijos (mesa, peana...). Con desde_suelo=True la caja va
    del suelo hasta z_sup (mesas, peanas y trípodes llegan al suelo).
  · TELEOP_FR5_CAMARAS: cada cámara (cuerpo) y, si se indica, su trípode hasta el suelo.
  · TELEOP_FR5_ZONAS_PROHIBIDAS: espacio donde el robot no debe entrar aunque no haya nada
    (puesto del operador, paso de personas...).
La IK diferencial se aparta de todas ellas a menos de d_col_activa (repulsión) y no acepta
posturas que las atraviesen (ModeloColision). Lo que tenga xyz/centro = None se ignora y se
avisa al arrancar: sin medir, no se protege.

Los ejes del mundo: el hombro del FR5 está en ~(0, 0, 0.157), el robot mira hacia −Y, la
T-pose es −X, z hacia arriba y el suelo en SOPORTE_SUELO_Z (−0.975).
"""
import numpy as np

from skill_shield.teleop_fr5.ik_teleop import Caja


def cajas_desde_config(g, z_suelo, verbose=True):
    """Lista de Caja a partir de los TELEOP_FR5_* de config (g = dict o globals())."""
    get = g.get if isinstance(g, dict) else (lambda k, d=None: getattr(g, k, d))
    cajas, avisos = [], []

    for o in get("TELEOP_FR5_OBSTACULOS", []) or []:
        if not o.get("activo", True):
            continue
        nombre = o.get("nombre", "obstaculo")
        if o.get("desde_suelo"):
            if o.get("xy") is None or o.get("z_sup") is None:
                avisos.append(nombre)
                continue
            z0 = float(z_suelo if z_suelo is not None else o.get("z_inf", 0.0))
            z1 = float(o["z_sup"])
            sx, sy = o["semi_xy"]
            cajas.append(Caja(nombre, (o["xy"][0], o["xy"][1], 0.5 * (z0 + z1)),
                              (sx, sy, 0.5 * abs(z1 - z0))))
        else:
            if o.get("centro") is None:
                avisos.append(nombre)
                continue
            cajas.append(Caja(nombre, o["centro"], o["semi"]))

    for cam in get("TELEOP_FR5_CAMARAS", []) or []:
        if not cam.get("activo", True):
            continue
        nombre = cam.get("nombre", "camara")
        if cam.get("xyz") is None:
            avisos.append(nombre)
            continue
        p = np.asarray(cam["xyz"], dtype=float)
        cajas.append(Caja(nombre, p, cam.get("semi", (0.06, 0.06, 0.04))))
        if cam.get("tripode", True) and z_suelo is not None and p[2] > z_suelo:
            r = float(cam.get("radio_tripode", 0.15))      # patas abiertas: caja ancha y baja
            alto = p[2] - z_suelo
            cajas.append(Caja(nombre + "_tripode", (p[0], p[1], z_suelo + 0.5 * alto),
                              (r, r, 0.5 * alto)))

    for z in get("TELEOP_FR5_ZONAS_PROHIBIDAS", []) or []:
        if not z.get("activo", True):
            continue
        nombre = z.get("nombre", "zona")
        if z.get("centro") is None:
            avisos.append(nombre)
            continue
        cajas.append(Caja(nombre, z["centro"], z["semi"], zona=True))

    if verbose:
        if cajas:
            print("🚧 Obstáculos de la IK: " + ", ".join(
                f"{c.nombre}{' (zona)' if c.zona else ''}" for c in cajas))
        if avisos:
            print("⚠ Sin posición (NO protegidos): " + ", ".join(avisos)
                  + " → mídelos y ponlos en config.py")
        if not any(c.zona for c in cajas):
            print("⚠ No hay zonas prohibidas definidas (TELEOP_FR5_ZONAS_PROHIBIDAS)")
    return cajas


def dibujar_en_visor(visor, entorno, rgba_obj=(1.0, 0.45, 0.0, 0.18), rgba_zona=(1.0, 0.0, 0.0, 0.15)):
    """Pinta las cajas (y las cápsulas del soporte) del entorno de la IK, translúcidas, en el
    visor de MuJoCo (user_scn): así se comprueba a ojo que coinciden con el decorado y, con el
    robot real, con el laboratorio. No tocan la física."""
    import mujoco
    scn = visor.user_scn
    n = 0
    for c in entorno.cajas:
        if scn.ngeom + n >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom + n], mujoco.mjtGeom.mjGEOM_BOX,
                            np.asarray(c.semi, dtype=float), np.asarray(c.centro, dtype=float),
                            np.asarray(c.R, dtype=float).flatten(),
                            np.asarray(rgba_zona if c.zona else rgba_obj, dtype=np.float32))
        n += 1
    for c in entorno.capsulas:
        if scn.ngeom + n >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom + n]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.asarray(rgba_obj, dtype=np.float32))
        conectar = getattr(mujoco, "mjv_connector", None) or getattr(mujoco, "mjv_makeConnector")
        try:
            conectar(g, mujoco.mjtGeom.mjGEOM_CAPSULE, float(c.r),
                     np.asarray(c.a, dtype=float), np.asarray(c.b, dtype=float))
        except TypeError:                             # firma antigua (coordenadas sueltas)
            conectar(g, mujoco.mjtGeom.mjGEOM_CAPSULE, float(c.r),
                     *[float(v) for v in c.a], *[float(v) for v in c.b])
        n += 1
    scn.ngeom += n
    return n
