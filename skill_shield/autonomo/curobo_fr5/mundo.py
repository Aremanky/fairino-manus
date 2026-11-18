"""
El mundo del planificador → mundo de cuRobo (17-nov). Sin cuRobo ni GPU (numpy + scipy): así se
puede probar en cualquier PC.

  · Rejilla de DISTANCIA CON SIGNO (ESDF, metros; negativo dentro de un obstáculo) con los cubos
    ocupados del planificador (ya sin lo excluido, con el macizo) y el «suelo» de las mesas y de las
    zonas de apoyo marcado como ocupado. cuRobo la usa como un VoxelGrid.
  · El entorno fijo del modelo de colisión: cápsulas (columna y cabezal del soporte…) y cajas
    (TELEOP_FR5_OBSTACULOS, cámaras…), y el suelo como una caja.
"""
import math

import numpy as np


def esdf(ocupado, res):
    """Distancia con signo (m) de cada cubo a la superficie de lo ocupado (rejilla booleana)."""
    from scipy.ndimage import distance_transform_edt
    ocupado = np.asarray(ocupado, bool)
    if not ocupado.any():
        return np.full(ocupado.shape, 1.0, np.float32)
    fuera = distance_transform_edt(~ocupado) * res            # libre: distancia al ocupado más cercano
    dentro = distance_transform_edt(ocupado) * res            # ocupado: distancia al libre más cercano
    return (np.where(ocupado, -(dentro - 0.5 * res), fuera - 0.5 * res)).astype(np.float32)


def rejilla(limites, res):
    lim = np.asarray(limites, float).reshape(3, 2)
    n = np.maximum(np.ceil((lim[:, 1] - lim[:, 0]) / res).astype(int), 1)
    return lim[:, 0], n


def ocupacion(centros, origen, n, res):
    """Rejilla booleana con los cubos de 'centros' (los de fuera se ignoran)."""
    occ = np.zeros(tuple(n), bool)
    C = np.asarray(centros, float).reshape(-1, 3)
    if len(C):
        idx = np.floor((C - origen) / res).astype(int)
        ok = np.all((idx >= 0) & (idx < n), axis=1)
        occ[idx[ok, 0], idx[ok, 1], idx[ok, 2]] = True
    return occ


def centros_rejilla(origen, n, res):
    ix, iy, iz = np.meshgrid(*[np.arange(k) for k in n], indexing="ij")
    return (np.stack([ix, iy, iz], -1).reshape(-1, 3) + 0.5) * res + origen


def _quat_wxyz(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]


def obstaculos_entorno(entorno, margen=0.0, ancho_suelo=4.0):
    """(cápsulas, cajas) del entorno fijo como dicts para cuRobo:
    cápsula = dict(name, base, tip, radius) · caja = dict(name, dims, pose=[x, y, z, qw, qx, qy, qz])."""
    caps, cajas = [], []
    if entorno is None:
        return caps, cajas
    for c in getattr(entorno, "capsulas", []) or []:
        caps.append(dict(name=f"cap_{c.nombre}", base=[float(v) for v in c.a], tip=[float(v) for v in c.b],
                         radius=float(c.r) + margen))
    for k, o in enumerate(getattr(entorno, "cajas", []) or []):
        cajas.append(dict(name=f"caja_{k}_{o.nombre}", dims=[float(2 * s + 2 * margen) for s in o.semi],
                          pose=[float(v) for v in o.centro] + [float(v) for v in _quat_wxyz(o.R)]))
    if getattr(entorno, "z_suelo", None) is not None:
        z = float(entorno.z_suelo)
        cajas.append(dict(name="suelo", dims=[ancho_suelo, ancho_suelo, 0.1],
                          pose=[0.0, 0.0, z - 0.05 + margen, 1.0, 0.0, 0.0, 0.0]))
    return caps, cajas


def desconocido_en(visto, P):
    """Máscara de los puntos P que caen en cubos que la cámara NO ha visto (lo de fuera de su rejilla, no)."""
    out = np.zeros(len(P), bool)
    if not visto or visto.get("conocido") is None:
        return out
    o, r, N = np.asarray(visto["origen"], float), float(visto["res"]), np.asarray(visto["n"])
    idx = np.floor((np.asarray(P, float) - o) / r).astype(int)
    ok = np.all((idx >= 0) & (idx < N), axis=1)
    out[ok] = ~visto["conocido"][idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return out


def mundo_del_planificador(pl, limites, res=None, desconocido_dist=None, margen=0.0):
    """ESDF (nx, ny, nz), centro y medidas de la rejilla para el estado ACTUAL del planificador
    `pl` (tarea.Planificador): sus cubos (sin lo excluido ni lo de las zonas) y su suelo.
    desconocido_dist (17-nov): en lo que la cámara NO ha visto (pl._visto) la distancia se rebaja a este
    valor (algo más que la esfera más gruesa del robot): cuRobo no lo trata como choque —el camino sigue
    siendo posible— pero, con una distancia de activación mayor, su optimizador lo PENALIZA y prefiere
    caminos por lo que sí se ha visto.
    margen (18-nov): se resta de la distancia (= inflar el robot contra el mapa y las mesas) con el mismo margen
    que usan las comprobaciones del proyecto. Las mesas del mundo simple (pl._losas) entran como ocupado."""
    res = float(res or pl._res)
    origen, n = rejilla(limites, res)
    occ = ocupacion(pl._arbol.data if pl._arbol is not None else np.zeros((0, 3)), origen, n, res)
    C = None
    if pl._suelo is not None or pl._discos:
        C = centros_rejilla(origen, n, res)
        occ |= pl._mascara_suelo(C).reshape(tuple(n))
    if getattr(pl, "_losas", None):
        C = centros_rejilla(origen, n, res) if C is None else C
        occ |= pl._mascara_losas(C, 0.0).reshape(tuple(n))
    sdf = esdf(occ, res) - np.float32(margen)
    if desconocido_dist is not None and getattr(pl, "_visto", None):
        C = centros_rejilla(origen, n, res) if C is None else C
        desc = desconocido_en(pl._visto, C).reshape(tuple(n)) & ~occ
        sdf = np.where(desc & (sdf > desconocido_dist), np.float32(desconocido_dist), sdf).astype(np.float32)
    centro = origen + 0.5 * n * res
    return sdf, centro, (n * res)
