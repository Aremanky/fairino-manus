"""
El mundo del planificador → mundo de cuRobo (17-nov). Sin cuRobo ni GPU (numpy + scipy): así se
puede probar en cualquier PC.

  · Rejilla de DISTANCIA CON SIGNO (ESDF, metros; negativo dentro de un obstáculo) con los cubos
    ocupados del planificador (ya sin lo excluido) y el «suelo» de las mesas y de las
    zonas de apoyo marcado como ocupado. cuRobo la usa como un VoxelGrid.
  · El entorno fijo del modelo de colisión: cápsulas (columna y cabezal del soporte…) y cajas
    (TELEOP_FR5_OBSTACULOS, cámaras…), y el suelo como una caja.
  · (20-nov) La ESDF se puede calcular en la GPU (edt: transformada de distancia exacta en PyTorch, sin cuRobo) y la
    pelota objetivo se le da a cuRobo como una esfera (capsula_objetivo) mientras se va hacia ella.
"""
import math

import numpy as np


# ══════════════ (20-nov) transformada de distancia EXACTA en la GPU (o en numpy) ══════════════
# La misma que scipy.ndimage.distance_transform_edt, pero separable por ejes (Saito-Toriwaki): la distancia al cuadrado
# a lo más cercano se calcula eje a eje con g[i] = min_j (f[j] + (i - j)²). Cada eje es un «mínimo más parábola»
# sobre la línea entera: n² operaciones por línea, nada para la GPU (360 000 cubos × 100 ≈ 36 M por eje) y EXACTO
# (no es una aproximación por saltos). Se hace por trozos para no pasarse de memoria.
_AVISO_EDT = None


def _edt2_eje(f, eje, xp, trozo):
    """Un eje del EDT al cuadrado: g = min_j f[..., j, ...] + (i - j)² a lo largo de 'eje'."""
    es_torch = xp.__name__ == "torch"
    a = xp.movedim(f, eje, -1) if es_torch else np.moveaxis(f, eje, -1)
    forma = a.shape
    n = int(forma[-1])
    a = a.reshape(-1, n)
    idx = xp.arange(n, dtype=a.dtype, device=a.device) if es_torch else np.arange(n, dtype=a.dtype)
    c = (idx[:, None] - idx[None, :]) ** 2                                  # c[i, j] = (i - j)²
    out = xp.empty_like(a)
    filas = max(1, int(trozo) // max(n * n, 1))
    for k in range(0, a.shape[0], filas):
        s = a[k:k + filas][:, None, :] + c[None, :, :]                       # (m, n_i, n_j)
        out[k:k + filas] = s.min(dim=2).values if es_torch else s.min(axis=2)
    out = out.reshape(forma)
    return xp.movedim(out, -1, eje) if es_torch else np.moveaxis(out, -1, eje)


def edt(objetivo, res=1.0, motor="auto", trozo=1 << 24):
    """Distancia (m) de cada celda a la celda True más cercana de 'objetivo' (0 en ellas): la de
    scipy.ndimage.distance_transform_edt(~objetivo) · res. motor: "auto" (GPU con PyTorch + CUDA si hay; si no
    scipy), "gpu" o "numpy" (scipy). Si la GPU falla, se avisa una vez y se sigue con scipy."""
    global _AVISO_EDT
    objetivo = np.asarray(objetivo, bool)
    if not objetivo.any():
        return np.full(objetivo.shape, np.inf, np.float32)
    if str(motor) in ("auto", "gpu", "cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch no ve ninguna GPU con CUDA")
            dev = torch.device("cuda")
            grande = float(sum(int(n) ** 2 for n in objetivo.shape)) + 1.0     # «infinito» que no desborda
            f = torch.where(torch.as_tensor(objetivo, device=dev), torch.zeros((), device=dev),
                            torch.full((), grande, device=dev)).float()
            for eje in range(f.dim()):
                f = _edt2_eje(f, eje, torch, trozo)
            return (torch.sqrt(f) * float(res)).cpu().numpy().astype(np.float32)
        except Exception as e:
            if str(motor) != "auto" and _AVISO_EDT != str(e):
                _AVISO_EDT = str(e)
                print(f"⚠️  ESDF en la GPU no disponible ({e}): sigo con scipy (CPU).")
    from scipy.ndimage import distance_transform_edt
    return (distance_transform_edt(~objetivo) * float(res)).astype(np.float32)


def edt_numpy_separable(objetivo, res=1.0, trozo=1 << 22):
    """El mismo algoritmo que la GPU, en numpy (para las pruebas: tiene que dar lo mismo que scipy)."""
    objetivo = np.asarray(objetivo, bool)
    grande = float(sum(int(n) ** 2 for n in objetivo.shape)) + 1.0
    f = np.where(objetivo, 0.0, grande).astype(np.float64)
    for eje in range(f.ndim):
        f = _edt2_eje(f, eje, np, trozo)
    return (np.sqrt(f) * float(res)).astype(np.float32)


def distancia_a_ocupado(ocupado, res, motor="auto"):
    """Distancia (m) de cada cubo al cubo ocupado más cercano (0 en ellos). Para MapaOcupacion.distancias()."""
    return edt(ocupado, res, motor)


def esdf(ocupado, res, motor="numpy"):
    """Distancia con signo (m) de cada cubo a la superficie de lo ocupado (rejilla booleana).
    (20-nov) motor="auto"/"gpu": las dos transformadas de distancia en la GPU (edt)."""
    ocupado = np.asarray(ocupado, bool)
    if not ocupado.any():
        return np.full(ocupado.shape, 1.0, np.float32)
    fuera = edt(ocupado, res, motor)                          # libre: distancia al ocupado más cercano
    dentro = edt(~ocupado, res, motor) if (~ocupado).any() else np.full(ocupado.shape, 1.0, np.float32)
    return (np.where(ocupado, -(dentro - 0.5 * res), fuera - 0.5 * res)).astype(np.float32)


def capsula_objetivo(objetivo, margen=0.0):
    """(20-nov) La pelota objetivo como obstáculo de cuRobo: dict(name, base, tip, radius) (una esfera: cápsula de
    largo casi 0), o None. objetivo: dict(c=centro (3,), r=radio) en el mundo."""
    if not objetivo:
        return None
    c = np.asarray(objetivo["c"], float).reshape(3)
    r = float(objetivo["r"]) + float(margen)
    if r <= 0:
        return None
    return dict(name="pelota_objetivo", base=[float(v) for v in c], tip=[float(c[0]), float(c[1]), float(c[2]) + 1e-4],
                radius=r)


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


def mundo_del_planificador(pl, limites, res=None, desconocido_dist=None, margen=0.0, motor_esdf="numpy"):
    """ESDF (nx, ny, nz), centro y medidas de la rejilla para el estado ACTUAL del planificador
    `pl` (tarea.Planificador): sus cubos (sin lo excluido ni lo de las zonas) y su suelo.
    desconocido_dist (17-nov): en lo que la cámara NO ha visto (pl._visto) la distancia se rebaja a este
    valor (algo más que la esfera más gruesa del robot): cuRobo no lo trata como choque —el camino sigue
    siendo posible— pero, con una distancia de activación mayor, su optimizador lo PENALIZA y prefiere
    caminos por lo que sí se ha visto.
    margen (18-nov): se resta de la distancia (= inflar el robot contra el mapa y las mesas) con el mismo margen
    que usan las comprobaciones del proyecto. Las mesas del mundo simple (pl._losas) entran como ocupado.
    motor_esdf (20-nov): "auto"/"gpu" = la ESDF en la GPU (edt), "numpy" = scipy."""
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
    sdf = esdf(occ, res, motor_esdf) - np.float32(margen)
    if desconocido_dist is not None and getattr(pl, "_visto", None):
        C = centros_rejilla(origen, n, res) if C is None else C
        desc = desconocido_en(pl._visto, C).reshape(tuple(n)) & ~occ
        sdf = np.where(desc & (sdf > desconocido_dist), np.float32(desconocido_dist), sdf).astype(np.float32)
    centro = origen + 0.5 * n * res
    return sdf, centro, (n * res)
