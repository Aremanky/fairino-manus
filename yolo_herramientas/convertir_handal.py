#!/usr/bin/env python3
"""
HANDAL (NVIDIA) → YOLO-seg con nuestras clases (herramienta + mango).

HANDAL viene en formato BOP: una carpeta por escena (vídeo) con rgb/, mask_visib/ (máscara visible de
cada objeto), scene_gt.json (pose de cada objeto en cada imagen) y scene_camera.json (intrínsecos).
La herramienta sale de mask_visib/. El MANGO se busca, por este orden:
  1. una carpeta de máscaras en la escena cuyo nombre lleve «afford», «handle» o «mango»;
  2. una malla del mango (*handle*.ply/.obj o *afford*) junto a los modelos: se proyecta con la pose
     de scene_gt.json y se recorta con la máscara visible del objeto;
  3. si no hay ninguna de las dos, la herramienta va sin mango (se avisa).

    python convertir_handal.py --inspeccionar              # PRIMERO: qué hay y qué va a usar
    python convertir_handal.py                             # convierte (1 de cada 10 imágenes)
    python convertir_handal.py --cada 5                    # más imágenes por vídeo
    python convertir_handal.py --solo <carpeta> --categoria martillo   # si la ruta no dice qué es

Las imágenes de un mismo vídeo son casi iguales: por eso --cada y por eso train/val se reparte por
escena, nunca por imagen.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

import cv2
import numpy as np

from comun import Escritor, cargar_config, dir_datos, id_clase, mascara_a_poligono, particion

PATRON_MANGO = re.compile(r"afford|handle|mango", re.I)
EXT_MALLA = (".ply", ".obj", ".stl", ".off")


# ───────────────────────────── búsqueda ─────────────────────────────

def escenas(raiz):
    out = []
    for d, subdirs, files in os.walk(raiz):
        if "scene_gt.json" in files and any(s.startswith("rgb") for s in subdirs):
            out.append(d)
    return sorted(out)


def categoria_de(ruta_rel, cfg):
    r = ruta_rel.lower()
    for palabra, clase in cfg["handal"].items():
        if palabra in r:
            return clase
    return None


def carpeta_modelos(escena, raiz):
    """La carpeta más cercana (subiendo desde la escena) que tenga subcarpetas «models*»."""
    d = os.path.dirname(escena)
    raiz = os.path.abspath(raiz)
    while True:
        try:
            if any(s.startswith("models") and os.path.isdir(os.path.join(d, s)) for s in os.listdir(d)):
                return d
        except OSError:
            pass
        if os.path.abspath(d) == raiz or os.path.dirname(d) == d:
            return None
        d = os.path.dirname(d)


_cache_indices = {}


def indice_mallas(base):
    """{obj_id: ruta} de las mallas de objeto entero y de mango bajo base/models*/ (una vez por carpeta)."""
    if base in _cache_indices:
        return _cache_indices[base]
    enteras, mangos = {}, {}
    for f in glob.glob(os.path.join(base, "models*", "**", "*"), recursive=True):
        if not f.lower().endswith(EXT_MALLA):
            continue
        m = re.search(r"(\d{6})", os.path.basename(f))
        if not m:
            continue
        oid = int(m.group(1))
        (mangos if PATRON_MANGO.search(os.path.relpath(f, base)) else enteras).setdefault(oid, f)
    _cache_indices[base] = (enteras, mangos)
    return enteras, mangos


def carpeta_mascara_mango(escena):
    for s in sorted(os.listdir(escena)):
        if os.path.isdir(os.path.join(escena, s)) and PATRON_MANGO.search(s):
            return os.path.join(escena, s)
    return None


def archivo(carpeta, *nombres):
    for n in nombres:
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(carpeta, n + ext)
            if os.path.exists(p):
                return p
    return None


# ───────────────────────────── proyección de mallas ─────────────────────────────

_cache_mallas = {}


def cargar_malla(ruta):
    if ruta not in _cache_mallas:
        import trimesh
        m = trimesh.load(ruta, force="mesh", process=False)
        caras = np.asarray(m.faces) if hasattr(m, "faces") else np.zeros((0, 3), int)
        _cache_mallas[ruta] = (np.asarray(m.vertices, np.float64), caras)
    return _cache_mallas[ruta]


def rasterizar(V, F, R, t, K, h, w, escala):
    P = (np.asarray(R, np.float64).reshape(3, 3) @ (V * escala).T).T + np.asarray(t, np.float64).reshape(3)
    delante = P[:, 2] > 1e-6
    uv = np.zeros((len(P), 2))
    uv[delante] = (np.asarray(K, np.float64).reshape(3, 3) @ (P[delante] / P[delante, 2:3]).T).T[:, :2]
    m = np.zeros((h, w), np.uint8)
    if len(F):
        F = F[delante[F].all(1)]
        if len(F):
            # Triángulo a triángulo: fillPoly con todos a la vez usa la regla par-impar y las caras que se
            # tapan (delante/detrás) se anulan entre sí. Muchos triángulos de la malla caen en los mismos
            # píxeles: se quitan los repetidos y los que quedan fuera de la imagen.
            tri = np.round(uv[F]).astype(np.int32)
            dentro = ((tri[:, :, 0].max(1) >= 0) & (tri[:, :, 0].min(1) < w) &
                      (tri[:, :, 1].max(1) >= 0) & (tri[:, :, 1].min(1) < h))
            tri = np.unique(tri[dentro].reshape(-1, 6), axis=0).reshape(-1, 3, 2)
            for t in tri:
                cv2.fillConvexPoly(m, t, 1)
    elif delante.sum() >= 3:                           # nube de puntos sin caras: envolvente
        cv2.fillPoly(m, [cv2.convexHull(np.round(uv[delante]).astype(np.int32))], 1)
    return m


def iou(a, b):
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


# ───────────────────────────── inspección ─────────────────────────────

def inspeccionar(raiz, cfg):
    esc = escenas(raiz)
    print(f"📂 {raiz}\n   escenas BOP encontradas: {len(esc)}")
    if not esc:
        print("   ✗ Ninguna carpeta con scene_gt.json y rgb/. ¿Está descomprimido? Primeros niveles:")
        for d, sub, fs in os.walk(raiz):
            nivel = os.path.relpath(d, raiz).count(os.sep)
            if nivel > 2:
                continue
            print("   " + "  " * nivel + os.path.basename(d) + "/  " + " ".join(sorted(fs)[:5]))
        return
    cats = Counter(categoria_de(os.path.relpath(e, raiz), cfg) for e in esc)
    print("   por categoría (según la ruta): " + " · ".join(f"{k or 'SIN CATEGORÍA'} {v}" for k, v in cats.items()))
    e = esc[0]
    print(f"\n   Ejemplo de escena: {os.path.relpath(e, raiz)}")
    for s in sorted(os.listdir(e)):
        p = os.path.join(e, s)
        if os.path.isdir(p):
            fs = sorted(os.listdir(p))
            print(f"     {s}/  ({len(fs)} archivos: {', '.join(fs[:3])}…)")
        else:
            print(f"     {s}")
    with open(os.path.join(e, "scene_gt.json")) as f:
        gt = json.load(f)
    k0 = sorted(gt, key=int)[0]
    print(f"     scene_gt.json[{k0}] = {json.dumps(gt[k0])[:300]}")
    with open(os.path.join(e, "scene_camera.json")) as f:
        print(f"     scene_camera.json[{k0}] = {json.dumps(json.load(f)[k0])[:300]}")
    cm = carpeta_mascara_mango(e)
    base = carpeta_modelos(e, raiz)
    enteras, mangos = indice_mallas(base) if base else ({}, {})
    print(f"\n   Carpeta de modelos: {base and os.path.relpath(base, raiz)} · mallas enteras {len(enteras)} · "
          f"mallas de mango {len(mangos)}")
    sospechosos = [f for f in glob.glob(os.path.join(raiz, "**", "*"), recursive=True)
                   if PATRON_MANGO.search(os.path.basename(f))][:15]
    print("   Archivos/carpetas con «afford|handle|mango» en el nombre: " + (", ".join(
        os.path.relpath(f, raiz) for f in sospechosos) or "NINGUNO"))
    if cm:
        print(f"\n   → MANGO desde las máscaras de {os.path.relpath(cm, raiz)}")
    elif mangos:
        print("\n   → MANGO proyectando las mallas de mango (método 2)")
    else:
        print("\n   → ⚠ NO se ha encontrado el mango: se convertiría solo la herramienta. Pásame esta salida.")


# ───────────────────────────── conversión ─────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", default=os.path.join(dir_datos(), "handal"))
    ap.add_argument("--inspeccionar", action="store_true")
    ap.add_argument("--cada", type=int, default=10, help="una imagen de cada N de cada vídeo")
    ap.add_argument("--solo", default=None, help="convertir solo las escenas bajo esta carpeta")
    ap.add_argument("--categoria", default=None, help="forzar la clase (si la ruta no la dice)")
    ap.add_argument("--sin-mango", action="store_true", help="solo la herramienta")
    ap.add_argument("--frac-val", type=float, default=0.1)
    ap.add_argument("--max-lado", type=int, default=1280)
    a = ap.parse_args()

    cfg = cargar_config()
    if not os.path.isdir(a.raiz):
        sys.exit(f"✗ No existe {a.raiz} (¿has hecho ./descargar.sh handal y descomprimido?)")
    if a.inspeccionar:
        inspeccionar(a.raiz, cfg)
        return
    id_mango = id_clase(cfg, "mango")
    raiz_busqueda = os.path.abspath(a.solo) if a.solo else a.raiz
    esc_todas = escenas(raiz_busqueda)
    if not esc_todas:
        sys.exit(f"✗ No hay escenas BOP en {raiz_busqueda}. Prueba con --inspeccionar")
    esc = Escritor("handal" if not a.solo else "handal_" + os.path.basename(raiz_busqueda.rstrip("/")),
                   cfg, max_lado=a.max_lado, limpiar=True)
    stats = Counter()
    escala_obj = {}                                    # (base, obj_id) → escala de la malla (o None)

    for n_esc, e in enumerate(esc_todas):
        rel = os.path.relpath(e, a.raiz)
        clase = a.categoria or categoria_de(rel, cfg)
        if clase is None:
            stats["escenas sin categoría (cocina u otra)"] += 1
            continue
        cid = id_clase(cfg, clase)
        split = particion(rel, a.frac_val)
        with open(os.path.join(e, "scene_gt.json")) as f:
            gt = json.load(f)
        with open(os.path.join(e, "scene_camera.json")) as f:
            cam = json.load(f)
        dir_rgb = next(os.path.join(e, s) for s in sorted(os.listdir(e)) if s.startswith("rgb"))
        dir_mv = os.path.join(e, "mask_visib") if os.path.isdir(os.path.join(e, "mask_visib")) else os.path.join(e, "mask")
        dir_mm = None if a.sin_mango else carpeta_mascara_mango(e)
        base = None if (a.sin_mango or dir_mm) else carpeta_modelos(e, a.raiz)
        enteras, mangos = indice_mallas(base) if base else ({}, {})
        stats["escenas"] += 1

        for fid in sorted(gt, key=int)[::max(1, a.cada)]:
            f6 = f"{int(fid):06d}"
            ruta_rgb = archivo(dir_rgb, f6)
            if ruta_rgb is None:
                continue
            img = cv2.imread(ruta_rgb)
            if img is None:
                continue
            h, w = img.shape[:2]
            inst = []
            for k, g in enumerate(gt[fid]):
                rm = archivo(dir_mv, f"{f6}_{k:06d}")
                if rm is None:
                    continue
                mv = cv2.imread(rm, cv2.IMREAD_GRAYSCALE)
                if mv is None or mv.shape != (h, w):
                    continue
                p = mascara_a_poligono(mv)
                if p is None:
                    continue
                inst.append((cid, p))
                mango = None
                if dir_mm:                                 # método 1: máscaras del mango
                    r = archivo(dir_mm, f"{f6}_{k:06d}", f6)
                    if r is not None:
                        mm = cv2.imread(r, cv2.IMREAD_GRAYSCALE)
                        if mm is not None and mm.shape == (h, w):
                            mango = (mm > 0) & (mv > 0)
                elif mangos.get(g.get("obj_id")) is not None:   # método 2: proyectar la malla del mango
                    oid = g["obj_id"]
                    K = cam[fid]["cam_K"]
                    clave = (base, oid)
                    if clave not in escala_obj:            # ¿mm o m? se comprueba con la malla entera
                        escala_obj[clave] = None
                        if oid in enteras:
                            V, F = cargar_malla(enteras[oid])
                            mejor = max((iou(rasterizar(V, F, g["cam_R_m2c"], g["cam_t_m2c"], K, h, w, s), mv), s)
                                        for s in (1.0, 1000.0, 0.001))
                            escala_obj[clave] = mejor[1] if mejor[0] > 0.5 else None
                            print(f"   malla {os.path.basename(enteras[oid])}: escala {mejor[1]:g}, "
                                  f"IoU con la máscara {mejor[0]:.2f}" + ("" if mejor[0] > 0.5 else " → NO se usa"))
                        else:
                            escala_obj[clave] = 1.0
                    if escala_obj[clave] is not None:
                        V, F = cargar_malla(mangos[oid])
                        mango = rasterizar(V, F, g["cam_R_m2c"], g["cam_t_m2c"], K, h, w, escala_obj[clave]) & (mv > 0)
                if mango is not None:
                    pm = mascara_a_poligono(mango)
                    if pm is not None:
                        inst.append((id_mango, pm))
                        stats["herramientas con mango"] += 1
                    else:
                        stats["herramientas con el mango tapado"] += 1
                else:
                    stats["herramientas sin mango"] += 1
            if inst:
                nombre = "handal_" + re.sub(r"[^A-Za-z0-9]+", "_", rel).strip("_") + "_" + f6
                esc.guardar(nombre, img, inst, split)
        if n_esc % 20 == 0:
            print(f"   escena {n_esc + 1}/{len(esc_todas)} · {dict(stats)}", flush=True)

    print("\n" + " · ".join(f"{k}: {v}" for k, v in stats.items()))
    if stats["herramientas sin mango"] and not stats["herramientas con mango"] and not a.sin_mango:
        print("⚠ No ha salido ningún mango: mira --inspeccionar y pásame la salida.")
    esc.cerrar(dict(stats))


if __name__ == "__main__":
    main()
