"""
Utilidades comunes de yolo_herramientas/: configuración, carpetas, máscara → polígono YOLO y escritura.

Formato intermedio (lo que deja cada convertir_*.py):
    <DATOS>/convertidos/<fuente>/images/{train,val}/<nombre>.jpg
    <DATOS>/convertidos/<fuente>/labels/{train,val}/<nombre>.txt     (YOLO-seg con NUESTROS ids)
    <DATOS>/convertidos/<fuente>/info.json                           (cuántas imágenes e instancias)
unir_datasets.py junta las fuentes que se le digan en <DATOS>/yolo/<nombre>/ con su data.yaml.
"""
import hashlib
import json
import os
import shutil
from collections import Counter

import cv2
import numpy as np
import yaml

AQUI = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(AQUI, "clases.yaml")


def cargar_config(ruta=CONFIG):
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    nombres = list(cfg["clases"])
    if len(set(nombres)) != len(nombres):
        raise SystemExit(f"✗ {ruta}: hay clases repetidas")
    cfg["_id"] = {n: i for i, n in enumerate(nombres)}
    return cfg


def id_clase(cfg, nombre):
    if nombre not in cfg["_id"]:
        raise SystemExit(f"✗ La clase «{nombre}» no está en la lista «clases» de clases.yaml")
    return cfg["_id"][nombre]


def dir_datos():
    """Carpeta de los datasets (FUERA del repo: pesan decenas de GB). HERRAMIENTAS_DATOS o ~/datasets_herramientas."""
    d = os.environ.get("HERRAMIENTAS_DATOS") or os.path.join(os.path.expanduser("~"), "datasets_herramientas")
    return os.path.abspath(os.path.expanduser(d))


def particion(clave, frac_val=0.1, semilla="herramientas"):
    """'train' o 'val' de forma estable a partir de una clave (la escena, el vídeo, la imagen…).
    Todas las imágenes de una misma escena caen en el mismo lado: si no, val se parece demasiado a train."""
    h = int(hashlib.md5(f"{semilla}:{clave}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < frac_val else "train"


# ───────────────────────────── máscara → polígono ─────────────────────────────

def _unir_contornos(cs):
    """Varios trozos de una misma instancia (p. ej. un mango tapado por un dedo) → UN polígono.
    Se recorre el primero y, en su punto más cercano al siguiente, se salta a este, se le da la
    vuelta entera y se vuelve por el mismo sitio (puente de anchura cero: no añade área)."""
    base = cs[0]
    for c in cs[1:]:
        d = ((base[:, None, :].astype(np.float64) - c[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(d.argmin()), d.shape)
        c_rot = np.roll(c, -j, axis=0)
        base = np.concatenate([base[:i + 1], c_rot, c_rot[:1], base[i:]])
    return base


def mascara_a_poligono(mascara, min_px=30, eps_px=1.0):
    """Máscara binaria (H, W) → polígono normalizado [x1, y1, x2, y2, …] (0-1) o None si no queda nada.
    Los trozos de menos de min_px píxeles se tiran; los demás se unen en un solo polígono."""
    m = (np.asarray(mascara) > 0).astype(np.uint8)
    h, w = m.shape[:2]
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in cs if cv2.contourArea(c) >= min_px]
    if not cs:
        return None
    cs.sort(key=cv2.contourArea, reverse=True)
    cs = [cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2) for c in cs]
    cs = [c for c in cs if len(c) >= 3]
    if not cs:
        return None
    p = _unir_contornos(cs).astype(np.float64)
    p[:, 0] = np.clip(p[:, 0] / w, 0, 1)
    p[:, 1] = np.clip(p[:, 1] / h, 0, 1)
    return p.reshape(-1).tolist()


def poligono_a_mascara(poligono, h, w):
    """Inversa (para revisar y para las pruebas): polígono normalizado → máscara (H, W) uint8."""
    p = np.asarray(poligono, np.float64).reshape(-1, 2) * [w, h]
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m


# ───────────────────────────── escritura ─────────────────────────────

class Escritor:
    """Escribe una fuente convertida en <DATOS>/convertidos/<fuente>/ y lleva la cuenta."""

    def __init__(self, fuente, cfg, max_lado=1280, limpiar=False):
        self.fuente, self.cfg, self.max_lado = fuente, cfg, max_lado
        self.raiz = os.path.join(dir_datos(), "convertidos", fuente)
        if limpiar and os.path.isdir(self.raiz):
            shutil.rmtree(self.raiz)
        for s in ("train", "val"):
            os.makedirs(os.path.join(self.raiz, "images", s), exist_ok=True)
            os.makedirs(os.path.join(self.raiz, "labels", s), exist_ok=True)
        self.n_img = Counter()
        self.n_inst = Counter()

    def guardar(self, nombre, imagen, instancias, split):
        """imagen: ruta de un archivo o array BGR. instancias: [(id_clase, poligono_normalizado), …].
        Las imágenes sin ninguna instancia también se guardan (sirven de «fondo») si se pasan."""
        destino = os.path.join(self.raiz, "images", split, nombre + ".jpg")
        if isinstance(imagen, str):
            img = None
            if self.max_lado or not imagen.lower().endswith((".jpg", ".jpeg")):
                img = cv2.imread(imagen)
                if img is None:
                    print(f"   ⚠ no se puede leer {imagen}: se salta")
                    return False
            else:
                shutil.copyfile(imagen, destino)
        else:
            img = imagen
        if img is not None:
            h, w = img.shape[:2]
            if self.max_lado and max(h, w) > self.max_lado:
                f = self.max_lado / max(h, w)
                img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(destino, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(self.raiz, "labels", split, nombre + ".txt"), "w") as f:
            for c, p in instancias:
                f.write(str(int(c)) + " " + " ".join(f"{v:.6f}" for v in p) + "\n")
                self.n_inst[(split, int(c))] += 1
        self.n_img[split] += 1
        return True

    def cerrar(self, extra=None):
        nombres = self.cfg["clases"]
        info = dict(fuente=self.fuente, imagenes=dict(self.n_img),
                    instancias={s: {nombres[c]: n for (s2, c), n in sorted(self.n_inst.items()) if s2 == s}
                                for s in ("train", "val")})
        info.update(extra or {})
        with open(os.path.join(self.raiz, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"\n✅ {self.fuente}: {self.n_img['train']} imágenes de train y {self.n_img['val']} de val → {self.raiz}")
        for s in ("train", "val"):
            if info["instancias"][s]:
                print(f"   {s}: " + " · ".join(f"{k} {v}" for k, v in info["instancias"][s].items()))
        return info
