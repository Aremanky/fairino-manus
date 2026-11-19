#!/usr/bin/env python3
"""
Resumen de los intentos del modo autónomo (18-nov): diagnosticos/intentos.jsonl lo escribe la Tarea sola, una
línea por evento (plan, afinado, agarre, reintento, fin). Aquí: tasa de éxito con su intervalo (con pocas pruebas
el número solo engaña), por qué falla, cuánto se equivocaba la cámara (lo que corrige el afinado encima de la
pelota) y si depende de la distancia o de la zona de la imagen.

    python diagnosticos/resumen_intentos.py                   # todo (sin los de simulacro)
    python diagnosticos/resumen_intentos.py --desde 2026-11-19 # desde ese día (p. ej. tras un cambio)
    python diagnosticos/resumen_intentos.py --simulacro        # también los de simulacro

Para comparar un cambio: 20 intentos antes y 20 después, con la pelota en los mismos sitios.
"""
import argparse
import json
import math
import os
import re
from collections import Counter, OrderedDict

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def leer(ruta, desde=None, simulacro=False):
    """{intento: [eventos]} en orden, filtrados."""
    out = OrderedDict()
    if not os.path.exists(ruta):
        return out
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            try:
                e = json.loads(linea)
            except ValueError:
                continue
            if desde and e.get("t", "") < desde:
                continue
            if e.get("simulacro") and not simulacro:
                continue
            out.setdefault(e.get("intento"), []).append(e)
    return out


def clasificar(eventos):
    """'éxito', 'éxito sin táctiles', 'no planifica', 'no la coge', 'toca algo al bajar', 'el robot no sigue',
    'controladora', 'cancelado', 'a medias' u 'otro'."""
    fin = next((e for e in reversed(eventos) if e["evento"] == "fin"), None)
    if fin is None:
        return "a medias"
    agarres = [e for e in eventos if e["evento"] == "agarre"]
    if fin.get("ok"):
        if agarres and agarres[-1].get("ok") is None:
            return "éxito sin táctiles"
        return "éxito"
    if fin.get("cancelada"):
        return "cancelado"
    motivo = str(fin.get("motivo", ""))
    planes = [e for e in eventos if e["evento"] == "plan"]
    if "No noto la pelota" in motivo:
        return "no la coge"
    if "Al bajar" in motivo or "ha tocado algo" in motivo:
        return "toca algo al bajar"
    if "no sigue" in motivo.lower() or "desv" in motivo.lower():
        return "el robot no sigue"
    if "controladora" in motivo.lower() or "no está conectado" in motivo:
        return "controladora"
    if planes and not planes[-1].get("ok") and motivo == planes[-1].get("motivo"):
        return "no planifica"
    if motivo.startswith(("No veo", "Todavía no son estables", "No hay pose")):
        return "no ve el objetivo"
    return "otro"


def wilson(k, n, z=1.96):
    """Intervalo del 95 % de una proporción k/n (Wilson: vale con pocas pruebas)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - m), min(1.0, c + m)


def motivo_corto(t):
    """El motivo de un plan fallido sin lo que cambia de una vez a otra (archivo, cm exactos)."""
    t = re.sub(r"\s*\(guardado en [^)]*\)", "", str(t))
    t = re.sub(r"\d+([.,]\d+)?", "#", t)
    return t[:160]


def resumir(intentos):
    clases = OrderedDict((k, clasificar(v)) for k, v in intentos.items())
    validos = {k: c for k, c in clases.items() if c not in ("cancelado", "a medias")}
    n = len(validos)
    L = []
    if not n:
        return ["Sin intentos terminados todavía (el registro se llena solo al usar «Planificar» y «Ejecutar»)."]
    exitos = sum(c.startswith("éxito") for c in validos.values())
    lo, hi = wilson(exitos, n)
    primera = sum(1 for k, c in validos.items() if c.startswith("éxito")
                  and not any(e["evento"] == "reintento" for e in intentos[k]))
    L += [f"{n} intentos terminados ({len(clases) - n} cancelados o a medias, no cuentan)",
          f"ÉXITO: {exitos}/{n} = {100 * exitos / n:.0f} %  (95 %: entre {100 * lo:.0f} y {100 * hi:.0f} %)"
          + f" · a la primera, sin reintentos: {primera}/{n}"]
    L.append("Cómo acaban: " + " · ".join(f"{c} {m}" for c, m in Counter(validos.values()).most_common()))
    planes = [e for v in intentos.values() for e in v if e["evento"] == "plan"]
    fallidos = [e for e in planes if not e.get("ok")]
    if planes:
        L.append(f"Planes: {len(planes)}, fallidos {len(fallidos)} ({100 * len(fallidos) / len(planes):.0f} %)")
        for m, k in Counter(motivo_corto(e.get("motivo", "")) for e in fallidos).most_common(4):
            L.append(f"   {k}× {m}")
    af = [e["cm"] for v in intentos.values() for e in v if e["evento"] == "afinado" and e.get("cm") is not None]
    if af:
        L.append(f"Afinado (lo que se había equivocado la cámara, medido encima de la pelota): mediana {np.median(af):.1f} cm, "
                 f"peor {max(af):.1f} cm ({len(af)} veces)")
    ag = [e for v in intentos.values() for e in v if e["evento"] == "agarre" and e.get("ok") is not None]
    if ag:
        L.append(f"Comprobación con los táctiles: {sum(bool(e['ok']) for e in ag)}/{len(ag)} la notan")

    def por(nombre, valor, cortes):
        filas = {}
        for k, c in validos.items():
            fin = next((e for e in reversed(intentos[k]) if e["evento"] == "fin"), {})
            x = valor(fin)
            if x is None:
                continue
            tramo = next((t for t, (a, b) in cortes.items() if a <= x < b), None)
            if tramo:
                filas.setdefault(tramo, []).append(c.startswith("éxito"))
        if len(filas) > 1:
            L.append(f"Por {nombre}: " + " · ".join(f"{t} {sum(v)}/{len(v)}" for t, v in filas.items()))

    por("distancia a la cámara", lambda f: f.get("distancia_camara_m"),
        OrderedDict([("< 0,8 m", (0, 0.8)), ("0,8-1 m", (0.8, 1.0)), ("> 1 m", (1.0, 9))]))
    por("zona de la imagen", lambda f: (f["centro_camara"][0] / f["centro_camara"][2]) if f.get("centro_camara") else None,
        OrderedDict([("izquierda", (-9, -0.2)), ("centro", (-0.2, 0.2)), ("derecha", (0.2, 9))]))
    return L


def main():
    ap = argparse.ArgumentParser(description="Resumen de los intentos del modo autónomo")
    ap.add_argument("--archivo", default=os.path.join(RAIZ, "diagnosticos", "intentos.jsonl"))
    ap.add_argument("--desde", default=None, help="AAAA-MM-DD[THH:MM]: solo desde ahí")
    ap.add_argument("--simulacro", action="store_true", help="contar también los de simulacro")
    a = ap.parse_args()
    print("\n".join(resumir(leer(a.archivo, a.desde, a.simulacro))))


if __name__ == "__main__":
    main()
