#!/usr/bin/env python3
"""
Prueba de la háptica REAL: yemas de la OrcaHand Touch física → vibración del MANUS
==================================================================================
Sin simulación ni cámaras: arranca el guante MANUS y el puente de los sensores táctiles y
hace vibrar cada dedo del guante según lo que mide la yema del mismo dedo de la mano física.

    python diagnosticos/probar_haptica_tactil.py             # en vivo: aprieta las yemas de la OrcaHand
    python diagnosticos/probar_haptica_tactil.py --dedos     # solo el guante: vibra cada dedo por turno
                                                            # (comprueba el orden pulgar → meñique)
    python diagnosticos/probar_haptica_tactil.py --umbral    # solo el guante: sube la vibración del índice
                                                            # poco a poco → apunta el 1er valor que notes y
                                                            # ponlo en MANUS_TACTIL_VIBRACION_MIN
    python diagnosticos/probar_haptica_tactil.py --umbral --dedo pulgar
    python diagnosticos/probar_haptica_tactil.py --mock      # sensores simulados (sin la mano física)
    python diagnosticos/probar_haptica_tactil.py --puerto /dev/ttyACM0   # puerto fijo de los sensores

En vivo:  Enter = nuevo cero de los sensores (sin tocar nada) · Ctrl+C = salir.
La tabla muestra, por dedo, la fuerza filtrada (N) y la vibración (%). Con ella se ajustan
MANUS_TACTIL_F_ON (umbral) y MANUS_TACTIL_F_MAX (100 %) en config.py.
"""
import argparse
import os
import select
import sys
import time

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del repo (este script está en diagnosticos/)
sys.path.insert(0, RAIZ)

from skill_shield.hardware import tactil_real                # noqa: E402
from skill_shield.mano import manus_guante                   # noqa: E402
from skill_shield.mano.manus_guante import GuanteManus       # noqa: E402

NOMBRES = ("pulgar", "índice", "corazón", "anular", "meñique")
NOMBRES_CLAVE = ("pulgar", "indice", "corazon", "anular", "menique")


def esperar_guante(g, segundos=15.0):
    t0 = time.time()
    while time.time() - t0 < segundos:
        if g.conectado and g.es_haptico():
            return True
        time.sleep(0.2)
    return False


def prueba_dedos(g):
    print("Vibrando cada dedo 1 s, de pulgar a meñique...")
    for i, n in enumerate(NOMBRES):
        p = [0.0] * 5
        p[i] = 0.8
        print(f"  → {n}")
        g.vibrar(p)
        time.sleep(1.0)
        g.vibrar([0.0] * 5)
        time.sleep(0.4)
    print("Hecho. Si no vibró el dedo que pone, avísame.")


def prueba_umbral(g, dedo):
    i = NOMBRES_CLAVE.index(dedo)
    print(f"Vibración del {NOMBRES[i]}, de menos a más (1.5 s cada valor). Apunta el PRIMER valor "
          "que notes y ponlo en config.py como MANUS_TACTIL_VIBRACION_MIN. Ctrl+C para parar.")
    for k in range(1, 26):
        v = 0.02 * k
        p = [0.0] * 5
        p[i] = v
        print(f"  {v:.2f}", end="", flush=True)
        g.vibrar(p)
        time.sleep(1.5)
        g.vibrar([0.0] * 5)
        time.sleep(0.6)
        print()
    print("Hecho (hasta 0.50).")


def en_vivo(g):
    print("Aprieta las yemas de la OrcaHand. Enter = nuevo cero · Ctrl+C = salir\n")
    print("  ".join(f"{n:^15s}" for n in NOMBRES) + "   estado")
    teclado = sys.stdin.isatty()
    while True:
        with g._lock:
            f = np.array(g.tacto_fuerza, dtype=float)
            v = np.array(g.tacto_vibracion, dtype=float)
        if g.fuente_activa != "real":
            f[:], v[:] = 0.0, 0.0
        celdas = "  ".join(f"{fi:5.2f} N {100 * vi:4.0f}%" for fi, vi in zip(f, v))
        estado = g.tactil.estado_txt() if g.tactil else "TACTIL: no iniciado"
        if not g.conectado:
            estado += " · MANUS conectando..."
        elif not g.es_haptico():
            estado += " · guante SIN háptica"
        print("\r" + celdas + "   " + estado + "      ", end="", flush=True)
        if not teclado:
            time.sleep(0.1)
        elif select.select([sys.stdin], [], [], 0.1)[0]:
            sys.stdin.readline()
            ok = g.tactil is not None and g.tactil.cero()
            print("\n" + ("cero pedido: no toques nada 2 s" if ok else "no hay puente con el que hablar"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dedos", action="store_true", help="solo vibrar cada dedo por turno")
    ap.add_argument("--umbral", action="store_true", help="buscar la vibración mínima que notas")
    ap.add_argument("--dedo", default="indice", choices=NOMBRES_CLAVE, help="dedo para --umbral")
    ap.add_argument("--mock", action="store_true", help="sensores simulados")
    ap.add_argument("--puerto", default=None, help="puerto de los sensores táctiles")
    a = ap.parse_args()

    extra = list(tactil_real.ORCA_TACTIL_ARGS_PUENTE)
    if a.mock:
        extra.append("--mock")
    if a.puerto:
        extra += ["--puerto", a.puerto]
    tactil_real.ORCA_TACTIL_ARGS_PUENTE = extra

    solo_guante = a.dedos or a.umbral
    g = GuanteManus(tactil=not solo_guante, fuente="real")   # solo los sensores físicos
    g.haptica_siempre = True                # vibra aunque no haya simulación mandando
    if not g.iniciar():
        return 1
    try:
        print("⏳ Esperando al guante MANUS...")
        if not esperar_guante(g):
            print("❌ El guante no aparece o no tiene háptica (¿dongle, guante encendido, "
                  "MANUS_MODO en config.py?). Sigo: verás las fuerzas pero no vibrará.")
        if a.umbral:
            prueba_umbral(g, a.dedo)
        elif a.dedos:
            prueba_dedos(g)
        else:
            en_vivo(g)
    except KeyboardInterrupt:
        print()
    finally:
        g.cerrar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
