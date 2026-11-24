#!/usr/bin/env python3
"""
Pruebas de cuRobo con el FR5 + OrcaHand, SIN mover el robot (17-nov). En el PC con la GPU:

    python diagnosticos/probar_curobo.py               # todo
    python diagnosticos/probar_curobo.py --n 50        # más tramos al azar

1. La GPU, PyTorch y cuRobo.
2. El URDF tiene la misma cinemática que la cadena del proyecto.
3. Tramos al azar en un mundo con una cajonera: % que cuRobo resuelve, cuánto tarda y % que pasan las
   comprobaciones del proyecto (mano, J4, soporte, mapa…).
4. El plan REAL del 17-nov (tests/datos/plan_pelota_pulgar_17nov.npz) con cuRobo y con el planificador
   propio: tiempo, tramos de cuRobo aceptados y holgura de la mano a la mesa.
"""
import argparse
import os
import sys
import time

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del repo (este script está en diagnosticos/)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tests"))


def titulo(t):
    print(f"\n━━━ {t} ━━━")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="tramos al azar")
    a = ap.parse_args()
    import config
    G = {k: getattr(config, k) for k in dir(config) if k.isupper()}

    titulo("1. GPU, PyTorch y cuRobo")
    import torch
    print(f"torch {torch.__version__} · CUDA {torch.version.cuda} · GPU: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO'} · "
          f"sm_120: {'sí' if torch.cuda.is_available() and 'sm_120' in torch.cuda.get_arch_list() else 'NO'}")
    import curobo
    print(f"cuRobo {curobo.__version__}")
    ruta_robot = os.path.join(RAIZ, G.get("AUTONOMO_CUROBO_ROBOT", "curobo_fr5/fr5_orca.yml"))
    if not os.path.exists(ruta_robot):
        sys.exit(f"✗ Falta {ruta_robot}: python -m skill_shield.autonomo.curobo_fr5.robot")

    titulo("2. URDF = cadena del proyecto")
    from skill_shield.autonomo.curobo_fr5.urdf import CADENA, HERRAMIENTA, fk_urdf
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    cad = cargar_cadena(config.RUTA_XML, verbose=False)
    urdf = os.path.join(os.path.dirname(ruta_robot), "fr5_orca.urdf")
    rng = np.random.default_rng(0)
    peor = max(float(np.abs(fk_urdf(urdf, q)[l] - cad.fk_marcos(q)[k]).max())
               for q in rng.uniform(-3, 3, (30, 6)) for k, l in enumerate(CADENA[1:] + (HERRAMIENTA,)))
    print(f"diferencia máxima: {peor:.1e} {'✓' if peor < 1e-6 else '✗ NO COINCIDE'}")

    titulo("3. Tramos al azar con una cajonera")
    from skill_shield.autonomo import tarea as TA
    from test_tarea import Q0, _cajonera
    t0 = time.perf_counter()
    pl = TA.Planificador(cad, dict(G, AUTONOMO_TAREA_PLANIFICADOR="curobo"))
    if pl.curobo is None:
        sys.exit("✗ cuRobo no arranca (mira el aviso de arriba)")
    print(f"planificador con cuRobo creado en {time.perf_counter() - t0:.1f} s")
    pl.poner_mapa(_cajonera(), 0.02)
    q0 = pl._empezar(Q0)
    validas = []
    while len(validas) < 2 * a.n:
        q = Q0 + rng.normal(0, 0.5, 6)
        if pl.motivo(q) is None:
            validas.append(q)
    ok = acept = 0
    tiempos = []
    for i in range(a.n):
        qa, qb = validas[2 * i], validas[2 * i + 1]
        pl.curobo.actualizar_mundo(pl, clave=("azar", pl._version_mundo))
        qs = pl.curobo.plan(qa, qb)
        tiempos.append(pl.curobo.t_ultimo)
        if qs is None:
            continue
        ok += 1
        try:
            d = pl._densificar(qs)
            if all(pl.motivo(q) is None for q in d[1:]):
                pl.comprobar_continuo(d)
                acept += 1
        except TA.ErrorPlan:
            pass
    print(f"cuRobo resuelve {ok}/{a.n} · pasan las comprobaciones del proyecto {acept}/{ok} · "
          f"tiempo mediano {1000 * np.median(tiempos):.0f} ms (máx {1000 * max(tiempos):.0f} ms)")

    titulo("4. El plan real del 17-nov (pelota + dejada)")
    from skill_shield.autonomo import calibracion as C
    d = np.load(os.path.join(RAIZ, "tests", "datos", "plan_pelota_pulgar_17nov.npz"))
    T = C.cargar(os.path.join(RAIZ, G.get("CALIB_ROBOT_ARCHIVO", "calibraciones/datos/camara_robot.json")))["T_mundo_cam"]
    for nombre, modo in (("propio", "propio"), ("cuRobo", "curobo")):
        p = TA.Planificador(cad, dict(G, AUTONOMO_TAREA_PLANIFICADOR=modo), T)
        p.poner_mapa(d["mapa"], float(d["res"]), [(e[:3], e[3]) for e in d["excluir"]])
        t0 = time.perf_counter()
        try:
            plan = p.planificar_pelota(d["q"], d["T"][0], 0.07, d["T"][1])
        except TA.ErrorPlan as e:
            print(f"{nombre:7s}: SIN PLAN ({e})")
            continue
        dur = TA.Ejecutor(plan, {"abierta": TA.MANO_ABIERTA}, np.radians(15), np.radians(30)).duracion
        print(f"{nombre:7s}: {time.perf_counter() - t0:5.1f} s planificando · {dur:4.0f} s de movimiento · "
              f"tramos de cuRobo aceptados {getattr(p, 'tramos_curobo', 0)}, rechazados {getattr(p, 'curobo_rechazos', 0)}")
    print("\n✅ Si todo sale con ✓ y los tramos de cuRobo se aceptan, prueba el modo autónomo en SIMULACRO.")


if __name__ == "__main__":
    main()
