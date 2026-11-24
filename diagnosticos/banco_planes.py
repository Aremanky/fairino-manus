"""
Banco de pruebas del planificador de la tarea (18-nov), con los planes GUARDADOS de verdad.

Cada vez que se planifica en el modo autónomo se guarda lo que se usó (pose, sitios, mapa, mesa…) en
diagnosticos/plan_ok_*.npz y diagnosticos/plan_fallido_*.npz. Este script los vuelve a planificar TODOS
con el código y el config.py de ahora y dice cuántos salen y cuánto tardan. Así cada cambio del
planificador se puede comparar con el anterior con datos, no a ojo:

    python diagnosticos/banco_planes.py                          # diagnosticos/ + tests/datos/
    python diagnosticos/banco_planes.py --guardar antes.json     # guardar los resultados…
    (cambiar algo)
    python diagnosticos/banco_planes.py --comparar antes.json    # …y compararlos con los de ahora
    python diagnosticos/banco_planes.py diagnosticos/plan_fallido_20261118_*.npz -v   # solo esos, con detalle

Sin robot ni cámara (con la GPU, si cuRobo está instalado y AUTONOMO_TAREA_PLANIFICADOR = "curobo").
Ojo: la escena fija que se usa es la de AHORA (escena_fija.json), no la del momento del plan.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)


def casos(rutas):
    if not rutas:
        rutas = [os.path.join(RAIZ, "diagnosticos"), os.path.join(RAIZ, "tests", "datos")]
    out = []
    for r in rutas:
        if os.path.isdir(r):
            out += sorted(glob.glob(os.path.join(r, "plan_*.npz")))
        else:
            out += sorted(glob.glob(r))
    return out


def planificar(pl, d, info):
    """Lo mismo que Tarea._planificar con los datos guardados. Devuelve el plan (o lanza ErrorPlan)."""
    excluir = [(e[:3], float(e[3])) for e in np.asarray(d["excluir"], float).reshape(-1, 4)]
    mapa = np.asarray(d["mapa"], float).reshape(-1, 3)
    pl.poner_mapa(mapa if len(mapa) else None, float(d["res"]), excluir)
    q = np.asarray(d["q"], float)
    T = np.asarray(d["T"], float).reshape(-1, 4, 4)
    if info.get("volver"):
        return pl.plan_volver(q, np.asarray(d["q_inicio"], float))
    if info.get("pelota"):
        kw = {}
        if info.get("apoyo_z") is not None:
            kw.update(apoyo_z=info["apoyo_z"], apoyo_poligono=info.get("apoyo_poligono"))
        if info.get("T_mundo_cam") is not None:
            kw["T_mundo_cam"] = np.asarray(info["T_mundo_cam"], float)
        return pl.planificar_pelota(q, T[0], float(info.get("diametro") or 0.07), T[1], **kw)
    return pl.planificar(q, *T[:3])


def main():
    ap = argparse.ArgumentParser(description="Vuelve a planificar los planes guardados y compara")
    ap.add_argument("rutas", nargs="*", help="archivos .npz o carpetas (por defecto diagnosticos/ y tests/datos/)")
    ap.add_argument("--guardar", help="guardar los resultados en este JSON")
    ap.add_argument("--comparar", help="JSON de una ejecución anterior (--guardar) para comparar")
    ap.add_argument("-v", "--detalle", action="store_true", help="el error completo de cada fallo")
    a = ap.parse_args()

    import config
    from skill_shield.autonomo import tarea as TA
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    G = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    lista = casos(a.rutas)
    if not lista:
        sys.exit("No hay planes guardados (diagnosticos/plan_*.npz).")
    T_cam = None
    ruta_cal = os.path.join(RAIZ, G.get("CALIB_ROBOT_ARCHIVO", "calibraciones/datos/camara_robot.json"))
    if os.path.exists(ruta_cal):
        from skill_shield.autonomo.calibracion import cargar
        T_cam = cargar(ruta_cal)["T_mundo_cam"]
    t0 = time.perf_counter()
    pl = TA.Planificador(cargar_cadena(G["RUTA_XML"], verbose=False), G, T_cam)
    print(f"Planificador listo en {time.perf_counter() - t0:.1f} s · mundo {'simple' if pl.mundo_simple else 'completo'}"
          f" · cuRobo {'sí' if pl.curobo is not None else 'no'} · GPU en lote {'sí' if pl._lote_torch is not None else 'no'}"
          f" · {len(lista)} planes\n")
    res = {}
    for ruta in lista:
        nombre = os.path.basename(ruta)
        d = np.load(ruta, allow_pickle=False)
        info = json.loads(str(d["info"]))
        antes = "fallido" if info.get("error") else "ok"
        t = time.perf_counter()
        try:
            plan = planificar(pl, d, info)
            seg = time.perf_counter() - t
            dur = TA.Ejecutor(plan, dict(abierta=TA.MANO_ABIERTA, agarre=TA.MANO_AGARRE, pelota_abierta=TA.MANO_PELOTA_ABIERTA,
                                         pelota_cerrada=TA.MANO_PELOTA_CERRADA, pelota_apretar=TA.MANO_PELOTA_APRETAR),
                              np.radians(G.get("AUTONOMO_TAREA_VEL_DEG_S", 15.0)),
                              np.radians(G.get("AUTONOMO_TAREA_ACEL_DEG_S2", 30.0)), fk_palma=pl._palma).duracion
            r = dict(ok=True, s=round(seg, 2), fases=len(plan.fases), duracion_s=round(dur, 1),
                     curobo=pl.tramos_curobo, rechazos=pl.curobo_rechazos, por_alto=pl.tramos_por_alto,
                     rodeos=pl.rodeos, avisos=list(pl.avisos), antes=antes)
            txt = (f"✅ {seg:6.1f} s  {len(plan.fases):2d} fases, {dur:5.1f} s de movimiento"
                   + (f", cuRobo {pl.tramos_curobo} (rechazados {pl.curobo_rechazos})" if pl.curobo is not None else "")
                   + (f", por alto {pl.tramos_por_alto}" if pl.tramos_por_alto else "")
                   + (f", rodeos {pl.rodeos}" if pl.rodeos else ""))
        except TA.ErrorPlan as e:
            seg = time.perf_counter() - t
            r = dict(ok=False, s=round(seg, 2), error=str(e), antes=antes, avisos=list(pl.avisos))
            txt = f"❌ {seg:6.1f} s  " + (str(e) if a.detalle else str(e)[:110] + ("…" if len(str(e)) > 110 else ""))
        res[nombre] = r
        print(f"{nombre:<42s} (antes {antes:<7s}) {txt}")
    ok = [r for r in res.values() if r["ok"]]
    ts = np.array([r["s"] for r in res.values()])
    print(f"\n{len(ok)}/{len(res)} salen · tiempo medio {ts.mean():.1f} s · máx {ts.max():.1f} s · "
          f"de los que fallaron al guardarse, ahora salen {sum(1 for r in ok if r['antes'] == 'fallido')}")
    if a.comparar:
        ant = json.load(open(a.comparar, encoding="utf-8"))
        comunes = [k for k in res if k in ant]
        mejor = [k for k in comunes if res[k]["ok"] and not ant[k]["ok"]]
        peor = [k for k in comunes if not res[k]["ok"] and ant[k]["ok"]]
        t_a = np.array([ant[k]["s"] for k in comunes])
        t_n = np.array([res[k]["s"] for k in comunes])
        print(f"Comparado con {a.comparar} ({len(comunes)} planes en los dos): ahora salen {len(mejor)} más y "
              f"{len(peor)} menos · tiempo medio {t_a.mean():.1f} → {t_n.mean():.1f} s")
        for k in peor:
            print(f"   ⚠️  ya no sale: {k}: {res[k].get('error', '')[:120]}")
    if a.guardar:
        with open(a.guardar, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"Resultados guardados en {a.guardar}")


if __name__ == "__main__":
    main()
