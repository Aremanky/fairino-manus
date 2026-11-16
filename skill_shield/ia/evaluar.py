"""
Evaluación de modelos sin robot ni MuJoCo (16-nov).

ENSAYO EN LAZO CERRADO («cinemático»): se reproduce la política sobre un episodio que NO ha visto
al entrenar como si el robot siguiera lo que se le manda (después del filtro de seguridad, con su
velocidad máxima y con el retraso de seguimiento que se mide en el propio episodio). La red ve SUS
PROPIAS posturas, no las de la demo, así que se notan los errores que se acumulan, si se queda
quieta al terminar y si sabe que ha terminado. Los táctiles y el objeto son los de la demo (no se
pueden simular aquí). Al final compara con la demo: dónde acaba la palma (cm) y los dedos (rad).
Es mejor criterio que la pérdida de validación para elegir checkpoint (act.py lo usa así); el
definitivo sigue siendo el robot (o MuJoCo, cuando esté el entorno de la pelota).

    python -m skill_shield.ia.evaluar <modelo>                 # con los episodios de validación
    python -m skill_shield.ia.evaluar <modelo> --todos         # con todos los de sus tareas
    python -m skill_shield.ia.evaluar <modelo> --checkpoints   # compara sus checkpoints guardados
"""
import argparse
import os

import numpy as np

from skill_shield.ia.episodios import crudo_de_episodio
from skill_shield.ia.politica import FiltroSeguridad, Politica

PROGRESO_FIN = 0.97


def seguimiento(q, a, minimo=0.05):
    """Cuánto se acerca cada fotograma la postura al comando: q[t+1] − q[t] ≈ α · (a[t] − q[t]),
    ajustado por mínimos cuadrados con el propio episodio (1 = llega en un fotograma)."""
    q, a = np.asarray(q, float), np.asarray(a, float)
    if q.size == 0 or len(q) < 3:
        return 1.0
    d, e = (q[1:] - q[:-1]).ravel(), (a[:-1] - q[:-1]).ravel()
    den = float(e @ e)
    return float(np.clip((d @ e) / den, minimo, 1.0)) if den > 1e-9 else 1.0


def _palma(cadena, q):
    return cadena.fk(np.asarray(q, float))["palma"]


def ensayo(red, ep, tarea=0, cadena=None, pasos_extra=30, progreso_fin=PROGRESO_FIN, v_max_brazo=1.0,
           peso_exp=0.15, lim_brazo=None, lim_mano=None):
    """Un episodio en lazo cerrado. → dict con ok, error, err_palma_cm, err_dedos_rad, terminado,
    pasos, desvio_max_cm (lo más lejos que se fue la palma de la de la demo) y puntuación (menor = mejor)."""
    receta = red["receta"]
    crudo = crudo_de_episodio(ep, receta)
    T = int(crudo["T"])
    pol = Politica(red, tarea=tarea, cadena=cadena, peso_exp=peso_exp)
    con_brazo = pol.mueve_brazo and crudo["q_brazo"].shape[1] == 6
    if con_brazo and pol.cadena is None:
        import config
        from skill_shield.teleop_fr5.cinematica import CadenaFR5
        pol.cadena = CadenaFR5.desde_mjcf(config.RUTA_XML)
    cad = pol.cadena
    a_b, a_m = np.asarray(crudo["a_brazo"], float), np.asarray(crudo["a_mano"], float)
    al_b = seguimiento(crudo["q_brazo"], a_b) if con_brazo else 1.0
    al_m = seguimiento(crudo["q_mano"], a_m)
    q_b = np.array(crudo["q_brazo"][0], float)
    q_m = np.array(crudo["q_mano"][0], float)
    tac, obj = crudo.get("tactil"), crudo.get("objeto")

    def obs(k):
        return dict(q_brazo=q_b, q_mano=q_m, tactil=None if tac is None else tac[min(k, T - 1)], objeto=obj)

    if not pol.reiniciar(obs(0)):
        return dict(ok=False, error=pol.error, puntuacion=100.0, terminado=False, pasos=0, T=T)
    filtro = FiltroSeguridad(lim_brazo=lim_brazo, lim_mano=lim_mano, v_max_brazo=v_max_brazo, dt=pol.dt,
                             salto_max=0.35 if pol.usa_ik else None,
                             max_dist_objeto=0.5 if pol.usa_ik else None)
    palma_demo = np.stack([_palma(cad, q) for q in a_b]) if con_brazo else None
    brazo_cmd = q_b.copy() if con_brazo else None
    mano_cmd = q_m.copy()
    error, terminado_en, desvio, k = None, None, 0.0, 0
    for k in range(T + int(pasos_extra)):
        p = pol.paso(obs(k))
        cmd = filtro.aplicar(p, brazo_cmd, mano_cmd)
        if cmd.parar:
            error = cmd.motivo
            break
        if con_brazo and cmd.brazo is not None:
            brazo_cmd = cmd.brazo
            q_b = q_b + al_b * (brazo_cmd - q_b)          # el robot se acerca a lo que se le manda
            desvio = max(desvio, float(np.linalg.norm(_palma(cad, q_b) - palma_demo[min(k, T - 1)])))
        if cmd.mano is not None:
            mano_cmd = cmd.mano
            q_m = q_m + al_m * (mano_cmd - q_m)
        if p.progreso is not None and p.progreso >= progreso_fin:
            terminado_en = k
            break
    err_palma = float(np.linalg.norm(_palma(cad, q_b) - palma_demo[-1])) * 100.0 if con_brazo else 0.0
    err_dedos = float(np.mean(np.abs(q_m - a_m[-1])))
    con_progreso = pol.ultimo is not None and pol.ultimo.progreso is not None
    terminado = terminado_en is not None
    puntuacion = err_palma + 10.0 * err_dedos + (3.0 if con_progreso and not terminado else 0.0) \
        + (20.0 if error else 0.0)
    return dict(ok=error is None, error=error, err_palma_cm=round(err_palma, 2), err_dedos_rad=round(err_dedos, 4),
                terminado=terminado, pasos=k + 1, T=T, desvio_max_cm=round(100.0 * desvio, 2),
                seguimiento=(round(al_b, 3), round(al_m, 3)), puntuacion=round(puntuacion, 3))


def evaluar(red, episodios, tarea_de=None, cadena=None, **kw):
    """Ensayo de cada episodio → (puntuación media, lista de resultados)."""
    tarea_de = list(tarea_de) if tarea_de is not None else [0] * len(episodios)
    res = [ensayo(red, ep, tarea=t, cadena=cadena, **kw) for ep, t in zip(episodios, tarea_de)]
    return (float(np.mean([r["puntuacion"] for r in res])) if res else float("inf")), res


def resumen(res):
    """Una línea con las medias de una lista de resultados de ensayo()."""
    if not res:
        return "sin episodios"
    ok = sum(r["ok"] for r in res)
    term = sum(bool(r.get("terminado")) for r in res)
    med = lambda k: np.mean([r.get(k, 0.0) for r in res if r.get(k) is not None])   # noqa: E731
    return (f"{ok}/{len(res)} sin parar · {term}/{len(res)} saben que han terminado · palma al final "
            f"{med('err_palma_cm'):.1f} cm · dedos {np.degrees(med('err_dedos_rad')):.1f}° · "
            f"desvío máx. {med('desvio_max_cm'):.1f} cm")


def _main(argv=None):
    from skill_shield.ia import act
    from skill_shield.ia.episodios import exito, leer_episodio, listar_episodios
    ap = argparse.ArgumentParser(description="Ensayo en lazo cerrado de un modelo entrenado")
    ap.add_argument("modelo", help="nombre del modelo (modelos/<modelo>.pth) o ruta al .pth")
    ap.add_argument("--todos", action="store_true", help="todos los episodios de sus tareas, no solo validación")
    ap.add_argument("--checkpoints", action="store_true", help="comparar los checkpoints de modelos/<modelo>/")
    a = ap.parse_args(argv)
    ruta = a.modelo if a.modelo.endswith(".pth") else os.path.join(act.MODELOS_DIR, f"{a.modelo}.pth")
    red = act.cargar_ia(ruta)
    if red is None:
        return 1
    receta = red["receta"]
    val = set(red.get("info", {}).get("episodios_val") or [])
    eps, tarea_de = [], []
    for ti, t in enumerate(receta.get("tareas") or []):
        for r in listar_episodios(os.path.join(act.EPISODIOS_DIR, t)):
            if not a.todos and val and f"{t}/{os.path.basename(r)}" not in val:
                continue
            ep = leer_episodio(r)
            if ep is not None and exito(ep):
                eps.append(ep)
                tarea_de.append(ti)
    if not eps:
        print("❌ No hay episodios para evaluar (¿modelo antiguo sin lista de validación? prueba --todos).")
        return 1
    print(f"🧪 {len(eps)} episodio(s){'' if a.todos or not val else ' de validación'}")
    rutas = [ruta]
    if a.checkpoints:
        d = ruta[:-4]
        rutas += sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pth")) if os.path.isdir(d) else []
    for r in rutas:
        red_r = red if r == ruta else act.cargar_ia(r, silencioso=True)
        if red_r is None:
            continue
        p, res = evaluar(red_r, eps, tarea_de)
        print(f"   {os.path.basename(r):28s} puntuación {p:7.2f} · {resumen(res)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
