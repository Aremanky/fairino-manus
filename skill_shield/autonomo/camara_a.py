"""
Cámara A del modo autónomo (19-nov): una OAK-D FIJA mirando la mesa de cerca.

La cámara de cabeza ve todo el espacio, pero la pelota le queda a ~0,9 m y el error del estéreo (que va a lo largo
del rayo, y crece con el cuadrado de la distancia) se come los últimos cm del agarre. Además, al pararse encima de
la pelota para afinar, el antebrazo suele taparla (está detrás del robot). La cámara A, fija en la mesa, mira la
zona de agarre a ~0,4-0,5 m y desde otro lado:

  · Proceso propio: el mismo skill_shield/autonomo/camara.py con --rol a (sin mapa de obstáculos: el mapa lo sigue
    haciendo solo la cabeza), con su calibración cámara→robot (AUTONOMO_CAM_A_CALIB_ARCHIVO, la saca
    calibrar_camara_robot.py --camara a) y sus canales (imagen, info y lista de pelotas/herramientas).
  · Tarea (tarea.py): si la cámara A ve la MISMA pelota que la cabeza (a ≤ AUTONOMO_CAM_A_MAX_DIF_M), estable y
    reciente, se planifica con la medida de la A, se desliza por SU rayo al apoyarla en la mesa y se afina con ella
    al llegar encima (lo que tape el brazo se mira desde la A). Si la A no la ve, todo sigue como antes (cabeza).

Esta parte no tiene hardware: se puede probar sola (tests/test_camara_a.py).
"""
import json
import time

import numpy as np

CANAL_IMG_A = "skill_autonomo_img_a"
CANAL_INFO_A = "skill_autonomo_caminfo_a"
CANAL_HERR_A = "skill_autonomo_herramientas_a"


def centro(h):
    return np.asarray(h["T_mundo"], float)[:3, 3]


def pelota_de_a(lista_a, c_ref, max_dif, edad_max=0.5, id_pref=None):
    """La pelota de la lista de la cámara A que corresponde a la que está en c_ref (centro en el mundo, el de la
    cabeza o el del plan): la más cercana, si está a ≤ max_dif m, es estable y se ha visto hace ≤ edad_max s.
    id_pref: el id que tenía en la A (si sigue ahí y vale, esa). None si la A no la ve (o no se puede fiar)."""
    if c_ref is None:
        return None
    c_ref = np.asarray(c_ref, float)
    vale = [h for h in (lista_a or []) if h.get("tipo") == "pelota" and h.get("T_mundo") is not None
            and h.get("estable") and float(h.get("edad_s") or 0.0) <= float(edad_max)]
    if not vale:
        return None
    if id_pref is not None:
        h = next((h for h in vale if h.get("id") == id_pref), None)
        if h is not None and np.linalg.norm(centro(h) - c_ref) <= float(max_dif):
            return h
    h = min(vale, key=lambda h: float(np.linalg.norm(centro(h) - c_ref)))
    return h if float(np.linalg.norm(centro(h) - c_ref)) <= float(max_dif) else None


def leer_canal(canal, max_edad_s=2.0, ahora=None):
    """(lista, T_mundo_cam) del canal de la cámara A, o ([], None) si no hay datos recientes o no está calibrada."""
    if canal is None:
        return [], None
    b, t = canal.leer()
    ahora = time.time() if ahora is None else ahora
    if not b or t is None or ahora - t > max_edad_s:
        return [], None
    try:
        d = json.loads(b)
    except ValueError:
        return [], None
    T = d.get("T_mundo_cam")
    return d.get("lista", []), (None if T is None else np.asarray(T, float))
