"""
Cámara A del modo autónomo (19-nov): una OAK-D FIJA mirando la mesa de cerca.

La cámara de cabeza ve todo el espacio, pero la pelota le queda a ~0,9 m y el error del estéreo (que va a lo largo
del rayo, y crece con el cuadrado de la distancia) se come los últimos cm del agarre. Además, al pararse encima de
la pelota para afinar, el antebrazo suele taparla (está detrás del robot). La cámara A, fija en la mesa, mira la
zona de agarre a ~0,4-0,5 m y desde otro lado:

  · Proceso propio: el mismo skill_shield/autonomo/camara.py con --rol a (sin mapa de obstáculos: el mapa lo sigue
    haciendo solo la cabeza), con su calibración cámara→robot (AUTONOMO_CAM_A_CALIB_ARCHIVO, la saca
    calibraciones/calibrar_camara_robot.py --camara a) y sus canales (imagen, info y lista de pelotas/herramientas).
  · Tarea (tarea.py): si la cámara A ve la MISMA pelota que la cabeza (a ≤ AUTONOMO_CAM_A_MAX_DIF_M), estable y
    reciente, se planifica con la medida de la A, se desliza por SU rayo al apoyarla en la mesa y se afina con ella
    al llegar encima (lo que tape el brazo se mira desde la A). Si la A no la ve, todo sigue como antes (cabeza).

  · (20-nov) También alimenta el MAPA de obstáculos: manda su profundidad ya preparada (sin sus pelotas, sin lo poco
    fiable, a la mitad de resolución) por CANAL_PROF_A y el proceso de la cabeza la funde en la misma TSDF
    (mapa.ConstructorMapa._fundir_a). Ver AUTONOMO_CAM_A_AL_MAPA.

Esta parte no tiene hardware: se puede probar sola (tests/test_camara_a.py, tests/test_mapa_20nov.py).
"""
import json
import struct
import time

import numpy as np

CANAL_IMG_A = "skill_autonomo_img_a"
CANAL_INFO_A = "skill_autonomo_caminfo_a"
CANAL_HERR_A = "skill_autonomo_herramientas_a"
CANAL_PROF_A = "skill_autonomo_prof_a"          # (20-nov) profundidad de la A para el mapa de la cabeza


def distancia_minima(mono_800p, extendida):
    """(20-nov) Distancia (m) por debajo de la cual la OAK-D Pro no mide con ese modo del estéreo (con el mismo margen
    que el AUTONOMO_MAPA_Z_MIN de siempre, 0,35 m a 400P): 95 píxeles de disparidad como mucho (190 con la
    extendida) y la focal en píxeles doble a 800P."""
    z = 0.70 if mono_800p else 0.35
    return z / 2.0 if extendida else z


# ── (20-nov) profundidad de la A → mapa de la cabeza ──
_CAB_PROF = struct.Struct("<I")


def preparar_prof(prof_mm, K, img_bgr=None, halos=None, T_mundo_cam=None, paso=2, z_min=0.0, libre_halos=True):
    """La profundidad de la A lista para el mapa: (profundidad reducida en mm uint16, K reducida, máscara reducida de
    los píxeles de sus pelotas o None). Lo poco fiable (píxeles voladores, saturados) y lo demasiado cerca, a 0.
    halos: los de DetectorPelotas.pixeles_halo() de la A: con libre_halos van en la máscara (en el mapa dicen «libre
    hasta aquí», sin superficie); sin él, a 0 (como antes)."""
    from skill_shield.autonomo.mapa import ConstructorMapa, pixeles_fiables
    prof = np.asarray(prof_mm)
    prof = prof.astype(np.uint16) if prof.dtype != np.uint16 else prof.copy()
    mascara = None
    if halos:
        m = ConstructorMapa.mascara_halos(prof, halos, K, T_mundo_cam)
        if libre_halos:
            mascara = m
        else:
            prof[m] = 0
    ok = pixeles_fiables(prof, img_bgr)
    if z_min:
        ok &= prof > int(round(float(z_min) * 1000.0))
    prof[~ok] = 0
    paso = max(1, int(paso))
    Ks = np.array(K, float).copy()
    Ks[0, 0] /= paso; Ks[0, 2] /= paso; Ks[1, 1] /= paso; Ks[1, 2] /= paso
    sub = np.ascontiguousarray(prof[::paso, ::paso])
    msub = None if mascara is None else np.ascontiguousarray(mascara[::paso, ::paso] & (sub > 0))
    return sub, Ks, msub


def empaquetar_prof(prof_mm, K, T_mundo_cam, mascara=None, t=None, dist_max=None):
    """Bytes del canal: cabecera JSON (K, T_mundo_cam, tamaño, tiempo, dist_max) + profundidad uint16 + máscara en bits."""
    prof = np.ascontiguousarray(prof_mm, np.uint16)
    h, w = prof.shape
    cab = json.dumps(dict(K=np.asarray(K, float).round(6).tolist(),
                          T=np.asarray(T_mundo_cam, float).round(7).tolist(), h=int(h), w=int(w),
                          t=float(time.time() if t is None else t), mascara=mascara is not None,
                          dist_max=None if dist_max is None else float(dist_max))).encode()
    partes = [_CAB_PROF.pack(len(cab)), cab, prof.tobytes()]
    if mascara is not None:
        partes.append(np.packbits(np.asarray(mascara, bool).ravel()).tobytes())
    return b"".join(partes)


def desempaquetar_prof(b):
    """dict(prof (h x w uint16 mm), K, T_mundo_cam, mascara (h x w bool) o None, t, dist_max) o None si no vale."""
    try:
        n = _CAB_PROF.unpack_from(b, 0)[0]
        d = json.loads(bytes(b[4:4 + n]))
        h, w = int(d["h"]), int(d["w"])
        i = 4 + n
        prof = np.frombuffer(b, np.uint16, h * w, i).reshape(h, w)
        i += 2 * h * w
        m = None
        if d.get("mascara"):
            m = np.unpackbits(np.frombuffer(b, np.uint8, (h * w + 7) // 8, i))[:h * w].reshape(h, w).astype(bool)
        return dict(prof=prof, K=np.asarray(d["K"], float), T_mundo_cam=np.asarray(d["T"], float), mascara=m,
                    t=float(d["t"]), dist_max=d.get("dist_max"))
    except (ValueError, KeyError, struct.error, TypeError):
        return None


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
