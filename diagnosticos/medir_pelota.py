#!/usr/bin/env python3
"""
¿Dónde está DE VERDAD la pelota y dónde cree la cámara que está? (18-nov) — paso 1 de la revisión.

La referencia es una HOJA con un tablero ChArUco puesta plana en la mesa: su pose sale de la imagen en
color (solvePnP, ~1 mm a 1 m con un tablero A4), así que NO depende ni del estéreo ni de la calibración
cámara→robot. La pelota se pone encima de una esquina del tablero: su centro de verdad es esa esquina +
el radio, hacia la cámara. Con eso se mide, en el marco de la CÁMARA:

  · lo que usa el robot: el centro que da pelotas.candidatas (la MISMA cuenta que el modo autónomo);
  · el centro sacado SOLO de la profundidad (esfera ajustada a sus puntos), de dos maneras: como el MAPA
    (píxel → rayo con K, sin quitar la distorsión de la lente) y como la PELOTA (quitándola). La que
    acierta dice qué modelo es el bueno para la profundidad alineada al color de la OAK;
  · la escala del estéreo en la propia hoja (profundidad del plano por PnP / la del estéreo).

Cada error se separa en DERECHA/IZQUIERDA y ABAJO/ARRIBA (de lado, vistos en la imagen) y LEJOS/CERCA (a lo
largo del rayo). El ruido del estéreo es a lo largo del rayo y a los dos lados; un error de LADO que se
repite es un sesgo (detección, alineación color/profundidad o intrínsecos), no ruido.

    python diagnosticos/medir_pelota.py --hoja                 # PNG para imprimir (A4 apaisado, al 100 %)
    python diagnosticos/medir_pelota.py --diametro 0.070       # medir (cierra antes el modo autónomo)
    python diagnosticos/medir_pelota.py --analizar diagnosticos/medidas_pelota/medida_….npz   # rehacer el análisis

Al medir: la hoja plana en la mesa; la pelota encima de la esquina que marca la ventana (círculo amarillo),
centrada a ojo mirando desde arriba (los 4 cuadros de alrededor iguales); ESPACIO para medir (~2 s quieta);
N para saltar esa esquina; Q o Esc para terminar. Mueve la hoja a 3-4 sitios (centro, izquierda, derecha,
más lejos) y mide 4-5 esquinas en cada uno: así se ve si el error cambia por la imagen o con la distancia.
Mide antes con calibre el lado de un cuadro impreso (--lado) y el diámetro de la pelota (--diametro).
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

# Hoja: 7 x 5 cuadros de 35 mm (245 x 175 mm, cabe en un A4 apaisado). Marcadores 4x4 con ids 30-46: no se
# confunden con el tablero de la calibración cámara→robot (0-9) ni con los de las herramientas (20-22).
HOJA = dict(cuadros=(7, 5), lado=0.035, marcador=0.026, diccionario="DICT_4X4_50", id0=30)
DPI = 300


# ════════════════════════════ tablero y hoja ════════════════════════════


def crear_tablero(h=HOJA):
    dic = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, h["diccionario"])
    )

    n = (h["cuadros"][0] * h["cuadros"][1]) // 2

    board = cv2.aruco.CharucoBoard(
        tuple(h["cuadros"]),
        float(h["lado"]),
        float(h["marcador"]),
        dic,
        np.arange(h["id0"], h["id0"] + n)
    )

    dp = cv2.aruco.DetectorParameters()

    # Permitir marcadores pequeños.
    dp.minMarkerPerimeterRate = 0.005
    dp.maxMarkerPerimeterRate = 0.5

    # Probar distintas ventanas de umbralización.
    dp.adaptiveThreshWinSizeMin = 3
    dp.adaptiveThreshWinSizeMax = 35
    dp.adaptiveThreshWinSizeStep = 8
    dp.adaptiveThreshConstant = 7

    # Mejorar la detección de los contornos.
    dp.polygonalApproxAccuracyRate = 0.03
    dp.minCornerDistanceRate = 0.03

    # Mejorar la reconstrucción de los marcadores.
    dp.perspectiveRemovePixelPerCell = 8
    dp.perspectiveRemoveIgnoredMarginPerCell = 0.13

    # Evitar aceptar marcadores con muy poco contraste interno.
    dp.minOtsuStdDev = 3.0

    # Refinamiento subpíxel de las esquinas de los marcadores.
    if hasattr(dp, "cornerRefinementMethod"):
        dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        dp.cornerRefinementWinSize = 5
        dp.cornerRefinementMaxIterations = 30
        dp.cornerRefinementMinAccuracy = 0.01

    cp = cv2.aruco.CharucoParameters()

    if hasattr(cp, "tryRefineMarkers"):
        cp.tryRefineMarkers = True

    detector = cv2.aruco.CharucoDetector(
        board,
        cp,
        dp
    )

    return board, detector



def generar_hoja(ruta, h=HOJA):
    """PNG a 300 ppp para imprimir al 100 % en A4 apaisado. Sin números encima del tablero (estropearían los
    marcadores): la ventana de la medida marca en cada momento la esquina en la que poner la pelota."""
    board, _ = crear_tablero(h)
    mm = DPI / 25.4
    nx, ny = h["cuadros"]
    w_b, h_b = int(round(nx * h["lado"] * 1000 * mm)), int(round(ny * h["lado"] * 1000 * mm))
    img = board.generateImage((w_b, h_b), marginSize=0)
    W, H = int(round(297 * mm)), int(round(210 * mm))
    hoja = np.full((H, W), 255, np.uint8)
    x0, y0 = (W - w_b) // 2, (H - h_b) // 2
    hoja[y0:y0 + h_b, x0:x0 + w_b] = img
    hoja = cv2.cvtColor(hoja, cv2.COLOR_GRAY2BGR)
    txt = (f"medir_pelota.py  -  {nx}x{ny} cuadros de {h['lado'] * 1000:.0f} mm (imprimir al 100 %, medir con calibre "
           f"y pasar --lado si no es exacto)  -  ids {h['id0']}-{h['id0'] + (nx * ny) // 2 - 1}")
    cv2.putText(hoja, txt, (x0, y0 + h_b + int(9 * mm)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(ruta, hoja)
    return ruta



def pose_tablero(detector, board, img, K, dist,
                 min_esquinas=8, max_reproj_px=1.5):

    gris = (
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3 else img
    )

    # Mejorar el contraste local sin alterar la imagen original.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gris_clahe = clahe.apply(gris)

    # Distintas escalas y tratamientos de imagen.
    variantes = [
        (gris, 1.0, "original"),
        (gris_clahe, 1.0, "clahe"),
    ]

    for escala in (1.5, 2.0):
        tam = (
            int(gris.shape[1] * escala),
            int(gris.shape[0] * escala)
        )

        original_ampliada = cv2.resize(
            gris, tam, interpolation=cv2.INTER_CUBIC
        )

        clahe_ampliada = cv2.resize(
            gris_clahe, tam, interpolation=cv2.INTER_CUBIC
        )

        variantes.append(
            (original_ampliada, escala, f"original_{escala}")
        )
        variantes.append(
            (clahe_ampliada, escala, f"clahe_{escala}")
        )

    mejor = None
    mejor_puntuacion = None

    obj_completo = board.getChessboardCorners().astype(
        np.float64
    )

    for imagen, escala, nombre in variantes:

        try:
            corners, ids, _, _ = detector.detectBoard(imagen)
        except cv2.error:
            continue

        if corners is None or ids is None:
            continue

        ids = ids.flatten().astype(int)

        if len(ids) < min_esquinas:
            continue

        # Descartar índices inválidos.
        validos = (ids >= 0) & (ids < len(obj_completo))
        ids = ids[validos]

        pts = corners.reshape(-1, 2)[validos].astype(
            np.float64
        )

        # Volver a las coordenadas de la imagen original.
        pts /= escala

        if len(ids) < min_esquinas:
            continue

        obj = obj_completo[ids]

        # Estimación robusta inicial de la pose.
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj,
                pts,
                K,
                dist,
                iterationsCount=100,
                reprojectionError=2.5,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        except cv2.error:
            continue

        if not ok or inliers is None:
            continue

        idx = inliers.flatten()

        if len(idx) < min_esquinas:
            continue

        # Refinar usando únicamente los puntos consistentes.
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj[idx],
                pts[idx],
                K,
                dist,
                rvec,
                tvec,
                True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        except cv2.error:
            continue

        if not ok:
            continue

        # Comprobar error de reproyección.
        proy, _ = cv2.projectPoints(
            obj[idx],
            rvec,
            tvec,
            K,
            dist
        )

        residuos = np.linalg.norm(
            proy.reshape(-1, 2) - pts[idx],
            axis=1
        )

        error = float(np.sqrt(np.mean(residuos ** 2)))

        if not np.isfinite(error) or error > max_reproj_px:
            continue

        # La prioridad es obtener más esquinas válidas.
        # En igualdad, gana la pose con menor error.
        puntuacion = (len(idx), -error)

        if mejor_puntuacion is None or puntuacion > mejor_puntuacion:
            T = np.eye(4)
            T[:3, :3] = cv2.Rodrigues(rvec)[0]
            T[:3, 3] = tvec.ravel()

            mejor = (T, len(idx))
            mejor_puntuacion = puntuacion

    if mejor is None:
        return None, 0

    return mejor



def promediar_poses(Ts):
    U, _, Vt = np.linalg.svd(np.mean([T[:3, :3] for T in Ts], axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.median([T_[:3, 3] for T_ in Ts], axis=0)
    return T


# ════════════════════════════ cuentas (sin cámara) ════════════════════════════

def centro_verdad(T_cam_tab, punto_tab, r):
    """Centro de la pelota apoyada en el punto (x, y) del tablero: el punto + r por la normal que mira a la cámara."""
    p = T_cam_tab[:3, :3] @ np.array([punto_tab[0], punto_tab[1], 0.0]) + T_cam_tab[:3, 3]
    n = T_cam_tab[:3, 2]
    if n @ p > 0:                                    # la normal tiene que apuntar hacia la cámara (hacia −p)
        n = -n
    return p + float(r) * n


def base_imagen(c):
    """(rayo, derecha, abajo): vectores unitarios para separar un error en el centro c (marco de la cámara)."""
    u = np.asarray(c, float) / np.linalg.norm(c)
    der = np.array([1.0, 0.0, 0.0]) - u[0] * u
    der /= np.linalg.norm(der)
    abajo = np.cross(u, der)                          # (y de la cámara: hacia abajo en la imagen)
    if abajo[1] < 0:
        abajo = -abajo
    return u, der, abajo


def descomponer(e, c_verdad):
    """Error e (m) → mm (derecha, abajo, lejos) vistos desde la cámara."""
    u, der, abajo = base_imagen(c_verdad)
    return np.array([e @ der, e @ abajo, e @ u]) * 1000.0


def rayos(us, vs, K, dist=None):
    """Rayos (x/z, y/z) de los píxeles; con dist, quitando la distorsión de la lente."""
    us, vs = np.asarray(us, float).ravel(), np.asarray(vs, float).ravel()
    if dist is not None and np.any(np.asarray(dist, float)):
        p = cv2.undistortPoints(np.column_stack([us, vs]).reshape(-1, 1, 2), np.asarray(K, float),
                                np.asarray(dist, float).ravel()).reshape(-1, 2)
        return p[:, 0], p[:, 1]
    return (us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1]


def ajustar_esfera(P, r, c0=None):
    """Centro de la esfera de radio r que mejor pasa por los puntos P (la parte que ve la cámara)."""
    from scipy.optimize import least_squares
    P = np.asarray(P, float).reshape(-1, 3)
    if c0 is None:
        m = np.median(P, axis=0)
        c0 = m + m / np.linalg.norm(m) * r
    sol = least_squares(lambda c: np.linalg.norm(P - c, axis=1) - r, np.asarray(c0, float), loss="huber",
                        f_scale=0.003)
    return sol.x


def centro_solo_profundidad(prof_mm, K, dist, u, v, r_px, diam, quitar_distorsion):
    """Centro (cámara, m) de la esfera ajustada SOLO a la profundidad dentro de la silueta (0,8 · r: sin los
    bordes, donde el estéreo mezcla pelota y fondo). quitar_distorsion=False: como el mapa (K a secas)."""
    h, w = prof_mm.shape[:2]
    rr = 0.8 * float(r_px)
    x0, x1 = max(int(u - rr), 0), min(int(u + rr) + 1, w)
    y0, y1 = max(int(v - rr), 0), min(int(v + rr) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    z = prof_mm[y0:y1, x0:x1].astype(np.float64) * 0.001
    sel = ((xx - u) ** 2 + (yy - v) ** 2 <= rr * rr) & (z > 0)
    if sel.sum() < 15:
        return None
    zs = z[sel]
    zm = float(np.median(zs))
    ok = np.abs(zs - zm) < 0.6 * diam                # fuera lo que sea fondo (mezcla en los bordes)
    if ok.sum() < 15:
        return None
    x, y = rayos(xx[sel][ok], yy[sel][ok], K, dist if quitar_distorsion else None)
    P = np.column_stack([x * zs[ok], y * zs[ok], zs[ok]])
    return ajustar_esfera(P, diam / 2.0)


def escala_en_hoja(T_cam_tab, h, prof_mm, K, dist, quitar_distorsion, excluir=None):
    """Cociente mediano Z_plano(PnP) / Z_estéreo en los píxeles de la hoja (sin la pelota) o None."""
    nx, ny = h["cuadros"]
    L = float(h["lado"])
    esq = np.array([[0, 0, 0], [nx * L, 0, 0], [nx * L, ny * L, 0], [0, ny * L, 0]], np.float64)
    rvec, _ = cv2.Rodrigues(T_cam_tab[:3, :3])
    px, _ = cv2.projectPoints(esq, rvec, T_cam_tab[:3, 3], K, dist)
    hh, ww = prof_mm.shape[:2]
    m = np.zeros((hh, ww), np.uint8)
    cv2.fillConvexPoly(m, np.round(px.reshape(-1, 2)).astype(np.int32), 255)
    m = cv2.erode(m, np.ones((9, 9), np.uint8))
    if excluir is not None:                          # la pelota (y su sombra en la profundidad)
        u, v, r = excluir
        cv2.circle(m, (int(u), int(v)), int(1.8 * r) + 4, 0, -1)
    vs, us = np.nonzero((m > 0) & (prof_mm > 0))
    if len(us) < 200:
        return None
    z = prof_mm[vs, us].astype(np.float64) * 0.001
    x, y = rayos(us, vs, K, dist if quitar_distorsion else None)
    n, t = T_cam_tab[:3, 2], T_cam_tab[:3, 3]
    den = x * n[0] + y * n[1] + n[2]
    ok = np.abs(den) > 1e-6
    q = (n @ t) / den[ok] / z[ok]
    q = q[np.isfinite(q) & (q > 0.8) & (q < 1.2)]
    return float(np.median(q)) if len(q) >= 200 else None


def medir_captura(fotos, profs, K, dist, T_tab, punto_tab, diam, cat, quitar_dist_pelota=True, ajuste=True,
                  min_relleno=0.7, h=HOJA):
    """Una captura (varias imágenes con la pelota quieta en la misma esquina) → dict con, por método, el centro
    medio (cámara, m), su error medio (mm: derecha, abajo, lejos) y su dispersión (mm)."""
    from skill_shield.autonomo.pelotas import candidatas
    c_v = centro_verdad(T_tab, punto_tab, diam / 2.0)
    uv_v = cv2.projectPoints(c_v.reshape(1, 3), np.zeros(3), np.zeros(3), K, dist)[0].reshape(2)
    met = {"robot": [], "prof_como_mapa": [], "prof_sin_distorsion": []}
    escalas = {"como_mapa": [], "sin_distorsion": []}
    uvs = []
    for img, prof in zip(fotos, profs):
        cands = candidatas(img, prof, K, cat, min_relleno=min_relleno, dist=dist if quitar_dist_pelota else None,
                           ajuste=ajuste)
        if not cands:
            continue
        e, c, d_est, (u, v, r) = min(cands, key=lambda k: np.hypot(k[3][0] - uv_v[0], k[3][1] - uv_v[1]))
        if np.hypot(u - uv_v[0], v - uv_v[1]) > 3 * r:   # no es la de la esquina
            continue
        uvs.append((u, v, r))
        met["robot"].append(c)
        for clave, qd in (("prof_como_mapa", False), ("prof_sin_distorsion", True)):
            cp = centro_solo_profundidad(prof, K, dist, u, v, r, diam, qd)
            if cp is not None:
                met[clave].append(cp)
        for clave, qd in (("como_mapa", False), ("sin_distorsion", True)):
            s = escala_en_hoja(T_tab, h, prof, K, dist, qd, excluir=(u, v, r))
            if s is not None:
                escalas[clave].append(s)
    out = dict(verdad=c_v, uv_verdad=uv_v, n=len(uvs), uv=np.median(uvs, axis=0) if uvs else None,
               distancia=float(np.linalg.norm(c_v)), escala={k: float(np.median(v)) if v else None
                                                             for k, v in escalas.items()})
    for k, cs in met.items():
        if not cs:
            out[k] = None
            continue
        C = np.array(cs)
        errs = np.array([descomponer(c - c_v, c_v) for c in C])
        out[k] = dict(centro=C.mean(0), error_mm=errs.mean(0), disp_mm=errs.std(0), n=len(C))
    return out


# ════════════════════════════ resumen ════════════════════════════

def _fmt(e):
    d, a, l = e
    return (f"{abs(d):4.1f} {'dcha' if d >= 0 else 'izda'} · {abs(a):4.1f} {'abajo' if a >= 0 else 'arriba'} · "
            f"{abs(l):4.1f} {'lejos' if l >= 0 else 'cerca'}")


def resumen(medidas, umbral_mm=5.0):
    """Texto: tabla por captura, media por método y lo que quiere decir."""
    ok = [m for m in medidas if m.get("robot")]
    if not ok:
        return ["Ninguna captura con la pelota y la hoja a la vez."]
    L = ["", "captura | distancia | píxel (u, v) | error de lo que usa el robot (mm)          | solo profundidad: "
         "como el mapa / sin distorsión (lado, mm)"]
    for k, m in enumerate(ok):
        pm, ps = m.get("prof_como_mapa"), m.get("prof_sin_distorsion")
        lado = lambda x: "—" if not x else f"{np.hypot(*x['error_mm'][:2]):4.1f}"   # noqa: E731
        L.append(f"{k:7d} | {m['distancia']:.2f} m  | ({m['uv'][0]:4.0f}, {m['uv'][1]:4.0f})  | "
                 f"{_fmt(m['robot']['error_mm'])} | {lado(pm)} / {lado(ps)}")
    E = np.array([m["robot"]["error_mm"] for m in ok])
    D = np.array([m["robot"]["disp_mm"] for m in ok])
    media, sd = E.mean(0), E.std(0)
    tot = np.linalg.norm(E, axis=1)
    L += ["", f"LO QUE USA EL ROBOT ({len(ok)} capturas): error medio {_fmt(media)} mm",
          f"   de una captura a otra varía ±{sd[0]:.1f} (lado) ±{sd[1]:.1f} (alto) ±{sd[2]:.1f} (rayo) mm; "
          f"dentro de una misma captura, ±{np.mean(D[:, 0]):.1f} / ±{np.mean(D[:, 1]):.1f} / ±{np.mean(D[:, 2]):.1f}",
          f"   error total: mediana {np.median(tot):.1f} mm, peor {tot.max():.1f} mm "
          f"({int((tot <= umbral_mm).sum())}/{len(tot)} capturas por debajo de {umbral_mm:.0f} mm)"]
    lado_robot = float(np.hypot(*media[:2]))
    res = {}
    for k in ("prof_como_mapa", "prof_sin_distorsion"):
        xs = [m[k]["error_mm"] for m in ok if m.get(k)]
        if xs:
            res[k] = np.mean(xs, axis=0)
            L.append(f"SOLO PROFUNDIDAD {'como el mapa (K sin distorsión)' if k == 'prof_como_mapa' else 'quitando la distorsión'}: "
                     f"error medio {_fmt(res[k])} mm")
    for k, txt in (("como_mapa", "como el mapa"), ("sin_distorsion", "quitando la distorsión")):
        es = [m["escala"][k] for m in ok if m["escala"].get(k)]
        if es:
            L.append(f"ESCALA del estéreo en la hoja ({txt}): {np.median(es):.4f} "
                     f"(a 1 m serían {abs(np.median(es) - 1) * 1000:.0f} mm {'de más' if np.median(es) < 1 else 'de menos'})")
    L += ["", "QUÉ QUIERE DECIR"]
    if np.median(tot) <= umbral_mm:
        L.append(f"✅ El centro que usa el robot para COGER la pelota está a menos de {umbral_mm:.0f} mm: si la mano no "
                 "llega bien a ella, el error está en otro eslabón (calibración cámara→robot, cinemática del brazo o "
                 "de la mano). Los avisos de abajo, si hay, son del MAPA (obstáculos y planificación).")
    if lado_robot > 4.0 and lado_robot > 2 * max(sd[0], sd[1]):
        L.append(f"⚠️  SESGO DE LADO de {lado_robot:.0f} mm que se repite: no es ruido del estéreo (ese va por el rayo).")
        if res:
            mejor = min(res, key=lambda k: np.hypot(*res[k][:2]))
            lm = float(np.hypot(*res[mejor][:2]))
            if lm < 0.5 * lado_robot:
                L.append(f"   La profundidad sola SÍ acierta de lado ({lm:.0f} mm): el sesgo está en la DETECCIÓN por color "
                         "(silueta mal ajustada: sombra, reflejos, el rango HSV o el centro del círculo).")
            else:
                L.append("   La profundidad sola tiene el mismo sesgo: está en la ALINEACIÓN color↔profundidad o en los "
                         "intrínsecos (K, distorsión) de la OAK, no en la detección.")
    if res:
        lado_prof = min(float(np.hypot(*v[:2])) for v in res.values())
        if lado_prof > 4.0 and lado_prof > lado_robot + 3.0:
            L.append(f"⚠️  PROFUNDIDAD CORRIDA de lado {lado_prof:.0f} mm respecto al color (el robot, que saca el lado "
                     "del color, va bien): en el MAPA los cubos de la pelota —y de todo— salen corridos a un lado. Es la "
                     "alineación color↔profundidad de la OAK (setDepthAlign) o sus intrínsecos.")
    if len(res) == 2:
        a, b = (float(np.hypot(*res[k][:2])) for k in ("prof_como_mapa", "prof_sin_distorsion"))
        if abs(a - b) > 1.5:
            L.append(f"⚠️  El mapa y la pelota no usan el mismo modelo de la lente y el bueno es "
                     f"{'QUITAR la distorsión (el mapa debería hacerlo)' if b < a else 'NO quitarla (la pelota no debería)'}: "
                     f"{min(a, b):.0f} frente a {max(a, b):.0f} mm de lado.")
    if abs(media[2]) > 4.0 and abs(media[2]) > 2 * sd[2]:
        L.append(f"⚠️  A lo largo del rayo sale {abs(media[2]):.0f} mm {'lejos' if media[2] > 0 else 'cerca'} de media: "
                 "escala del estéreo (mira la ESCALA de arriba: AUTONOMO_CAM_ESCALA_PROF) o el paso de la cara al centro.")
    if max(sd) > 6.0:
        L.append(f"⚠️  Varía mucho de una captura a otra (±{max(sd):.0f} mm): mira en la tabla si depende del sitio de la "
                 "imagen (bordes: distorsión) o de la distancia (estéreo).")
    return L


# ════════════════════════════ captura con la cámara ════════════════════════════

ORDEN_ESQUINAS = (8, 11, 14, 22, 19, 16, 9, 13, 21, 17)   # repartidas por la hoja (de las 24 interiores de 7x5)


def capturar(a, h):
    import config as cfg
    from config import ALTO_CAM, ANCHO_CAM
    from skill_shield.autonomo.pelotas import catalogo
    from skill_shield.vision.camara_oak import crear_pipeline_oak
    from skill_shield.vision.multicamara import abrir_dispositivo, activar_ir, intrinsecos_color
    board, det = crear_tablero(h)
    g = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    cat = catalogo(g)
    quitar_dist = bool(g.get("AUTONOMO_PELOTAS_QUITAR_DISTORSION", True))
    escala = float(g.get("AUTONOMO_CAM_ESCALA_PROF", 1.0) or 1.0)
    esquinas = board.getChessboardCorners()
    # la MISMA profundidad que el modo autónomo (sin rellenar huecos ni filtro temporal), con su escala
    dev = abrir_dispositivo(crear_pipeline_oak(
        con_hd=False, rellenar_huecos=g.get("AUTONOMO_CAM_RELLENAR_HUECOS", 0),
        temporal=g.get("AUTONOMO_CAM_FILTRO_TEMPORAL", False), confianza=g.get("AUTONOMO_CAM_CONFIANZA", 200)),
        a.mxid or g.get("AUTONOMO_CAM_MXID"))
    medidas, crudo, k_obj, txt = [], [], 0, ""
    with dev:
        activar_ir(dev, "cabeza", int(g.get("AUTONOMO_CAM_IR_MA", 800)))
        K, dist = intrinsecos_color(dev, ANCHO_CAM, ALTO_CAM)
        qc = dev.getOutputQueue("color", maxSize=1, blocking=False)
        qd = dev.getOutputQueue("depth", maxSize=1, blocking=False)
        prof = None
        print("ESPACIO = medir la esquina marcada · N = otra esquina · Q/Esc = terminar")
        while True:
            pd = qd.tryGet()
            if pd is not None:
                prof = pd.getFrame()
                if abs(escala - 1.0) > 1e-6:
                    prof = np.clip(prof.astype(np.float32) * escala, 0, 65535).astype(np.uint16)
            pc = qc.tryGet()
            if pc is None:
                time.sleep(0.003)
                continue
            img = pc.getCvFrame()
            T, n_esq = pose_tablero(det, board, img, K, dist)
            obj = ORDEN_ESQUINAS[k_obj % len(ORDEN_ESQUINAS)]
            vis = img.copy()
            if T is not None:
                c = centro_verdad(T, esquinas[obj][:2], a.diametro / 2)
                proyectados, _ = cv2.projectPoints(
                    c.reshape(1, 3),
                    np.zeros(3),
                    np.zeros(3),
                    K,
                    dist
                )
                uu, vv = proyectados.reshape(-1, 2)[0]
                r_px = a.diametro / 2 * K[0, 0] / c[2]
                cv2.circle(vis, (int(uu), int(vv)), int(r_px), (0, 255, 255), 2)
                cv2.putText(vis, f"esquina {obj}", (int(uu + r_px + 4), int(vv)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)
            estado = (
                    f"ChArUco: {'OK' if T is not None else 'NO DETECTADO'} "
                    f"({n_esq}/24 esquinas) · capturas {len(medidas)}"
                )
            for i, t_ in enumerate([estado] + txt.split("\n")):
                cv2.putText(vis, t_, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, t_, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("medir_pelota", vis)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (27, ord("q")):
                break
            if tecla == ord("n"):
                k_obj += 1
            if tecla != ord(" ") or prof is None:
                continue
            fotos, profs, Ts, t0 = [], [], [], time.monotonic()
            while len(fotos) < a.fotos and time.monotonic() - t0 < 6.0:   # ~2 s con la pelota quieta
                pd, pc = qd.tryGet(), qc.tryGet()
                if pd is not None:
                    prof = pd.getFrame()
                    if abs(escala - 1.0) > 1e-6:
                        prof = np.clip(prof.astype(np.float32) * escala, 0, 65535).astype(np.uint16)
                if pc is None or prof is None:
                    time.sleep(0.003)
                    continue
                im = pc.getCvFrame()
                Ti, _ = pose_tablero(det, board, im, K, dist)
                if Ti is not None:
                    fotos.append(im)
                    profs.append(prof.copy())
                    Ts.append(Ti)
            if len(fotos) < 5:
                txt = "no se ve bien la hoja: más luz o menos inclinada"
                continue
            Tm = promediar_poses(Ts)
            m = medir_captura(fotos, profs, K, dist, Tm, esquinas[obj][:2], a.diametro, cat, quitar_dist)
            if not m.get("robot"):
                txt = "no veo la pelota en la esquina marcada"
                continue
            m.update(esquina=int(obj), T_tab=Tm)
            medidas.append(m)
            crudo.append(dict(fotos=[cv2.imencode(".png", f)[1] for f in fotos[::4]], profs=profs[::4], T=Tm,
                              esquina=int(obj)))
            txt = f"esquina {obj}: {_fmt(m['robot']['error_mm'])} mm"
            print(f"📏 {txt}  (a {m['distancia']:.2f} m)")
            k_obj += 1
    cv2.destroyAllWindows()
    return medidas, crudo, K, dist


def guardar(ruta, crudo, K, dist, a, h):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    datos = dict(K=K, dist=np.asarray(dist, float), info=json.dumps(dict(diametro=a.diametro, hoja=h)))
    for i, c in enumerate(crudo):
        datos[f"c{i}_T"] = c["T"]
        datos[f"c{i}_esquina"] = c["esquina"]
        datos[f"c{i}_profs"] = np.array(c["profs"])
        for j, f in enumerate(c["fotos"]):
            datos[f"c{i}_foto{j}"] = np.frombuffer(f.tobytes(), np.uint8)
    np.savez_compressed(ruta, **datos)


def analizar(ruta):
    """Rehace las cuentas con los datos guardados (las mismas funciones; vale para probar cambios)."""
    import config as cfg
    from skill_shield.autonomo.pelotas import catalogo
    d = np.load(ruta)
    info = json.loads(str(d["info"]))
    h = dict(info["hoja"])
    h["cuadros"] = tuple(h["cuadros"])
    board, _ = crear_tablero(h)
    g = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    K, dist = d["K"], d["dist"]
    medidas, i = [], 0
    while f"c{i}_T" in d:
        fotos = [cv2.imdecode(d[f"c{i}_foto{j}"], cv2.IMREAD_COLOR) for j in range(len(d[f"c{i}_profs"]))]
        m = medir_captura(fotos, list(d[f"c{i}_profs"]), K, dist, d[f"c{i}_T"],
                          board.getChessboardCorners()[int(d[f"c{i}_esquina"])][:2], info["diametro"], catalogo(g),
                          bool(g.get("AUTONOMO_PELOTAS_QUITAR_DISTORSION", True)), h=h)
        medidas.append(m)
        i += 1
    return medidas


def main():
    import config as cfg
    ap = argparse.ArgumentParser(description="Error de la posición de la pelota (paso 1 de la revisión)")
    ap.add_argument("--hoja", action="store_true", help="genera el PNG para imprimir y sale")
    ap.add_argument("--diametro", type=float, default=None, help="diámetro de la pelota en m (con calibre)")
    ap.add_argument("--lado", type=float, default=HOJA["lado"], help="lado del cuadro impreso en m (con calibre)")
    ap.add_argument("--fotos", type=int, default=20, help="imágenes por captura")
    ap.add_argument("--mxid", default=None)
    ap.add_argument("--analizar", default=None, help="archivo .npz de una medida anterior")
    a = ap.parse_args()
    h = dict(HOJA, lado=a.lado, marcador=a.lado * HOJA["marcador"] / HOJA["lado"])
    if a.hoja:
        ruta = generar_hoja(os.path.join(RAIZ, "diagnosticos", "hoja_medir_pelota.png"), h)
        print(f"🖨️  {ruta}: imprímela al 100 % (sin «ajustar a la página») y mide un cuadro con calibre.")
        return
    if a.analizar:
        print("\n".join(resumen(analizar(a.analizar))))
        return
    if not a.diametro:
        sys.exit("Falta --diametro (m): mide la pelota con un calibre (p. ej. --diametro 0.070).")
    medidas, crudo, K, dist = capturar(a, h)
    if not medidas:
        sys.exit("Sin capturas.")
    ruta = os.path.join(RAIZ, "diagnosticos", "medidas_pelota", time.strftime("medida_%Y%m%d_%H%M%S.npz"))
    guardar(ruta, crudo, K, dist, a, h)
    lineas = resumen(medidas)
    print("\n".join(lineas))
    with open(ruta.replace(".npz", ".txt"), "w") as f:
        f.write("\n".join(lineas) + "\n")
    print(f"\n💾 Datos en {ruta} (y el resumen en .txt): mándalos para revisarlos.")


if __name__ == "__main__":
    main()
