"""
(18-nov) diagnosticos/medir_pelota.py con escenas SINTÉTICAS: la hoja ChArUco en la mesa y la pelota encima de
una esquina, renderizadas con los intrínsecos y la distorsión REALES de la OAK de cabeza (camara_robot.json).
Comprueba que la herramienta da error ~0 cuando todo cuadra y que separa bien cada tipo de fallo:
  · la profundidad corrida de lado respecto al color (alineación) → lo dice la profundidad sola, no el robot;
  · la detección por color corrida → lo dice el robot, no la profundidad;
  · el modelo de la lente: la profundidad sola acierta quitando la distorsión y falla «como el mapa».

    python -m pytest tests/test_medir_pelota.py -q
"""
import json
import os
import sys

import cv2
import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "diagnosticos"))

import medir_pelota as MP                                                    # noqa: E402
from skill_shield.autonomo.pelotas import catalogo                           # noqa: E402

D = 0.070
_cal = json.load(open(os.path.join(RAIZ, "camara_robot.json")))["intrinsecos"]
K = np.array(_cal["K"], float)
K[:2] *= 640 / _cal["ancho"]
DIST = np.array(_cal["dist"], float)
CAT = catalogo(dict(AUTONOMO_PELOTAS=[dict(nombre="Pelota roja", colores=[((0, 120, 60), (7, 255, 255)),
                                                                          ((168, 120, 60), (179, 255, 255))],
                                           diametro=None, color_gemelo=(0.9, 0.1, 0.15))]))


def _pose_hoja(px=(220, 330), dist_m=0.95, inclinacion_deg=50.0):
    """Hoja en la mesa: cámara mirando hacia abajo inclinacion_deg, centro de la hoja en el píxel px."""
    th = np.radians(inclinacion_deg)
    z = np.array([0.0, np.cos(th), np.sin(th)])            # normal de la hoja, hacia la mesa (lejos de la cámara)
    x = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    p = cv2.undistortPoints(np.array([[px]], float), K, DIST).reshape(2)
    centro = dist_m * np.r_[p, 1.0] / np.linalg.norm(np.r_[p, 1.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centro - R @ np.array([7 * 0.035 / 2, 5 * 0.035 / 2, 0.0])
    return T


def _rayos(Kr, w, h, distorsion):
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    if distorsion:
        p = cv2.undistortPoints(np.column_stack([u.ravel(), v.ravel()]).reshape(-1, 1, 2), Kr, DIST).reshape(-1, 2)
    else:
        p = np.column_stack([(u.ravel() - Kr[0, 2]) / Kr[0, 0], (v.ravel() - Kr[1, 2]) / Kr[1, 1]])
    return np.column_stack([p, np.ones(len(p))])


def _cortes(d, T, c, r):
    """t (= Z, el rayo tiene z = 1) del plano de la hoja y de la pelota por cada rayo."""
    n, p0 = T[:3, 2], T[:3, 3]
    t_pl = (n @ p0) / (d @ n)
    b = d @ c
    a = np.sum(d * d, axis=1)
    disc = b * b - a * (c @ c - r * r)
    t_b = np.where(disc > 0, (b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
    return t_pl, t_b


def render(T, esquina, desplaza_prof_px=0, desplaza_color_m=0.0, prof_con_distorsion=True, ruido=0.0, rng=None):
    board, _ = MP.crear_tablero()
    c = MP.centro_verdad(T, board.getChessboardCorners()[esquina][:2], D / 2)
    # color a resolución doble y luego reducido (como el ISP): que los marcadores salgan nítidos
    K2 = K.copy()
    K2[:2] *= 2
    d2 = _rayos(K2, 1280, 960, True)
    cc = c + np.array([desplaza_color_m, 0.0, 0.0])
    t_pl, t_b = _cortes(d2, T, cc, D / 2)
    P = d2 * t_pl[:, None]
    loc = (P - T[:3, 3]) @ T[:3, :3]
    s = 4000 / (7 * 0.035)
    tex = board.generateImage((4000, int(round(5 * 0.035 * s))), marginSize=0)
    mx, my = (loc[:, 0] * s).astype(np.float32).reshape(960, 1280), (loc[:, 1] * s).astype(np.float32).reshape(960, 1280)
    gris = cv2.remap(cv2.GaussianBlur(tex, (0, 0), 6), mx, my, cv2.INTER_LINEAR, borderValue=235)
    img = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    bola = (t_b < t_pl).reshape(960, 1280)
    img[bola] = (40, 30, 200)
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    # profundidad (mm): en la rejilla del color (con su distorsión) o como si fuera de una cámara sin distorsión
    d1 = _rayos(K, 640, 480, prof_con_distorsion)
    t_pl, t_b = _cortes(d1, T, c, D / 2)
    z = np.minimum(t_pl, t_b).reshape(480, 640)
    if ruido and rng is not None:
        z = z + rng.normal(0, ruido, z.shape) * z * z
    if desplaza_prof_px:
        z = np.roll(z, desplaza_prof_px, axis=1)
    return img, (np.clip(z, 0, 10) * 1000).astype(np.uint16)


def _medir(T, esquina, n=4, **kw):
    board, det = MP.crear_tablero()
    rng = np.random.default_rng(esquina)
    fotos, profs, Ts = [], [], []
    for _ in range(n):
        img, prof = render(T, esquina, rng=rng, **kw)
        Ti, n_esq = MP.pose_tablero(det, board, img, K, DIST)
        assert Ti is not None, f"no detecta la hoja sintética ({n_esq} esquinas)"
        fotos.append(img)
        profs.append(prof)
        Ts.append(Ti)
    return MP.medir_captura(fotos, profs, K, DIST, MP.promediar_poses(Ts), board.getChessboardCorners()[esquina][:2],
                            D, CAT)


def _lado(m, k):
    return float(np.hypot(*m[k]["error_mm"][:2]))


def test_la_hoja_da_la_pose_de_verdad():
    T = _pose_hoja()
    board, det = MP.crear_tablero()
    img, _ = render(T, 14)
    Ti, n = MP.pose_tablero(det, board, img, K, DIST)
    assert Ti is not None and n >= 12
    assert np.linalg.norm(Ti[:3, 3] - T[:3, 3]) < 0.004                     # < 4 mm a ~1 m


@pytest.mark.parametrize("esquina", [8, 22])
def test_todo_cuadra_error_pequeno(esquina):
    m = _medir(_pose_hoja(), esquina, ruido=0.004)
    assert m["robot"] and np.linalg.norm(m["robot"]["error_mm"]) < 5.0, m["robot"]["error_mm"]
    assert _lado(m, "prof_sin_distorsion") < 3.0
    assert abs(m["escala"]["sin_distorsion"] - 1.0) < 0.004
    texto = "\n".join(MP.resumen([m, m]))
    assert "SESGO" not in texto and "corrida" not in texto


def test_profundidad_corrida_de_lado_respecto_al_color():
    """6 px a ~1 m ≈ 1,1 cm: el robot (la posición de lado sale del color) apenas lo nota; la profundidad sola sí,
    y eso es lo que pone los cubos de la pelota a un lado en el mapa."""
    m = _medir(_pose_hoja(), 14, desplaza_prof_px=6)
    assert _lado(m, "robot") < 5.0
    assert _lado(m, "prof_sin_distorsion") > 8.0
    assert "PROFUNDIDAD CORRIDA" in "\n".join(MP.resumen([m, m]))


def test_deteccion_por_color_corrida():
    m = _medir(_pose_hoja(), 14, desplaza_color_m=0.012)
    assert _lado(m, "robot") > 8.0 and _lado(m, "prof_sin_distorsion") < 4.0
    assert "DETECCIÓN por color" in "\n".join(MP.resumen([m, m]))


def test_modelo_de_la_lente_del_mapa():
    """Pelota en una esquina de la imagen (donde la distorsión pesa): la profundidad sola acierta quitando la
    distorsión y falla tratándola como el mapa (K a secas)."""
    m = _medir(_pose_hoja(px=(120, 400), dist_m=0.9), 19)
    assert _lado(m, "prof_sin_distorsion") < 3.0
    assert _lado(m, "prof_como_mapa") > _lado(m, "prof_sin_distorsion") + 2.0
    assert "QUITAR la distorsión" in "\n".join(MP.resumen([m, m]))


def test_hoja_y_analisis_guardado(tmp_path):
    ruta = MP.generar_hoja(str(tmp_path / "hoja.png"))
    img = cv2.imread(ruta)
    assert img.shape[1] == round(297 * MP.DPI / 25.4)
    board, det = MP.crear_tablero()
    corners, ids, _, _ = det.detectBoard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert ids is not None and len(ids) == 24                                # se lee entera


# (19-nov) La hoja vista DE CANTO (como en la caja, ~60-70° respecto a su normal). solvePnPRansac (núcleo EPnP con
# 5 puntos) se degeneraba con un tablero plano tan inclinado: se quedaba con UNA fila de esquinas alineadas
# («8/24 esquinas; detectadas 24» en la ventana) y daba una pose falsa, a >10 cm, con «OK».
def _esquinas_proyectadas(T, ruido_px=0.2, seed=0):
    board, _ = MP.crear_tablero()
    obj = board.getChessboardCorners().astype(np.float64)
    uv = cv2.projectPoints(obj, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, DIST)[0].reshape(-1, 2)
    return obj, uv + np.random.default_rng(seed).normal(0, ruido_px, uv.shape)


@pytest.mark.parametrize("inclinacion", [8.0, 15.0, 25.0, 35.0])
def test_pose_con_la_hoja_de_canto(inclinacion):
    T = _pose_hoja(inclinacion_deg=inclinacion)
    obj, uv = _esquinas_proyectadas(T)
    r = MP._pose_robusta(obj, uv, K, DIST, 1.5, 0.035)
    assert r is not None, "no da pose con las 24 esquinas exactas"
    Ti, idx = r[0], r[1]
    assert len(idx) == 24, f"solo {len(idx)}/24 esquinas como buenas"
    err = np.linalg.norm((Ti[:3, :3] @ obj.T).T + Ti[:3, 3] - ((T[:3, :3] @ obj.T).T + T[:3, 3]), axis=1).max()
    assert err < 0.003, f"pose a {err * 1000:.1f} mm"


def test_pose_quita_esquinas_malas():
    T = _pose_hoja(inclinacion_deg=15.0)
    obj, uv = _esquinas_proyectadas(T)
    malas = [3, 10, 17, 20]
    uv[malas] += [[9, -6], [-7, 8], [8, 7], [-9, -5]]
    r = MP._pose_robusta(obj, uv, K, DIST, 1.5, 0.035)
    assert r is not None
    assert not set(malas) & set(r[1].tolist()) and len(r[1]) == 20


def test_una_fila_sola_no_da_pose():
    """Esquinas alineadas (una fila) no fijan la inclinación de la hoja: mejor «NO DETECTADO» que una pose falsa."""
    T = _pose_hoja(inclinacion_deg=15.0)
    obj, uv = _esquinas_proyectadas(T)
    fila = np.arange(6, 12)
    assert MP._pose_robusta(obj[fila], uv[fila], K, DIST, 1.5, 0.035) is None
