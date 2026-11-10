"""
Calibración cámara de cabeza → robot (paso 4 de la hoja de ruta) — funciones puras.

Montaje «ojo fuera de la mano» (eye-to-hand): la cámara va fija en el cabezal del soporte y
el tablero ChArUco en la brida/mano. Como la cámara y el robot van en la MISMA peana, la
transformación cámara→robot no cambia al mover la peana: se calibra una vez y luego solo
se comprueba (calibrar_camara_robot.py --verificar).

Marcos (todo en metros):
  mundo    = el del MJCF (fairino_mjcf.xml), el mismo que usan el gemelo y la IK.
  brida    = marco de wrist3 ya girado (CadenaFR5.fk_marcos(q)[5]).
  cam      = cámara de color de la OAK-D (convención OpenCV: +Z adelante, +Y abajo).
  tablero  = el del ChArUco (solvePnP).
Incógnitas:  T_mundo_cam (lo que se busca)  y  T_brida_tablero (cómo quedó montado el tablero).
Para cada muestra i:  T_mundo_brida_i · T_brida_tablero  =  T_mundo_cam · T_cam_tablero_i
"""
import datetime
import json
import math

import cv2
import numpy as np

# Métodos de cv2.calibrateHandEye (si esta versión de OpenCV lo trae) + uno propio en numpy
METODOS = {"TSAI": cv2.CALIB_HAND_EYE_TSAI, "PARK": cv2.CALIB_HAND_EYE_PARK,
           "HORAUD": cv2.CALIB_HAND_EYE_HORAUD, "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
           "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS} if hasattr(cv2, "calibrateHandEye") else {}
METODOS["AX=XB"] = None


# ── utilidades ───────────────────────────────────────────────────────────────
def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def T_de(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float).ravel()
    return T


def angulo_R(R):
    """Ángulo (grados) de una rotación."""
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def proyectar_a_so3(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def promediar(Ts):
    R = proyectar_a_so3(np.mean([T[:3, :3] for T in Ts], axis=0))
    return T_de(R, np.mean([T[:3, 3] for T in Ts], axis=0))


def puntos_tablero(cuadros, lado):
    """Esquinas interiores del ChArUco en el marco del tablero (N x 3), como OpenCV."""
    nx, ny = int(cuadros[0]) - 1, int(cuadros[1]) - 1
    return np.array([[(i + 1) * lado, (j + 1) * lado, 0.0] for j in range(ny) for i in range(nx)])


def _transformar(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


# ── diversidad de las posturas ───────────────────────────────────────────────
def diversidad(T_mundo_brida):
    """(giro máximo entre posturas en grados, recorrido de la brida en mm). Hacen falta giros
    variados (≥ 30-40°, en varios ejes) para que la calibración quede bien condicionada."""
    if len(T_mundo_brida) < 2:
        return 0.0, 0.0
    Rs = [T[:3, :3] for T in T_mundo_brida]
    giro = max(angulo_R(Ra.T @ Rb) for k, Ra in enumerate(Rs) for Rb in Rs[k + 1:])
    P = np.array([T[:3, 3] for T in T_mundo_brida])
    return giro, float(np.linalg.norm(P.max(0) - P.min(0)) * 1000)


# ── errores ──────────────────────────────────────────────────────────────────
def errores_mm(T_mundo_cam, T_brida_tablero, T_mundo_brida, T_cam_tablero, P_tablero):
    """Error por muestra (mm): distancia media entre las esquinas del tablero donde las ve la
    cámara y donde deberían estar según el robot y la calibración (todo en el mundo)."""
    out = []
    for Tb, Tc in zip(T_mundo_brida, T_cam_tablero):
        visto = _transformar(T_mundo_cam @ Tc, P_tablero)
        previsto = _transformar(Tb @ T_brida_tablero, P_tablero)
        out.append(float(np.mean(np.linalg.norm(visto - previsto, axis=1)) * 1000))
    return np.array(out)


# ── cálculo ──────────────────────────────────────────────────────────────────
def resolver_ax_xb(A, B):
    """X tal que A_k·X = X·B_k (Park y Martin, 1994): giro por mínimos cuadrados sobre los
    ejes de giro y traslación por mínimos cuadrados lineales."""
    M = np.zeros((3, 3))
    for Ak, Bk in zip(A, B):
        a = cv2.Rodrigues(Ak[:3, :3])[0].ravel()
        b = cv2.Rodrigues(Bk[:3, :3])[0].ravel()
        M += np.outer(b, a)
    w, V = np.linalg.eigh(M.T @ M)
    R = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T @ M.T
    R = proyectar_a_so3(R)
    C = np.vstack([Ak[:3, :3] - np.eye(3) for Ak in A])
    d = np.concatenate([R @ Bk[:3, 3] - Ak[:3, 3] for Ak, Bk in zip(A, B)])
    t = np.linalg.lstsq(C, d, rcond=None)[0]
    return T_de(R, t)


def _hand_eye(T_mundo_brida, T_cam_tablero, metodo):
    """T_mundo_cam por el método dado. Con cv2.calibrateHandEye en modo ojo fuera de la mano
    se le pasan las poses del robot INVERTIDAS (mundo→brida) y devuelve cam→mundo."""
    if METODOS[metodo] is None:
        # T_b_i·Y = X·T_c_i  →  (T_b_j·T_b_i⁻¹)·X = X·(T_c_j·T_c_i⁻¹)  para cada par i < j
        A, B = [], []
        n = len(T_mundo_brida)
        for i in range(n):
            for j in range(i + 1, n):
                A.append(T_mundo_brida[j] @ inv_T(T_mundo_brida[i]))
                B.append(T_cam_tablero[j] @ inv_T(T_cam_tablero[i]))
        T_mundo_cam = resolver_ax_xb(A, B)
    else:
        Tbg = [inv_T(T) for T in T_mundo_brida]
        R, t = cv2.calibrateHandEye([T[:3, :3] for T in Tbg], [T[:3, 3] for T in Tbg],
                                    [T[:3, :3] for T in T_cam_tablero], [T[:3, 3] for T in T_cam_tablero],
                                    method=METODOS[metodo])
        T_mundo_cam = T_de(R, t)
    T_brida_tablero = promediar([inv_T(Tb) @ T_mundo_cam @ Tc
                                 for Tb, Tc in zip(T_mundo_brida, T_cam_tablero)])
    return T_mundo_cam, T_brida_tablero


def _refinar(T_mundo_cam, T_brida_tablero, T_mundo_brida, T_cam_tablero, P_tablero):
    """Ajuste fino conjunto de las dos incógnitas minimizando el error 3D de las esquinas."""
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return T_mundo_cam, T_brida_tablero

    def a_vec(T):
        return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])

    def a_T(v):
        return T_de(cv2.Rodrigues(np.asarray(v[:3], float).reshape(3, 1))[0], v[3:6])

    def residuos(x):
        Tc, Tt = a_T(x[:6]), a_T(x[6:])
        r = []
        for Tb, Tct in zip(T_mundo_brida, T_cam_tablero):
            r.append((_transformar(Tc @ Tct, P_tablero) - _transformar(Tb @ Tt, P_tablero)).ravel())
        return np.concatenate(r)

    sol = least_squares(residuos, np.concatenate([a_vec(T_mundo_cam), a_vec(T_brida_tablero)]),
                        loss="huber", f_scale=0.003)
    return a_T(sol.x[:6]), a_T(sol.x[6:])


def calibrar(T_mundo_brida, T_cam_tablero, P_tablero, refinar=True):
    """Prueba los métodos de OpenCV, se queda con el de menor error y lo refina.
    Devuelve dict con T_mundo_cam, T_brida_tablero, método, errores por muestra y resumen."""
    if len(T_mundo_brida) < 3:
        raise ValueError("Hacen falta al menos 3 muestras (mejor 15-25)")
    resultados = {}
    for nombre in METODOS:
        try:
            Tc, Tt = _hand_eye(T_mundo_brida, T_cam_tablero, nombre)
            e = errores_mm(Tc, Tt, T_mundo_brida, T_cam_tablero, P_tablero)
            if np.all(np.isfinite(e)):
                resultados[nombre] = (float(np.mean(e)), Tc, Tt)
        except (cv2.error, np.linalg.LinAlgError):
            continue
    if not resultados:
        raise RuntimeError("Ningún método de calibración dio un resultado válido")
    metodo = min(resultados, key=lambda k: resultados[k][0])
    _, Tc, Tt = resultados[metodo]
    if refinar:
        Tc2, Tt2 = _refinar(Tc, Tt, T_mundo_brida, T_cam_tablero, P_tablero)
        if np.mean(errores_mm(Tc2, Tt2, T_mundo_brida, T_cam_tablero, P_tablero)) <= resultados[metodo][0]:
            Tc, Tt, metodo = Tc2, Tt2, metodo + "+refinado"
    e = errores_mm(Tc, Tt, T_mundo_brida, T_cam_tablero, P_tablero)
    return dict(T_mundo_cam=Tc, T_brida_tablero=Tt, metodo=metodo, errores_mm=e,
                error_medio_mm=float(np.mean(e)), error_max_mm=float(np.max(e)),
                otros_metodos={k: round(v[0], 2) for k, v in resultados.items()})


def descartar_peores(T_mundo_brida, T_cam_tablero, P_tablero, max_descartes=3, umbral_mm=8.0):
    """Calibra, quita la peor muestra si pasa de umbral_mm y repite (hasta max_descartes).
    Devuelve (resultado, índices usados)."""
    idx = list(range(len(T_mundo_brida)))
    res = calibrar([T_mundo_brida[i] for i in idx], [T_cam_tablero[i] for i in idx], P_tablero)
    for _ in range(max_descartes):
        if len(idx) <= 8 or res["error_max_mm"] <= umbral_mm:
            break
        peor = idx[int(np.argmax(res["errores_mm"]))]
        idx.remove(peor)
        res = calibrar([T_mundo_brida[i] for i in idx], [T_cam_tablero[i] for i in idx], P_tablero)
    return res, idx


def verificar(calib, T_mundo_brida, T_cam_tablero, P_tablero):
    """Errores (mm) de muestras NUEVAS con una calibración guardada."""
    return errores_mm(calib["T_mundo_cam"], calib["T_brida_tablero"], T_mundo_brida, T_cam_tablero, P_tablero)


# ── vista de MuJoCo desde la cámara calibrada ────────────────────────────────
def camara_gl(T_mundo_cam, K, ancho, alto, cerca):
    """Parámetros de un mjvGLCamera que reproduce la cámara real (pinhole con su punto
    principal): pos, forward, up y el frustum en el plano cercano. OpenCV mira por +Z con +Y
    hacia abajo; OpenGL/MuJoCo usa 'up' = -Y de OpenCV."""
    R, t = T_mundo_cam[:3, :3], T_mundo_cam[:3, 3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return dict(pos=t.copy(), forward=R[:, 2].copy(), up=-R[:, 1].copy(),
                frustum_near=cerca, frustum_top=cerca * cy / fy, frustum_bottom=-cerca * (alto - cy) / fy,
                frustum_center=cerca * (ancho / 2.0 - cx) / fx)


# ── archivo ──────────────────────────────────────────────────────────────────
def guardar(ruta, res, *, mxid, K, dist, ancho, alto, tablero, n_muestras, usadas):
    d = dict(
        descripcion="Calibración cámara de cabeza → robot (calibrar_camara_robot.py). Marco 'mundo' = "
                    "el del MJCF (fairino_mjcf.xml). T_mundo_cam: punto en cámara (OpenCV) → mundo.",
        fecha=datetime.datetime.now().isoformat(timespec="seconds"), mxid=mxid,
        T_mundo_cam=np.asarray(res["T_mundo_cam"]).tolist(),
        T_brida_tablero=np.asarray(res["T_brida_tablero"]).tolist(),
        metodo=res["metodo"], error_medio_mm=round(res["error_medio_mm"], 2),
        error_max_mm=round(res["error_max_mm"], 2), otros_metodos=res.get("otros_metodos"),
        n_muestras=int(n_muestras), muestras_usadas=[int(i) for i in usadas],
        intrinsecos=dict(K=np.asarray(K).tolist(), dist=np.asarray(dist).ravel().tolist(),
                         ancho=int(ancho), alto=int(alto)),
        tablero=tablero)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    return d


def cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    d["T_mundo_cam"] = np.array(d["T_mundo_cam"], float)
    d["T_brida_tablero"] = np.array(d["T_brida_tablero"], float)
    d["intrinsecos"]["K"] = np.array(d["intrinsecos"]["K"], float)
    return d


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRACIÓN AUTOMÁTICA: el programa elige las posturas (10-nov)
# ══════════════════════════════════════════════════════════════════════════════
# 1. EXPLORAR: desde una postura en la que la cámara ya ve el tablero, movimientos pequeños
#    (sobre todo de la muñeca) → 6-10 muestras → calibración APROXIMADA.
# 2. PLANIFICAR: con esa calibración se sabe dónde aparecerá el tablero en la imagen para
#    cualquier postura. Se sortean miles de posturas cerca de la de inicio y se quedan las que
#    (a) no chocan según el modelo de colisión de la teleoperación (FR5 + mano + tablero contra
#    soporte, suelo, mesa, cámara…), (b) dejan el tablero dentro de la imagen, de cara a la
#    cámara y a buena distancia. De ellas se eligen ~20 lo más distintas posible (giros en los
#    3 ejes y posiciones repartidas) y se ordenan para que los trayectos sean cortos. Cada
#    trayecto se comprueba simulando cómo lo moverá fr5_servo; si uno choca, se pasa por la
#    postura de inicio o se descarta esa postura.
# 3. MEDIR: el robot las recorre (con el pedal) y se calcula la calibración definitiva.

GIROS_EXPLORAR_DEG = [  # (joint, grados) relativos a la postura de inicio
    (5, 15), (5, -15), (4, 12), (4, -12), (3, 10), (3, -10), (0, 6), (0, -6), (1, 5), (2, -6)]


class Colisiones:
    """Envuelve ModeloColision: una postura vale si no aparece NINGUNA colisión nueva respecto a
    la postura de inicio y las que ya hubiera no empeoran (el modelo es conservador: si la
    postura de inicio ya «choca» en el modelo, es que el modelo no cuadra del todo)."""

    def __init__(self, modelo, q_inicio, tolerancia=0.005):
        self.m = modelo
        self.base = {(a, b): p for a, b, p in modelo.colisiones(np.asarray(q_inicio, float))}
        self.tol = float(tolerancia)

    def libre(self, q):
        for a, b, p in self.m.colisiones(np.asarray(q, float)):
            if (a, b) not in self.base or p > self.base[(a, b)] + self.tol:
                return False
        return True

    def camino_libre(self, qa, qb, v_max_deg=15.0, a_max_deg=300.0, dt=0.04):
        return all(self.libre(q) for q in simular_camino(qa, qb, v_max_deg, a_max_deg, dt))


def simular_camino(qa, qb, v_max_deg=15.0, a_max_deg=300.0, dt=0.04, max_pasos=4000):
    """Posturas intermedias (rad) del trayecto qa→qb tal como lo genera fr5_servo: cada joint
    con velocidad y aceleración limitadas por separado (no es una recta en el espacio articular)."""
    from skill_shield.hardware.fr5_servo import paso_limitado
    q = np.degrees(np.asarray(qa, float))
    obj = np.degrees(np.asarray(qb, float))
    v = np.zeros_like(q)
    out = [np.radians(q)]
    for _ in range(max_pasos):
        q, v = paso_limitado(q, v, obj, v_max_deg, a_max_deg, dt)
        out.append(np.radians(q))
        if np.max(np.abs(obj - q)) < 1e-3 and np.max(np.abs(v)) < 1e-3:
            break
    return out


def tablero_en_imagen(T_cam_tablero, K, ancho, alto, P, margen=0.08, z_min=0.25, z_max=1.6, ang_max=50.0):
    """(¿visible?, ángulo de visión en grados). Todas las esquinas dentro de la imagen con un
    margen, delante de la cámara a una distancia razonable y no demasiado de canto."""
    Pc = _transformar(T_cam_tablero, P)
    if np.any(Pc[:, 2] < z_min) or np.any(Pc[:, 2] > z_max):
        return False, 90.0
    uv = Pc[:, :2] / Pc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    mx, my = margen * ancho, margen * alto
    if np.any(uv[:, 0] < mx) or np.any(uv[:, 0] > ancho - mx) or np.any(uv[:, 1] < my) or np.any(uv[:, 1] > alto - my):
        return False, 90.0
    n = T_cam_tablero[:3, 2]
    c = Pc.mean(0)
    ang = math.degrees(math.acos(float(np.clip(abs(n @ c) / np.linalg.norm(c), 0, 1))))
    return ang <= ang_max, ang


def posturas_explorar(q0, colisiones, limites, v_max_deg=15.0):
    """Pequeños movimientos desde q0 (ida y vuelta) que no chocan. Devuelve lista de q."""
    out = []
    for j, g in GIROS_EXPLORAR_DEG:
        q = np.array(q0, float)
        q[j] += math.radians(g)
        if not (limites[j, 0] <= q[j] <= limites[j, 1]):
            continue
        if colisiones.libre(q) and colisiones.camino_libre(q0, q, v_max_deg):
            out.append(q)
    return out


def planificar_posturas(q0, cadena, colisiones, T_mundo_cam, T_brida_tablero, K, ancho, alto, P,
                        n=20, candidatas=4000, amplitud_deg=(30, 20, 25, 45, 45, 70), v_max_deg=15.0,
                        semilla=0):
    """Posturas para la calibración definitiva. Devuelve (lista de dicts {q, capturar}, info).
    Las de capturar=False son pasos intermedios (volver a la postura de inicio)."""
    rng = np.random.default_rng(semilla)
    q0 = np.asarray(q0, float)
    lim = cadena.limites
    amp = np.radians(amplitud_deg)
    T_cam_mundo = inv_T(T_mundo_cam)
    buenas = []                                     # (q, T_cam_tablero)
    rechazos = dict(limites=0, imagen=0, choque=0)
    for _ in range(int(candidatas)):
        q = q0 + rng.uniform(-amp, amp)
        if np.any(q < lim[:, 0]) or np.any(q > lim[:, 1]):
            rechazos["limites"] += 1
            continue
        Tct = T_cam_mundo @ cadena.fk_marcos(q)[5] @ T_brida_tablero
        if not tablero_en_imagen(Tct, K, ancho, alto, P)[0]:
            rechazos["imagen"] += 1
            continue
        if not colisiones.libre(q):
            rechazos["choque"] += 1
            continue
        buenas.append((q, Tct))
    if not buenas:
        return [], dict(rechazos=rechazos, candidatas_validas=0)

    # elegir las más distintas: muestreo del punto más lejano (giro en grados + 100·distancia en m)
    def dist(a, b):
        return angulo_R(a[1][:3, :3].T @ b[1][:3, :3]) + 100.0 * float(np.linalg.norm(a[1][:3, 3] - b[1][:3, 3]))
    elegidas = [min(range(len(buenas)), key=lambda i: float(np.linalg.norm(buenas[i][0] - q0)))]
    dmin = np.array([dist(buenas[i], buenas[elegidas[0]]) for i in range(len(buenas))])
    while len(elegidas) < min(n, len(buenas)):
        k = int(np.argmax(dmin))
        if dmin[k] <= 0:
            break
        elegidas.append(k)
        dmin = np.minimum(dmin, [dist(buenas[i], buenas[k]) for i in range(len(buenas))])

    # ordenar (vecino más cercano en joints) y comprobar cada trayecto
    pendientes = [buenas[i][0] for i in elegidas]
    plan, actual, descartadas, por_inicio = [], q0, 0, 0
    while pendientes:
        k = int(np.argmin([np.max(np.abs(q - actual)) for q in pendientes]))
        q = pendientes.pop(k)
        if colisiones.camino_libre(actual, q, v_max_deg):
            plan.append(dict(q=q.tolist(), capturar=True))
            actual = q
        elif colisiones.camino_libre(actual, q0, v_max_deg) and colisiones.camino_libre(q0, q, v_max_deg):
            plan.append(dict(q=q0.tolist(), capturar=False))
            plan.append(dict(q=q.tolist(), capturar=True))
            actual, por_inicio = q, por_inicio + 1
        else:
            descartadas += 1
    if plan and not colisiones.camino_libre(actual, q0, v_max_deg):
        pass                                        # al acabar se queda donde está (fr5 deshabilitado)
    Tb = [cadena.fk_marcos(np.array(p["q"]))[5] for p in plan if p["capturar"]]
    giro, recorrido = diversidad(Tb)
    return plan, dict(rechazos=rechazos, candidatas_validas=len(buenas), elegidas=len(Tb),
                      descartadas_por_trayecto=descartadas, pasos_por_inicio=por_inicio,
                      giro_deg=round(giro, 1), recorrido_mm=round(recorrido, 1))
