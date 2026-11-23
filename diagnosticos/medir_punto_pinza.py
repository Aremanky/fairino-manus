#!/usr/bin/env python3
"""
¿Dónde está DE VERDAD el punto de pinza de la OrcaHand? (23-nov) — método de los 4 puntos del TCP, sin cámara.

Por qué: la mano del gemelo (mano_mount en fairino_mjcf.xml) no está montada como la real. En las dos calibraciones
cámara→robot, el tablero pegado a la palma sale girado ~50° alrededor del eje de la brida respecto a la palma del
gemelo, y el planificador lleva a la pelota un punto de pinza que está a ~7,8 cm del eje: por eso coge la pelota
unos cm a la derecha. Este script mide el punto de pinza REAL (en el marco de la brida, wrist3) y lo compara con el
del modelo. No mueve el robot ni usa cámaras: solo hace cuentas con las poses que le des.

Cómo medir (10-15 min):
  1. Fija la pelota de verdad para que no ruede (sobre un vaso o un rollo de cinta), al alcance del brazo.
  2. Pon la mano en la pinza de CONTACTO para esa pelota (la misma que usa la tarea):
        python diagnosticos/medir_punto_pinza.py --mano --diametro 0.070      (Ctrl+C la suelta)
     (necesita el puerto de la mano libre: cierra antes el modo autónomo y la teleoperación).
  3. Mueve el brazo (arrastre en la WebApp, o la ventana de sliders) hasta que la pelota quede ENTRE las yemas de
     pulgar e índice, centrada y tocándolas sin empujarla. Apunta los 6 ángulos de la controladora (los de la
     WebApp, o el botón «Imprimir pose» de los sliders).
  4. Repite desde 4-6 orientaciones de la mano MUY distintas (J6 ±40°, muñeca más o menos inclinada…) SIN mover la
     pelota. Cuanto más distintas, mejor sale.
  5. Pega las poses en un archivo (una por línea, con punto decimal; vale «pose3: 120.0, -43.0, …  # nota») y:
        python diagnosticos/medir_punto_pinza.py poses_pinza.txt --diametro 0.070

El brazo no se mueve solo: lo mueves tú. Sale: el punto de pinza real en la brida, cuánto se aleja del modelo, el
giro de la mano que lo explica y la línea de <body name="mano_mount" …> para fairino_mjcf.xml con ese giro. Tras
cambiarla, repite el paso 5 con las mismas poses: el modelo y lo medido deberían quedar a menos de ~5 mm. Si el giro
ya cuadra pero el punto sigue lejos (pulgar sin calibrar, dedos), te da AUTONOMO_TAREA_PINZA_PUNTO_BRIDA_M.
"""
import argparse
import math
import os
import re
import sys

import numpy as np


def _raiz(arg):
    r = arg or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, r)
    return r


def leer_poses(ruta):
    """Poses (grados de la controladora, N x 6) de un archivo de texto: 6 números por línea (lo que hay antes de
    «:» y después de «#» no cuenta)."""
    poses = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.split("#")[0]
            if ":" in linea:
                linea = linea.split(":", 1)[1]
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", linea)
            if len(nums) >= 6:
                poses.append([float(v) for v in nums[:6]])
    return np.array(poses, float).reshape(-1, 6)


def a_mjcf(q_deg, cfg):
    """Grados de la controladora → rad del MJCF (inversa de q_FR5 = SIGNO·grados(q) + OFFSET, como fr5_servo)."""
    signo = np.asarray(getattr(cfg, "FR5_SIGNO", (1,) * 6), float)
    offset = np.asarray(getattr(cfg, "FR5_OFFSET_DEG", (0,) * 6), float)
    return np.radians(np.asarray(q_deg, float) - offset) * signo


def resolver(Ts):
    """Punto p (marco de la brida) y pelota b (mundo) con R_i·p + t_i = b para todas las poses (mínimos cuadrados)."""
    A = np.zeros((3 * len(Ts), 6))
    y = np.zeros(3 * len(Ts))
    for i, T in enumerate(Ts):
        A[3 * i:3 * i + 3, :3] = T[:3, :3]
        A[3 * i:3 * i + 3, 3:] = -np.eye(3)
        y[3 * i:3 * i + 3] = -T[:3, 3]
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    sv = np.linalg.svd(A, compute_uv=False)
    p, b = x[:3], x[3:]
    res = np.array([np.linalg.norm(T[:3, :3] @ p + T[:3, 3] - b) for T in Ts])
    return p, b, res, sv[-1] / sv[0]


def postura_contacto(cfg, diametro, ruta_xml):
    """(postura de la mano {ctrl: rad}, punto de pinza del MODELO en la brida) para esa pelota: lo mismo que
    Planificador.pinza (tarea.py), con las posturas de config."""
    from skill_shield.autonomo.choque_mano import yemas_mano
    ab, ce = dict(cfg.AUTONOMO_TAREA_MANO_PINZA_ABIERTA), dict(cfg.AUTONOMO_TAREA_MANO_PINZA_CERRADA)

    def post(s):
        return {k: ab.get(k, 0.0) + s * (ce.get(k, 0.0) - ab.get(k, 0.0)) for k in set(ab) | set(ce)}
    filas = []
    for s in np.linspace(0.0, 1.0, 21):
        y = yemas_mano(ruta_xml, tuple(sorted((k, round(v, 4)) for k, v in post(s).items())), 0.004)
        filas.append((s, float(np.linalg.norm(y["pulgar"] - y["indice"])), (y["pulgar"] + y["indice"]) / 2))
    k = next((i for i, f in enumerate(filas) if f[1] <= diametro), None)
    if not k:
        raise SystemExit(f"Una pelota de {diametro * 100:.1f} cm no cabe en la pinza del modelo.")
    (s0, g0, p0), (s1, g1, p1) = filas[k - 1], filas[k]
    u = (g0 - diametro) / max(g0 - g1, 1e-9)
    return post(s0 + u * (s1 - s0)), p0 + u * (p1 - p0)


def linea_mano_mount(ruta_xml, theta_deg):
    """La línea <body name="mano_mount" …> de ruta_xml girada theta alrededor del eje de la brida (z de wrist3)."""
    from skill_shield.autonomo.choque_mano import _T_local
    import xml.etree.ElementTree as ET
    el = next(b for b in ET.parse(ruta_xml).getroot().iter("body") if b.get("name") == "mano_mount")
    T = _T_local(el)
    a = math.radians(theta_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    p, R = Rz @ T[:3, 3], Rz @ T[:3, :3]
    w = math.sqrt(max(0.0, 1 + np.trace(R))) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    yq = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return (f'<body name="mano_mount" pos="{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" '
            f'quat="{w:.6f} {x:.6f} {yq:.6f} {z:.6f}"')


def sujetar_mano(cfg, postura):
    from skill_shield.autonomo.tarea import ManoActiva
    m = ManoActiva(getattr(cfg, "PUERTO_HARDWARE", "/dev/ttyUSB0"))
    if not m.conectar():
        raise SystemExit(f"No se pudo conectar la mano: {m.error}")
    m.enviar(postura)
    print("✋ Mano en la pinza de contacto y con par. Mueve el brazo y apunta las poses. Ctrl+C para soltarla.")
    try:
        import time
        while True:
            time.sleep(0.5)
            m.enviar(postura)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            m.cerrar()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Punto de pinza REAL de la mano (método de los 4 puntos del TCP)")
    ap.add_argument("poses", nargs="?", help="archivo con las poses (grados de la controladora, 6 por línea)")
    ap.add_argument("--diametro", type=float, default=0.070, help="diámetro de la pelota (m)")
    ap.add_argument("--mano", action="store_true", help="poner la mano en la pinza de contacto y sujetarla")
    ap.add_argument("--raiz", default=None, help="carpeta del proyecto (por defecto, la de arriba de este script)")
    a = ap.parse_args()
    _raiz(a.raiz)
    import config as cfg
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    postura, p_mod = postura_contacto(cfg, a.diametro, cfg.RUTA_XML)
    if a.mano:
        print("Postura de contacto:", {k: round(v, 3) for k, v in sorted(postura.items())})
        return sujetar_mano(cfg, postura)
    if not a.poses:
        ap.error("falta el archivo de poses (o --mano)")
    Q = leer_poses(a.poses)
    if len(Q) < 3:
        raise SystemExit(f"Hacen falta al menos 3 poses (mejor 5-6); en {a.poses} hay {len(Q)}.")
    cad = cargar_cadena(cfg.RUTA_XML, verbose=False)
    Ts = [cad.fk_marcos(a_mjcf(q, cfg))[5] for q in Q]
    giros = [math.degrees(math.acos(np.clip((np.trace(Ti[:3, :3].T @ Tj[:3, :3]) - 1) / 2, -1, 1)))
             for k, Ti in enumerate(Ts) for Tj in Ts[k + 1:]]
    p, b, res, cond = resolver(Ts)
    print(f"{len(Q)} poses · giro máximo entre ellas {max(giros):.0f}°")
    if max(giros) < 30 or cond < 1e-3:
        print("⚠️  Las orientaciones se parecen demasiado: el resultado no es fiable. Gira más la mano entre poses.")
    for i, r in enumerate(res, 1):
        print(f"   pose {i}: {r * 1000:5.1f} mm" + ("   ← ¿pelota movida o mal centrada?" if r > 0.008 else ""))
    print(f"error medio {res.mean() * 1000:.1f} mm (bueno: < 4 mm)")
    print(f"\nPelota (mundo del MJCF): {np.round(b * 100, 1)} cm")
    print(f"Punto de pinza REAL en la brida:     {np.round(p * 100, 1)} cm "
          f"(a {math.hypot(p[0], p[1]) * 100:.1f} cm del eje, azimut {math.degrees(math.atan2(p[1], p[0])):.0f}°)")
    print(f"Punto de pinza del MODELO (gemelo):  {np.round(p_mod * 100, 1)} cm "
          f"(a {math.hypot(p_mod[0], p_mod[1]) * 100:.1f} cm del eje, azimut "
          f"{math.degrees(math.atan2(p_mod[1], p_mod[0])):.0f}°)")
    print(f"→ el modelo se equivoca en {np.linalg.norm(p - p_mod) * 100:.1f} cm")
    th = math.degrees(math.atan2(p[1], p[0]) - math.atan2(p_mod[1], p_mod[0]))
    th = (th + 180.0) % 360.0 - 180.0
    a_ = math.radians(th)
    Rz = np.array([[math.cos(a_), -math.sin(a_), 0.0], [math.sin(a_), math.cos(a_), 0.0], [0.0, 0.0, 1.0]])
    resto = p - Rz @ p_mod
    print(f"\nGiro de la mano real respecto al gemelo, alrededor del eje de la brida: {th:+.1f}°")
    print(f"Lo que quedaría tras girar el gemelo: {np.linalg.norm(resto) * 100:.1f} cm "
          f"(radial {(math.hypot(p[0], p[1]) - math.hypot(p_mod[0], p_mod[1])) * 100:+.1f}, "
          f"a lo largo del eje {resto[2] * 100:+.1f} cm)")
    if abs(th) >= 2.0:
        print("\nEn fairino_mjcf.xml (y en modelo/fairino_description/mjcf/fairino_mjcf.xml), cambia la línea de "
              "mano_mount por (y deja CALIB_MANO_GIRO_DEG = 0 en config.py):")
        print("   " + linea_mano_mount(cfg.RUTA_XML, th) + ' childclass="mano" gravcomp="1">')
        print("y regenera los archivos de cuRobo:\n   python -m skill_shield.autonomo.curobo_fr5.urdf\n"
              "   python -m skill_shield.autonomo.curobo_fr5.robot\n"
              "Luego vuelve a pasar este script con las MISMAS poses: el giro debería salir < 2°.")
    else:
        print("El giro ya cuadra (< 2°).")
        if np.linalg.norm(resto) > 0.005:
            print(f"Pero el punto de pinza del modelo sigue a {np.linalg.norm(resto) * 100:.1f} cm del real (dedos, pulgar"
                  "…). Para que la tarea use el MEDIDO, en config.py:\n"
                  f"   AUTONOMO_TAREA_PINZA_PUNTO_BRIDA_M = ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")


if __name__ == "__main__":
    main()
