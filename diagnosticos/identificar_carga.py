"""
Carga de la brida (masa y CENTRO DE MASAS de la mano + adaptador) medida CON EL ROBOT (18-nov).

Con FR5_CARGA_CDM_MM = None la controladora se queda con el centro de masas que tuviera (a menudo (0,0,0),
en la brida): su modelo dinámico espera otros pares en la muñeca y la detección de colisiones salta sin
choque. Hay tres formas de arreglarlo:

  1. La identificación de carga de la WebApp del FR5 (la «oficial»).
  2. Este script: el SDK tiene LoadIdentify* (LoadIdentifyDynFilterInit / DynVarInit / Main / GetResult),
     pero FAIRINO no documenta ni la trayectoria que hay que hacer ni qué es «gain» (su ejemplo mete números
     fijos), así que aquí se mide de otra forma que sí se puede comprobar: con el robot QUIETO en ~12
     posturas de la muñeca (J4, J5, J6), el par de gravedad que miden J4-J6 depende solo de la masa y del
     centro de masas de lo que cuelga de ellos. Restando lo de los eslabones de la muñeca (masas del MJCF),
     es un ajuste lineal por mínimos cuadrados. La masa sale también: si se parece a la pesada en la báscula
     (FR5_CARGA_KG), el centro de masas es de fiar.
  3. diagnosticos/estimar_cdm_mano.py: solo con la geometría del modelo (sin mover nada).

    python diagnosticos/identificar_carga.py --leer        # solo leer lo que tiene ahora la controladora
    python diagnosticos/identificar_carga.py               # medir: mueve SOLO la muñeca, con el pedal
    python diagnosticos/identificar_carga.py --recalcular  # rehacer el ajuste con las muestras guardadas

Para medir: deja el robot en una postura cómoda, con la mano lejos de todo (la muñeca va a girar ±90° el J6
y ±45° el J5: mira los cables), cierra el modo autónomo y mantén la tecla del pedal (P) con la ventana delante.
Cada postura se comprueba antes (límites, choques del brazo y la mano a 5 cm del brazo, como la tarea) y el
robot va a ≤ 8 °/s. La mano tiene que estar QUIETA y sin tocar nada (sin par: el par de la mano mueve el centro
de masas de los dedos, pero poco). Tarda ~3 min.
"""
import argparse
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

G_MUNDO = np.array([0.0, 0.0, -9.81])
MUNECA = ("wrist1_Link", "wrist2_Link", "wrist3_Link")       # cuerpos de los marcos 3, 4 y 5 de la cadena
MUESTRAS = os.path.join(RAIZ, "diagnosticos", "carga_muestras.json")
DELTAS_DEG = [(0, 0, 0), (0, 30, 0), (0, -30, 0), (0, 0, 60), (0, 0, -60), (0, 30, 60), (0, -30, -60),
              (20, 0, 0), (-20, 0, 0), (20, 30, -60), (-20, -30, 60), (0, 45, 0), (0, -45, 0), (0, 0, 90),
              (0, 0, -90), (15, -45, 90), (-15, 45, -90)]


def inerciales_muneca(ruta_xml):
    """{cuerpo: (masa kg, centro de masas en su marco)} de los eslabones de la muñeca, del MJCF."""
    out = {}
    for b in ET.parse(ruta_xml).getroot().iter("body"):
        if b.get("name") in MUNECA:
            i = b.find("inertial")
            if i is not None:
                out[b.get("name")] = (float(i.get("mass", 0.0)),
                                      np.array([float(v) for v in i.get("pos", "0 0 0").split()]))
    return out


def filas(cadena, q, eslabones, g=G_MUNDO):
    """Para J4, J5 y J6 en la postura q: (A (3 x 4), par conocido (3,)), con
    par_gravedad_j = A_j · [m, m·cx, m·cy, m·cz] + conocido_j, siendo c el centro de masas de la carga en el
    marco de wrist3 (el 5 de la cadena) y 'conocido' lo de los eslabones de la muñeca."""
    m = cadena.fk_marcos(np.asarray(q, float))
    R5, o5 = m[5][:3, :3], m[5][:3, 3]
    A, conocido = np.zeros((3, 4)), np.zeros(3)
    for fila, j in enumerate((3, 4, 5)):                      # J4, J5, J6: eje z de los marcos 3, 4, 5
        a, oj = m[j][:3, 2], m[j][:3, 3]
        A[fila, 0] = a @ np.cross(o5 - oj, g)
        A[fila, 1:] = R5.T @ np.cross(g, a)                   # a·((R5 mc) × g) = (mc)·(R5ᵀ (g × a))
        for k, nombre in ((3, "wrist1_Link"), (4, "wrist2_Link"), (5, "wrist3_Link")):
            if k < j or nombre not in eslabones:
                continue
            mb, cb = eslabones[nombre]
            pb = m[k][:3, :3] @ cb + m[k][:3, 3]
            conocido[fila] += mb * (a @ np.cross(pb - oj, g))
    return A, conocido


def ajustar(cadena, Qs, Pares, eslabones, d6):
    """Mínimos cuadrados con un desfase por joint. Prueba los dos signos del par medido (el motor sujeta la
    gravedad: el par medido es el de gravedad cambiado de signo, según el convenio de la controladora) y se
    queda con el de masa positiva y menos residuo. Devuelve dict o None."""
    filas_A, filas_b = [], []
    for q, tau in zip(Qs, Pares):
        A, k = filas(cadena, q, eslabones)
        filas_A.append((A, k, np.asarray(tau, float)[3:6]))
    if len(filas_A) < 5:
        return None
    mejor = None
    for s in (1.0, -1.0):
        M, y = [], []
        for A, k, t in filas_A:
            for r in range(3):
                off = np.zeros(3)
                off[r] = 1.0
                M.append(np.r_[A[r], off])
                y.append(s * t[r] - k[r])
        M, y = np.array(M), np.array(y)
        x, *_ = np.linalg.lstsq(M, y, rcond=None)
        res = float(np.sqrt(np.mean((M @ x - y) ** 2)))
        if x[0] > 0 and (mejor is None or res < mejor["residuo_nm"]):
            c = x[1:4] / x[0]
            mejor = dict(masa_kg=float(x[0]), cdm_wrist3_m=c.tolist(), signo=s, residuo_nm=res,
                         desfases_nm=x[4:].tolist(),
                         cdm_brida_mm=((c - np.array([0.0, 0.0, d6])) * 1000.0).tolist(),
                         condicion=float(np.linalg.cond(M)), n=len(filas_A))
    return mejor


def informe(r, masa_bascula=None):
    c = r["cdm_brida_mm"]
    print(f"\nMasa: {r['masa_kg']:.2f} kg" + (f"  (báscula: {masa_bascula:.2f} kg)" if masa_bascula else "")
          + f" · residuo {r['residuo_nm']:.3f} N·m · {r['n']} posturas · condición {r['condicion']:.0f}")
    print(f"Centro de masas en la brida: x={c[0]:.1f}  y={c[1]:.1f}  z={c[2]:.1f} mm")
    fiable = True
    if masa_bascula and abs(r["masa_kg"] - masa_bascula) > 0.2 * masa_bascula:
        print("⚠️  La masa no cuadra con la báscula (>20 %): el par de la muñeca tiene rozamiento o desfases grandes; "
              "repite con la mano sin par y quieta, o usa la identificación de la WebApp.")
        fiable = False
    if r["condicion"] > 1e4:
        print("⚠️  Las posturas no bastan para separar bien el centro de masas (mal condicionado): añade más giro de J5/J6.")
        fiable = False
    print(f"\n→ config.py:  FR5_CARGA_CDM_MM = ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})"
          + ("" if fiable else "   (revísalo antes de usarlo)"))
    print("   (y FR5_CARGA_KG con lo de la báscula; fr5_servo los escribe en la controladora al conectar)")


def leer_controladora(ip):
    import config as cfg
    if getattr(cfg, "FR5_SDK_RUTA", None):
        sys.path.insert(0, cfg.FR5_SDK_RUTA)
    from fairino import Robot
    robot = Robot.RPC(ip)
    rw, rc = robot.GetTargetPayload(0), robot.GetTargetPayloadCog(0)
    print(f"Carga en la controladora: {rw[1]} kg · centro de masas {rc[1]} mm (códigos {rw[0]}, {rc[0]})")
    try:
        robot.CloseRPC()
    except Exception:
        pass


def posturas(q0, pl):
    """Posturas de la muñeca alrededor de q0 que pasan las comprobaciones de la tarea (límites, choques del
    brazo, mano a AUTONOMO_TAREA_MANO_HOLGURA del brazo), en un orden en el que cada tramo también vale."""
    cands = [q0 + np.radians([0, 0, 0, d4, d5, d6]) for d4, d5, d6 in DELTAS_DEG]
    validas = [q for q in cands if pl.motivo(q) is None]
    orden, actual = [], q0
    resto = validas[1:] if validas and np.allclose(validas[0], q0) else validas
    while resto:
        resto.sort(key=lambda q: float(np.max(np.abs(q - actual))))
        for i, q in enumerate(resto):
            n = max(2, int(math.ceil(float(np.max(np.abs(q - actual))) / math.radians(2.0))) + 1)
            if all(pl.motivo(actual + (q - actual) * u) is None for u in np.linspace(0, 1, n)[1:]):
                orden.append(q)
                actual = resto.pop(i)
                break
        else:
            break
    return [q0] + orden


def medir(args, cfg, G):
    import cv2
    from skill_shield.autonomo.tarea import Planificador
    from skill_shield.hardware.fr5_servo import ENVIANDO, FR5Servo
    from skill_shield.teleop_fr5.cinematica import cargar_cadena
    from skill_shield.teleop_fr5.pedal import Pedal
    cadena = cargar_cadena(cfg.RUTA_XML, verbose=False)
    pl = Planificador(cadena, dict(G, AUTONOMO_TAREA_PLANIFICADOR="propio"))
    pedal = Pedal.desde_config(cfg)
    if not pedal.hombre_muerto:
        sys.exit("✗ Hace falta un pedal de HOMBRE MUERTO (TELEOP_FR5_PEDAL = \"teclado\" o \"evdev\").")
    c = {k: getattr(cfg, k) for k in dir(cfg) if k.startswith("FR5_")}
    c["FR5_SIMULACRO"] = False
    c["FR5_VEL_MAX_DEG_S"] = min(float(c.get("FR5_VEL_MAX_DEG_S", 60.0)), args.vel)
    fr5 = FR5Servo.desde_config(c)
    if not fr5.iniciar():
        sys.exit("✗ Sin conexión con el FR5.")
    muestras = []
    try:
        t0 = time.time()
        while fr5.q_real_rad() is None and time.time() - t0 < 5:
            time.sleep(0.05)
        q0 = fr5.q_real_rad()
        if q0 is None:
            sys.exit("✗ No llega la pose real del FR5.")
        m0 = pl.motivo(q0)
        if m0:
            sys.exit(f"✗ La postura de inicio no pasa las comprobaciones: {m0}. Lleva el robot a otra.")
        objetivos = posturas(q0, pl)
        print(f"🦾 {len(objetivos)} posturas de la muñeca. Mantén {pedal.descripcion}. Q/Esc para salir.")
        if len(objetivos) < 6:
            sys.exit("✗ Desde aquí no salen bastantes posturas válidas: lleva la mano más lejos del brazo y de todo.")
        vel = math.radians(args.vel) * 0.8
        q_cmd, i, armado_antes = q0.copy(), 0, False
        quieto_desde, pares = None, []
        lienzo = np.zeros((120, 520, 3), np.uint8)
        while i < len(objetivos):
            cv2.imshow("Identificar carga (mantén el pedal)", lienzo)
            k = cv2.waitKeyEx(10)
            if k in (27, ord("q")):
                break
            pedal.tecla(k)
            pedal.actualizar()
            q_real = fr5.q_real_rad()
            if fr5.en_error:
                print("✗ La controladora marca error: paro.")
                break
            if pedal.armado and not armado_antes:
                q_cmd = q_real.copy() if q_real is not None else q_cmd
                fr5.publicar(q_cmd)
                fr5.habilitar(True)
            if not pedal.armado and armado_antes:
                fr5.habilitar(False)
            armado_antes = pedal.armado
            lienzo[:] = 0
            txt = f"postura {i + 1}/{len(objetivos)} " + ("· moviendo" if pedal.armado else "· PEDAL SUELTO")
            cv2.putText(lienzo, txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if not pedal.armado or q_real is None:
                quieto_desde, pares = None, []
                continue
            d = objetivos[i] - q_cmd
            paso = vel * 0.01
            q_cmd = objetivos[i].copy() if float(np.max(np.abs(d))) <= paso else q_cmd + d * (paso / float(np.max(np.abs(d))))
            m = pl.motivo(q_cmd)
            if m:
                print(f"✗ {m}: paro.")
                break
            fr5.publicar(q_cmd)
            llegado = np.allclose(q_cmd, objetivos[i]) and float(np.max(np.abs(q_real - objetivos[i]))) < math.radians(0.3)
            if not llegado or fr5.estado()["estado"] != ENVIANDO:
                quieto_desde, pares = None, []
                continue
            ahora = time.time()
            quieto_desde = quieto_desde or ahora
            if ahora - quieto_desde > 1.0:                        # 1 s para que deje de oscilar…
                p = fr5.par_real()
                if p is not None:
                    pares.append(p)
            if ahora - quieto_desde > 2.5 and len(pares) >= 20:   # …y 1,5 s de medida
                muestras.append(dict(q=q_real.tolist(), par=np.median(pares, axis=0).tolist()))
                print(f"   ✓ postura {i + 1}: par J4-J6 = {np.round(np.median(pares, axis=0)[3:], 3).tolist()} N·m")
                i, quieto_desde, pares = i + 1, None, []
    finally:
        fr5.habilitar(False)
        fr5.parar()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    if muestras:
        with open(MUESTRAS, "w", encoding="utf-8") as f:
            json.dump(dict(fecha=time.strftime("%Y-%m-%d %H:%M:%S"), muestras=muestras), f, indent=1)
        print(f"💾 {len(muestras)} muestras en {MUESTRAS}")
    return muestras


def main():
    import config as cfg
    from skill_shield.teleop_fr5.cinematica import DH_OFICIAL_FR5, cargar_cadena
    ap = argparse.ArgumentParser(description="Masa y centro de masas de la carga de la brida, con el robot")
    ap.add_argument("--leer", action="store_true", help="solo leer la carga que tiene la controladora")
    ap.add_argument("--recalcular", action="store_true", help="ajustar con las muestras guardadas")
    ap.add_argument("--vel", type=float, default=8.0, help="°/s como mucho")
    a = ap.parse_args()
    G = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    if a.leer:
        return leer_controladora(G.get("FR5_IP", "192.168.58.2"))
    if a.recalcular:
        muestras = json.load(open(MUESTRAS, encoding="utf-8"))["muestras"]
    else:
        muestras = medir(a, cfg, G)
    cadena = cargar_cadena(cfg.RUTA_XML, verbose=False)
    r = ajustar(cadena, [m["q"] for m in muestras], [m["par"] for m in muestras],
                inerciales_muneca(cfg.RUTA_XML), float(DH_OFICIAL_FR5["d6"]))
    if r is None:
        sys.exit("✗ Muy pocas muestras para el ajuste (hacen falta 5 o más).")
    informe(r, G.get("FR5_CARGA_KG"))


if __name__ == "__main__":
    main()
