#!/usr/bin/env python3
"""
Ajustar el GEMELO DIGITAL al FR5 real (9-nov).

Los ángulos del simulador ya son los del robot (las líneas [GEMELO] coinciden), pero el
dibujo no: el brazo real apunta a otro sitio. Hay dos posibles causas y este script las
separa, SOLO LEYENDO el robot (no manda ningún movimiento):

  1) CADENA (convenio de los joints): se compara la posición de la BRIDA que da la propia
     controladora (en su marco base) con la del MJCF para los mismos ángulos, probando
     todos los giros del marco base alrededor de z y los signos de los joints. Si cuadra
     con signos +1 (error de pocos mm), la cadena del MJCF es correcta.
  2) MONTAJE: cómo está atornillada la base del FR5 en la placa del soporte (giro alrededor
     del eje de joint1). Se ve en el visor: el brazo de MuJoCo sigue al robot real en
     directo y escribes en la terminal el giro (0, 90, 180, 270 o cualquier número) hasta
     que coincida. MUEVE EL ROBOT (WebApp, modo arrastre) para comprobarlo en varias poses.
     Con "g" se guarda en fairino_mjcf.xml (copia de seguridad en .bak_gemelo).

Uso (con Skill_Shield_Nivel_6 CERRADO: un solo programa conectado al robot):
    python diagnosticos/ajustar_gemelo.py                 # robot en 192.168.58.2
    python diagnosticos/ajustar_gemelo.py --sin-robot 65.2 -60.8 69.8 -8.2 97.4 2.0
"""
import argparse
import itertools
import math
import os
import re
import shutil
import sys
import threading
import time

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import mujoco                    # noqa: E402
import mujoco.viewer             # noqa: E402
import config as cfg             # noqa: E402

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
D6_BRIDA = 0.100                 # brida = origen de wrist3_Link + 100 mm en su eje z (DH oficial)


# ── cuaterniones (w, x, y, z) ──
def q_mult(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def q_z(grados):
    t = math.radians(grados) / 2
    return np.array([math.cos(t), 0.0, 0.0, math.sin(t)])


def leer_quat_base(ruta):
    m = re.search(r'<body name="base_link" pos="[^"]*" quat="([^"]*)"', open(ruta, encoding="utf-8").read())
    if not m:
        sys.exit(f"✗ No encuentro <body name=\"base_link\" ... quat=\"...\"> en {ruta}")
    return np.array([float(v) for v in m.group(1).split()])


def guardar_quat_base(ruta, q):
    txt = open(ruta, encoding="utf-8").read()
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    nuevo = " ".join(f"{v:.9g}" for v in q)
    txt2, n = re.subn(r'(<body name="base_link" pos="[^"]*" quat=")[^"]*(")', rf"\g<1>{nuevo}\g<2>", txt, count=1)
    if n != 1:
        sys.exit("✗ No he podido escribir el quat de base_link")
    shutil.copy2(ruta, ruta + ".bak_gemelo")
    open(ruta, "w", encoding="utf-8").write(txt2)
    return nuevo


# ── robot (solo lectura) ──
class LectorFR5(threading.Thread):
    def __init__(self, ip):
        super().__init__(daemon=True)
        if getattr(cfg, "FR5_SDK_RUTA", None):
            sys.path.insert(0, cfg.FR5_SDK_RUTA)
        from fairino import Robot
        self.robot = Robot.RPC(ip)
        if not getattr(Robot.RPC, "is_connect", True):
            sys.exit(f"✗ Sin conexión con el FR5 en {ip}. ¿Está cerrado Skill_Shield_Nivel_6?")
        self.q = None
        self.brida = None
        self.activo = True

    def run(self):
        while self.activo:
            try:
                r = self.robot.GetActualJointPosDegree(1)
                if isinstance(r, (tuple, list)) and r[0] == 0:
                    self.q = [float(v) for v in r[1]]
                r = self.robot.GetActualToolFlangePose(1)
                if isinstance(r, (tuple, list)) and r[0] == 0:
                    self.brida = [float(v) for v in r[1]]
            except Exception as e:
                print(f"⚠ lectura: {e}")
            time.sleep(0.05)

    def cerrar(self):
        self.activo = False
        try:
            self.robot.CloseRPC()
        except Exception:
            pass


# ── cinemática del MJCF ──
class Gemelo:
    def __init__(self, ruta):
        self.m = mujoco.MjModel.from_xml_path(ruta)
        self.d = mujoco.MjData(self.m)
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.w3 = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "wrist3_Link")
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINTS]
        self.qadr = [int(self.m.jnt_qposadr[j]) for j in jid]
        self.quat0 = self.m.body_quat[self.base].copy()
        self.signo = np.asarray(getattr(cfg, "FR5_SIGNO", (1,) * 6), float)
        self.offset = np.asarray(getattr(cfg, "FR5_OFFSET_DEG", (0,) * 6), float)

    def poner(self, q_fr5_deg):
        """Ángulos de la controladora → MJCF (inversa de fr5_servo)."""
        q = np.radians((np.asarray(q_fr5_deg, float) - self.offset) * self.signo)
        for a, v in zip(self.qadr, q):
            self.d.qpos[a] = v
        mujoco.mj_kinematics(self.m, self.d)

    def girar_base(self, grados):
        self.m.body_quat[self.base] = q_mult(self.quat0, q_z(grados))

    def brida_en_base(self, q_mjcf_deg):
        """Posición (m) de la brida en el marco de base_link para ángulos MJCF en grados."""
        for a, v in zip(self.qadr, np.radians(q_mjcf_deg)):
            self.d.qpos[a] = v
        mujoco.mj_kinematics(self.m, self.d)
        Rb = self.d.xmat[self.base].reshape(3, 3)
        R3 = self.d.xmat[self.w3].reshape(3, 3)
        p = self.d.xpos[self.w3] + R3[:, 2] * D6_BRIDA
        return Rb.T @ (p - self.d.xpos[self.base])


def comprobar_cadena(gem, q_fr5, brida_mm):
    """Busca el giro del marco base (z) y los signos de los joints que mejor explican la
    brida que da la controladora."""
    p_ctrl = np.asarray(brida_mm[:3], float) / 1000.0
    q_mj = (np.asarray(q_fr5, float) - gem.offset) * gem.signo
    giros = np.radians(np.arange(0, 360, 1.0))
    Rz = np.stack([np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
                   for t in giros])
    resultados = []
    for signos in itertools.product((1, -1), repeat=6):
        p = gem.brida_en_base(q_mj * np.asarray(signos))
        err = np.linalg.norm(Rz @ p - p_ctrl, axis=1)
        k = int(np.argmin(err))
        resultados.append((float(err[k]), signos, k))
    resultados.sort(key=lambda r: r[0])
    base = next(r for r in resultados if r[1] == (1,) * 6)
    print("\n── 1) CADENA: brida de la controladora vs brida del MJCF ──")
    print(f"   controladora (marco base, mm): {np.round(np.asarray(brida_mm[:3]), 1)}")
    print(f"   MJCF con los signos actuales : error {1000 * base[0]:.1f} mm (marco base girado {base[2]}°)")
    mejor = resultados[0]
    if base[0] < 0.010:
        print("   ✓ La cadena del MJCF coincide con la del robot: el problema es solo el MONTAJE.")
    else:
        print(f"   ⚠ NO coincide. Mejor combinación: signos {mejor[1]} → error {1000 * mejor[0]:.1f} mm "
              f"(marco base girado {mejor[2]}°)")
        print("     Mándale esta salida a quien mantenga el código: hay que corregir el convenio de joints.")
    return base[0] < 0.010


def main():
    ap = argparse.ArgumentParser(description="Ajustar el gemelo digital al FR5 real (solo lectura)")
    ap.add_argument("--ip", default=getattr(cfg, "FR5_IP", "192.168.58.2"))
    ap.add_argument("--sin-robot", nargs=6, type=float, metavar="Q", help="ángulos fijos (°), sin conectar")
    a = ap.parse_args()

    ruta = os.path.join(RAIZ, "fairino_description", "mjcf", "fairino_mjcf.xml")
    gem = Gemelo(ruta)
    lector = None
    if a.sin_robot:
        q_fijo = list(a.sin_robot)
    else:
        lector = LectorFR5(a.ip)
        lector.start()
        t0 = time.time()
        while (lector.q is None or lector.brida is None) and time.time() - t0 < 5:
            time.sleep(0.1)
        if lector.q is None:
            sys.exit("✗ No llega la pose del robot")
        q_fijo = None
        print(f"✓ FR5: joints (°) {np.round(lector.q, 1)}")
        if lector.brida is not None and any(abs(v) > 1e-6 for v in lector.brida[:3]):
            comprobar_cadena(gem, lector.q, lector.brida)
        else:
            print("⚠ La controladora no da la pose de la brida: me salto la comprobación de la cadena")

    estado = {"giro": 0.0, "salir": False}
    print("\n── 2) MONTAJE: compara el visor con el robot real ──")
    print("   Escribe el giro de la base en grados (0, 90, 180, 270, -15...) y Enter.")
    print("   'g' = guardar el giro actual en fairino_mjcf.xml · 'q' = salir sin guardar\n")

    def leer_teclado():
        while not estado["salir"]:
            try:
                t = input(f"giro actual {estado['giro']:.0f}° > ").strip().lower()
            except EOFError:
                t = "q"
            if t in ("q", "salir"):
                estado["salir"] = True
            elif t == "g":
                nuevo = guardar_quat_base(ruta, q_mult(gem.quat0, q_z(estado["giro"])))
                print(f"💾 Guardado en fairino_mjcf.xml: quat=\"{nuevo}\" (copia en .bak_gemelo)")
                print("   Pon también  giro_base_grados = "
                      f"{estado['giro']:.0f}  en decoracion/crear_soporte_hombros.py (MEDIDAS)")
                estado["salir"] = True
            else:
                try:
                    estado["giro"] = float(t)
                except ValueError:
                    print("   ? escribe un número, 'g' o 'q'")
    threading.Thread(target=leer_teclado, daemon=True).start()

    with mujoco.viewer.launch_passive(gem.m, gem.d) as visor:
        while visor.is_running() and not estado["salir"]:
            gem.girar_base(estado["giro"])
            gem.poner(q_fijo if q_fijo is not None else lector.q)
            visor.sync()
            time.sleep(0.03)
    if lector is not None:
        lector.cerrar()


if __name__ == "__main__":
    main()
