#!/usr/bin/env python3
"""
EPISODIO por poses (9-nov): el BRAZO va de pose en pose con Enter y la MANO se teleopera con
el guante MANUS como siempre (OrcaHand Touch real + háptica de sus sensores).

    ./venv_nivel6/bin/python episodio_poses.py episodio_poses.txt
    ./venv_nivel6/bin/python episodio_poses.py episodio_poses.txt --grabar coger_pieza
    ./venv_nivel6/bin/python episodio_poses.py episodio_poses.txt --simulacro    # el FR5 no se mueve
    ./venv_nivel6/bin/python episodio_poses.py episodio_poses.txt --sin-robot    # solo simulador

El fichero tiene una pose por línea: 6 ángulos en GRADOS DE LA CONTROLADORA (los de la WebApp
y los de la ventana de sliders; el botón "Imprimir pose" de los sliders escribe la línea ya
hecha). La 1ª es la posición INICIAL; luego cada Enter lleva a la siguiente. Admite comentarios
con # y un nombre opcional delante:  agarre: 12.5, -80, 95, -60, 90, 0

Teclas (en ESTA terminal, seguidas de Enter):
    Enter = ir a la siguiente pose · s = PARAR (deja de mandar al FR5) · e = borrar el error
    del FR5 (colisión…) · r = volver a la inicial y repetir · q = salir. La seta física sigue
    siendo la parada de verdad.

Con --grabar TAREA se graba (brazo + mano, el mismo formato que [G]) desde que sale de la pose
inicial hasta que llega a la última, en episodios/TAREA/. Cada "r" es un episodio nuevo.

Requisitos: Skill_Shield_Nivel_6 CERRADO (un solo programa con el robot, la mano y el guante),
FR5 habilitado y en modo automático (como para la teleoperación).
"""
import argparse
import math
import os
import queue
import re
import sys
import threading
import time

import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)

import config as cfg                                                     # noqa: E402
from skill_shield.hardware.fr5_servo import FR5Servo, paso_limitado      # noqa: E402
from skill_shield.interfaz.sliders_brazo import limites_sliders           # noqa: E402

TOL_LLEGADA_DEG = 1.0


# ══════════════════════════════════════════════════════════════════════════════
def leer_poses(ruta):
    poses = []
    for n, linea in enumerate(open(ruta, encoding="utf-8"), 1):
        linea = linea.split("#", 1)[0].strip()
        if not linea:
            continue
        nombre = None
        if ":" in linea:
            nombre, linea = (t.strip() for t in linea.split(":", 1))
        nums = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", linea)]
        if len(nums) != 6:
            sys.exit(f"✗ {ruta}:{n}: hacen falta 6 ángulos y hay {len(nums)}")
        poses.append((nombre or f"pose {len(poses) + 1}", nums))
    if len(poses) < 2:
        sys.exit(f"✗ {ruta}: hacen falta al menos 2 poses (inicial + destino)")
    return poses


def comprobar_limites(poses, lim):
    mal = []
    for nombre, q in poses:
        fuera = [f"J{i + 1}={v:.1f}° (rango {a:.0f}..{b:.0f})" for i, (v, (a, b)) in enumerate(zip(q, lim))
                 if not a <= v <= b]
        if fuera:
            mal.append(f"   {nombre}: " + ", ".join(fuera))
    if mal:
        sys.exit("✗ Poses fuera del rango permitido (el de los sliders):\n" + "\n".join(mal))


def fmt(q):
    return "[" + ", ".join(f"{v:7.1f}" for v in q) + "]"


class Teclado(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.cola = queue.Queue()

    def run(self):
        while True:
            try:
                self.cola.put(input().strip().lower())
            except EOFError:
                self.cola.put("q")
                return

    def leer(self):
        try:
            return self.cola.get_nowait()
        except queue.Empty:
            return None


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Episodio: brazo por poses con Enter + mano con MANUS")
    ap.add_argument("poses", help="fichero con una pose por línea (6 ángulos, grados de la controladora)")
    ap.add_argument("--grabar", metavar="TAREA", help="grabar el episodio en episodios/TAREA/")
    ap.add_argument("--vel", type=float, default=None, help="velocidad máx. por joint (°/s)")
    ap.add_argument("--simulacro", action="store_true", help="conecta al FR5 pero NO lo mueve")
    ap.add_argument("--sin-robot", action="store_true", help="sin FR5: el brazo solo en el simulador")
    ap.add_argument("--sin-mano", action="store_true", help="sin OrcaHand física")
    ap.add_argument("--sin-guante", action="store_true", help="sin guante MANUS")
    ap.add_argument("--sin-visor", action="store_true", help="sin ventana de MuJoCo")
    a = ap.parse_args()

    espacio = vars(cfg)
    poses = leer_poses(a.poses)
    comprobar_limites(poses, limites_sliders(espacio))
    vel = min(float(a.vel or cfg.EPISODIO_FR5_VEL_MAX_DEG_S),
              float(getattr(cfg, "FR5_VEL_MAX_DEG_S", 60.0)))
    signo = np.asarray(getattr(cfg, "FR5_SIGNO", (1,) * 6), float)
    offset = np.asarray(getattr(cfg, "FR5_OFFSET_DEG", (0,) * 6), float)
    a_rad = lambda q_deg: np.radians((np.asarray(q_deg, float) - offset) * signo)   # noqa: E731
    a_deg = lambda q_rad: signo * np.degrees(np.asarray(q_rad, float)) + offset     # noqa: E731

    print("📋 Poses del episodio (grados de la controladora):")
    for nombre, q in poses:
        print(f"   {nombre:>12s}: {fmt(q)}")
    print(f"   velocidad máx. {vel:.0f} °/s por joint (la 1ª ida a la inicial, a "
          f"{getattr(cfg, 'FR5_VEL_APROX_DEG_S', 10.0):.0f} °/s)")

    # ── FR5 (proceso con fork: ANTES de abrir visor, guante y mano) ──
    fr5 = None
    simulacro = a.simulacro or bool(getattr(cfg, "EPISODIO_FR5_SIMULACRO", False))
    if not a.sin_robot:
        par = dict(espacio)
        par.update(FR5_SIMULACRO=simulacro, FR5_VEL_MAX_DEG_S=vel)
        fr5 = FR5Servo.desde_config(par)
        if not fr5.iniciar():
            sys.exit("✗ Sin conexión con el FR5 (¿Skill_Shield cerrado? ¿cable?). Usa --sin-robot para probar.")

    import mujoco
    from skill_shield.hardware.sincronizacion import construir_hardware_target
    from skill_shield.ia.grabador import GrabadorEpisodio
    from skill_shield.interfaz.fantasma import FantasmaBrazo

    modelo = mujoco.MjModel.from_xml_path(cfg.RUTA_XML)
    datos = mujoco.MjData(modelo)
    jids = [mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_JOINT, j) for j in cfg.ARM_JOINTS]
    qadr = [int(modelo.jnt_qposadr[j]) for j in jids]
    dadr = [int(modelo.jnt_dofadr[j]) for j in jids]
    cidx = [mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{j}") for j in cfg.ARM_JOINTS]
    if min(cidx) < 0:
        cidx = list(range(6))
    for i, v in enumerate(cfg.POSICION_STANDBY):
        datos.ctrl[cfg.OFFSET_MANO + i] = float(v)

    # pose de partida del brazo: la real (o la inicial del fichero sin robot)
    if fr5 is not None:
        q_ini_rad = fr5.q_real_rad(max_edad_s=5.0)
        q_cmd_deg = a_deg(q_ini_rad) if q_ini_rad is not None else np.asarray(poses[0][1], float)
    else:
        q_cmd_deg = np.asarray(poses[0][1], float)
    v_cmd = np.zeros(6)
    for k in range(6):
        datos.qpos[qadr[k]] = datos.ctrl[cidx[k]] = a_rad(q_cmd_deg)[k]
    mujoco.mj_forward(modelo, datos)
    print(f"🦾 Brazo ahora: {fmt(q_cmd_deg)}")

    # ── mano física y guante, como en Skill_Shield ──
    mano = None
    if not a.sin_mano:
        from orca_hardware import OrcaHardwareWrapper
        mano = OrcaHardwareWrapper(puerto=cfg.PUERTO_HARDWARE)
        mano.conectar()
        if not mano.conectado:
            print("⚠ Mano física sin conectar: sigo sin ella")
            mano = None
    guante = None
    if not a.sin_guante:
        from skill_shield.mano.manus_guante import GuanteManus
        guante = GuanteManus()
        if guante.iniciar():
            guante.aplicar = True              # los dedos siguen al guante desde el principio
            guante.haptica(modelo, datos)      # (le da el modelo para la háptica simulada)
        else:
            print("⚠ Guante MANUS sin iniciar: la mano se queda en reposo")
            guante = None

    grabador = None
    if a.grabar:
        grabador = GrabadorEpisodio.desde_modelo(
            modelo, cfg.ARM_JOINTS, cfg.OFFSET_MANO, cfg.N_ACT,
            fps=float(getattr(cfg, "IA_FPS", 30.0)), robot=str(getattr(cfg, "ROBOT_BRAZO", "FR5")),
            recortar=bool(getattr(cfg, "IA_RECORTAR_QUIETOS", True)))
        carpeta_ep = os.path.join("episodios", a.grabar)

    visor = None
    fantasma = None
    if not a.sin_visor:
        import mujoco.viewer
        visor = mujoco.viewer.launch_passive(modelo, datos)
        try:
            fantasma = FantasmaBrazo(modelo, cfg.ARM_JOINTS)
        except Exception as e:
            print(f"⚠ sin brazo fantasma ({e})")

    teclado = Teclado()
    teclado.start()

    # ── estado del episodio ──
    idx = -1               # pose objetivo actual (-1 = aún no se ha mandado nada)
    moviendo = False       # ¿el brazo está yendo hacia el objetivo? (FR5: recibiendo ServoJ)
    llegado = True
    t_llegada = None
    dt = float(modelo.opt.timestep)
    t_sig = time.perf_counter()
    t_mano = t_visor = t_fant = 0.0

    def ir_a(i):
        nonlocal idx, moviendo, llegado
        idx = i
        llegado = False
        if fr5 is not None and not moviendo:
            fr5.publicar(a_rad(poses[idx][1]))
            fr5.habilitar(True)              # ServoMoveStart desde la pose real
        moviendo = True
        print(f"▶ Hacia '{poses[idx][0]}' {fmt(poses[idx][1])}")

    def parar(motivo):
        nonlocal moviendo, v_cmd
        if fr5 is not None and moviendo:
            fr5.habilitar(False)
        moviendo = False
        v_cmd = np.zeros(6)
        print(f"⏹ {motivo}")

    def parar_grabacion():
        if grabador is not None and grabador.activo:
            ruta, n = grabador.parar(carpeta_ep)
            if ruta:
                print(f"💾 Episodio guardado: {ruta} ({n} fotogramas)")

    print("\n⏎  Enter = ir a la posición INICIAL ('{}') · s = parar · e = reset errores FR5 · r = repetir · q = salir"
          .format(poses[0][0]))
    try:
        while True:
            # ── teclado ──
            orden = teclado.leer()
            if orden is not None:
                if orden in ("q", "salir"):
                    break
                if orden == "s":
                    parar("PARADO (Enter para seguir hacia el mismo objetivo)")
                    parar_grabacion()
                elif orden == "e":
                    if fr5 is not None:
                        fr5.resetear()
                        print("🔄 RESET del FR5 pedido (ResetAllError). Enter para seguir")
                elif orden == "r":
                    parar_grabacion()
                    ir_a(0)
                elif orden == "":
                    if idx >= 0 and not moviendo and not llegado:
                        ir_a(idx)                                    # reanudar tras "s"
                    elif idx < 0:
                        ir_a(0)
                    elif not llegado:
                        print(f"   espera a que llegue a '{poses[idx][0]}' (o s = parar)")
                    elif idx + 1 < len(poses):
                        if idx == 0 and grabador is not None and not grabador.activo:
                            grabador.iniciar(a.grabar, datos)
                            print(f"🔴 GRABANDO (tarea '{a.grabar}')")
                        ir_a(idx + 1)
                    else:
                        print("ℹ Ya está en la última pose: r = repetir, q = salir")
                else:
                    print("   ? Enter, s, e, r o q")

            # ── brazo: objetivo al FR5 / paso en el simulador ──
            if fr5 is not None:
                if fr5.en_error and moviendo:
                    _c = fr5.estado()["codigo"]
                    parar(f"FR5 en ERROR (código {_c[0]}/{_c[1]}): e = borrar el error, luego Enter")
                    parar_grabacion()
                if moviendo:
                    fr5.publicar(a_rad(poses[idx][1]))               # (watchdog: cada vuelta)
                e = fr5.estado()
                q_cmd_deg = np.asarray(e["q_deg"], float)
                q_real_rad = None if simulacro else fr5.q_real_rad(max_edad_s=0.3)
            else:
                if idx >= 0 and moviendo:
                    q_cmd_deg, v_cmd = paso_limitado(q_cmd_deg, v_cmd, np.asarray(poses[idx][1], float),
                                                     vel, float(getattr(cfg, "FR5_ACEL_MAX_DEG_S2", 300.0)), dt)
                q_real_rad = None
            q_med_deg = a_deg(q_real_rad) if q_real_rad is not None else q_cmd_deg

            if idx >= 0 and not llegado and \
                    np.max(np.abs(q_med_deg - np.asarray(poses[idx][1]))) <= TOL_LLEGADA_DEG:
                llegado = True
                print(f"✓ En '{poses[idx][0]}'" + ("  (Enter = siguiente)" if idx + 1 < len(poses)
                                                     else "  — FIN: r = repetir, q = salir"))
                if idx + 1 == len(poses):
                    t_llegada = time.perf_counter()

            if t_llegada is not None and time.perf_counter() - t_llegada > 1.0:
                t_llegada = None
                parar_grabacion()

            # ── simulador: brazo (gemelo = real) + mano (guante) ──
            for k in range(6):
                datos.ctrl[cidx[k]] = math.radians((q_cmd_deg[k] - offset[k]) * signo[k])
            q_vis = q_real_rad if q_real_rad is not None else a_rad(q_cmd_deg)
            for k in range(6):
                datos.qpos[qadr[k]] = float(q_vis[k])
                datos.qvel[dadr[k]] = 0.0
            if guante is not None:
                guante.escribir_ctrl(datos, dt)
            if grabador is not None:
                grabador.muestrear(datos)
            mujoco.mj_step(modelo, datos)

            ahora = time.perf_counter()
            if mano is not None and ahora - t_mano >= 0.02:          # mano física a 50 Hz
                t_mano = ahora
                mano.enviar_posiciones(construir_hardware_target(datos))
            if visor is not None and ahora - t_visor >= 1 / 60:
                t_visor = ahora
                if not visor.is_running():
                    break
                if fantasma is not None and idx >= 0 and ahora - t_fant >= 1 / 20:
                    t_fant = ahora
                    mujoco.mj_kinematics(modelo, datos)
                    fantasma.dibujar(visor, a_rad(poses[idx][1]), punta_real=datos.xpos[fantasma.punta].copy())
                visor.sync()

            # ── ritmo de la física (tiempo real) ──
            t_sig += dt
            resto = t_sig - time.perf_counter()
            if resto > 0:
                time.sleep(resto)
            elif resto < -0.1:
                t_sig = time.perf_counter()
    except KeyboardInterrupt:
        print("\n(Ctrl+C)")
    finally:
        parar("Saliendo: FR5 quieto")
        parar_grabacion()
        if grabador is not None:
            grabador.esperar()
        if fr5 is not None:
            time.sleep(0.1)
            fr5.parar()
        if guante is not None:
            guante.cerrar()
        if mano is not None:
            mano.desconectar()
        if visor is not None:
            visor.close()
        print("👋 Fin del episodio.")


if __name__ == "__main__":
    main()
