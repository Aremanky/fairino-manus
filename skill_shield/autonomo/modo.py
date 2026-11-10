"""
Funcionamiento autónomo (10-nov) — paso 2 de la hoja de ruta: el esqueleto.

Lo lanza la última opción del menú de arranque (MODO_EJECUCION = "AUTONOMO"), con su propio
bucle, separado del main() de teleoperación de Skill_Shield_Nivel_6.py.

Qué hace ahora (criterio de «hecho» del paso 2):
  1. FR5Servo, arrancado LO PRIMERO (usa fork), SIEMPRE en simulacro y sin habilitar:
     solo lee la pose real. En este paso el robot NO se puede mover de ninguna forma.
  2. Proceso de cámara (skill_shield/autonomo/camara.py): la cámara de cabeza.
  3. Proceso web (skill_shield/autonomo/web.py): imagen + estado, abierta a la red local.
  4. MuJoCo con el brazo copiando la pose REAL del FR5 (gemelo digital).

Reparto en procesos (como fr5_servo y el puente táctil) para que la visión, la web y el
gemelo no se frenen entre sí. Se comunican por memoria compartida (canal.py):
  cámara ──imagen JPEG / info──▶ web
  este proceso ──estado JSON──▶ web

Para salir: cerrar la ventana de MuJoCo o Ctrl+C en la terminal.
"""
import json
import os
import subprocess
import sys
import time

CANAL_IMG = "skill_autonomo_img"
CANAL_INFO = "skill_autonomo_caminfo"
CANAL_ESTADO = "skill_autonomo_estado"


def _lanzar(cmd, cwd, nombre):
    try:
        proc = subprocess.Popen(cmd, cwd=cwd)
        print(f"   ▶ {nombre}: PID {proc.pid}")
        return proc
    except OSError as e:
        print(f"⚠️  No se pudo lanzar {nombre} ({e}).")
        return None


def _cerrar(proc, nombre, timeout=3.0):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        print(f"⚠️  {nombre} no terminó a tiempo → kill")
        proc.kill()
        proc.wait(1.0)


def _estado_fr5(fr5, simulacro):
    if fr5 is None:
        return dict(texto="desactivado (AUTONOMO_CON_FR5 = False)", en_linea=False)
    e = fr5.estado()
    q = fr5.q_real_rad()
    return dict(texto=e["texto"], en_linea=fr5.en_linea, en_error=fr5.en_error,
                colision=e["colision"], codigo=list(e["codigo"]), simulacro=simulacro,
                q_real_deg=[round(v, 2) for v in e["q_real_deg"]] if q is not None else None)


def ejecutar(cfg):
    """Punto de entrada del modo autónomo. `cfg` = el config.py ya cargado (un dict, p. ej.
    los globals() de Skill_Shield_Nivel_6.py). Devuelve 0 al terminar bien."""
    from skill_shield.autonomo.canal import Canal
    from skill_shield.hardware.fr5_servo import FR5Servo, sincronizar_mujoco

    raiz = os.path.dirname(os.path.abspath(sys.modules["config"].__file__)) \
        if "config" in sys.modules else os.getcwd()
    print("🧠 Funcionamiento autónomo — paso 2 (esqueleto): cámara + gemelo + web, SIN mover el robot.")

    # ── 1. FR5: lo primero (fork, antes de abrir el visor y de lanzar procesos) ──
    # Simulacro FORZADO: aunque config diga otra cosa, en este paso no se manda ningún ServoJ
    # y tampoco se escriben carga/TCP/límites en la controladora (solo se leen y se enseñan).
    fr5, simulacro = None, True
    if cfg.get("AUTONOMO_CON_FR5", True):
        c = dict(cfg)
        c["FR5_SIMULACRO"] = simulacro
        fr5 = FR5Servo.desde_config(c)
        if not fr5.iniciar():
            print("   Sin conexión con el FR5: el gemelo enseñará el brazo en reposo.")

    # ── 2. Canales de memoria compartida (este proceso es el dueño: los borra al salir) ──
    canal_img = Canal(CANAL_IMG, capacidad=2 << 20, crear=True)
    canal_info = Canal(CANAL_INFO, capacidad=64 << 10, crear=True)
    canal_estado = Canal(CANAL_ESTADO, capacidad=64 << 10, crear=True)

    # ── 3. Procesos de cámara y web ──
    py_vision = cfg.get("AUTONOMO_PYTHON_PERCEPCION") or sys.executable
    pid = str(os.getpid())
    cmd_cam = [py_vision, "-m", "skill_shield.autonomo.camara",
               "--canal-img", CANAL_IMG, "--canal-info", CANAL_INFO, "--padre", pid]
    if cfg.get("AUTONOMO_CAM_MXID"):
        cmd_cam += ["--mxid", str(cfg["AUTONOMO_CAM_MXID"])]
    if cfg.get("AUTONOMO_CAM_VENTANA", False):
        cmd_cam.append("--ventana")
    cmd_web = [sys.executable, "-m", "skill_shield.autonomo.web",
               "--host", str(cfg.get("AUTONOMO_WEB_HOST", "0.0.0.0")),
               "--puerto", str(cfg.get("AUTONOMO_WEB_PUERTO", 8080)),
               "--canal-img", CANAL_IMG, "--canal-info", CANAL_INFO,
               "--canal-estado", CANAL_ESTADO, "--padre", pid]
    proc_cam = _lanzar(cmd_cam, raiz, "cámara de cabeza")
    proc_web = _lanzar(cmd_web, raiz, "web")

    visor = None
    try:
        # ── 4. Gemelo en MuJoCo ──
        import mujoco
        import mujoco.viewer
        from skill_shield.simulacion.entorno_taller import EntornoTaller
        from skill_shield.simulacion.entorno_taladro import EntornoTaladro

        modelo = mujoco.MjModel.from_xml_path(cfg["RUTA_XML"])
        datos = mujoco.MjData(modelo)
        arm = cfg["ARM_JOINTS"]
        jid = [mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm]
        qpos_idx = [modelo.jnt_qposadr[j] for j in jid]
        dof_idx = [modelo.jnt_dofadr[j] for j in jid]
        ctrl_idx = [mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{j}") for j in arm]
        if min(ctrl_idx) < 0:
            ctrl_idx = list(range(6))
        EntornoTaller(modelo, datos)                    # se cargan escondidos (paso 6: herramientas)
        EntornoTaladro(modelo, datos)
        reposo = cfg.get("POSICION_STANDBY_BRAZO", [0.0] * 6)
        sincronizar_mujoco(modelo, datos, reposo, qpos_idx, dof_idx, ctrl_idx)
        off = int(cfg.get("OFFSET_MANO", 6))
        for i, v in enumerate(cfg.get("POSICION_STANDBY", [])):
            if off + i < modelo.nu:
                datos.ctrl[off + i] = float(v)
        mujoco.mj_forward(modelo, datos)
        visor = mujoco.viewer.launch_passive(modelo, datos)

        periodo = 1.0 / float(cfg.get("AUTONOMO_HZ", 60.0))
        t_estado, avisado_cam, avisado_web, sincronizado = 0.0, False, False, None
        print("✅ Modo autónomo en marcha. Cierra la ventana de MuJoCo o pulsa Ctrl+C para salir.")
        while visor.is_running():
            t0 = time.monotonic()
            q = fr5.q_real_rad() if fr5 is not None else None
            if q is not None:
                with visor.lock():
                    sincronizar_mujoco(modelo, datos, q, qpos_idx, dof_idx, ctrl_idx)
            if (q is not None) != sincronizado:
                sincronizado = q is not None
                print("🔗 Gemelo: " + ("copiando la pose REAL del FR5" if sincronizado
                                       else "sin pose real del FR5 (brazo quieto en el simulador)"))
            visor.sync()

            if t0 - t_estado >= 0.1:
                t_estado = t0
                canal_estado.escribir(json.dumps(dict(
                    modo="AUTÓNOMO · paso 2 (solo lectura)",
                    fr5=_estado_fr5(fr5, simulacro),
                    gemelo="copiando la pose real" if sincronizado else "sin pose real",
                )).encode())
                if proc_cam is not None and proc_cam.poll() is not None and not avisado_cam:
                    avisado_cam = True
                    print(f"⚠️  El proceso de cámara terminó (código {proc_cam.returncode}). "
                          f"El gemelo y la web siguen.")
                if proc_web is not None and proc_web.poll() is not None and not avisado_web:
                    avisado_web = True
                    print(f"⚠️  El proceso web terminó (código {proc_web.returncode}).")
            time.sleep(max(0.0, periodo - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("\n⏹  Ctrl+C")
    finally:
        print("🧹 Cerrando el modo autónomo…")
        _cerrar(proc_web, "web")
        _cerrar(proc_cam, "cámara")
        if fr5 is not None:
            fr5.parar()
        if visor is not None:
            try:
                visor.close()
            except Exception:
                pass
        for c in (canal_img, canal_info, canal_estado):
            c.cerrar()
    return 0
