"""
Funcionamiento autónomo (10-nov) — paso 2 de la hoja de ruta: el esqueleto.

Lo lanza la última opción del menú de arranque (MODO_EJECUCION = "AUTONOMO"), con su propio
bucle, separado del main() de teleoperación de Skill_Shield_Nivel_6.py.

Qué hace ahora (criterio de «hecho» del paso 2):
  1. FR5Servo, arrancado LO PRIMERO (usa fork), SIEMPRE en simulacro y sin habilitar:
     solo lee la pose real. En este paso el robot NO se puede mover de ninguna forma.
  2. Proceso de cámara (skill_shield/autonomo/camara.py): la cámara de cabeza.
  3. Proceso web (skill_shield/autonomo/web.py, FastAPI): imagen, estado y mandos.
  4. MuJoCo con el brazo copiando la pose REAL del FR5 (gemelo digital), en una ventana
     (AUTONOMO_VENTANA_MUJOCO) y en la web (render_gemelo.py: solo la escena, sin menús).
  5. Estado de la OrcaHand (solo lectura, sin activar el par) y de sus táctiles.
  6. Estado de la calibración cámara→robot (camara_robot.json, paso 4).
  7. Mapa de obstáculos (mapa.py): lo calcula el proceso de la cámara con la profundidad; aquí
     solo se le pasa la pose real del brazo (para quitarlo) y se dibuja en la ventana.
  8. Herramientas con ArUco (herramientas.py): las detecta el proceso de la cámara; la web las
     lista y deja elegir la objetivo; aquí se dibujan en la ventana.

Mandos de la web (los decide ESTE proceso, la web solo los pide):
  · Pedal de la web: «movimiento permitido» solo mientras llega su latido. Con el pedal
    desactivado (opción de la web, por defecto activado) se permite sin pedal.
    En el paso 2 no hay nada que mover: el permiso se calcula y se enseña, nada más.
  · Reinicio tras falsa colisión: FR5Servo.resetear() → ResetAllError, disable, enable.

Reparto en procesos, comunicados por memoria compartida (canal.py):
  cámara ──imagen JPEG / info──▶ web
  este proceso ──estado JSON──▶ web ──mandos JSON──▶ este proceso

Para salir: cerrar la ventana de MuJoCo o Ctrl+C en la terminal.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

CANAL_IMG = "skill_autonomo_img"
CANAL_INFO = "skill_autonomo_caminfo"
CANAL_ESTADO = "skill_autonomo_estado"
CANAL_MANDO = "skill_autonomo_mando"
CANAL_QPOS = "skill_autonomo_qpos"
CANAL_SIM = "skill_autonomo_sim"
CANAL_SIMCAM = "skill_autonomo_simcam"
CANAL_QBRAZO = "skill_autonomo_qbrazo"
CANAL_MAPA = "skill_autonomo_mapa"
CANAL_HERRAMIENTAS = "skill_autonomo_herramientas"


def estado_calibracion(raiz, cfg):
    """Resumen de camara_robot.json (+ su última verificación) para la web."""
    ruta = cfg.get("CALIB_ROBOT_ARCHIVO", "camara_robot.json")
    ruta = ruta if os.path.isabs(ruta) else os.path.join(raiz, ruta)
    if not os.path.exists(ruta):
        return dict(estado="sin", texto="sin calibrar")
    try:
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        return dict(estado="error", texto=f"archivo ilegible: {e}")
    umbral = float(cfg.get("CALIB_ROBOT_ERROR_MAX_MM", 5.0))
    out = dict(fecha=d.get("fecha"), error_medio_mm=d.get("error_medio_mm"), n=len(d.get("muestras_usadas", [])))
    ok = d.get("error_medio_mm") is not None and float(d["error_medio_mm"]) <= umbral
    ver = os.path.splitext(ruta)[0] + "_verificacion.json"
    if os.path.exists(ver) and os.path.getmtime(ver) >= os.path.getmtime(ruta):
        try:
            with open(ver, encoding="utf-8") as f:
                v = json.load(f)
            out["verificacion"] = dict(fecha=v.get("fecha"), ok=v.get("ok"),
                                       error_max_mm=max(v.get("errores_mm") or [0]))
            ok = ok and bool(v.get("ok"))
        except (OSError, ValueError):
            pass
    out["estado"] = "ok" if ok else "aviso"
    out["texto"] = f"{d.get('error_medio_mm')} mm" + ("" if ok else " (revisar)")
    return out


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


class Control:
    """Mandos de la web → permiso de movimiento y peticiones de reinicio.

    Seguro por defecto: sin noticias de la web (o con noticias viejas) el pedal está
    OBLIGATORIO y SUELTO, así que no se permite moverse."""

    def __init__(self, latido_s=0.35, web_viva_s=1.0):
        self.latido_s = float(latido_s)
        self.web_viva_s = float(web_viva_s)
        self._reset_visto = None
        self.pedal_obligatorio = True
        self.pedal_pisado = False
        self.web_viva = False
        self.ver_obstaculos = True

    def actualizar(self, mando, t_mando, ahora=None):
        """Lee el último mando de la web. Devuelve True si pide un reinicio NUEVO."""
        ahora = time.time() if ahora is None else ahora
        self.web_viva = mando is not None and t_mando is not None and ahora - t_mando < self.web_viva_s
        if not self.web_viva:
            self.pedal_obligatorio, self.pedal_pisado = True, False
            return False
        self.pedal_obligatorio = bool(mando.get("pedal_obligatorio", True))
        self.ver_obstaculos = bool(mando.get("ver_obstaculos", True))
        self.pedal_pisado = bool(mando.get("pedal")) and ahora - float(mando.get("pedal_t") or 0) < self.latido_s
        rid = mando.get("reset_id")
        nuevo = rid is not None and rid != self._reset_visto
        if rid is not None:
            self._reset_visto = rid
        return nuevo

    def movimiento_permitido(self, robot_listo):
        return bool(robot_listo) and (self.pedal_pisado or not self.pedal_obligatorio)

    def estado(self, robot_listo, mueve=False):
        permitido = self.movimiento_permitido(robot_listo)
        if not robot_listo:
            motivo = "FR5 sin conexión o con error"
        elif self.pedal_obligatorio and not self.pedal_pisado:
            motivo = "esperando al pedal"
        else:
            motivo = "pedal pisado" if self.pedal_obligatorio else "pedal desactivado"
        return dict(pedal_obligatorio_robot=self.pedal_obligatorio, pedal_pisado=self.pedal_pisado,
                    movimiento_permitido=permitido, motivo=motivo, web_viva=self.web_viva,
                    paso_mueve=mueve)


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
    # El reinicio de la web (ResetAllError → disable → enable) sí actúa: no mueve el robot.
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
    canal_mando = Canal(CANAL_MANDO, capacidad=4 << 10, crear=True)
    canal_sim = Canal(CANAL_SIM, capacidad=2 << 20, crear=True)
    canal_simcam = Canal(CANAL_SIMCAM, capacidad=2 << 20, crear=True)
    canal_qpos = None                                   # se crea al conocer nq (tras cargar el modelo)
    con_mapa = bool(cfg.get("AUTONOMO_MAPA", True))
    canal_qbrazo = Canal(CANAL_QBRAZO, capacidad=64, crear=True) if con_mapa else None
    canal_mapa = Canal(CANAL_MAPA, capacidad=16 + 12 * int(cfg.get("AUTONOMO_MAPA_MAX_DIBUJO", 6000)),
                       crear=True) if con_mapa else None
    con_herr = bool(cfg.get("AUTONOMO_HERRAMIENTAS"))
    canal_herr = Canal(CANAL_HERRAMIENTAS, capacidad=64 << 10, crear=True) if con_herr else None

    # ── 3. Procesos de cámara y web ──
    py_vision = cfg.get("AUTONOMO_PYTHON_PERCEPCION") or sys.executable
    pid = str(os.getpid())
    cmd_cam = [py_vision, "-m", "skill_shield.autonomo.camara",
               "--canal-img", CANAL_IMG, "--canal-info", CANAL_INFO, "--padre", pid]
    if cfg.get("AUTONOMO_CAM_MXID"):
        cmd_cam += ["--mxid", str(cfg["AUTONOMO_CAM_MXID"])]
    if cfg.get("AUTONOMO_CAM_VENTANA", False):
        cmd_cam.append("--ventana")
    if con_mapa:
        cmd_cam += ["--canal-qbrazo", CANAL_QBRAZO, "--canal-mapa", CANAL_MAPA]
    if con_herr:
        cmd_cam += ["--canal-herramientas", CANAL_HERRAMIENTAS]
    if con_mapa or con_herr:
        cmd_cam += ["--canal-mando", CANAL_MANDO]
    cmd_web = [sys.executable, "-m", "skill_shield.autonomo.web",
               "--host", str(cfg.get("AUTONOMO_WEB_HOST", "0.0.0.0")),
               "--puerto", str(cfg.get("AUTONOMO_WEB_PUERTO", 8080)),
               "--canal-img", CANAL_IMG, "--canal-info", CANAL_INFO,
               "--canal-estado", CANAL_ESTADO, "--canal-mando", CANAL_MANDO,
               "--canal-sim", CANAL_SIM, "--canal-simcam", CANAL_SIMCAM, "--padre", pid] \
        + (["--canal-herramientas", CANAL_HERRAMIENTAS] if con_herr else [])
    proc_cam = _lanzar(cmd_cam, raiz, "cámara de cabeza")
    proc_web = _lanzar(cmd_web, raiz, "web")
    proc_render = None

    # ── 4. OrcaHand (solo lectura) y táctiles ──
    mano, tactil = None, None
    from skill_shield.autonomo.mano import MonitorMano, estado_tactil
    if cfg.get("AUTONOMO_CON_MANO", True):
        mano = MonitorMano().iniciar()
        print(f"🖐️  OrcaHand: vigilando {mano.puerto} en solo lectura (no se activa el par).")
    if cfg.get("AUTONOMO_CON_TACTIL", True) and cfg.get("ORCA_TACTIL_ACTIVO", True):
        try:
            from skill_shield.hardware.tactil_real import TactilReal
            tactil = TactilReal()
            tactil.iniciar()
        except Exception as e:
            print(f"⚠️  Táctiles no disponibles ({e}).")
            tactil = None

    control = Control(latido_s=float(cfg.get("AUTONOMO_PEDAL_LATIDO_S", 0.35)))
    visor = None
    try:
        # ── 5. Gemelo en MuJoCo ──
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
        # ── gemelo en la web: proceso de render sin ventana (solo la escena, sin menús) ──
        if cfg.get("AUTONOMO_SIM_WEB", True):
            canal_qpos = Canal(CANAL_QPOS, capacidad=8 * modelo.nq, crear=True)
            canal_qpos.escribir(datos.qpos.astype("float64").tobytes())
            proc_render = _lanzar([sys.executable, "-m", "skill_shield.autonomo.render_gemelo",
                                   "--canal-qpos", CANAL_QPOS, "--canal-sim", CANAL_SIM,
                                   "--canal-simcam", CANAL_SIMCAM, "--canal-mando", CANAL_MANDO,
                                   "--padre", pid] + (["--canal-mapa", CANAL_MAPA] if con_mapa else [])
                                  + (["--canal-herramientas", CANAL_HERRAMIENTAS] if con_herr else []),
                                  raiz, "gemelo para la web")
        con_ventana = bool(cfg.get("AUTONOMO_VENTANA_MUJOCO", True))
        if con_ventana:
            visor = mujoco.viewer.launch_passive(modelo, datos)
        else:
            print("🧊 Sin ventana de MuJoCo (AUTONOMO_VENTANA_MUJOCO = False): el gemelo se ve en la web.")

        periodo = 1.0 / float(cfg.get("AUTONOMO_HZ", 60.0))
        t_estado, avisado_cam, avisado_web, sincronizado, permitido_antes = 0.0, False, False, None, None
        avisado_render, t_calib, calib_estado = False, 0.0, estado_calibracion(raiz, cfg)
        t_mapa = 0.0
        from skill_shield.autonomo.mapa import desempaquetar, dibujar_voxeles
        from skill_shield.autonomo.herramientas import dibujar_herramientas
        print("✅ Modo autónomo en marcha. " + ("Cierra la ventana de MuJoCo o pulsa" if con_ventana else "Pulsa")
              + " Ctrl+C para salir.")
        while visor is None or visor.is_running():
            t0 = time.monotonic()
            q = fr5.q_real_rad() if fr5 is not None else None
            if q is not None:
                if visor is not None:
                    with visor.lock():
                        sincronizar_mujoco(modelo, datos, q, qpos_idx, dof_idx, ctrl_idx)
                else:
                    sincronizar_mujoco(modelo, datos, q, qpos_idx, dof_idx, ctrl_idx)
            if (q is not None) != sincronizado:
                sincronizado = q is not None
                print("🔗 Gemelo: " + ("copiando la pose REAL del FR5" if sincronizado
                                       else "sin pose real del FR5 (brazo quieto en el simulador)"))
            if visor is not None:
                visor.sync()
            if canal_qpos is not None:
                canal_qpos.escribir(datos.qpos.astype("float64").tobytes())
            if canal_qbrazo is not None:                # pose del brazo para quitarlo del mapa (1 = real)
                q_b = q if q is not None else np.array([datos.qpos[a] for a in qpos_idx])
                canal_qbrazo.escribir(np.r_[np.asarray(q_b, float), 1.0 if q is not None else 0.0].tobytes())
            if visor is not None and (canal_mapa is not None or canal_herr is not None) and t0 - t_mapa >= 0.25:
                t_mapa = t0                             # obstáculos y herramientas en la ventana de MuJoCo
                ver = cfg.get("AUTONOMO_MAPA_VER_VISOR", True) and control.ver_obstaculos
                b_m, t_m = canal_mapa.leer() if canal_mapa is not None else (None, None)
                b_h, t_h = canal_herr.leer() if canal_herr is not None else (None, None)
                with visor.lock():
                    visor.user_scn.ngeom = 0
                    if ver and b_m and t_m and time.time() - t_m < 3.0:
                        res_m, cen_m = desempaquetar(b_m)
                        dibujar_voxeles(visor.user_scn, cen_m, res_m)
                    if b_h and t_h and time.time() - t_h < 3.0:
                        try:
                            d_h = json.loads(b_h)
                            dibujar_herramientas(visor.user_scn, d_h.get("lista", []), d_h.get("objetivo"))
                        except ValueError:
                            pass

            # ── mandos de la web ──
            m_bytes, t_m = canal_mando.leer()
            try:
                mando = json.loads(m_bytes) if m_bytes else None
            except ValueError:
                mando = None
            if control.actualizar(mando, t_m):
                if fr5 is not None and fr5.en_linea:
                    print("🔄 Web: reinicio tras colisión → ResetAllError · disable · enable")
                    fr5.resetear()
                else:
                    print("⚠️  Web: reinicio pedido, pero no hay conexión con el FR5.")
            robot_listo = fr5 is not None and fr5.conectado
            permitido = control.movimiento_permitido(robot_listo)
            if permitido != permitido_antes:
                permitido_antes = permitido
                print("🟢 Movimiento PERMITIDO (" if permitido else "⛔ Movimiento bloqueado (",
                      control.estado(robot_listo)["motivo"], ") — en el paso 2 el robot no se mueve", sep="")

            if t0 - t_estado >= 0.1:
                t_estado = t0
                canal_estado.escribir(json.dumps(dict(
                    modo="AUTÓNOMO · paso 2 (solo lectura)",
                    fr5=_estado_fr5(fr5, simulacro),
                    mano=mano.estado() if mano is not None else dict(estado="desactivado",
                                                                     texto="desactivada en config"),
                    tactil=estado_tactil(tactil),
                    gemelo="copiando la pose real" if sincronizado else "sin pose real",
                    control=control.estado(robot_listo, mueve=False),
                    calibracion=calib_estado,
                )).encode())
                if t0 - t_calib >= 2.0:                 # por si se recalibra con el modo en marcha
                    t_calib, calib_estado = t0, estado_calibracion(raiz, cfg)
                if proc_render is not None and proc_render.poll() is not None and not avisado_render:
                    avisado_render = True
                    print(f"⚠️  El render del gemelo para la web terminó (código {proc_render.returncode}).")
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
        _cerrar(proc_render, "gemelo web")
        _cerrar(proc_cam, "cámara")
        if tactil is not None:
            tactil.cerrar()
        if mano is not None:
            mano.cerrar()
        if fr5 is not None:
            fr5.parar()
        if visor is not None:
            try:
                visor.close()
            except Exception:
                pass
        for c in (canal_img, canal_info, canal_estado, canal_mando, canal_sim, canal_simcam, canal_qpos,
                  canal_qbrazo, canal_mapa, canal_herr):
            if c is not None:
                c.cerrar()
    return 0
