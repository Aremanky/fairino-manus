"""
Funcionamiento autónomo (desde el 10-nov): cámara(s) + gemelo + web + tarea.

Lo lanza la última opción del menú de arranque (MODO_EJECUCION = "AUTONOMO"), con su propio
bucle, separado del main() de teleoperación de Skill_Shield_Nivel_6.py.

Qué hace:
  1. FR5Servo, arrancado LO PRIMERO (usa fork). Con AUTONOMO_FR5_SIMULACRO = False la tarea mueve
     el FR5 REAL, solo con el pedal de la web (o con el pedal desactivado desde la web, si
     AUTONOMO_PERMITIR_SIN_PEDAL). Se reconecta solo si se apaga (AUTONOMO_FR5_RECONECTAR).
     (23-nov) Sin FR5 (apagado, sin conexión o AUTONOMO_CON_FR5 = False) la tarea se planifica
     desde la pose del gemelo y se hace solo en el gemelo; el gemelo arranca en
     AUTONOMO_GEMELO_REPOSO_DEG.
  2. Proceso de cámara (skill_shield/autonomo/camara.py): la cámara de cabeza.
  3. Proceso web (skill_shield/autonomo/web.py, FastAPI): imagen, estado y mandos.
  4. MuJoCo con el brazo copiando la pose REAL del FR5 (gemelo digital), en una ventana
     (AUTONOMO_VENTANA_MUJOCO) y en la web (render_gemelo.py: solo la escena, sin menús).
  5. Estado de la OrcaHand (solo lectura, sin activar el par) y de sus táctiles.
     11-nov: también su POSTURA (muñeca de la Orca y dedos) se copia al gemelo, y las piezas
     de la mano con esa postura van al mapa para quitar la mano real de los obstáculos.
  6. Estado de la calibración cámara→robot (camara_robot.json, paso 4).
  7. Mapa de obstáculos (mapa.py): lo calcula el proceso de la cámara con la profundidad; aquí
     solo se le pasa la pose real del brazo (para quitarlo) y se dibuja en la ventana.
  8. Herramientas con ArUco (herramientas.py): las detecta el proceso de la cámara; la web las
     lista y deja elegir la objetivo; aquí se dibujan en la ventana.
  9. (19-nov) Cámara A (camara_a.py): OAK-D fija mirando la mesa de cerca, en su propio proceso
     (camara.py --rol a, sin mapa). La tarea la usa para medir la pelota al planificar y al afinar.
     (20-nov) Con AUTONOMO_CAM_A_AL_MAPA su profundidad también va al mapa de la cabeza (canal CANAL_PROF_A).

Mandos de la web (los decide ESTE proceso, la web solo los pide):
  · Pedal de la web: «movimiento permitido» solo mientras llega su latido. Con el pedal
    desactivado (opción de la web, por defecto activado) la tarea avanza sin pedal mientras el
    proceso de la web siga vivo (si se cae o se reinicia, vuelve a «pedal obligatorio» y se para).
  · Reinicio tras falsa colisión: FR5Servo.resetear() → ResetAllError, disable, enable.

Reparto en procesos, comunicados por memoria compartida (canal.py):
  cámara ──imagen JPEG / info──▶ web
  este proceso ──estado JSON──▶ web ──mandos JSON──▶ este proceso

Para salir: cerrar la ventana de MuJoCo o Ctrl+C en la terminal.
"""
import contextlib
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
CANAL_VISTO = "skill_autonomo_visto"     # (17-nov) qué ha visto la cámara (bits por cubo)
CANAL_HERRAMIENTAS = "skill_autonomo_herramientas"
CANAL_CONGELAR = "skill_autonomo_congelar"   # (18-nov) hay tarea → la cámara no aplica la autocorrección
from skill_shield.autonomo.camara_a import CANAL_HERR_A, CANAL_IMG_A, CANAL_INFO_A, CANAL_PROF_A   # noqa: E402


def estado_calibracion(raiz, cfg, ruta=None):
    """Resumen de camara_robot.json (+ su última verificación) para la web. ruta (19-nov): la de otra cámara."""
    ruta = ruta or cfg.get("CALIB_ROBOT_ARCHIVO", "calibraciones/datos/camara_robot.json")
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


def _estado_fr5(fr5, simulacro, est_carga=None):
    if fr5 is None:
        return dict(texto="desactivado (AUTONOMO_CON_FR5 = False)", en_linea=False)
    e = fr5.estado()
    q = fr5.q_real_rad()
    carga = None
    if est_carga is not None:                   # (23-nov) carga MEDIDA en la brida (mano incluida), para la web
        try:
            tgt, carga_ctrl = fr5.diag_par()
            carga = est_carga.actualizar(q, fr5.par_real(), habilitado=fr5.habilitado_ctrl(),
                                         carga_ctrl=carga_ctrl, par_tgt=tgt)
        except Exception as ex:
            carga = dict(texto=f"error: {ex}", kg=None)
    return dict(texto=e["texto"], en_linea=fr5.en_linea, en_error=fr5.en_error,
                colision=e["colision"], codigo=list(e["codigo"]), simulacro=simulacro,
                q_real_deg=[round(v, 2) for v in e["q_real_deg"]] if q is not None else None,
                carga=carga)


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

    def permiso_tarea(self):
        """(23-nov) Lo que se le pasa a la tarea como «pedal»: pedal pisado o, con el pedal desactivado
        desde la web, siempre que el proceso de la web siga vivo (si se cae, se para)."""
        return self.web_viva and (self.pedal_pisado or not self.pedal_obligatorio)

    def movimiento_permitido(self, robot_listo):
        return bool(robot_listo) and self.permiso_tarea()

    def estado(self, robot_listo, mueve=False, gemelo=False):
        permitido = self.movimiento_permitido(robot_listo)
        if not robot_listo:
            motivo = "FR5 sin conexión o con error"
        elif gemelo and self.permiso_tarea():
            motivo = ("pedal pisado" if self.pedal_obligatorio else "pedal desactivado") + " · solo el gemelo"
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
    # ── 1. FR5: lo primero (fork, antes de abrir el visor y de lanzar procesos) ──
    # 11-nov (paso 10, tarea): con AUTONOMO_TAREA y AUTONOMO_FR5_SIMULACRO = False el FR5 se mueve
    # DE VERDAD, pero solo mientras la tarea está en marcha y el pedal de la web pisado, y con su
    # propio tope de velocidad (AUTONOMO_FR5_VEL_MAX_DEG_S). Fuera de simulacro fr5_servo también
    # escribe carga, TCP y límites en la controladora (como en la teleoperación).
    # En simulacro no se manda ningún ServoJ: la tarea se ve solo en el gemelo.
    simulacro = bool(cfg.get("AUTONOMO_FR5_SIMULACRO", True)) or not cfg.get("AUTONOMO_TAREA", True)
    # (23-nov) escena fija AUTOMÁTICA: no se arrastra la de otra sesión (la mesa o el robot pueden haberse movido).
    # Se planifica sin ella y, si un plan no sale, la tarea la escanea sola (Tarea._lanzar_escaneo)
    if cfg.get("AUTONOMO_ESCENA_FIJA_AUTO", True) and cfg.get("AUTONOMO_ESCENA_FIJA", True):
        try:
            from skill_shield.autonomo import escena_fija as EF
            ruta_ef = EF.ruta(cfg)
            if os.path.exists(ruta_ef):
                EF.olvidar(ruta_ef)
                print(f"🧱 Escena fija de la sesión anterior apartada ({os.path.basename(ruta_ef)}.bak): se planifica "
                      "sin ella y, si hace falta, se escanea sola.")
        except Exception as e:
            print(f"⚠️  No pude apartar la escena fija anterior: {e}")
    print("🧠 Funcionamiento autónomo: cámara + gemelo + web" +
          (" · tarea con el FR5 REAL (solo con el pedal)" if not simulacro else " · FR5 en simulacro (no se mueve)"))
    fr5 = None
    est_carga = None
    if cfg.get("AUTONOMO_CON_FR5", True):
        c = dict(cfg)
        c["FR5_SIMULACRO"] = simulacro
        c["FR5_VEL_MAX_DEG_S"] = min(float(cfg.get("FR5_VEL_MAX_DEG_S", 60.0)),
                                     float(cfg.get("AUTONOMO_FR5_VEL_MAX_DEG_S", 20.0)))
        # 13-nov: si el FR5 se apaga (o no está encendido al arrancar), fr5_servo se reconecta
        # solo cuando vuelve, como la OrcaHand y los táctiles. Nunca se arma solo.
        c["FR5_RECONECTAR"] = bool(cfg.get("AUTONOMO_FR5_RECONECTAR", True))
        # (18-nov) ServoJ por UDP en este modo (menos latencia y menos tirones que XML-RPC cada 8 ms); si la
        # controladora no lo acepta, fr5_servo vuelve solo a XML-RPC (_LlamadaServoJ)
        if cfg.get("AUTONOMO_FR5_CMD_TYPE") is not None:
            c["FR5_CMD_TYPE"] = int(cfg["AUTONOMO_FR5_CMD_TYPE"])
        if not simulacro and cfg.get("FR5_CARGA_CDM_MM") is None:
            print("ℹ️  FR5_CARGA_CDM_MM sin poner: se queda el centro de masas que tenga la controladora (fr5_servo lo "
                  "enseña al conectar). Para medirlo con el robot: python diagnosticos/identificar_carga.py")
        fr5 = FR5Servo.desde_config(c)
        # (23-nov) carga de la brida medida con el par de los joints (la web la enseña en «Robot FR5», siempre)
        try:
            from skill_shield.hardware.carga_fr5 import EstimadorCarga
            est_carga = EstimadorCarga(cfg["RUTA_XML"], signo=cfg.get("FR5_PAR_SIGNO_GRAVEDAD"))
        except Exception as ex:
            print(f"⚠️  No se puede calcular la carga del FR5: {ex}")
        if not fr5.iniciar():
            print("   Sin conexión con el FR5: el gemelo enseñará el brazo en reposo"
                  + (" y se conectará solo cuando el FR5 esté encendido." if c["FR5_RECONECTAR"] else "."))

    # ── 2. Canales de memoria compartida (este proceso es el dueño: los borra al salir) ──
    canal_img = Canal(CANAL_IMG, capacidad=2 << 20, crear=True)
    canal_info = Canal(CANAL_INFO, capacidad=64 << 10, crear=True)
    canal_estado = Canal(CANAL_ESTADO, capacidad=64 << 10, crear=True)
    canal_mando = Canal(CANAL_MANDO, capacidad=4 << 10, crear=True)
    canal_sim = Canal(CANAL_SIM, capacidad=2 << 20, crear=True)
    canal_simcam = Canal(CANAL_SIMCAM, capacidad=2 << 20, crear=True)
    canal_qpos = None                                   # se crea al conocer nq (tras cargar el modelo)
    con_mapa = bool(cfg.get("AUTONOMO_MAPA", True))
    # pose del brazo + cápsulas de las piezas de la mano (11-nov): hasta ~280 piezas
    canal_qbrazo = Canal(CANAL_QBRAZO, capacidad=16 << 10, crear=True) if con_mapa else None
    canal_mapa = Canal(CANAL_MAPA, capacidad=16 + 12 * int(cfg.get("AUTONOMO_MAPA_MAX_DIBUJO", 6000)),
                       crear=True) if con_mapa else None
    canal_visto = Canal(CANAL_VISTO, capacidad=2 << 20, crear=True) if con_mapa else None
    con_herr = bool(cfg.get("AUTONOMO_HERRAMIENTAS"))
    canal_herr = Canal(CANAL_HERRAMIENTAS, capacidad=64 << 10, crear=True) if con_herr else None
    canal_congelar = Canal(CANAL_CONGELAR, capacidad=16 << 10, crear=True) if con_mapa else None   # (23-nov) + escaneo
    # (19-nov) cámara A: OAK-D fija mirando la mesa de cerca (pelotas y herramientas; sin mapa)
    con_cam_a = bool(cfg.get("AUTONOMO_CAM_A", False)) and bool(cfg.get("AUTONOMO_CAM_A_MXID"))
    if cfg.get("AUTONOMO_CAM_A", False) and not con_cam_a:
        print("⚠️  AUTONOMO_CAM_A = True pero sin AUTONOMO_CAM_A_MXID: no se abre la cámara A.")
    canal_img_a = Canal(CANAL_IMG_A, capacidad=2 << 20, crear=True) if con_cam_a else None
    canal_info_a = Canal(CANAL_INFO_A, capacidad=64 << 10, crear=True) if con_cam_a else None
    canal_herr_a = Canal(CANAL_HERR_A, capacidad=64 << 10, crear=True) if con_cam_a else None
    ruta_calib_a = cfg.get("AUTONOMO_CAM_A_CALIB_ARCHIVO", "calibraciones/datos/camara_a_robot.json")
    # (20-nov) la profundidad de la A también va al mapa de la cabeza (la misma TSDF)
    con_prof_a = con_cam_a and con_mapa and bool(cfg.get("AUTONOMO_CAM_A_AL_MAPA", True))
    canal_prof_a = Canal(CANAL_PROF_A, capacidad=1 << 20, crear=True) if con_prof_a else None

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
        cmd_cam += ["--canal-qbrazo", CANAL_QBRAZO, "--canal-mapa", CANAL_MAPA, "--canal-visto", CANAL_VISTO,
                    "--canal-congelar", CANAL_CONGELAR]
        if con_prof_a:
            cmd_cam += ["--canal-prof-a", CANAL_PROF_A]
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
        + (["--canal-herramientas", CANAL_HERRAMIENTAS] if con_herr else []) \
        + (["--canal-img-a", CANAL_IMG_A, "--canal-info-a", CANAL_INFO_A] if con_cam_a else [])
    # (18-nov) cada proceso, con su señal de vida: si se cae o se cuelga, el vigilante lo reinicia solo
    from skill_shield.autonomo.vigilante import Vigilado, Vigilante, comprobar_http
    vig_kw = dict(gracia_s=float(cfg.get("AUTONOMO_VIGILAR_GRACIA_S", 25.0)),
                  sin_datos_s=float(cfg.get("AUTONOMO_VIGILAR_SIN_DATOS_S", 6.0)),
                  espera_max_s=float(cfg.get("AUTONOMO_VIGILAR_ESPERA_MAX_S", 30.0)))
    import secrets
    # la web reiniciada reconoce las sesiones abiertas (firmadas con este secreto): no hay que volver a entrar
    env_web = dict(os.environ, SKILL_SHIELD_WEB_SECRETO=secrets.token_hex(16))
    if not cfg.get("AUTONOMO_WEB_CLAVE") and str(cfg.get("AUTONOMO_WEB_HOST", "0.0.0.0")) not in ("127.0.0.1", "localhost"):
        # sin contraseña configurada, la web se inventaba una en cada arranque: con los reinicios cambiaría.
        # Se inventa aquí UNA para toda la sesión y se le pasa a cada web que se lance
        env_web["SKILL_SHIELD_WEB_CLAVE"] = secrets.token_urlsafe(6)
        print(f"   🔑 Sin SKILL_SHIELD_WEB_CLAVE ni clave_web.txt: contraseña de esta sesión = "
              f"{env_web['SKILL_SHIELD_WEB_CLAVE']}")
    puerto_web = int(cfg.get("AUTONOMO_WEB_PUERTO", 8080))
    vig_cam = Vigilado("cámara de cabeza", cmd_cam, raiz, senal=canal_img.tiempo, **vig_kw)
    vig_web = Vigilado("web", cmd_web, raiz, senal=canal_mando.tiempo, env=env_web,
                       extra=comprobar_http(f"http://127.0.0.1:{puerto_web}/api/estado"), **vig_kw)
    vigilados = [vig_cam, vig_web]
    if con_cam_a:                                       # (19-nov) cámara A: el mismo programa, con --rol a
        cmd_cam_a = [py_vision, "-m", "skill_shield.autonomo.camara", "--rol", "a",
                     "--mxid", str(cfg["AUTONOMO_CAM_A_MXID"]), "--calibracion", str(ruta_calib_a),
                     "--canal-img", CANAL_IMG_A, "--canal-info", CANAL_INFO_A,
                     "--canal-herramientas", CANAL_HERR_A, "--padre", pid]
        if con_mapa or con_herr:
            cmd_cam_a += ["--canal-mando", CANAL_MANDO]
        if con_prof_a:
            cmd_cam_a += ["--canal-prof-a", CANAL_PROF_A]
        if cfg.get("AUTONOMO_CAM_VENTANA", False):
            cmd_cam_a.append("--ventana")
        vigilados.append(Vigilado("cámara A (mesa)", cmd_cam_a, raiz, senal=canal_img_a.tiempo, **vig_kw))
        print(f"📷 Cámara A (mesa): {cfg['AUTONOMO_CAM_A_MXID']} · calibración {ruta_calib_a}")
    vigilante = Vigilante(vigilados, activo=cfg.get("AUTONOMO_VIGILAR", True)).arrancar()
    avisados = set()

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
    tarea = None
    if cfg.get("AUTONOMO_TAREA", True):
        from skill_shield.autonomo.tarea import Tarea
        tarea = Tarea(cfg, raiz, fr5, simulacro, mano, tactil=lambda: estado_tactil(tactil))

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
        # (23-nov) sin FR5, la tarea parte de aquí: una postura sin choques (la de reposo de la
        # teleoperación deja la mano pegada al soporte y el planificador no saldría de ella)
        rep_deg = cfg.get("AUTONOMO_GEMELO_REPOSO_DEG")
        reposo = (np.radians(rep_deg).tolist() if rep_deg is not None
                  else cfg.get("POSICION_STANDBY_BRAZO", [0.0] * 6))
        sincronizar_mujoco(modelo, datos, reposo, qpos_idx, dof_idx, ctrl_idx)
        off = int(cfg.get("OFFSET_MANO", 6))
        for i, v in enumerate(cfg.get("POSICION_STANDBY", [])):
            if off + i < modelo.nu:
                datos.ctrl[off + i] = float(v)
        mujoco.mj_forward(modelo, datos)
        # Muñeca de la Orca y dedos (11-nov): joint del MJCF de cada articulación de orca_core,
        # por el mismo actuador que usa la teleoperación (OFFSET_MANO + JOINT_A_CTRL).
        from skill_shield.hardware.orca_hardware import OrcaHardwareWrapper
        from skill_shield.autonomo.mapa import PiezasMano, empaquetar_pose
        mano_qpos = {}
        for nombre_j, k in OrcaHardwareWrapper.JOINT_A_CTRL.items():
            a = off + int(k)
            if a < modelo.nu and int(modelo.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
                mano_qpos[nombre_j] = (int(modelo.jnt_qposadr[int(modelo.actuator_trnid[a, 0])]), a)
        piezas_mano = None
        if canal_qbrazo is not None and len(mano_qpos) == len(OrcaHardwareWrapper.JOINT_A_CTRL):
            try:
                piezas_mano = PiezasMano(modelo)
            except Exception as e:
                print(f"⚠️  Mapa: no se pueden sacar las piezas de la mano del modelo ({e}): "
                      "se quitará con una cápsula recta.")
        sincronizada_mano = None
        # ── gemelo en la web: proceso de render sin ventana (solo la escena, sin menús) ──
        if cfg.get("AUTONOMO_SIM_WEB", True):
            canal_qpos = Canal(CANAL_QPOS, capacidad=8 * modelo.nq, crear=True)
            canal_qpos.escribir(datos.qpos.astype("float64").tobytes())
            vigilante.agregar(Vigilado(
                "gemelo para la web", [sys.executable, "-m", "skill_shield.autonomo.render_gemelo",
                                       "--canal-qpos", CANAL_QPOS, "--canal-sim", CANAL_SIM,
                                       "--canal-simcam", CANAL_SIMCAM, "--canal-mando", CANAL_MANDO,
                                       "--padre", pid] + (["--canal-mapa", CANAL_MAPA] if con_mapa else [])
                + (["--canal-herramientas", CANAL_HERRAMIENTAS] if con_herr else []),
                raiz, senal=canal_sim.tiempo, **vig_kw))
        con_ventana = bool(cfg.get("AUTONOMO_VENTANA_MUJOCO", True))
        if con_ventana:
            visor = mujoco.viewer.launch_passive(modelo, datos)
        else:
            print("🧊 Sin ventana de MuJoCo (AUTONOMO_VENTANA_MUJOCO = False): el gemelo se ve en la web.")

        periodo = 1.0 / float(cfg.get("AUTONOMO_HZ", 60.0))
        t_estado, sincronizado, permitido_antes = 0.0, None, None
        t_calib, calib_estado = 0.0, estado_calibracion(raiz, cfg)
        calib_a_estado = estado_calibracion(raiz, cfg, ruta_calib_a) if con_cam_a else None
        t_mapa = 0.0
        from skill_shield.autonomo.mapa import desempaquetar, dibujar_voxeles
        from skill_shield.autonomo.herramientas import dibujar_herramientas
        from skill_shield.autonomo import escena_fija as EF                # (17-nov) cajas fijas en azul
        escena_ver = EF.EscenaFija(cfg) if cfg.get("AUTONOMO_ESCENA_FIJA_VER", True) else None
        print("✅ Modo autónomo en marcha. " + ("Cierra la ventana de MuJoCo o pulsa" if con_ventana else "Pulsa")
              + " Ctrl+C para salir.")
        q_tarea_sim, t_ant = None, time.monotonic()
        t_congelar = 0.0
        while visor is None or visor.is_running():
            t0 = time.monotonic()
            dt_vuelta, t_ant = min(t0 - t_ant, 0.1), t0
            q = fr5.q_real_rad() if fr5 is not None else None
            # la tarea conectó la mano con par: desde entonces se lee de ella (el monitor ya está cerrado)
            mano_ver = tarea.mano_activa if tarea is not None and tarea.mano_activa is not None else mano
            qm = mano_ver.postura() if mano_ver is not None else None      # muñeca de la Orca y dedos (rad)
            q_vista = q_tarea_sim if q_tarea_sim is not None else q    # simulacro: el gemelo hace la tarea
            if q_vista is not None or qm is not None:
                with visor.lock() if visor is not None else contextlib.nullcontext():
                    if qm is not None:
                        for nombre_j, v in qm.items():
                            if nombre_j in mano_qpos:
                                adr, a = mano_qpos[nombre_j]
                                datos.qpos[adr] = v
                                datos.ctrl[a] = v
                    if q_vista is not None:
                        sincronizar_mujoco(modelo, datos, q_vista, qpos_idx, dof_idx, ctrl_idx)
                    else:
                        mujoco.mj_forward(modelo, datos)
            if (qm is not None) != sincronizada_mano:
                sincronizada_mano = qm is not None
                print("🖐️  Gemelo: " + ("copiando la postura REAL de la mano (muñeca de la Orca y dedos)"
                                       if sincronizada_mano else
                                       "sin la postura real de la mano (se queda como estaba)"))
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
                # + las piezas de la mano, solo si brazo Y mano son reales (si no, cápsula recta)
                caps = (piezas_mano.capsulas(datos) if piezas_mano is not None and q is not None
                        and qm is not None and q_tarea_sim is None else None)
                canal_qbrazo.escribir(empaquetar_pose(q_b, q is not None, caps))
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
                    if ver and escena_ver is not None:
                        escena_ver.actualizar()
                        EF.dibujar(visor.user_scn, escena_ver.cajas)
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
            # ── tarea (paso 10): órdenes de la web y un paso de ejecución (solo con el pedal) ──
            if tarea is not None:
                def _contexto():
                    b_h, t_h = canal_herr.leer() if canal_herr is not None else (None, None)
                    lista, T_mundo_cam = [], None
                    if b_h and t_h and time.time() - t_h < 2.0:
                        try:
                            d_h = json.loads(b_h)
                            lista = d_h.get("lista", [])
                            T_mundo_cam = d_h.get("T_mundo_cam")    # (18-nov) con la autocorrección de ahora
                        except ValueError:
                            pass
                    b_mp, t_mp = canal_mapa.leer() if canal_mapa is not None else (None, None)
                    mp = desempaquetar(b_mp) if b_mp and t_mp and time.time() - t_mp < 3.0 else None
                    if mp is not None and len(mp[1]) >= int(cfg.get("AUTONOMO_MAPA_MAX_DIBUJO", 6000)):
                        print("⚠️  Tarea: el mapa tiene más cubos de los que se publican "
                              "(AUTONOMO_MAPA_MAX_DIBUJO): se planifica con una muestra.")
                    visto = None
                    if canal_visto is not None:
                        from skill_shield.autonomo.mapa import desempaquetar_visto
                        b_v, t_v = canal_visto.leer()
                        visto = desempaquetar_visto(b_v) if b_v and t_v and time.time() - t_v < 3.0 else None
                    # (19-nov) lo que ve la cámara A (pelotas, con su calibración): la tarea lo usa si ve la misma
                    from skill_shield.autonomo.camara_a import leer_canal
                    lista_a, T_a = leer_canal(canal_herr_a)
                    q_gem = np.array([datos.qpos[a] for a in qpos_idx], float)   # (23-nov) sin FR5
                    return dict(q_real=q, q_gemelo=q_gem, herramientas=lista,
                                objetivo_id=(mando or {}).get("objetivo_id"), mapa=mp, visto=visto,
                                T_mundo_cam=T_mundo_cam, herramientas_a=lista_a, T_mundo_cam_a=T_a)
                tarea.orden(mando, _contexto)
                q_tarea_sim = tarea.paso(dt_vuelta, q, control.permiso_tarea(), _contexto)
            if canal_congelar is not None and t0 - t_congelar >= 0.2:
                t_congelar = t0                         # (18-nov) con tarea en marcha: autocorrección congelada
                pe = getattr(tarea, "peticion_escaneo", None) if tarea is not None else None
                canal_congelar.escribir(json.dumps(dict(
                    congelar=bool(tarea is not None and getattr(tarea, "congelar_autocalib", False)),
                    # (23-nov) la tarea pide escanear la escena fija (sin escena el plan no salió)
                    **({"fijo_escanear_id": pe["id"], "fijo_t": pe["t"], "fijo_excluir": pe["excluir"]}
                       if pe else {}))).encode())
            # (23-nov) sin FR5 la tarea va al gemelo: también puede «moverse» (solo en el gemelo)
            en_gemelo = tarea is not None and tarea.solo_gemelo()
            robot_listo = (fr5 is not None and fr5.conectado) or en_gemelo
            permitido = control.movimiento_permitido(robot_listo)
            if permitido != permitido_antes:
                permitido_antes = permitido
                print("🟢 Movimiento permitido (" if permitido else "⛔ Movimiento bloqueado (",
                      control.estado(robot_listo, gemelo=en_gemelo)["motivo"], ")"
                      + ("" if tarea is not None and not en_gemelo
                         else " — el FR5 está en simulacro" if simulacro else " — sin FR5: solo el gemelo"), sep="")

            if t0 - t_estado >= 0.1:
                t_estado = t0
                canal_estado.escribir(json.dumps(dict(
                    modo=("AUTÓNOMO · FR5 en simulacro" if simulacro or tarea is None
                          else "AUTÓNOMO · sin FR5: solo el gemelo" if en_gemelo
                          else "AUTÓNOMO · tarea · FR5 REAL"),
                    fr5=_estado_fr5(fr5, simulacro, est_carga),
                    mano=mano_ver.estado() if mano_ver is not None else dict(estado="desactivado",
                                                                             texto="desactivada en config"),
                    tactil=estado_tactil(tactil),
                    gemelo="copiando la pose real" if sincronizado else "sin pose real",
                    control=control.estado(robot_listo, mueve=tarea is not None and not en_gemelo, gemelo=en_gemelo),
                    calibracion=calib_estado,
                    calibracion_a=calib_a_estado,                 # (19-nov) None = sin cámara A
                    tarea=tarea.estado_web() if tarea is not None else None,
                    procesos=vigilante.estado(),            # (18-nov) cámara, web y gemelo: vivos / reinicios
                )).encode())
                if t0 - t_calib >= 2.0:                 # por si se recalibra con el modo en marcha
                    t_calib, calib_estado = t0, estado_calibracion(raiz, cfg)
                    if con_cam_a:
                        calib_a_estado = estado_calibracion(raiz, cfg, ruta_calib_a)
                if not vigilante.activo:                # (con el vigilante, los reinicia él en su hilo)
                    vigilante.avisos_sin_vigilar(avisados)
            time.sleep(max(0.0, periodo - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("\n⏹  Ctrl+C")
    finally:
        print("🧹 Cerrando el modo autónomo…")
        vigilante.parar()                               # (18-nov) web, gemelo y cámara
        if tactil is not None:
            tactil.cerrar()
        if tarea is not None:
            tarea.cerrar()                              # ServoMoveEnd y, si tenía par, suelta la mano
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
                  canal_qbrazo, canal_mapa, canal_herr, canal_visto, canal_congelar,
                  canal_img_a, canal_info_a, canal_herr_a, canal_prof_a):
            if c is not None:
                c.cerrar()
    return 0
