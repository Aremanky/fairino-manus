"""
Proceso `fr5_servo`: envía ServoJ al FAIRINO FR5 REAL cada 8 ms (5-nov).

Hasta ahora el FR5 solo existía en MuJoCo. Este módulo arranca un PROCESO aparte (no un
hilo: el GIL del bucle principal — cámaras, MediaPipe, IK, torch — metería retrasos de
decenas de ms) que habla con la controladora del FR5 por el SDK Python de FAIRINO y le
manda `ServoJ` a ritmo fijo (FR5_PERIODO_S = 0.008).

Reparto:
  · Bucle principal / HiloFisica → `FR5Servo.publicar(q_rad)`: deja el último objetivo de
    joint1..6 (rad, convenio del MJCF) en memoria compartida. No espera a nada.
  · Proceso fr5_servo, cada 8 ms:
      1. lee el objetivo (si lleva más de FR5_WATCHDOG_S sin llegar → mantiene la pose)
      2. rad → grados (signo/offset por joint), recorta a los límites del FR5 con margen
      3. genera un paso con VELOCIDAD y ACELERACIÓN limitadas (frena a tiempo, sin pasarse)
      4. robot.ServoJ(...)  — o solo lo calcula si FR5_SIMULACRO = True
  · Al habilitar se parte de la pose REAL del robot (GetActualJointPosDegree); si la del
    simulador está lejos (> FR5_DESFASE_MAX_DEG) el robot se acerca DESPACIO
    (FR5_VEL_APROX_DEG_S) y solo después pasa a la velocidad normal.

Seguridad (no sustituye a la seta de emergencia ni a las zonas de seguridad del FR5):
  · Si el proceso principal muere, este proceso lo detecta (cambia el ppid) y termina con
    ServoMoveEnd.
  · FR5_ERRORES_MAX ServoJ fallidos seguidos → deja de enviar y avisa.

Prueba aislada (sin cámaras ni MuJoCo), por ejemplo mañana al llegar el robot:
    python -m skill_shield.hardware.fr5_servo                 # conecta, lee y SIMULA 5 s
    python -m skill_shield.hardware.fr5_servo --real          # mantiene la pose actual 5 s
    python -m skill_shield.hardware.fr5_servo --real --oscilar 2   # joint6 ±2° despacio

Requiere el SDK Python de FAIRINO (no está en PyPI): carpeta `fairino/` del repositorio
oficial fairino-python-sdk, en el PYTHONPATH o en FR5_SDK_RUTA.
"""
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

# Estados del proceso (FR5Servo.estado()["estado"])
INICIANDO, ESPERA, ACERCANDO, ENVIANDO, ERROR, TERMINADO = range(6)
NOMBRE_ESTADO = {INICIANDO: "iniciando", ESPERA: "conectado (en espera)",
                 ACERCANDO: "acercándose a la pose del simulador", ENVIANDO: "enviando ServoJ",
                 ERROR: "ERROR", TERMINADO: "terminado"}

# Rangos reales del FR5 en grados (los mismos que LIMITES_CTRL_BRAZO de config.py)
LIMITES_FR5_DEG = [(-175.0, 175.0), (-265.0, 85.0), (-160.0, 160.0),
                   (-265.0, 85.0), (-175.0, 175.0), (-175.0, 175.0)]

PARAMETROS_POR_DEFECTO = dict(
    FR5_IP="192.168.58.2",          # IP de fábrica de la controladora FAIRINO
    FR5_SDK_RUTA=None,              # carpeta que CONTIENE fairino/ (None = ya está en el path)
    FR5_PERIODO_S=0.008,            # 8 ms entre ServoJ (también es el cmdT que se le pasa)
    FR5_CMD_TYPE=0,                 # 0 = XML-RPC · 1 = UDP (puerto 20007, menos latencia)
    FR5_SIMULACRO=True,             # True = conecta y lee, pero NO manda ServoJ
    FR5_VEL_MAX_DEG_S=60.0,         # tope de velocidad por joint (el FR5 admite ~180°/s)
    FR5_ACEL_MAX_DEG_S2=300.0,      # tope de aceleración por joint
    FR5_VEL_APROX_DEG_S=10.0,       # velocidad al acercarse a la pose del simulador
    FR5_DESFASE_MAX_DEG=5.0,        # por debajo de esto se considera "sincronizado"
    FR5_MARGEN_LIMITES_DEG=3.0,     # no se piden posiciones a menos de esto de los topes
    FR5_WATCHDOG_S=0.25,            # sin objetivos nuevos durante esto → mantener pose
    FR5_ERRORES_MAX=10,             # ServoJ fallidos SEGUIDOS antes de dejar de enviar
    FR5_SIGNO=(1, 1, 1, 1, 1, 1),   # MJCF → controladora (comprobar joint a joint)
    FR5_OFFSET_DEG=(0, 0, 0, 0, 0, 0),
    FR5_PRIORIDAD_RT=50,            # SCHED_FIFO si hay permisos (0 = no intentarlo)
    FR5_CPU=None,                   # núcleo fijo para el proceso (None = cualquiera)
    FR5_TIMEOUT_CONEXION_S=5.0,
    FR5_FIJAR_LIMITES_BLANDOS=False,  # escribir los límites blandos en la controladora al conectar
    FR5_LIMITES_BLANDOS_DEG=None,     # ((min, max) × 6) en grados; None = no tocar
    # 9-nov: detección de colisiones (None = no tocar lo que tenga la controladora)
    FR5_CARGA_KG=None,                # masa de lo que cuelga de la brida (mano + adaptador), kg
    FR5_CARGA_CDM_MM=None,            # su centro de masas en el marco de la brida (x, y, z) mm
    FR5_ANGULO_INSTALACION=None,      # (inclinación, giro) en grados, como en la WebApp
    FR5_NIVEL_COLISION=None,          # 6 niveles 1-10 (más alto = hace falta más fuerza)
    FR5_ESTRATEGIA_COLISION=None,     # 0 pausa con error · 2 para con error · (ver SDK)
    FR5_PARAR_SI_COLISION=True,       # colisión/fallo durante el ServoJ → dejar de enviar YA
    # 10-nov: herramienta (TCP) en la palma de la ORCA (None = no tocar la de la controladora)
    FR5_TCP_MM=None,                  # (x, y, z, rx, ry, rz) mm/° del TCP respecto a la brida
    FR5_HERRAMIENTA_ID=1,             # nº de sistema de herramienta en la controladora (1-14)
    FR5_RESET_REHABILITAR=True,       # 10-nov: el reset también hace disable → enable
)

# Índices de la memoria compartida de estado
_E_ESTADO, _E_CICLOS, _E_RETRASOS, _E_ERR, _E_PER_MED, _E_PER_MAX, _E_SJ_MED, _E_SJ_MAX = range(8)
_E_QCMD = 8           # 8..13: último comando enviado (o pose real si no se envía), grados
_E_QREAL = 14         # 14..19: pose REAL leída del robot (grados)
_E_TREAL = 20         # time.monotonic() de esa lectura
_E_MAIN, _E_SUB, _E_COLIS, _E_RSTATE = 21, 22, 23, 24   # 9-nov: códigos de fallo de la controladora
_N_ESTADO = 25
PERIODO_LECTURA_S = 0.02    # pose real: 50 Hz (del paquete de estado en tiempo real, sin RPC)


# ══════════════════════════════════════════════════════════════════════════════
# LADO DEL PROCESO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class FR5Servo:
    """Asa del proceso fr5_servo para el programa principal."""

    def __init__(self, **parametros):
        self.p = dict(PARAMETROS_POR_DEFECTO)
        self.p.update({k: v for k, v in parametros.items() if k in PARAMETROS_POR_DEFECTO})
        # fork: arrancamos ANTES de abrir el visor/cámaras (main() lo hace así), y el hijo
        # solo usa sockets. "spawn" reimportaría Skill_Shield_Nivel_6 entero (torch, MediaPipe…).
        self._ctx = mp.get_context("fork")
        self._obj = self._ctx.Array("d", 8)      # q0..q5 (rad), t_publicación, nº de secuencia
        self._est = self._ctx.Array("d", _N_ESTADO)
        self._habilitado = self._ctx.Value("i", 0)
        self._reset = self._ctx.Value("i", 0)    # 9-nov: petición de ResetAllError
        self._salir = self._ctx.Event()
        self._listo = self._ctx.Event()          # el hijo ya intentó conectar
        self._proc = None
        self._seq = 0

    @classmethod
    def desde_config(cls, espacio):
        """Toma los FR5_* de un diccionario (p. ej. globals() tras `from config import *`)."""
        return cls(**{k: espacio[k] for k in PARAMETROS_POR_DEFECTO if k in espacio})

    def iniciar(self, esperar=True):
        """Arranca el proceso y (opcionalmente) espera a saber si conectó. Devuelve True si OK."""
        self._proc = self._ctx.Process(
            target=_proceso_fr5_servo, name="fr5_servo", daemon=True,
            args=(self.p, self._obj, self._est, self._habilitado, self._salir, self._listo, self._reset))
        self._proc.start()
        if esperar:
            self._listo.wait(self.p["FR5_TIMEOUT_CONEXION_S"] + 2.0)
        ok = self.conectado
        modo = "SIMULACRO (no se envía nada)" if self.p["FR5_SIMULACRO"] else "REAL"
        print(f"🦾 fr5_servo [{modo}] {self.p['FR5_IP']}: {self.estado()['texto']}")
        return ok

    def publicar(self, q_rad):
        """Último objetivo de joint1..6 en rad (convenio del MJCF). Barato: llamable a 500 Hz."""
        self._seq += 1
        with self._obj.get_lock():
            self._obj[0:6] = [float(v) for v in q_rad[:6]]
            self._obj[6] = time.monotonic()
            self._obj[7] = float(self._seq)

    def habilitar(self, si=True):
        """True: empieza a mandar ServoJ (desde la pose real) · False: ServoMoveEnd y espera."""
        self._habilitado.value = 1 if si else 0

    def resetear(self):
        """9-nov: ServoMoveEnd + ResetAllError en la controladora (tras una colisión o un fallo).
        10-nov: y después disable → enable (FR5_RESET_REHABILITAR), para las falsas colisiones.
        El brazo queda DESARMADO: hay que volver a armar (pedal / ARMAR)."""
        self._habilitado.value = 0
        self._reset.value = 1

    @property
    def conectado(self):
        """Conectado y SIN error: se le pueden mandar comandos."""
        return self._proc is not None and self._proc.is_alive() and \
            int(self._est[_E_ESTADO]) in (ESPERA, ACERCANDO, ENVIANDO)

    @property
    def en_linea(self):
        """Conectado, aunque esté en ERROR (la pose real se sigue leyendo)."""
        return self._proc is not None and self._proc.is_alive() and \
            int(self._est[_E_ESTADO]) in (ESPERA, ACERCANDO, ENVIANDO, ERROR)

    @property
    def en_error(self):
        return self.en_linea and int(self._est[_E_ESTADO]) == ERROR

    def estado(self):
        e = list(self._est[:])
        cod = int(e[_E_ESTADO])
        return dict(estado=cod, texto=NOMBRE_ESTADO.get(cod, "?"), ciclos=int(e[_E_CICLOS]),
                    retrasos=int(e[_E_RETRASOS]), errores_servoj=int(e[_E_ERR]),
                    periodo_medio_ms=e[_E_PER_MED], periodo_max_ms=e[_E_PER_MAX],
                    servoj_medio_ms=e[_E_SJ_MED], servoj_max_ms=e[_E_SJ_MAX],
                    q_deg=e[_E_QCMD:_E_QCMD + 6], q_real_deg=e[_E_QREAL:_E_QREAL + 6],
                    edad_real_s=(time.monotonic() - e[_E_TREAL]) if e[_E_TREAL] else None,
                    codigo=(int(e[_E_MAIN]), int(e[_E_SUB])), colision=bool(e[_E_COLIS]),
                    estado_robot=int(e[_E_RSTATE]))

    def q_real_rad(self, max_edad_s=0.3):
        """Pose REAL del FR5 en el convenio del MJCF (rad), o None si no hay lectura reciente.
        Inversa de q_FR5[°] = SIGNO · grados(q_MJCF) + OFFSET. (También en ERROR: el gemelo
        sigue enseñando el robot.)"""
        if not self.en_linea:
            return None
        e = self._est[:]
        if not e[_E_TREAL] or time.monotonic() - e[_E_TREAL] > max_edad_s:
            return None
        q_deg = np.asarray(e[_E_QREAL:_E_QREAL + 6], float)
        return np.radians(q_deg - np.asarray(self.p["FR5_OFFSET_DEG"], float)) \
            * np.asarray(self.p["FR5_SIGNO"], float)

    @property
    def enviando(self):
        """True mientras el proceso manda ServoJ (habilitado y sin error)."""
        return self.conectado and int(self._est[_E_ESTADO]) in (ACERCANDO, ENVIANDO)

    def parar(self, timeout=2.0):
        """ServoMoveEnd + CloseRPC en el hijo y espera a que termine."""
        if self._proc is None:
            return
        self._habilitado.value = 0
        self._salir.set()
        self._proc.join(timeout)
        if self._proc.is_alive():
            print("⚠ fr5_servo no terminó a tiempo → terminate()")
            self._proc.terminate()
            self._proc.join(1.0)
        self._proc = None


def sincronizar_mujoco(modelo, datos, q, qpos_idx, dof_idx, ctrl_idx):
    """Pone el brazo del simulador EXACTAMENTE en q (rad): posición, velocidad 0 y ctrl = q,
    para que la simulación, la IK y el generador suave arranquen de la postura REAL."""
    import mujoco
    for k in range(6):
        datos.qpos[qpos_idx[k]] = float(q[k])
        datos.qvel[dof_idx[k]] = 0.0
        datos.ctrl[ctrl_idx[k]] = float(q[k])
    mujoco.mj_forward(modelo, datos)


# ══════════════════════════════════════════════════════════════════════════════
# LADO DEL PROCESO fr5_servo
# ══════════════════════════════════════════════════════════════════════════════

def _importar_sdk(ruta):
    if ruta:
        sys.path.insert(0, ruta)
    from fairino import Robot     # noqa: E402  (SDK oficial de FAIRINO)
    return Robot


def _prioridad_tiempo_real(p):
    if p["FR5_CPU"] is not None:
        try:
            os.sched_setaffinity(0, {int(p["FR5_CPU"])})
        except Exception as e:
            print(f"  fr5_servo: no puedo fijar la CPU {p['FR5_CPU']} ({e})")
    if p["FR5_PRIORIDAD_RT"]:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(p["FR5_PRIORIDAD_RT"])))
            print(f"  fr5_servo: prioridad de tiempo real SCHED_FIFO {p['FR5_PRIORIDAD_RT']}")
        except (PermissionError, AttributeError, OSError):
            try:
                os.nice(-10)
            except OSError:
                pass   # sin permisos: se queda con la prioridad normal (suele bastar)


class _LlamadaServoJ:
    """Adapta la firma de ServoJ: SDK v2 (con axisPos) o v1 antiguo (sin axisPos)."""

    def __init__(self, robot, p):
        self.robot, self.p = robot, p
        self.v2 = True
        self.epos = [0.0, 0.0, 0.0, 0.0]

    def __call__(self, q_deg, cmd_id):
        q = [float(v) for v in q_deg]
        if self.v2:
            try:
                return self.robot.ServoJ(joint_pos=q, axisPos=self.epos, acc=0.0, vel=0.0,
                                         cmdT=self.p["FR5_PERIODO_S"], filterT=0.0, gain=0.0,
                                         id=cmd_id, cmdType=self.p["FR5_CMD_TYPE"])
            except TypeError:
                self.v2 = False
                print("  fr5_servo: SDK antiguo (ServoJ sin axisPos/cmdType) → firma v1")
        return self.robot.ServoJ(q, 0.0, 0.0, self.p["FR5_PERIODO_S"], 0.0, 0.0)


class _LectorPoseReal:
    """Pose real (°): del paquete de estado en tiempo real del SDK (lo actualiza su hilo de
    estado, sin RPC) y, si este SDK no lo tiene, GetActualJointPosDegree (RPC) a 10 Hz."""

    def __init__(self, robot):
        self.robot = robot
        self.rapido = True
        self.t_rpc = 0.0

    def __call__(self, ahora):
        if self.rapido:
            try:
                r = self.robot.GetRobotRealTimeState()
                pkg = r[1] if isinstance(r, (tuple, list)) else r
                q = [float(v) for v in list(pkg.jt_cur_pos)[:6]]
                if len(q) == 6 and any(abs(v) > 1e-9 for v in q):
                    return q
            except Exception:
                pass
            self.rapido = False
            print("  fr5_servo: este SDK no da el estado en tiempo real → pose real por RPC a 10 Hz")
        if ahora - self.t_rpc < 0.1:
            return None
        self.t_rpc = ahora
        try:
            r = self.robot.GetActualJointPosDegree(1)
            if _ret(r) == 0:
                return [float(v) for v in list(r[1])[:6]]
        except Exception:
            pass
        return None


def _limites_blandos(robot, p, q_real):
    """Escribe (o, en simulacro, solo compara) los límites blandos de los joints en la
    controladora. Devuelve los límites que hay que respetar (grados, (6, 2)) o None."""
    lim = p.get("FR5_LIMITES_BLANDOS_DEG")
    if lim is None:
        return None
    lim = np.asarray(lim, dtype=float).reshape(6, 2)
    try:
        r = robot.GetJointSoftLimitDeg(0)
        actuales = np.asarray(r[1], dtype=float).reshape(6, 2) if _ret(r) == 0 else None
    except Exception:
        actuales = None
    if actuales is not None:
        print("  fr5_servo: límites blandos en la controladora (°): "
              + "  ".join(f"j{i + 1}[{a:.0f},{b:.0f}]" for i, (a, b) in enumerate(actuales)))
    if not p.get("FR5_FIJAR_LIMITES_BLANDOS"):
        return actuales
    fuera = [i + 1 for i in range(6) if not (lim[i, 0] <= q_real[i] <= lim[i, 1])]
    if fuera:
        print(f"⚠ fr5_servo: el robot está FUERA de los límites blandos nuevos en j{fuera} → "
              f"no los escribo (llévalo dentro con el mando y vuelve a arrancar)")
        return actuales
    if actuales is not None and np.allclose(actuales, lim, atol=0.05):
        print("✓ fr5_servo: límites blandos ya puestos")
        return lim
    if p["FR5_SIMULACRO"]:
        print("  fr5_servo [SIMULACRO]: pondría los límites blandos (°): "
              + "  ".join(f"j{i + 1}[{a:.0f},{b:.0f}]" for i, (a, b) in enumerate(lim)))
        return actuales
    try:
        e_pos = _ret(robot.SetLimitPositive([float(v) for v in lim[:, 1]]))
        e_neg = _ret(robot.SetLimitNegative([float(v) for v in lim[:, 0]]))
    except Exception as e:
        e_pos = e_neg = f"excepción: {e}"
    try:
        r = robot.GetJointSoftLimitDeg(0)
        leidos = np.asarray(r[1], dtype=float).reshape(6, 2) if _ret(r) == 0 else None
    except Exception:
        leidos = None
    if e_pos == 0 and e_neg == 0 and leidos is not None and np.allclose(leidos, lim, atol=0.05):
        print("✓ fr5_servo: límites blandos escritos y comprobados en la controladora")
        return lim
    print(f"⚠ fr5_servo: NO se pudieron fijar los límites blandos (SetLimitPositive → {e_pos}, "
          f"SetLimitNegative → {e_neg}). Ponlos a mano en la web del FR5 (ajustes de seguridad).")
    return leidos if leidos is not None else actuales


def _ret(r):
    """Los métodos del SDK devuelven err o (err, datos)."""
    return r[0] if isinstance(r, (tuple, list)) else r


def _leer_codigos(robot):
    """(main_code, sub_code, colisión, estado del robot) del paquete de estado (sin RPC)."""
    try:
        r = robot.GetRobotRealTimeState()
        pkg = r[1] if isinstance(r, (tuple, list)) else r
        return (int(pkg.main_code), int(pkg.sub_code), int(getattr(pkg, "collisionState", 0)),
                int(pkg.robot_state))
    except Exception:
        return None


def _configurar_colisiones(robot, p):
    """9-nov: enseña (y si config lo pide, ESCRIBE) lo que decide las colisiones falsas:
    carga en la brida, ángulo de instalación, niveles y estrategia de colisión."""
    try:
        r = robot.GetRobotInstallAngle()
        if _ret(r) == 0:
            print(f"  fr5_servo: ángulo de instalación en la controladora: inclinación "
                  f"{float(r[1][0]):.1f}°, giro {float(r[1][1]):.1f}°")
    except Exception:
        pass
    try:
        r = robot.GetRobotRealTimeState()
        pkg = r[1] if isinstance(r, (tuple, list)) else r
        print("  fr5_servo: niveles de colisión en la controladora: "
              + " ".join(f"j{i + 1}={int(v)}" for i, v in enumerate(list(pkg.collisionLevel)[:6])))
    except Exception:
        pass
    ordenes = []
    if p.get("FR5_CARGA_KG") is not None:
        ordenes.append(("carga", lambda: robot.SetLoadWeight(0, float(p["FR5_CARGA_KG"])),
                        f"{float(p['FR5_CARGA_KG']):.2f} kg"))
    if p.get("FR5_CARGA_CDM_MM") is not None:
        x, y, z = (float(v) for v in p["FR5_CARGA_CDM_MM"])
        ordenes.append(("centro de masas", lambda: robot.SetLoadCoord(x, y, z, 0), f"({x:.0f}, {y:.0f}, {z:.0f}) mm"))
    if p.get("FR5_ANGULO_INSTALACION") is not None:
        ya, za = (float(v) for v in p["FR5_ANGULO_INSTALACION"])
        ordenes.append(("ángulo de instalación", lambda: robot.SetRobotInstallAngle(ya, za),
                        f"inclinación {ya:.1f}°, giro {za:.1f}°"))
    if p.get("FR5_NIVEL_COLISION") is not None:
        niv = [float(v) for v in p["FR5_NIVEL_COLISION"]]
        ordenes.append(("niveles de colisión", lambda: robot.SetAnticollision(0, niv, 0),
                        " ".join(f"{v:.0f}" for v in niv)))
    if p.get("FR5_ESTRATEGIA_COLISION") is not None:
        est = int(p["FR5_ESTRATEGIA_COLISION"])
        ordenes.append(("estrategia de colisión", lambda: robot.SetCollisionStrategy(est), str(est)))
    if p.get("FR5_TCP_MM") is not None:
        # 10-nov: TCP en la palma. No cambia nada de lo que hace este programa (manda ServoJ en
        # joints, con su propia cinemática); sirve para la WebApp, la detección de colisiones y
        # los movimientos cartesianos de la controladora. Tipo 0 = herramienta, instalada en el
        # extremo del robot (0), carga nº 0 (la misma de SetLoadWeight).
        tcp = [float(v) for v in p["FR5_TCP_MM"]]
        hid = int(p.get("FR5_HERRAMIENTA_ID", 1))
        ordenes.append(("herramienta (TCP)", lambda: robot.SetToolCoord(hid, tcp, 0, 0, 0, 0),
                        f"nº {hid}: ({tcp[0]:.1f}, {tcp[1]:.1f}, {tcp[2]:.1f}) mm, "
                        f"({tcp[3]:.1f}, {tcp[4]:.1f}, {tcp[5]:.1f})°"))
    for nombre, f, txt in ordenes:
        if p["FR5_SIMULACRO"]:
            print(f"  fr5_servo [SIMULACRO]: pondría {nombre} = {txt}")
            continue
        try:
            rc = _ret(f())
        except Exception as e:
            rc = f"excepción: {e}"
        print(f"  fr5_servo: {nombre} = {txt} → " + ("✓" if rc == 0 else f"⚠ código {rc}"))
    if p.get("FR5_TCP_MM") is not None:
        try:                                    # comprobar qué TCP tiene activo la controladora
            r = robot.GetTCPOffset(0)
            if _ret(r) == 0 and r[1] is not None:
                print("  fr5_servo: TCP activo en la controladora: ("
                      + ", ".join(f"{float(v):.1f}" for v in r[1]) + ") mm/°")
        except Exception:
            pass


def paso_limitado(q, v, objetivo, v_max, a_max, dt):
    """
    Un paso hacia `objetivo` con |v| ≤ v_max y |a| ≤ a_max, frenando a tiempo para no
    pasarse (v ≤ √(2·a·|e|)). Vectorizado por joint. Devuelve (q, v) nuevos.
    """
    e = objetivo - q
    # velocidad máx. que aún permite parar en el objetivo (frenada discreta: v·dt + v²/2a ≤ |e|)
    v_frenada = a_max * (np.sqrt(dt * dt + 2.0 * np.abs(e) / a_max) - dt)
    v_des = np.sign(e) * np.minimum(v_max, v_frenada)
    v = v + np.clip(v_des - v, -a_max * dt, a_max * dt)
    q_nuevo = q + v * dt
    # si el paso cruzara el objetivo (p. ej. el objetivo saltó hacia atrás), se queda en él
    cruza = np.sign(objetivo - q_nuevo) * np.sign(e) < 0
    q_nuevo = np.where(cruza, objetivo, q_nuevo)
    v = np.where(cruza, 0.0, v)
    return q_nuevo, v


def _proceso_fr5_servo(p, obj, est, habilitado, salir, listo, reset=None):
    est[_E_ESTADO] = INICIANDO
    ppid = os.getppid()
    robot = None
    try:
        # ── Conexión ──────────────────────────────────────────────────────────
        try:
            Robot = _importar_sdk(p["FR5_SDK_RUTA"])
        except ImportError as e:
            print(f"✗ fr5_servo: no encuentro el SDK de FAIRINO ({e}). Copia la carpeta "
                  f"fairino/ del fairino-python-sdk o pon su ruta en FR5_SDK_RUTA.")
            est[_E_ESTADO] = ERROR
            return
        try:
            robot = Robot.RPC(p["FR5_IP"])
            r = robot.GetActualJointPosDegree(0)
            err, q_real = (r[0], list(r[1])) if isinstance(r, (tuple, list)) else (r, None)
        except Exception as e:
            err, q_real = f"excepción: {e}", None
        if err != 0 or q_real is None or len(q_real) < 6:
            print(f"✗ fr5_servo: sin conexión con el FR5 en {p['FR5_IP']} "
                  f"(GetActualJointPosDegree → {err}). ¿Cable, IP del PC en 192.168.58.x, "
                  f"robot encendido?")
            est[_E_ESTADO] = ERROR
            return
        est[_E_QCMD:_E_QCMD + 6] = q_real[:6]
        est[_E_QREAL:_E_QREAL + 6] = q_real[:6]
        est[_E_TREAL] = time.monotonic()
        est[_E_ESTADO] = ESPERA
        print(f"✓ fr5_servo: FR5 conectado, pose real (°) = "
              f"[{', '.join(f'{v:.1f}' for v in q_real[:6])}]")
        lim_blandos = _limites_blandos(robot, p, q_real)
        _configurar_colisiones(robot, p)
        listo.set()
        _prioridad_tiempo_real(p)

        # ── Constantes ────────────────────────────────────────────────────────
        dt = float(p["FR5_PERIODO_S"])
        signo = np.asarray(p["FR5_SIGNO"], float)
        offset = np.asarray(p["FR5_OFFSET_DEG"], float)
        m = float(p["FR5_MARGEN_LIMITES_DEG"])
        lim_inf = np.array([a + m for a, _ in LIMITES_FR5_DEG])
        lim_sup = np.array([b - m for _, b in LIMITES_FR5_DEG])
        if lim_blandos is not None:            # nunca pedir nada que la controladora rechace
            lim_inf = np.maximum(lim_inf, lim_blandos[:, 0] + 0.5)
            lim_sup = np.minimum(lim_sup, lim_blandos[:, 1] - 0.5)
        servoj = _LlamadaServoJ(robot, p)

        q_cmd = np.asarray(q_real[:6], float)
        v_cmd = np.zeros(6)
        enviando, acercando, cmd_id, errores_seg = False, False, 0, 0
        n_ciclos = n_retrasos = 0
        per_med = sj_med = 0.0
        per_max = sj_max = 0.0
        t_ult = None
        t_dead = time.perf_counter() + dt
        t_ult_lectura = 0.0
        leer_real = _LectorPoseReal(robot)
        t_ult_codigos = 0.0

        def _publicar_real(ahora_):
            q_ = leer_real(ahora_)
            if q_ is not None:
                est[_E_QREAL:_E_QREAL + 6] = q_
                est[_E_TREAL] = time.monotonic()

        # ── Bucle a 8 ms ──────────────────────────────────────────────────────
        while not salir.is_set():
            if os.getppid() != ppid:                     # el programa principal murió
                print("⚠ fr5_servo: el proceso principal ha desaparecido → paro")
                break
            ahora = time.perf_counter()

            # 9-nov: códigos de fallo / colisión (paquete de estado, sin RPC) cada 0,1 s
            if ahora - t_ult_codigos >= 0.1:
                t_ult_codigos = ahora
                c_ = _leer_codigos(robot)
                if c_ is not None:
                    est[_E_MAIN], est[_E_SUB], est[_E_COLIS], est[_E_RSTATE] = c_
                    if enviando and p.get("FR5_PARAR_SI_COLISION", True) and (c_[2] or c_[0]):
                        print(f"✗ fr5_servo: la controladora marca "
                              f"{'COLISIÓN' if c_[2] else 'FALLO'} (código {c_[0]}/{c_[1]}) → dejo de "
                              "enviar. [X] / RESET para borrar el error y volver a armar")
                        try:
                            robot.ServoMoveEnd()
                        except Exception:
                            pass
                        enviando = False
                        habilitado.value = 0
                        est[_E_ESTADO] = ERROR

            # 9-nov: recuperación rápida (ResetAllError) a petición del programa
            if reset is not None and reset.value:
                reset.value = 0
                habilitado.value = 0
                if enviando and not p["FR5_SIMULACRO"]:
                    try:
                        robot.ServoMoveEnd()
                    except Exception:
                        pass
                enviando = False
                try:
                    rc = _ret(robot.ResetAllError())
                except Exception as e:
                    rc = f"excepción: {e}"
                time.sleep(0.3)
                if p.get("FR5_RESET_REHABILITAR", True):
                    # 10-nov: falsa colisión por el peso → borrar error, deshabilitar y volver a
                    # habilitar (lo mismo que se hace a mano en la WebApp). No mueve el robot.
                    pasos = []
                    for nombre_, estado_ in (("disable", 0), ("enable", 1)):
                        try:
                            r_ = _ret(robot.RobotEnable(estado_))
                        except Exception as e:
                            r_ = f"excepción: {e}"
                        pasos.append(f"{nombre_} → {'✓' if r_ == 0 else r_}")
                        time.sleep(0.5)
                    print("🔄 fr5_servo: ResetAllError → " + ("✓" if rc == 0 else str(rc))
                          + " · " + " · ".join(pasos))
                c_ = _leer_codigos(robot)
                if c_ is not None:
                    est[_E_MAIN], est[_E_SUB], est[_E_COLIS], est[_E_RSTATE] = c_
                errores_seg = 0
                if rc == 0 and (c_ is None or (c_[0] == 0 and not c_[2])):
                    est[_E_ESTADO] = ESPERA
                    print("🔄 fr5_servo: errores borrados (ResetAllError) → en espera; vuelve a armar")
                else:
                    print(f"⚠ fr5_servo: ResetAllError → {rc}; código {c_[:2] if c_ else '?'}. Si sigue, "
                          "puede que haya que sacarlo del tope o rearmarlo en la WebApp")
                    est[_E_ESTADO] = ERROR if (c_ and (c_[0] or c_[2])) else ESPERA
                t_dead, t_ult = time.perf_counter() + dt, None
                continue

            if not habilitado.value:
                if enviando:
                    if not p["FR5_SIMULACRO"]:
                        robot.ServoMoveEnd()
                    enviando = False
                    print("⏸  fr5_servo: ServoMoveEnd (en espera)")
                if est[_E_ESTADO] != ERROR:
                    est[_E_ESTADO] = ESPERA
                if ahora - t_ult_lectura >= PERIODO_LECTURA_S:   # pose real (sincroniza MuJoCo)
                    t_ult_lectura = ahora
                    _publicar_real(ahora)
                    est[_E_QCMD:_E_QCMD + 6] = est[_E_QREAL:_E_QREAL + 6]
                time.sleep(dt)
                t_dead, t_ult = time.perf_counter() + dt, None
                continue

            if not enviando:                              # ── arranque del streaming ──
                r = robot.GetActualJointPosDegree(0)
                if _ret(r) != 0:
                    print(f"✗ fr5_servo: no puedo leer la pose real ({_ret(r)}); no envío")
                    habilitado.value = 0
                    est[_E_ESTADO] = ERROR
                    continue
                q_cmd = np.asarray(list(r[1])[:6], float)
                v_cmd[:] = 0.0
                cmd_id += 1
                if not p["FR5_SIMULACRO"]:
                    ret = _ret(robot.ServoMoveStart())
                    if ret != 0:
                        print(f"✗ fr5_servo: ServoMoveStart → {ret} (¿modo automático y "
                              f"robot habilitado?)")
                        habilitado.value = 0
                        est[_E_ESTADO] = ERROR
                        continue
                enviando, acercando, errores_seg = True, True, 0
                print("▶  fr5_servo: ServoMoveStart — enviando ServoJ cada "
                      f"{dt * 1000:.0f} ms" + (" (SIMULACRO)" if p["FR5_SIMULACRO"] else ""))

            # 1) objetivo (o mantener si el principal no publica)
            with obj.get_lock():
                q_obj_rad = np.array(obj[0:6])
                t_pub, seq = obj[6], obj[7]
            fresco = seq > 0 and (time.monotonic() - t_pub) <= p["FR5_WATCHDOG_S"]
            if fresco:
                q_obj = np.clip(signo * np.degrees(q_obj_rad) + offset, lim_inf, lim_sup)
            else:
                q_obj = q_cmd.copy()                       # mantener la pose
                v_cmd[:] = 0.0

            # 2) acercamiento lento si el simulador está lejos del robot real
            desfase = float(np.max(np.abs(q_obj - q_cmd)))
            if acercando and fresco and desfase <= p["FR5_DESFASE_MAX_DEG"]:
                acercando = False
                print("✓ fr5_servo: sincronizado con el simulador → velocidad normal")
            v_max = p["FR5_VEL_APROX_DEG_S"] if acercando else p["FR5_VEL_MAX_DEG_S"]
            q_cmd, v_cmd = paso_limitado(q_cmd, v_cmd, q_obj, float(v_max),
                                         float(p["FR5_ACEL_MAX_DEG_S2"]), dt)
            q_cmd = np.clip(q_cmd, lim_inf, lim_sup)

            # 3) ServoJ
            t_sj = time.perf_counter()
            if p["FR5_SIMULACRO"]:
                ret = 0
            else:
                try:
                    ret = _ret(servoj(q_cmd, cmd_id))
                except Exception as e:
                    ret = f"excepción: {e}"
            d_sj = time.perf_counter() - t_sj
            if ret != 0:
                errores_seg += 1
                est[_E_ERR] += 1
                if errores_seg >= p["FR5_ERRORES_MAX"]:
                    print(f"✗ fr5_servo: {errores_seg} ServoJ fallidos seguidos (último: {ret})"
                          " → dejo de enviar")
                    try:
                        robot.ServoMoveEnd()
                    except Exception:
                        pass
                    enviando = False
                    habilitado.value = 0
                    est[_E_ESTADO] = ERROR
                    continue
            else:
                errores_seg = 0

            # 4) estadísticas para el principal
            n_ciclos += 1
            if t_ult is not None:
                per = ahora - t_ult
                per_med += 0.02 * (per - per_med) if per_med else per
                per_max = max(per_max * 0.999, per)
            t_ult = ahora
            sj_med += 0.02 * (d_sj - sj_med) if sj_med else d_sj
            sj_max = max(sj_max * 0.999, d_sj)
            est[_E_CICLOS], est[_E_RETRASOS] = n_ciclos, n_retrasos
            est[_E_PER_MED], est[_E_PER_MAX] = per_med * 1e3, per_max * 1e3
            est[_E_SJ_MED], est[_E_SJ_MAX] = sj_med * 1e3, sj_max * 1e3
            est[_E_QCMD:_E_QCMD + 6] = list(q_cmd)
            est[_E_ESTADO] = ACERCANDO if acercando else ENVIANDO
            if ahora - t_ult_lectura >= PERIODO_LECTURA_S:          # pose real
                t_ult_lectura = ahora
                _publicar_real(ahora)

            # 5) esperar al siguiente tick (plazos absolutos: el error no se acumula)
            t_dead += dt
            resto = t_dead - time.perf_counter()
            if resto < 0:                                 # nos hemos pasado del plazo
                n_retrasos += 1
                if resto < -dt:                           # más de un ciclo: se resincroniza
                    t_dead = time.perf_counter() + dt
                continue
            if resto > 0.0015:
                time.sleep(resto - 0.001)
            while time.perf_counter() < t_dead:           # último ms en espera activa
                pass

        if enviando and not p["FR5_SIMULACRO"]:
            robot.ServoMoveEnd()
            print("⏹  fr5_servo: ServoMoveEnd")
    except KeyboardInterrupt:
        pass
    finally:
        listo.set()
        if robot is not None:
            try:
                if est[_E_ESTADO] in (ACERCANDO, ENVIANDO) and not p["FR5_SIMULACRO"]:
                    robot.ServoMoveEnd()
            except Exception:
                pass
            try:
                robot.CloseRPC()
            except Exception:
                pass
        if est[_E_ESTADO] != ERROR:
            est[_E_ESTADO] = TERMINADO


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBA AISLADA:  python -m skill_shield.hardware.fr5_servo [--real] [--oscilar GRADOS]
# ══════════════════════════════════════════════════════════════════════════════

def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Prueba del proceso fr5_servo (ServoJ cada 8 ms)")
    ap.add_argument("--ip", default=None)
    ap.add_argument("--real", action="store_true", help="manda ServoJ de verdad (sin esto: simulacro)")
    ap.add_argument("--segundos", type=float, default=5.0)
    ap.add_argument("--oscilar", type=float, default=0.0,
                    help="amplitud (°) de una oscilación lenta de joint6 alrededor de la pose actual")
    a = ap.parse_args()

    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        import config as cfg
        par = {k: getattr(cfg, k) for k in PARAMETROS_POR_DEFECTO if hasattr(cfg, k)}
    except Exception:
        par = {}
    if a.ip:
        par["FR5_IP"] = a.ip
    par["FR5_SIMULACRO"] = not a.real

    fr5 = FR5Servo(**par)
    if not fr5.iniciar():
        sys.exit(1)
    # pose real (°) → convenio del MJCF (rad): inversa de signo·deg(q) + offset
    q0 = fr5.q_real_rad(max_edad_s=5.0)
    fr5.publicar(q0)
    fr5.habilitar(True)
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < a.segundos:
            q = q0.copy()
            if a.oscilar:
                q[5] += math.radians(a.oscilar) * math.sin(2 * math.pi * 0.2 * (time.monotonic() - t0))
            fr5.publicar(q)
            time.sleep(0.01)
            e = fr5.estado()
            if e["estado"] == ERROR:
                break
    finally:
        e = fr5.estado()
        fr5.habilitar(False)
        time.sleep(0.05)
        fr5.parar()
    print(f"ciclos={e['ciclos']}  periodo medio={e['periodo_medio_ms']:.2f} ms "
          f"(máx {e['periodo_max_ms']:.2f})  ServoJ medio={e['servoj_medio_ms']:.2f} ms "
          f"(máx {e['servoj_max_ms']:.2f})  retrasos={e['retrasos']}  "
          f"errores ServoJ={e['errores_servoj']}")


if __name__ == "__main__":
    _main()
