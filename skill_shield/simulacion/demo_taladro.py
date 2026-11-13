"""
Demo automática del taladro — tecla [5].

(Extraído de Skill_Shield_Nivel_6.py — código sin cambios, solo reubicado.)
"""
import numpy as np
import mujoco
import math

from config import *


# ══════════════════════════════════════════════════════════════════════════════
# DEMO AUTOMÁTICA DEL TALADRO — TECLA [5]
# ══════════════════════════════════════════════════════════════════════════════
# El brazo ARCTOS + OrcaHand DERECHA, sin cámara:
#   1) se acerca al taladro por su lado DERECHO (palma contra el lado derecho de
#      la empuñadura, pulgar arriba, dedos rodeándola: agarre de pistola),
#   2) lo agarra y lo levanta de la peana,
#   3) lo lleva bajo el techo con la broca mirando hacia ARRIBA,
#   4) aprieta el gatillo MÁS FUERTE con el índice → el motor arranca,
#   5) empuja la broca contra el techo y deja el agujero,
#   6) suelta el gatillo, devuelve el taladro a la peana y vuelve a reposo.
#
# Todo se calcula con la cinemática REAL del modelo (brazo + OrcaHand + taladro):
# la mano alarga el brazo, así que el alcance y la altura del techo se miden
# con la mano incluida, no con el brazo solo.
#
# Agarre: al cerrar la mano el taladro se "suelda" a la mano (weld de MuJoCo,
# <equality> en arctos_mjcf.xml). Con la mano simulada y 0.6 kg el taladro acaba
# resbalando; así la demo sale siempre igual. Mientras espera en la peana va
# soldado a la peana, para que la mano no lo tumbe al acercarse.

def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _minjerk(s):
    """0→1 con velocidad y aceleración nulas al principio y al final (movimiento suave)."""
    s = min(max(float(s), 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def _vector_giro(R):
    """Matriz de rotación → vector de giro (eje * ángulo, rad)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).reshape(9))
    if q[0] < 0:
        q = -q
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return np.zeros(3)
    return q[1:] / s * (2.0 * math.atan2(s, q[0]))


def _eq_activa(modelo, datos, eq, activa: bool):
    """Activa/desactiva una restricción (MuJoCo 3.x: datos.eq_active; 2.x: modelo.eq_active)."""
    arr = getattr(datos, "eq_active", None)
    if arr is None:
        arr = modelo.eq_active
    arr[eq] = 1 if activa else 0


class DemoTaladro:
    NOMBRES_DEDOS = ("i", "m", "r", "p")

    def __init__(self, modelo, entorno_taladro):
        self.ok = False
        self.env = entorno_taladro
        self.terminada = True
        self.segmentos, self.seg, self.t_seg = [], None, 0.0
        self.fase = "-"
        self.z_techo = None
        self.n_agujero = 0
        if not entorno_taladro.disponible:
            print("⚠ Demo taladro [5] NO disponible: falta el escenario del taladro [4].")
            return

        _id = lambda tipo, n: mujoco.mj_name2id(modelo, tipo, n)
        B, G, J, E = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM,
                      mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_EQUALITY)
        self.eq_mano  = _id(E, "taladro_en_mano")
        self.eq_peana = _id(E, "taladro_en_peana")
        if self.eq_mano < 0 or self.eq_peana < 0:
            print("⚠ Demo taladro [5] NO disponible: añade en arctos_mjcf.xml el bloque\n"
                  "    <equality>\n"
                  "      <weld name=\"taladro_en_mano\"  body1=\"mano_mount\"      body2=\"taladro\" active=\"false\" solref=\"0.01 1\"/>\n"
                  "      <weld name=\"taladro_en_peana\" body1=\"taladro_soporte\" body2=\"taladro\" active=\"false\" solref=\"0.01 1\"/>\n"
                  "    </equality>")
            return

        self.bm      = _id(B, "mano_mount")
        self.b_tal   = entorno_taladro.b_taladro
        self.b_sop   = entorno_taladro.b_soporte
        self.g_punta = _id(G, "taladro_punta")
        self.g_mango = _id(G, "taladro_col_empunadura")
        self.g_sop   = _id(G, "taladro_soporte_geom")
        j_mcp = {d: _id(J, f"right_{d}-mcp") for d in self.NOMBRES_DEDOS}
        j_pip_m = _id(J, "right_m-pip")
        if min(self.bm, self.g_punta, self.g_mango, self.g_sop, j_pip_m, *j_mcp.values()) < 0:
            print("⚠ Demo taladro [5] NO disponible: no encuentro mano_mount, los dedos "
                  "de la OrcaHand o las piezas del taladro en el modelo.")
            return

        self.qidx = np.array([modelo.jnt_qposadr[_id(J, j)] for j in ARM_JOINTS], dtype=int)
        self.didx = np.array([modelo.jnt_dofadr[_id(J, j)] for j in ARM_JOINTS], dtype=int)
        self.lo = np.array([l for l, _ in LIMITES_CTRL_BRAZO], dtype=float)
        self.hi = np.array([h for _, h in LIMITES_CTRL_BRAZO], dtype=float)
        self.dk = mujoco.MjData(modelo)          # datos "borrador" solo para cinemática / IK

        # ── 1) Marco de agarre de la MANO (en ejes locales de mano_mount) ──────
        # f = hacia donde apuntan los dedos · n = hacia donde se cierran (normal de
        # la palma) · l = hacia el índice/pulgar. Se miden en el propio modelo.
        d = self.dk
        mujoco.mj_kinematics(modelo, d)
        Rm, pm = d.xmat[self.bm].reshape(3, 3).copy(), d.xpos[self.bm].copy()
        anc = lambda j: d.xanchor[j].copy()
        f = _unit(anc(j_pip_m) - anc(j_mcp["m"]))
        pip_antes = anc(j_pip_m)
        qa_mcp = modelo.jnt_qposadr[j_mcp["m"]]
        d.qpos[qa_mcp] = 0.8                                   # flexionar el corazón...
        mujoco.mj_kinematics(modelo, d)
        mov = anc(j_pip_m) - pip_antes                          # ...¿hacia dónde se cierra?
        d.qpos[qa_mcp] = modelo.qpos0[qa_mcp]
        mujoco.mj_kinematics(modelo, d)
        n = _unit(mov - (mov @ f) * f)
        lat = anc(j_mcp["i"]) - anc(j_mcp["p"])                 # meñique → índice
        l = _unit(lat - (lat @ f) * f - (lat @ n) * n)
        if float(np.cross(f, n) @ l) < 0:
            print("⚠ Demo taladro: la mano no parece DERECHA (índice y meñique cambiados); "
                  "el taladro quedaría al revés en la mano.")
            l = np.cross(f, n)
        nudillos = np.mean([anc(j_mcp[k]) for k in self.NOMBRES_DEDOS], axis=0)
        self.H_loc = Rm.T @ np.column_stack([f, n, l])           # columnas f, n, l

        # ── 2) Geometría del TALADRO (en ejes del propio taladro) ─────────────
        d.qpos[entorno_taladro.qa:entorno_taladro.qa + 7] = [0, 0, 0, 1, 0, 0, 0]
        mujoco.mj_kinematics(modelo, d)
        Rg = d.geom_xmat[self.g_mango].reshape(3, 3).copy()          # cápsula de la empuñadura
        self.c_mango = d.geom_xpos[self.g_mango].copy()
        self.r_mango = float(modelo.geom_size[self.g_mango][0])
        eje_mango = Rg[:, 2] if Rg[2, 2] >= 0 else -Rg[:, 2]      # batería → cabeza
        self.bit_d = _unit(d.xaxis[entorno_taladro.j_giro].copy())   # eje de la broca
        self.tip_d = (d.geom_xpos[self.g_punta].copy()
                      + self.bit_d * float(modelo.geom_size[self.g_punta][0]))
        ex = _unit(self.bit_d - (self.bit_d @ eje_mango) * eje_mango)
        self.E = np.column_stack([ex, np.cross(eje_mango, ex), eje_mango])
        d.qpos[:] = modelo.qpos0
        mujoco.mj_kinematics(modelo, d)

        # Punto de la mano que debe coincidir con el eje de la empuñadura: delante
        # de la palma (media mano + radio del mango), a la altura de los nudillos.
        o_f, o_n, o_l = TALADRO_DEMO_AGARRE_OFFSET
        agarre = nudillos + f * o_f + n * (o_n + self.r_mango) + l * o_l
        self.G_loc = Rm.T @ (agarre - pm)

        # Agarre "de diseño": pose del taladro en ejes de mano_mount
        self.R_md_dis = self.H_loc @ self.E.T
        self.p_md_dis = self.G_loc - self.R_md_dis @ self.c_mango

        # Semillas para la IK (varias posturas de partida del brazo)
        # (elevación, codo) en rad → joint2/joint3 con ARM_IMITACION. Con el ARCTOS salen
        # exactamente las de antes: (0.5, 1.0), (1.0, 0.2), (-0.5, 1.2).
        from skill_shield.brazo.angulos_humano import angulos_humano_a_q_brazo
        _posturas = [tuple(angulos_humano_a_q_brazo(0.0, e, c)[1:3])
                     for e, c in ((2.6416, 0.52), (2.1416, 1.32), (3.6416, 0.32))]
        self.semillas = [np.array([j1, j2, j3, 0.0, JOINT5_FIJO + j5, 0.0])
                         for j1 in np.linspace(-2.4, 2.4, 7)
                         for (j2, j3) in _posturas
                         for j5 in (0.6, -0.6)]

        # Manos (17 valores de ctrl)
        self.mano_abierta = self._mano(TALADRO_DEMO_MANO_ABIERTA)
        self.mano_agarre  = self._mano(TALADRO_DEMO_MANO_AGARRE)
        self.mano_gatillo = self._mano({**TALADRO_DEMO_MANO_AGARRE, **TALADRO_DEMO_MANO_GATILLO})
        g0, g1 = self.mano_agarre[11], self.mano_gatillo[11]
        self.umbral_gatillo = g0 + 0.7 * (g1 - g0)

        # Autocolisiones: eslabones del ARCTOS (sus propios geoms) y mano. Los pares que
        # ya se tocan en posturas normales (mallas que solapan en las uniones) no cuentan.
        self.cuerpos_brazo = {int(modelo.jnt_bodyid[_id(J, j)]) for j in ARM_JOINTS}   # ARCTOS y FR5
        self.cuerpos_mano = set(entorno_taladro.cuerpos_mano)
        self.geoms_brazo = [g for g in range(modelo.ngeom) if modelo.geom_bodyid[g] in self.cuerpos_brazo]
        self.col_brazo_orig = {g: (int(modelo.geom_contype[g]), int(modelo.geom_conaffinity[g]))
                               for g in self.geoms_brazo}
        self.pares_base = set()
        for q_b in (np.zeros(6), np.array(POSICION_STANDBY_BRAZO, dtype=float)):
            self.pares_base |= set(self._pares_en_contacto(modelo, q_b, umbral=0.0))
        self.integral = np.zeros(len(ARM_JOINTS))
        self._q_cmd_prev = None
        self._t_fin = None

        # Estado dinámico
        self.sujeto, self.omega, self.motor = False, 0.0, False
        self.agujero_colocado, self.progreso = False, 0.0
        self.error = None
        self.ok = True
        fmt = lambda v: "(" + ", ".join(f"{x:+.2f}" for x in v) + ")"
        print(f"🛠️  Demo taladro [5] lista (mango r = {self.r_mango * 1000:.0f} mm, "
              f"broca a {np.linalg.norm(self.tip_d - self.c_mango) * 100:.1f} cm del mango).\n"
              f"    Mano en ejes de mano_mount: dedos {fmt(self.H_loc[:, 0])} · palma "
              f"{fmt(self.H_loc[:, 1])} · índice {fmt(self.H_loc[:, 2])} · "
              f"agarre {fmt(self.G_loc * 100)} cm")

    # ── utilidades ────────────────────────────────────────────────────────────
    @staticmethod
    def _mano(valores: dict):
        m = np.zeros(N_ACT)
        m[13:17] = POSICION_STANDBY[13:17]
        for i, v in valores.items():
            lo, hi = LIMITES_CTRL[i]
            m[i] = float(np.clip(v, lo, hi))
        return m

    def _nombre_cuerpo(self, modelo, b):
        return mujoco.mj_id2name(modelo, mujoco.mjtObj.mjOBJ_BODY, int(b)) or f"cuerpo {b}"

    def _es_par_vigilado(self, b1, b2):
        """Brazo↔brazo o brazo↔mano (lo de dentro de la mano no cuenta)."""
        br1, br2 = b1 in self.cuerpos_brazo, b2 in self.cuerpos_brazo
        return (br1 and (br2 or b2 in self.cuerpos_mano)) or (br2 and b1 in self.cuerpos_mano)

    def _pares_en_contacto(self, modelo, q, umbral=0.003):
        """{(b1, b2): penetración} de autocolisiones si el brazo estuviera en q.
        Se mira con las colisiones ORIGINALES del brazo aunque la demo las tenga apagadas."""
        d = self.dk
        guarda = {g: (int(modelo.geom_contype[g]), int(modelo.geom_conaffinity[g]))
                  for g in self.geoms_brazo}
        for g, (ct, ca) in self.col_brazo_orig.items():
            modelo.geom_contype[g], modelo.geom_conaffinity[g] = ct, ca
        d.qpos[self.qidx] = q
        mujoco.mj_kinematics(modelo, d)
        mujoco.mj_collision(modelo, d)
        pares = {}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(modelo.geom_bodyid[c.geom1]), int(modelo.geom_bodyid[c.geom2])
            if self._es_par_vigilado(b1, b2) and -float(c.dist) > umbral:
                par = (min(b1, b2), max(b1, b2))
                pares[par] = max(pares.get(par, 0.0), -float(c.dist))
        for g, (ct, ca) in guarda.items():
            modelo.geom_contype[g], modelo.geom_conaffinity[g] = ct, ca
        return pares

    def _autocolision(self, modelo, q):
        """(penetración máx. en m, texto) de autocolisiones NUEVAS en la postura q."""
        pares = {k: v for k, v in self._pares_en_contacto(modelo, q).items()
                 if k not in self.pares_base}
        if not pares:
            return 0.0, ""
        (b1, b2), pen = max(pares.items(), key=lambda kv: kv[1])
        return pen, f"{self._nombre_cuerpo(modelo, b1)} ↔ {self._nombre_cuerpo(modelo, b2)}"

    def _fk(self, modelo, q):
        d = self.dk
        d.qpos[self.qidx] = q
        mujoco.mj_kinematics(modelo, d)
        mujoco.mj_comPos(modelo, d)
        return d.xmat[self.bm].reshape(3, 3).copy(), d.xpos[self.bm].copy()

    def _ik(self, modelo, q_ini, p_loc, p_obj, ejes, iters=100, w_rot=0.3, lam=0.02, paso=0.2):
        """IK del brazo completo (6 joints): lleva el punto p_loc (ejes de mano_mount)
        a p_obj y alinea los ejes. ejes = [(eje_local, eje_mundo_objetivo), ...];
        con UN solo eje, el giro alrededor de él queda libre."""
        d = self.dk
        q = np.clip(np.asarray(q_ini, dtype=float), self.lo, self.hi)
        p_loc, p_obj = np.asarray(p_loc, dtype=float), np.asarray(p_obj, dtype=float)
        A_loc = np.column_stack([a for a, _ in ejes])
        T = np.column_stack([_unit(t) for _, t in ejes])
        jacp, jacr = np.zeros((3, modelo.nv)), np.zeros((3, modelo.nv))

        def errores(R, p0):
            e_p = p_obj - (p0 + R @ p_loc)
            A = R @ A_loc
            if len(ejes) == 1:
                a, t = A[:, 0], T[:, 0]
                c = np.cross(a, t)
                ang = math.atan2(float(np.linalg.norm(c)), float(a @ t))
                e_r = _unit(c) * ang if np.linalg.norm(c) > 1e-9 else np.zeros(3)
            else:
                e_r = _vector_giro(T @ A.T)
            return e_p, e_r, A

        for _ in range(iters):
            R, p0 = self._fk(modelo, q)
            e_p, e_r, A = errores(R, p0)
            if np.linalg.norm(e_p) < 3e-4 and np.linalg.norm(e_r) < 2e-3:
                break
            mujoco.mj_jac(modelo, d, jacp, jacr, p0 + R @ p_loc, self.bm)
            Jp, Jr = jacp[:, self.didx], jacr[:, self.didx]
            if len(ejes) == 1:                                # el giro sobre el eje no importa
                a = A[:, 0]
                Jr = (np.eye(3) - np.outer(a, a)) @ Jr
            Jt = np.vstack([Jp, w_rot * Jr])
            e = np.concatenate([e_p, w_rot * e_r])
            dq = Jt.T @ np.linalg.solve(Jt @ Jt.T + lam ** 2 * np.eye(6), e)
            m = float(np.abs(dq).max())
            if m > paso:
                dq *= paso / m
            q = np.clip(q + dq, self.lo, self.hi)
        R, p0 = self._fk(modelo, q)
        e_p, _, A = errores(R, p0)
        ang = max(math.acos(float(np.clip(A[:, k] @ T[:, k], -1.0, 1.0))) for k in range(len(ejes)))
        return q, float(np.linalg.norm(e_p)), ang

    def _ik_multi(self, modelo, q_pref, p_loc, p_obj, ejes, extra=()):
        """IK desde muchas posturas de partida; se queda con la que llega (≤3 mm, ≤3°)
        moviendo menos el brazo respecto a q_pref."""
        q_pref = np.asarray(q_pref, dtype=float)
        mejor, clave_mejor = None, None
        for k, s in enumerate([q_pref, *extra, *self.semillas]):
            q, ep, ea = self._ik(modelo, s, p_loc, p_obj, ejes, iters=150 if k == 0 else 100)
            bueno = ep < 0.003 and ea < math.radians(3.0)
            if bueno and self._autocolision(modelo, q)[0] > 0.0:
                coste = 5.0 + ep                      # llega, pero la mano choca con el brazo
            else:
                coste = float(np.abs(q - q_pref).sum()) if bueno else ep + 0.05 * ea + 10.0
            if clave_mejor is None or coste < clave_mejor:
                mejor, clave_mejor = (q, ep, ea), coste
            # Si desde la postura actual ya llega sin retorcerse, no hace falta buscar más
            if k == 0 and bueno and coste < 2.5:
                break
        return mejor

    def _camino(self, modelo, q0, p_loc, pA, pB, ejes, n=10):
        """Línea recta de pA a pB (mismo agarre), resuelta en n puntos con IK encadenada."""
        qs, peor = [np.asarray(q0, dtype=float)], 0.0
        pA, pB = np.asarray(pA, dtype=float), np.asarray(pB, dtype=float)
        for k in range(1, n + 1):
            q, ep, _ = self._ik(modelo, qs[-1], p_loc, pA + (k / n) * (pB - pA), ejes, iters=50)
            qs.append(q)
            peor = max(peor, ep)
        return qs, peor

    @staticmethod
    def _avisar(nombre, ep, ea):
        if ep > 0.01 or ea > math.radians(8):
            print(f"   ⚠ {nombre}: la IK no llega bien (error {ep * 100:.1f} cm, "
                  f"{math.degrees(ea):.0f}°). Revisa TALADRO_POS_SOPORTE / TALADRO_DEMO_AGUJERO_XY.")
        else:
            print(f"   ✓ {nombre}: error {ep * 1000:.1f} mm, {math.degrees(ea):.1f}°")

    def _soldar(self, modelo, datos, eq, b1, b2):
        """Activa el weld eq con la pose relativa ACTUAL de b2 respecto a b1 (sin saltos)."""
        mujoco.mj_kinematics(modelo, datos)
        R1, p1 = datos.xmat[b1].reshape(3, 3), datos.xpos[b1]
        q1inv, qrel = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q1inv, datos.xquat[b1])
        mujoco.mju_mulQuat(qrel, q1inv, datos.xquat[b2])
        modelo.eq_data[eq, 0:3]  = 0.0                               # ancla en el origen de b2
        modelo.eq_data[eq, 3:6]  = R1.T @ (datos.xpos[b2] - p1)      # b2 visto desde b1
        modelo.eq_data[eq, 6:10] = qrel
        _eq_activa(modelo, datos, eq, True)

    def _pose(self, datos, b):
        return datos.xmat[b].reshape(3, 3).copy(), datos.xpos[b].copy()

    # ── planificación ─────────────────────────────────────────────────────────
    def _agujero_xy(self):
        k = self.n_agujero
        pasos = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        dx, dy = pasos[k % len(pasos)]
        s = TALADRO_DEMO_SEP_AGUJEROS
        return np.array([TALADRO_DEMO_AGUJERO_XY[0] + dx * s, TALADRO_DEMO_AGUJERO_XY[1] + dy * s])

    def _calcular_techo(self, modelo, q_ini, xy):
        """Altura del techo = lo más alto que llega la PUNTA DE LA BROCA (brazo + mano +
        taladro, broca vertical) sobre el punto del agujero, menos un margen."""
        tip_m = self.p_md_dis + self.R_md_dis @ self.tip_d
        bit_m = self.R_md_dis @ self.bit_d
        eje = [(bit_m, np.array([0.0, 0.0, 1.0]))]
        llega = lambda r: r[1] < 0.003 and r[2] < math.radians(3.0)
        q, z_ok, z_mal = np.asarray(q_ini, dtype=float), None, None
        for z in np.arange(0.60, 2.01, 0.05):                # búsqueda gruesa (5 cm)
            r = self._ik_multi(modelo, q, tip_m, (xy[0], xy[1], z), eje)
            if llega(r):
                q, z_ok = r[0], float(z)
            else:
                z_mal = float(z)
                break
        if z_ok is None:
            print("   ⚠ La broca no llega en vertical sobre el punto del agujero: "
                  "acerca TALADRO_DEMO_AGUJERO_XY a la base del brazo.")
            return 0.9
        if z_mal is not None:                                 # afinar (1 cm)
            for z in np.arange(z_ok + 0.01, z_mal, 0.01):
                r = self._ik(modelo, q, tip_m, (xy[0], xy[1], z), eje, iters=80)
                if not llega(r):
                    break
                q, z_ok = r[0], float(z)
        print(f"   ✓ Alcance máx. de la broca (brazo + OrcaHand + taladro): z = {z_ok:.2f} m")
        return z_ok - TALADRO_DEMO_MARGEN_TECHO

    # ── trayectorias cartesianas: la mano va en LÍNEA RECTA, sin barridos ────
    # (con interpolación ARTICULAR entre dos posturas lejanas la mano puede dar
    #  un rodeo por abajo y quedarse enganchada bajo la peana: es lo que pasaba)
    @staticmethod
    def _slerp_R(RA, RB, u):
        qa, qb = np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(qa, np.ascontiguousarray(RA, dtype=float).reshape(9))
        mujoco.mju_mat2Quat(qb, np.ascontiguousarray(RB, dtype=float).reshape(9))
        c = float(qa @ qb)
        if c < 0.0:
            qb, c = -qb, -c
        if c > 0.9995:
            q = qa + u * (qb - qa)
        else:
            th = math.acos(c)
            q = (math.sin((1.0 - u) * th) * qa + math.sin(u * th) * qb) / math.sin(th)
        q = q / np.linalg.norm(q)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3)

    def _pose_q(self, modelo, q):
        """(rotación de mano_mount, punto de agarre) si el brazo estuviera en q."""
        R, p = self._fk(modelo, q)
        return R, p + R @ self.G_loc

    def _recta(self, modelo, q0, R_B, P_B, paso_m=0.015, paso_rad=0.06):
        """Desde la pose de q0, la mano va en línea recta hasta P_B girando poco a poco
        hasta R_B. IK encadenada (cada punto parte del anterior) → sin saltos.
        Devuelve (qs, error_final_m, error_final_rad, mayor_salto_rad)."""
        R_A, P_A = self._pose_q(modelo, q0)
        ang = float(np.linalg.norm(_vector_giro(R_B @ R_A.T)))
        n = max(2, int(math.ceil(max(np.linalg.norm(P_B - P_A) / paso_m, ang / paso_rad))))
        qs, I, salto = [np.asarray(q0, dtype=float)], np.eye(3), 0.0
        ep = ea = 0.0
        for k in range(1, n + 1):
            u = k / n
            R = self._slerp_R(R_A, R_B, u)
            q, ep, ea = self._ik(modelo, qs[-1], self.G_loc, P_A + u * (P_B - P_A),
                                 [(I[:, j], R[:, j]) for j in range(3)], iters=60 if k < n else 150)
            salto = max(salto, float(np.abs(q - qs[-1]).max()))
            qs.append(q)
        return qs, ep, ea, salto

    def _punto_bajo_cerca_del_taladro(self, modelo, qs, n=40):
        """Recorre un movimiento ARTICULAR y devuelve cuánto baja la mano (m) por
        debajo de la altura segura cuando está cerca del taladro (0 = nunca)."""
        cx, cy = self.pose_peana[1][:2]
        peor = 0.0
        for a, b in zip(qs[:-1], qs[1:]):
            for u in np.linspace(0.0, 1.0, n):
                _, P = self._pose_q(modelo, a + u * (b - a))
                if math.hypot(P[0] - cx, P[1] - cy) < 0.25:
                    peor = max(peor, self.z_segura - P[2])
        return peor

    def _tramo(self, modelo, nombre, q0, R_B, P_B, q_alt=None, q_fin=None):
        """Tramo en línea recta; si es imposible (límites, singularidad), movimiento
        articular hasta una postura alternativa (q_alt o la mejor IK), pasando por
        arriba si ese movimiento bajara la mano junto al taladro.
        q_fin: postura EXACTA en la que debe acabar (enlaza con el tramo siguiente)."""
        qs, ep, ea, salto = self._recta(modelo, q0, R_B, P_B)
        pen, par = max((self._autocolision(modelo, q) for q in qs[::3] + [qs[-1]]),
                       key=lambda r: r[0])
        if pen > 0.0:
            print(f"   ⚠ {nombre}: en línea recta la mano chocaría con el brazo ({par}, "
                  f"{pen * 1000:.0f} mm)")
            ep = max(ep, 1.0)
        if ep < 0.005 and ea < math.radians(4) and salto < 1.2:
            if q_fin is None:
                return qs
            if float(np.abs(qs[-1] - q_fin).max()) < 0.15:
                return qs[:-1] + [np.asarray(q_fin, dtype=float)]
            ep = ea = 0.0                     # llega, pero con otra postura del brazo
        if q_fin is not None:
            q_alt = q_fin
        I = np.eye(3)
        if q_alt is None:
            q_alt, ep2, ea2 = self._ik_multi(modelo, q0, self.G_loc, P_B,
                                             [(I[:, j], R_B[:, j]) for j in range(3)])
            self._avisar(f"{nombre} (postura alternativa)", ep2, ea2)
        print(f"   ⚠ {nombre}: no se puede en línea recta (error {ep * 100:.1f} cm, "
              f"{math.degrees(ea):.0f}°) → movimiento articular")
        camino = [np.asarray(q0, dtype=float), np.asarray(q_alt, dtype=float)]
        if self._punto_bajo_cerca_del_taladro(modelo, camino) > 0.02:
            R_A, P_A = self._pose_q(modelo, q0)
            P_m = 0.5 * (P_A + P_B)
            P_m[2] = max(P_A[2], P_B[2], self.z_segura) + 0.10
            R_m = self._slerp_R(R_A, R_B, 0.5)
            q_via, ep3, ea3 = self._ik_multi(modelo, q0, self.G_loc, P_m,
                                             [(I[:, j], R_m[:, j]) for j in range(3)])
            camino = [camino[0], q_via, camino[1]]
            baja = self._punto_bajo_cerca_del_taladro(modelo, camino)
            print(f"     ↳ con un punto de paso por arriba (la mano baja {baja * 100:.0f} cm "
                  f"de la altura segura junto al taladro)")
        return camino

    # ── planificación ─────────────────────────────────────────────────────────
    def _plan_recogida(self, modelo, datos, q_act):
        print("🧮 Demo taladro: planificando la recogida...")
        R_wd, p_wd = self._pose(datos, self.b_tal)
        self.pose_peana = (R_wd, p_wd)
        H = R_wd @ self.E                           # f, n, l que debe tener la mano
        R_g = H @ self.H_loc.T                      # rotación de mano_mount al agarrar
        P_g = p_wd + R_wd @ self.c_mango            # centro del mango
        arriba = np.array([0.0, 0.0, 1.0])
        P_pre = P_g - H[:, 1] * TALADRO_DEMO_APROX  # al lado DERECHO del mango
        R_0, P_0 = self._pose_q(modelo, q_act)
        # Altura segura: el punto de agarre por encima de la cabeza del taladro
        z_seg = P_g[2] + TALADRO_DEMO_ALTURA_SEGURA
        self.z_segura = z_seg
        P_sobre = np.array([P_pre[0], P_pre[1], z_seg])

        I = np.eye(3)
        ejes_g = [(I[:, j], R_g[:, j]) for j in range(3)]
        # El final del acercamiento se planifica HACIA ATRÁS desde el agarre, con una
        # postura de brazo natural (cercana a la de reposo), para que la mano entre
        # siempre igual venga de donde venga:
        #   agarre ← entrar ← punto de entrada ← bajar ← sobre el taladro
        q_ref = np.array(POSICION_STANDBY_BRAZO, dtype=float)
        q_agarre, ep, ea = self._ik_multi(modelo, q_ref, self.G_loc, P_g, ejes_g, extra=(q_act,))
        self._avisar("agarre de la empuñadura", ep, ea)
        pen, par = self._autocolision(modelo, q_agarre)
        if pen > 0.0:
            print(f"   ⚠ En la postura de agarre la mano toca el brazo ({par}, {pen * 1000:.0f} mm). "
                  f"Prueba a mover TALADRO_POS_SOPORTE.")
        salir, ep1, _, _ = self._recta(modelo, q_agarre, R_g, P_pre)
        alzar, ep2, _, _ = self._recta(modelo, salir[-1], R_g, P_sobre)
        if max(ep1, ep2) > 0.005:
            print(f"   ⚠ La entrada lateral al mango no sale recta (error {max(ep1, ep2) * 100:.1f} cm)")
        entrar, bajar = salir[::-1], alzar[::-1]
        # Desde donde esté el brazo: subir en vertical e ir por arriba
        subir = self._tramo(modelo, "subir la mano", q_act, R_0,
                            np.array([P_0[0], P_0[1], max(z_seg, P_0[2])]))
        ir = self._tramo(modelo, "ir sobre el taladro", subir[-1], R_g, P_sobre, q_fin=bajar[0])
        levantar = self._tramo(modelo, "levantar", q_agarre, R_g,
                               P_g + np.array([0.0, 0.0, TALADRO_DEMO_ALTURA_PASO]))
        return dict(subir=subir, ir=ir, bajar=bajar, entrar=entrar, levantar=levantar)

    def _plan_techo_y_vuelta(self, modelo, datos):
        """Tras agarrar: pose REAL del taladro en la mano → perforar y devolverlo."""
        print("🧮 Demo taladro: planificando el taladrado...")
        R_wm, p_wm = self._pose(datos, self.bm)
        R_wd, p_wd = self._pose(datos, self.b_tal)
        R_md, p_md = R_wm.T @ R_wd, R_wm.T @ (p_wd - p_wm)
        tip_m, bit_m = p_md + R_md @ self.tip_d, R_md @ self.bit_d
        arriba = np.array([0.0, 0.0, 1.0])
        eje = [(bit_m, arriba)]
        x, y = self.xy
        zt = self.z_techo
        q_lev = self.plan1["levantar"][-1]

        # Pose bajo el techo con la broca vertical (el giro alrededor de ella, libre)
        q_bajo, ep, ea = self._ik_multi(modelo, q_lev, tip_m, (x, y, zt - TALADRO_DEMO_BAJO_TECHO), eje)
        self._avisar("broca bajo el techo", ep, ea)
        R_b, P_b = self._pose_q(modelo, q_bajo)
        al_techo = self._tramo(modelo, "llevar al techo", q_lev, R_b, P_b, q_alt=q_bajo)
        q0 = al_techo[-1]
        acercar, ep1 = self._camino(modelo, q0, tip_m, (x, y, zt - TALADRO_DEMO_BAJO_TECHO),
                                    (x, y, zt - 0.001), eje)
        perforar, ep2 = self._camino(modelo, acercar[-1], tip_m, (x, y, zt - 0.001),
                                     (x, y, zt + TALADRO_DEMO_PROFUNDIDAD), eje, n=8)
        self._avisar("perforación", max(ep1, ep2), 0.0)
        atras = perforar[::-1] + acercar[::-1][1:]            # termina en q0

        # Devolver el taladro EXACTAMENTE a donde estaba en la peana
        R_wd0, p_wd0 = self.pose_peana
        R_obj = R_wd0 @ R_md.T                                # mano_mount al dejarlo
        P_obj = (p_wd0 - R_obj @ p_md) + R_obj @ self.G_loc   # su punto de agarre
        n_w = R_obj @ self.H_loc[:, 1]
        P_arriba = P_obj + arriba * TALADRO_DEMO_ALTURA_PASO
        volver  = self._tramo(modelo, "volver sobre la peana", q0, R_obj, P_arriba)
        dejar   = self._tramo(modelo, "dejar en la peana", volver[-1], R_obj, P_obj)
        R_f, P_f = self._pose_q(modelo, dejar[-1])
        self._avisar("taladro en la peana", float(np.linalg.norm(P_f - P_obj)),
                     float(np.linalg.norm(_vector_giro(R_obj @ R_f.T))))
        retirar = self._tramo(modelo, "retirar la mano", dejar[-1], R_obj, P_obj - n_w * TALADRO_DEMO_APROX)
        alejar  = self._tramo(modelo, "subir la mano", retirar[-1], R_obj,
                              P_obj - n_w * TALADRO_DEMO_APROX + arriba * TALADRO_DEMO_ALTURA_PASO)
        q_rep = np.array(POSICION_STANDBY_BRAZO, dtype=float)
        R_s, P_s = self._pose_q(modelo, q_rep)
        # A reposo: la mano va en línea recta hasta su sitio de reposo (lejos del
        # taladro) y, si el brazo llega con otra postura, se recoloca ya allí.
        reposo, ep, ea, salto = self._recta(modelo, alejar[-1], R_s, P_s)
        if ep < 0.005 and ea < math.radians(4) and salto < 1.2:
            reposo = reposo + [q_rep]
        else:
            reposo = self._tramo(modelo, "volver a reposo", alejar[-1], R_s, P_s, q_fin=q_rep)
        return dict(al_techo=al_techo, acercar=acercar, perforar=perforar, atras=atras,
                    volver=volver, dejar=dejar, retirar=retirar, alejar=alejar, reposo=reposo)

    # ── segmentos (trayectorias por tramos) ───────────────────────────────────
    def _seg(self, nombre, dur, qs, mano=None, al_terminar=None, vibrar=False):
        qs = [np.asarray(q, dtype=float) for q in qs]
        if len(qs) == 1:
            qs = qs * 2
        dur = dur / max(TALADRO_DEMO_VELOCIDAD, 0.05)
        # Nunca más deprisa que TALADRO_DEMO_VEL_MAX (rad/s) en ninguna articulación:
        # el perfil suave tiene un pico de 1.875 × la velocidad media.
        recorrido = float(sum(np.abs(b - a).max() for a, b in zip(qs[:-1], qs[1:])))
        dur = max(dur, 1.875 * recorrido / TALADRO_DEMO_VEL_MAX, 0.05)
        return dict(nombre=nombre, dur=dur, qs=qs, mano=mano, al_terminar=al_terminar, vibrar=vibrar)

    def iniciar(self, modelo, datos, visor=None) -> bool:
        if not self.ok:
            print("⚠ Demo taladro [5] no disponible (mira los avisos del arranque).")
            return False
        env = self.env
        if not env.visible:
            env.mostrar(modelo, datos)
        # Estado limpio: sin soldaduras, taladro de pie en la peana
        _eq_activa(modelo, datos, self.eq_mano, False)
        _eq_activa(modelo, datos, self.eq_peana, False)
        env.recolocar_taladro(modelo, datos)
        env._colisiones(modelo, True)
        self._fisica_demo(modelo, True)
        self._soldar(modelo, datos, self.eq_peana, self.b_sop, self.b_tal)
        modelo.body_gravcomp[self.b_tal] = 1.0
        self.sujeto, self.omega, self.agujero_colocado, self.progreso = False, 0.0, False, 0.0
        self.error = None
        self.integral[:] = 0.0
        self._q_cmd_prev = None
        # Los ESLABONES del brazo dejan de chocar con la mano y el taladro durante la
        # demo (bit de colisión 2): un roce de mallas en la muñeca ya no puede dejar
        # el brazo atascado. El plan igualmente evita posturas con autocolisión.
        if TALADRO_DEMO_SIN_AUTOCOLISION:
            for g in self.geoms_brazo:
                modelo.geom_contype[g] = modelo.geom_conaffinity[g] = 2

        # La peana y el techo no chocan en la demo: el taladro va sujeto por los welds
        # y así la mano nunca puede quedarse enganchada bajo la peana.
        modelo.geom_contype[self.g_sop] = modelo.geom_conaffinity[self.g_sop] = 0
        modelo.geom_contype[env.g_techo] = modelo.geom_conaffinity[env.g_techo] = 0

        q_act = datos.qpos[self.qidx].copy()
        self.plan1 = self._plan_recogida(modelo, datos, q_act)

        # Techo a la altura que alcanza DE VERDAD la broca (una sola vez)
        self.xy = self._agujero_xy()
        if TALADRO_DEMO_TECHO_Z is not None:
            self.z_techo = float(TALADRO_DEMO_TECHO_Z)
        elif self.z_techo is None:
            self.z_techo = self._calcular_techo(modelo, self.plan1["levantar"][-1], self.xy)
        datos.mocap_pos[env.m_techo] = (TALADRO_TECHO_XY[0], TALADRO_TECHO_XY[1], self.z_techo + 0.015)
        datos.mocap_quat[env.m_techo] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(modelo, datos)
        print(f"🔩 Techo a z = {self.z_techo:.2f} m · agujero nº {self.n_agujero + 1} en "
              f"({self.xy[0]:+.2f}, {self.xy[1]:+.2f})")

        p1 = self.plan1
        T = TALADRO_DEMO_TIEMPOS
        self.segmentos = [
            self._seg("Subiendo la mano", T["subir"], p1["subir"], self.mano_abierta),
            self._seg("Yendo sobre el taladro", T["acercar"], p1["ir"]),
            self._seg("Bajando junto al mango", T["bajar"], p1["bajar"]),
            self._seg("Metiendo la mano (lado derecho)", T["entrar"], p1["entrar"]),
            self._seg("Cerrando la mano", T["cerrar"], [p1["entrar"][-1]], self.mano_agarre,
                      al_terminar=self._al_agarrar),
        ]
        self.seg, self.t_seg, self.terminada = None, 0.0, False
        self._mano_ini = datos.ctrl[OFFSET_MANO:OFFSET_MANO + N_ACT].copy()
        if visor is not None:
            with visor.lock():
                sp = np.array(TALADRO_POS_SOPORTE, dtype=float)
                cam = TALADRO_DEMO_CAMARA
                visor.cam.lookat[:] = (0.5 * (sp[0] + self.xy[0]), 0.5 * (sp[1] + self.xy[1]),
                                       0.5 * (sp[2] + self.z_techo))
                visor.cam.distance  = cam["distancia"]
                visor.cam.azimuth   = cam["azimut"]
                visor.cam.elevation = cam["elevacion"]
        return True

    def _comprobar_agarre(self, datos):
        """¿Está de verdad el mango dentro de la mano? Devuelve (ok, texto)."""
        R_m, p_m = self._pose(datos, self.bm)
        R_d, p_d = self._pose(datos, self.b_tal)
        P = p_m + R_m @ self.G_loc                    # donde la mano espera el mango
        c = p_d + R_d @ self.c_mango                  # donde está el mango
        v = R_d.T @ (P - c)                           # en ejes del taladro
        a = self.E[:, 2]                              # eje del mango (ejes del taladro)
        perp = float(np.linalg.norm(v - (v @ a) * a))
        eje_mano = R_d.T @ (R_m @ self.H_loc[:, 2])
        ang = math.degrees(math.acos(float(np.clip(eje_mano @ a, -1.0, 1.0))))
        txt = (f"mano a {perp * 100:.1f} cm del eje del mango, {v @ a * 100:+.1f} cm a lo largo, "
               f"{ang:.0f}° de giro (x broca {v[0] * 100:+.1f} · y izq {v[1] * 100:+.1f} · "
               f"z arriba {v[2] * 100:+.1f} cm)")
        ok = perp <= TALADRO_DEMO_TOL_AGARRE and abs(v @ a) <= 0.04 and ang <= 35.0
        return ok, txt

    def _contactos_dedos(self, modelo, datos):
        """Contactos REALES mano↔taladro (comprobación física, independiente del plan)."""
        return self.env.contactos_mano(modelo, datos)

    def _al_agarrar(self, modelo, datos):
        ok, txt = self._comprobar_agarre(datos)
        p1, T = self.plan1, TALADRO_DEMO_TIEMPOS
        if not ok:
            # NO se suelda un taladro que no está en la mano: se aborta con calma
            self.error = "Agarre fallido: la mano no esta en el mango"
            print(f"❌ Agarre FALLIDO: {txt}\n   → la mano se retira. Envía esta línea para afinarlo.")
            salida = p1["entrar"][::-1] + p1["bajar"][::-1][1:] + p1["ir"][::-1][1:]
            self.segmentos = [
                self._seg("Abriendo la mano", T["abrir"], [p1["entrar"][-1]], self.mano_abierta),
                self._seg("Retirando la mano", T["retirar"], salida),
                self._seg("Volviendo a reposo", T["reposo"], [salida[-1], POSICION_STANDBY_BRAZO],
                          np.array(POSICION_STANDBY, dtype=float), al_terminar=self._al_acabar),
            ]
            return
        # Taladro: de la peana a la mano
        _eq_activa(modelo, datos, self.eq_peana, False)
        self._soldar(modelo, datos, self.eq_mano, self.bm, self.b_tal)
        modelo.body_gravcomp[self.b_tal] = 1.0
        self.sujeto = True
        n_c = self._contactos_dedos(modelo, datos)
        print(f"✊ Taladro AGARRADO ({txt}; {n_c} contactos dedos↔taladro)")
        if n_c < TALADRO_DEMO_CONTACTOS_MIN:
            print("   ⚠ La mano está donde manda el plan pero apenas toca el mango: el taladro "
                  "se sujeta igual (soldadura), pero ajusta TALADRO_DEMO_AGARRE_OFFSET o "
                  "TALADRO_DEMO_MANO_AGARRE para que los dedos lo rodeen de verdad.")
        p2 = self._plan_techo_y_vuelta(modelo, datos)
        q_top = p2["acercar"][-1]
        self.segmentos += [
            self._seg("Levantando el taladro", T["levantar"], p1["levantar"]),
            self._seg("Llevandolo al techo", T["al_techo"], p2["al_techo"]),
            self._seg("Acercando la broca al techo", T["acercar_broca"], p2["acercar"]),
            self._seg("Apretando el gatillo", T["gatillo"], [q_top], self.mano_gatillo),
            self._seg("Motor en marcha", T["arrancar"], [q_top]),
            self._seg("PERFORANDO el techo", T["perforar"], p2["perforar"], vibrar=True),
            self._seg("Sacando la broca", T["sacar"], p2["atras"]),
            self._seg("Soltando el gatillo", T["soltar_gatillo"], [p2["atras"][-1]], self.mano_agarre),
            self._seg("Devolviendo el taladro", T["volver"], p2["volver"]),
            self._seg("Dejandolo en la peana", T["dejar"], p2["dejar"], al_terminar=self._al_dejar),
            self._seg("Abriendo la mano", T["abrir"], [p2["dejar"][-1]], self.mano_abierta),
            self._seg("Retirando la mano", T["retirar"], p2["retirar"]),
            self._seg("Subiendo la mano", T["levantar"], p2["alejar"]),
            self._seg("Volviendo a reposo", T["reposo"], p2["reposo"],
                      np.array(POSICION_STANDBY, dtype=float), al_terminar=self._al_acabar),
        ]

    def _al_dejar(self, modelo, datos):
        _eq_activa(modelo, datos, self.eq_mano, False)
        self._soldar(modelo, datos, self.eq_peana, self.b_sop, self.b_tal)
        self.sujeto = False
        print("🖐️  Taladro devuelto a la peana")

    def _al_acabar(self, modelo, datos):
        if self.error is None:
            self.n_agujero += 1
            print("✅ Demo del taladro TERMINADA — [5] repetir (nuevo agujero) · [S] volver al teleop")
        else:
            print("⚠ Demo del taladro interrumpida — [5] reintentar · [S] volver al teleop")

    # ── cada paso de física ───────────────────────────────────────────────────
    def paso(self, modelo, datos, dt):
        """Se llama ANTES de cada mj_step: escribe ctrl del brazo y la mano."""
        if self.terminada:
            self._motor(modelo, datos, dt)
            # Diagnóstico: con la demo acabada el brazo vuelve a los SERVOS. Si al
            # segundo no sostienen la postura de reposo, el problema está en el modelo
            # (peso sin compensar, servos flojos o roces) y afecta también al teleop.
            if self._t_fin is not None:
                self._t_fin += dt
                if self._t_fin >= 1.0:
                    self._t_fin = None
                    dq = np.degrees(datos.qpos[self.qidx] - datos.ctrl[:len(ARM_JOINTS)])
                    if float(np.abs(dq).max()) > 1.0:
                        print("   ⚠ Con los servos, el brazo no sostiene el reposo: real−mando (°) "
                              + "  ".join(f"j{k + 1} {v:+.1f}" for k, v in enumerate(dq)))
                        choques = self._choques_vivos(modelo, datos)
                        print("      contactos del brazo/mano: "
                              + (", ".join(choques) if choques else "ninguno"))
                    else:
                        print("   ✓ Con los servos, el brazo sostiene bien el reposo.")
            return
        if self.seg is None:
            if not self.segmentos:
                self.terminada, self.fase = True, "Terminada"
                self._t_fin = 0.0
                return
            self.seg = self.segmentos.pop(0)
            self.t_seg = 0.0
            self.fase = self.seg["nombre"]
            self._mano_ini = datos.ctrl[OFFSET_MANO:OFFSET_MANO + N_ACT].copy()
        seg = self.seg
        self.t_seg += dt
        s = _minjerk(self.t_seg / seg["dur"])

        qs = seg["qs"]
        u = s * (len(qs) - 1)
        i = min(int(u), len(qs) - 2)
        q = qs[i] + (u - i) * (qs[i + 1] - qs[i])
        if seg["vibrar"] and self.omega > 1.0:            # vibración del taladro
            a = TALADRO_DEMO_VIBRACION
            q = q + np.array([0, 0, 0, a * math.sin(2 * math.pi * 23 * self.t_seg),
                              a * math.cos(2 * math.pi * 31 * self.t_seg), 0])
        if TALADRO_DEMO_BRAZO_CINEMATICO:
            # Brazo CINEMÁTICO: sus articulaciones se colocan EXACTAMENTE en el plan en
            # cada paso de física (como en una animación). Ni el peso de la mano, ni un
            # roce, ni una soldadura pueden desviarlo. Los dedos siguen siendo físicos.
            q_prev = self._q_cmd_prev if self._q_cmd_prev is not None else datos.qpos[self.qidx].copy()
            datos.qpos[self.qidx] = q
            datos.qvel[self.didx] = (q - q_prev) / max(dt, 1e-6)
            datos.ctrl[:len(ARM_JOINTS)] = q
            self._q_cmd_prev = q.copy()
            if seg["mano"] is not None:
                datos.ctrl[OFFSET_MANO:OFFSET_MANO + N_ACT] = (
                    self._mano_ini + s * (np.asarray(seg["mano"]) - self._mano_ini))
            self._motor(modelo, datos, dt)
            self._fin_de_segmento(modelo, datos, seg)
            return
        # Corrección integral: si el brazo se queda corto de forma sostenida (peso, un
        # roce...), el mando se adelanta un poco hasta que las articulaciones llegan.
        # Solo con el plan casi quieto: el retraso normal del servo al moverse no cuenta.
        err = q - datos.qpos[self.qidx]
        q_prev = getattr(self, "_q_plan_prev", q)
        quieto = float(np.abs(q - q_prev).max()) < 0.05 * dt
        self._q_plan_prev = q
        if TALADRO_DEMO_KI > 0.0 and quieto and float(np.abs(err).max()) < 0.3:
            self.integral = np.clip(self.integral + TALADRO_DEMO_KI * err * dt, -0.25, 0.25)
        datos.ctrl[:len(ARM_JOINTS)] = q + self.integral
        if seg["mano"] is not None:
            datos.ctrl[OFFSET_MANO:OFFSET_MANO + N_ACT] = (
                self._mano_ini + s * (np.asarray(seg["mano"]) - self._mano_ini))

        self._motor(modelo, datos, dt)
        self._fin_de_segmento(modelo, datos, seg)

    def _fin_de_segmento(self, modelo, datos, seg):
        if self.t_seg >= seg["dur"]:
            self.seg = None
            # ¿Ha llegado el brazo de verdad? (si no, algo lo frena: un choque)
            _, p_plan = self._fk(modelo, seg["qs"][-1])
            desvio = float(np.linalg.norm(datos.xpos[self.bm] - p_plan))
            if desvio > 0.02:
                dq = np.degrees(datos.qpos[self.qidx] - seg["qs"][-1])
                print(f"   ⚠ Al acabar '{seg['nombre']}' la mano estaba a {desvio * 100:.1f} cm "
                      f"de donde debía.\n      joints real−plan (°): "
                      + "  ".join(f"j{k + 1} {v:+.1f}" for k, v in enumerate(dq))
                      + "\n      corrección integral (°): "
                      + "  ".join(f"{v:+.1f}" for v in np.degrees(self.integral)))
                choques = self._choques_vivos(modelo, datos)
                print("      contactos del brazo/mano: " + (", ".join(choques) if choques else "ninguno"))
            if seg["al_terminar"] is not None:
                seg["al_terminar"](modelo, datos)

    def _choques_vivos(self, modelo, datos):
        """Contactos REALES ahora mismo del brazo o la mano con cualquier cosa
        (salvo los de dentro de la propia mano y mano↔taladro)."""
        vistos = {}
        for i in range(datos.ncon):
            c = datos.contact[i]
            b1, b2 = int(modelo.geom_bodyid[c.geom1]), int(modelo.geom_bodyid[c.geom2])
            implicados = self.cuerpos_brazo | self.cuerpos_mano
            if b1 not in implicados and b2 not in implicados:
                continue
            if b1 in self.cuerpos_mano and b2 in self.cuerpos_mano:
                continue
            if {b1, b2} & self.env.cuerpos_taladro and {b1, b2} & self.cuerpos_mano:
                continue
            k = f"{self._nombre_cuerpo(modelo, b1)} ↔ {self._nombre_cuerpo(modelo, b2)}"
            vistos[k] = max(vistos.get(k, 0.0), -float(c.dist))
        return [f"{k} ({v * 1000:.0f} mm)" for k, v in sorted(vistos.items(), key=lambda kv: -kv[1])[:4]]

    def _motor(self, modelo, datos, dt):
        """Gatillo: el motor gira si el índice aprieta MÁS que en el agarre normal.
        La broca que gira y está dentro del techo va haciendo el agujero."""
        env = self.env
        encendido = self.sujeto and datos.ctrl[OFFSET_MANO + 11] >= self.umbral_gatillo
        w_obj = TALADRO_DEMO_RPM * 2 * math.pi / 60 if encendido else 0.0
        self.omega += float(np.clip(w_obj - self.omega, -120 * dt, 120 * dt))   # rampa
        if self.omega > 0.0 or encendido:
            datos.qvel[env.da_giro] = self.omega
        self.motor = encendido
        if not (encendido and env.m_agujeros):
            return
        # ¿La punta ya está en el techo?
        r = float(modelo.geom_size[self.g_punta][0])
        bit_w = _unit(datos.xaxis[env.j_giro])
        tip = datos.geom_xpos[self.g_punta] + bit_w * r
        zt = self.z_techo
        if tip[2] < zt - 0.002:
            return
        k = self.n_agujero % len(env.m_agujeros)
        b, m = env.b_agujeros[k], env.m_agujeros[k]
        g = modelo.body_geomadr[b]
        if not self.agujero_colocado:
            if not hasattr(self, "_r_agujero"):
                self._r_agujero = float(modelo.geom_size[g][0])
            datos.mocap_pos[m] = (tip[0], tip[1], zt - 0.0012)
            datos.mocap_quat[m] = [1.0, 0.0, 0.0, 0.0]
            self.agujero_colocado = True
            print("🌀 La broca toca el techo: perforando...")
        self.progreso = max(self.progreso, float(np.clip((tip[2] - zt) / TALADRO_DEMO_PROFUNDIDAD, 0, 1)))
        modelo.geom_size[g][0] = 0.0015 + self.progreso * (self._r_agujero - 0.0015)

    def _fisica_demo(self, modelo, activar: bool):
        """Ajustes de física SOLO durante la demo (al salir se restauran):
          · El taladro hereda del <default> del brazo damping=3 y armature=0.02 en su
            freejoint (pensados para las articulaciones del ARCTOS): frenan su giro y
            la soldadura con la mano cede poco a poco → al bajar a la peana el taladro
            acababa girado dentro de la mano y luego la mano lo atravesaba al soltarlo.
          · noslip (del escenario [4]) hace el rozamiento de los dedos "rígido" y
            también empuja al taladro contra la soldadura.
          · Soldaduras más rígidas: el taladro no se mueve en la mano ni en la peana."""
        da = self.env.da
        if activar:
            if getattr(self, "_fisica_orig", None) is None:
                self._fisica_orig = dict(
                    damping=modelo.dof_damping[da:da + 6].copy(),
                    armature=modelo.dof_armature[da:da + 6].copy(),
                    friccion=modelo.dof_frictionloss[da:da + 6].copy(),
                    noslip=int(modelo.opt.noslip_iterations),
                    solref={eq: modelo.eq_solref[eq].copy() for eq in (self.eq_mano, self.eq_peana)},
                    solimp={eq: modelo.eq_solimp[eq].copy() for eq in (self.eq_mano, self.eq_peana)})
            modelo.dof_damping[da:da + 6] = 0.0
            modelo.dof_armature[da:da + 6] = 0.0
            modelo.dof_frictionloss[da:da + 6] = 0.0
            modelo.opt.noslip_iterations = 0
            for eq in (self.eq_mano, self.eq_peana):
                modelo.eq_solref[eq][:2] = (TALADRO_DEMO_SOLDADURA_RIGIDEZ, 1.0)
                modelo.eq_solimp[eq][:3] = (0.95, 0.99, 0.001)
        elif getattr(self, "_fisica_orig", None) is not None:
            o = self._fisica_orig
            modelo.dof_damping[da:da + 6] = o["damping"]
            modelo.dof_armature[da:da + 6] = o["armature"]
            modelo.dof_frictionloss[da:da + 6] = o["friccion"]
            modelo.opt.noslip_iterations = o["noslip"]
            for eq in (self.eq_mano, self.eq_peana):
                modelo.eq_solref[eq] = o["solref"][eq]
                modelo.eq_solimp[eq] = o["solimp"][eq]
            self._fisica_orig = None

    def terminar(self, modelo, datos):
        """Al salir de la demo: todo como estaba (el techo y los agujeros se quedan)."""
        if not self.ok:
            return
        _eq_activa(modelo, datos, self.eq_mano, False)
        _eq_activa(modelo, datos, self.eq_peana, False)
        self.sujeto, self.omega, self.motor = False, 0.0, False
        self.env._colisiones(modelo, True)       # el techo vuelve a ser sólido
        self._fisica_demo(modelo, False)
        for g, (ct, ca) in self.col_brazo_orig.items():
            modelo.geom_contype[g], modelo.geom_conaffinity[g] = ct, ca
        self.integral[:] = 0.0
        datos.ctrl[:len(ARM_JOINTS)] = datos.qpos[self.qidx]   # sin tirón al salir
        if self.env.visible:
            self.env.recolocar_taladro(modelo, datos)
        for i, v in enumerate(POSICION_STANDBY):
            datos.ctrl[OFFSET_MANO + i] = float(v)
        self.segmentos, self.seg, self.terminada = [], None, True
        mujoco.mj_forward(modelo, datos)
