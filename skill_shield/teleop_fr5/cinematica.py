"""
Cinemática del FAIRINO FR5 en numpy puro (sin MuJoCo).

· La cadena se LEE de fairino_mjcf.xml (base_link → shoulder → … → wrist3 → mano_mount), de
  modo que la IK usa exactamente la misma geometría que la simulación. También se puede
  construir desde los parámetros DH oficiales del FR5 (ver `CadenaFR5.desde_dh`).
· FK de todos los marcos, jacobiana geométrica de cualquier punto y, como el FR5 es un brazo
  tipo UR (ejes 2-3-4 paralelos, muñeca con desplazamiento d4), IK ANALÍTICA con sus 8
  configuraciones (hombro izq/der × codo arriba/abajo × muñeca normal/volteada).

Medidas (corregido el 5-nov): el MJCF salía del URDF ANTIGUO del FR5 (fr5_robot.urdf de
frcobot_ros: d1 = 155 mm, d4 = 130 mm), que pone la muñeca 28 mm desplazada respecto al robot
real en cualquier postura. Ahora fairino_mjcf.xml tiene las del URDF OFICIAL del FR5 v6
(fr5v6.urdf / fairino5_v6.urdf): d1 = 152, a2 = 425, a3 = 395,01, d4 = 102,1, d5 = 102 mm
(alcance de catálogo, 922 mm ≈ a2 + a3 + d5). `CadenaFR5.desde_dh()` construye la cadena
oficial; `comparar_con_dh()` imprime las diferencias (deben salir < 0,1 mm).

Convenciones:
  q  = ángulos de joint1..joint6 tal como los usa MuJoCo (= theta de la DH estándar).
  Marco "herramienta" = mano_mount (hand-forward = +Y de mano_mount).
  Puntos de interés (en mano_mount): muñeca OrcaHand (0, 0.0923, 0.010), palma (0, 0.137, 0.007).
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

# ── puntos de la OrcaHand en el marco de mano_mount (medidos en orcahand_right_body.xml) ──
OFFSET_MUNECA_ORCA = np.array([0.0, 0.0923, 0.010])     # eje de la articulación right_wrist
OFFSET_PALMA_ORCA  = np.array([0.0, 0.137, 0.007])      # centro de la palma
OFFSET_NUDILLOS    = np.array([0.0008, 0.181, 0.004])

# Parámetros DH oficiales del FR5 (manual FAIRINO, tabla 2.4-5, y URDF oficial fr5v6.urdf),
# DH estándar tipo UR. d6 = brida (en el MJCF la cadena llega hasta mano_mount, más allá).
DH_OFICIAL_FR5 = dict(d1=0.152, a2=-0.425, a3=-0.39501, d4=0.1021, d5=0.102, d6=0.100)


# ══════════════════════════════════════════════════════════════════════════════
# utilidades de rotación
# ══════════════════════════════════════════════════════════════════════════════

def quat_a_mat(q):
    """Cuaternión (w, x, y, z) → matriz 3x3."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_eje(eje, ang):
    """Rodrigues: giro de 'ang' rad alrededor del eje unitario."""
    x, y, z = np.asarray(eje, dtype=float) / max(np.linalg.norm(eje), 1e-12)
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def euler_xyz_a_mat(e):
    """Euler 'xyz' de MuJoCo (eulerseq por defecto): R = Rx(a) · Ry(b) · Rz(c)."""
    a, b, c = e
    return rot_eje([1, 0, 0], a) @ rot_eje([0, 1, 0], b) @ rot_eje([0, 0, 1], c)


def T_de(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv_T(T):
    R, p = T[:3, :3], T[:3, 3]
    return T_de(R.T, -R.T @ p)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def mat_desde_z_y_roll(z, roll, ref=None):
    """Matriz de rotación cuyo eje Z es 'z' (unitario); el giro alrededor de Z (roll) se mide
    desde una referencia: la proyección de 'ref' (por defecto el eje vertical del mundo)
    perpendicular a z. Sirve para construir la orientación objetivo de la mano a partir de la
    dirección del antebrazo (z) y la pronosupinación (roll)."""
    z = np.asarray(z, dtype=float)
    z = z / max(np.linalg.norm(z), 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if ref is None else np.asarray(ref, dtype=float)
    x = ref - float(ref @ z) * z
    if np.linalg.norm(x) < 0.2:                       # z casi vertical: otra referencia
        ref = np.array([1.0, 0.0, 0.0])
        x = ref - float(ref @ z) * z
    x /= np.linalg.norm(x)
    x = rot_eje(z, roll) @ x
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# ══════════════════════════════════════════════════════════════════════════════
# cadena cinemática
# ══════════════════════════════════════════════════════════════════════════════

class CadenaFR5:
    """
    Cadena serie de 6 ejes de revolución + marco de herramienta fijo.

    eslabones: lista de 6 (T_fijo 4x4, eje 3) — T_fijo lleva del marco anterior (ya girado)
               al marco del joint; el joint gira 'eje' en ese marco.
    T_base:    mundo → marco del primer eslabón (base_link del MJCF, con su inclinación).
    T_tool:    marco del último joint (wrist3, ya girado) → marco de herramienta (mano_mount).
    limites:   (6, 2) rad.
    """

    def __init__(self, eslabones, T_base, T_tool, limites, nombres=None, origen="?"):
        assert len(eslabones) == 6
        self.eslabones = [(np.asarray(T, dtype=float), np.asarray(a, dtype=float) / np.linalg.norm(a))
                          for T, a in eslabones]
        self._ejes = np.array([a for _, a in self.eslabones])          # (6, 3) ejes en su marco
        self.T_base = np.asarray(T_base, dtype=float)
        self.T_tool = np.asarray(T_tool, dtype=float)
        self.limites = np.asarray(limites, dtype=float).reshape(6, 2)
        self.nombres = list(nombres) if nombres else [f"joint{k + 1}" for k in range(6)]
        self.origen = origen
        self._dh = None

    # ── construcción ─────────────────────────────────────────────────────────
    @classmethod
    def desde_mjcf(cls, ruta, body_herramienta="mano_mount"):
        """Lee base_link → joint1..6 → mano_mount de un MJCF del FR5 (fairino_mjcf.xml)."""
        root = ET.parse(ruta).getroot()
        base = None
        for b in root.iter("body"):
            if b.get("name") == "base_link":
                base = b
                break
        if base is None:
            raise ValueError(f"{ruta}: no hay <body name='base_link'>")

        def R_de(b):
            if b.get("quat"):
                return quat_a_mat([float(v) for v in b.get("quat").split()])
            if b.get("euler"):
                return euler_xyz_a_mat([float(v) for v in b.get("euler").split()])
            if b.get("axisangle"):
                x, y, z, a = [float(v) for v in b.get("axisangle").split()]
                return rot_eje([x, y, z], a)
            return np.eye(3)

        def p_de(b):
            return np.array([float(v) for v in b.get("pos", "0 0 0").split()])

        T_base = T_de(R_de(base), p_de(base))
        eslabones, limites, nombres = [], [], []
        T_acum = np.eye(4)            # transformación fija acumulada desde el último joint
        cuerpo, T_tool = base, None
        while True:
            hijos = cuerpo.findall("body")
            if not hijos:
                break
            # el hijo que sigue la cadena del brazo (el que tiene joint o es la herramienta)
            sig = None
            for h in hijos:
                if h.get("name") == body_herramienta or h.find("joint") is not None:
                    sig = h
                    break
            if sig is None:
                break
            T_acum = T_acum @ T_de(R_de(sig), p_de(sig))
            if sig.get("name") == body_herramienta:
                T_tool = T_acum
                break
            j = sig.find("joint")
            eje = np.array([float(v) for v in j.get("axis", "0 0 1").split()])
            rng = [float(v) for v in j.get("range", "-3.1416 3.1416").split()]
            eslabones.append((T_acum, eje))
            limites.append(rng)
            nombres.append(j.get("name"))
            T_acum = np.eye(4)
            cuerpo = sig
        if len(eslabones) != 6:
            raise ValueError(f"{ruta}: se esperaban 6 joints en la cadena, hay {len(eslabones)}")
        if T_tool is None:
            raise ValueError(f"{ruta}: no se encontró el body de herramienta '{body_herramienta}'")
        return cls(eslabones, T_base, T_tool, limites, nombres, origen=os.path.basename(ruta))

    @classmethod
    def desde_dh(cls, d1=None, a2=None, a3=None, d4=None, d5=None, d6=None,
                 T_base=None, T_flange_tool=None, limites=None, params=None):
        """
        Cadena UR estándar (alpha = π/2, 0, 0, π/2, −π/2, 0) con los parámetros dados (por
        defecto los oficiales del FR5). El marco DH 6 se convierte al marco 'flange' del MJCF
        (girado +90° en Z) y de ahí a mano_mount con T_flange_tool (por defecto el del MJCF:
        pos (0, 0.0099, 0.160), euler (π/2, 0, 0) respecto a la brida a d6).
        """
        p = dict(DH_OFICIAL_FR5)
        if params:
            p.update(params)
        for k, v in dict(d1=d1, a2=a2, a3=a3, d4=d4, d5=d5, d6=d6).items():
            if v is not None:
                p[k] = v
        alphas = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]
        aa = [0.0, p["a2"], p["a3"], 0.0, 0.0, 0.0]
        dd = [p["d1"], 0.0, 0.0, p["d4"], p["d5"], 0.0]
        # DH estándar: T_i = Rz(theta) Tz(d) Tx(a) Rx(alpha). Eslabón i (fijo, antes del giro):
        # Tz(d_i) Tx(a_i) Rx(alpha_i) del eslabón ANTERIOR seguido del giro Rz(theta_i).
        eslabones = []
        T_prev = np.eye(4)
        for i in range(6):
            eslabones.append((T_prev, np.array([0.0, 0.0, 1.0])))
            T_prev = T_de(p=[0, 0, dd[i]]) @ T_de(p=[aa[i], 0, 0]) @ T_de(rot_eje([1, 0, 0], alphas[i]))
        # tras joint6: Tz(d6) → brida (DH) · Rz(+90°) = brida del MJCF
        T_brida = T_de(p=[0, 0, p["d6"]]) @ T_de(rot_eje([0, 0, 1], math.pi / 2))
        if T_flange_tool is None:
            # mano_mount según fairino_mjcf.xml: pos (0, 0.0099, 0.160) y euler (π/2,0,0) en wrist3,
            # cuyo origen está d6 antes de la brida → respecto a la brida: z = 0.160 − d6
            T_flange_tool = T_de(euler_xyz_a_mat([math.pi / 2, 0, 0]), [0.0, 0.0099, 0.160 - p["d6"]])
        T_tool = T_brida @ T_flange_tool
        if limites is None:
            limites = [(-3.0543, 3.0543), (-4.6251, 1.4835), (-2.8274, 2.8274),
                       (-4.6251, 1.4835), (-3.0543, 3.0543), (-3.0543, 3.0543)]
        return cls(eslabones, np.eye(4) if T_base is None else T_base, T_tool, limites,
                   origen=f"DH {p}")

    # ── FK ───────────────────────────────────────────────────────────────────
    def fk_marcos(self, q):
        """Lista de 7 matrices 4x4 en el MUNDO: marco de cada joint YA GIRADO (6) + herramienta.
        (28-oct: Rodrigues en forma cerrada sin matrices intermedias; mismo resultado, ~3× más
        rápido: la IK diferencial la llama varias veces por fotograma.)"""
        T = self.T_base
        marcos = []
        G = np.eye(4)
        for (Tf, eje), qi in zip(self.eslabones, q):
            x, y, z = eje
            c, s_ = math.cos(qi), math.sin(qi)
            C = 1.0 - c
            G[0, 0], G[0, 1], G[0, 2] = c + x * x * C, x * y * C - z * s_, x * z * C + y * s_
            G[1, 0], G[1, 1], G[1, 2] = y * x * C + z * s_, c + y * y * C, y * z * C - x * s_
            G[2, 0], G[2, 1], G[2, 2] = z * x * C - y * s_, z * y * C + x * s_, c + z * z * C
            T = T @ Tf @ G
            marcos.append(T)
        marcos.append(T @ self.T_tool)
        return marcos

    def fk(self, q):
        """Dict con lo que usa la teleoperación (todo en el mundo)."""
        m = self.fk_marcos(q)
        Tt = m[6]
        return {
            "marcos": m,
            "hombro": m[1][:3, 3].copy(),             # eje de joint2
            "codo": m[2][:3, 3].copy(),               # eje de joint3
            "muneca_fr5": m[4][:3, 3].copy(),         # eje de joint5 (centro de muñeca del FR5)
            "brida": m[5][:3, 3].copy(),              # origen de wrist3 (joint6)
            "T_tool": Tt,
            "muneca": Tt[:3, :3] @ OFFSET_MUNECA_ORCA + Tt[:3, 3],   # articulación de la OrcaHand
            "palma": Tt[:3, :3] @ OFFSET_PALMA_ORCA + Tt[:3, 3],
            "adelante": Tt[:3, 1].copy(),             # hand-forward (+Y de mano_mount)
            "ejes": [m_[:3, :3] @ e for m_, (_, e) in zip(m, self.eslabones)],
        }

    def punto(self, q, offset_tool):
        """Punto fijo en el marco de herramienta → mundo."""
        Tt = self.fk_marcos(q)[6]
        return Tt[:3, :3] @ np.asarray(offset_tool, dtype=float) + Tt[:3, 3]

    # ── Jacobiana ────────────────────────────────────────────────────────────
    def jacobiana(self, q, offset_tool=OFFSET_MUNECA_ORCA, marcos=None):
        """(J_lin 3x6, J_ang 3x6) geométrica en el mundo del punto 'offset_tool' de la herramienta."""
        m = self.fk_marcos(q) if marcos is None else marcos
        Tt = m[6]
        p = Tt[:3, :3] @ np.asarray(offset_tool, dtype=float) + Tt[:3, 3]
        W = np.einsum("nij,nj->ni", np.array([mi[:3, :3] for mi in m[:6]]), self._ejes)
        O = np.array([mi[:3, 3] for mi in m[:6]])
        return np.cross(W, p - O).T, W.T

    def jacobiana_punto_marco(self, q, k, marcos=None):
        """Jacobiana lineal del ORIGEN del marco k (0..6) — p. ej. k=2 → codo, k=4 → muñeca FR5.
        Solo dependen de los joints anteriores."""
        m = self.fk_marcos(q) if marcos is None else marcos
        p = m[k][:3, 3]
        J = np.zeros((3, 6))
        n = min(k, 6)
        if n:
            W = np.einsum("nij,nj->ni", np.array([mi[:3, :3] for mi in m[:n]]), self._ejes[:n])
            O = np.array([mi[:3, 3] for mi in m[:n]])
            J[:, :n] = np.cross(W, p - O).T
        return J

    # ── parámetros DH equivalentes ────────────────────────────────────────────
    def dh_equivalente(self):
        """
        Extrae (d1, a2, a3, d4, d5, d6) y las transformaciones fijas que hacen que esta cadena
        sea EXACTAMENTE una cadena DH tipo UR:  T_tool(q) = T_base · T_dh(q) · T_6tool.
        Comprueba la estructura (ejes 2‖3‖4, 1⟂2, 5⟂4, 6⟂5). Se cachea.
        """
        if self._dh is not None:
            return self._dh
        Tb_inv = inv_T(self.T_base)
        m = [Tb_inv @ M for M in self.fk_marcos(np.zeros(6))]    # en el marco base
        ejes = [m[i][:3, :3] @ self.eslabones[i][1] for i in range(6)]
        ori = [m[i][:3, 3] for i in range(6)]
        z = ejes[0]
        ok = (abs(abs(ejes[1] @ ejes[2]) - 1) < 1e-6 and abs(abs(ejes[2] @ ejes[3]) - 1) < 1e-6
              and abs(ejes[0] @ ejes[1]) < 1e-6 and abs(ejes[3] @ ejes[4]) < 1e-6
              and abs(ejes[4] @ ejes[5]) < 1e-6)
        if not ok:
            raise ValueError("La cadena no tiene estructura UR (ejes 2‖3‖4, 1⟂2, 4⟂5, 5⟂6)")
        # Marco DH 0: origen en ori[0], z = eje1. x0: de modo que, con q=0, el hombro (ori[1])
        # esté en la dirección... En la DH estándar UR con theta=0, el brazo (a2<0) va hacia −x0.
        d1 = float((ori[1] - ori[0]) @ z)
        u = ori[2] - ori[1]                               # hombro → codo (⟂ eje2)
        a2 = -float(np.linalg.norm(u))
        x0 = u / np.linalg.norm(u) * -1.0                 # a2 negativo → el brazo apunta a −x0
        y0 = np.cross(z, x0)
        R0 = np.column_stack([x0, y0, z])
        a3 = -float(np.linalg.norm(ori[3] - ori[2]))
        # d4: desplazamiento del centro de muñeca (eje5) a lo largo del eje 4 (= eje 2)
        d4 = float((ori[4] - ori[3]) @ ejes[3])
        d5 = float((ori[5] - ori[4]) @ ejes[4])
        # d6: del origen de joint6 a la brida DH, a lo largo del eje 6. La brida DH no está en
        # la cadena: tomamos como "brida" el punto del eje 6 más cercano a mano_mount (origen).
        p_tool = m[6][:3, 3]
        d6 = float((p_tool - ori[5]) @ ejes[5])
        # Comprobar signos: en la DH UR estándar, con theta=0: eje2 = −y0 ... lo validamos
        # numéricamente construyendo la cadena DH y comparando (ver comparar_con_dh).
        T_0 = T_de(R0, ori[0])                            # base_link → marco DH 0
        p = dict(d1=d1, a2=a2, a3=a3, d4=d4, d5=d5, d6=d6)
        ref = CadenaFR5.desde_dh(params=p, T_base=self.T_base @ T_0, T_flange_tool=np.eye(4))
        # T_6tool: lo que hay que añadir tras la brida DH (ya girada +90°) para llegar a mano_mount
        T_6tool = inv_T(ref.fk_marcos(np.zeros(6))[6]) @ self.fk_marcos(np.zeros(6))[6]
        # signos de los joints: la DH equivale a la cadena si sus ejes coinciden; si alguno está
        # invertido, el ángulo cambia de signo (lo detectamos y lo guardamos)
        signos = np.ones(6)
        ref_m = [inv_T(self.T_base) @ M for M in ref.fk_marcos(np.zeros(6))]
        for i in range(6):
            e_ref = ref_m[i][:3, :3] @ np.array([0, 0, 1.0])
            signos[i] = 1.0 if float(e_ref @ ejes[i]) > 0 else -1.0
        self._dh = dict(params=p, T_0=T_0, T_6tool=T_6tool, signos=signos)
        return self._dh

    def comparar_con_dh(self, params_oficiales=DH_OFICIAL_FR5, verbose=True):
        """Diferencias entre la cadena (MJCF) y los parámetros DH oficiales (mm)."""
        p = self.dh_equivalente()["params"]
        dif = {k: 1000.0 * (p[k] - params_oficiales[k]) for k in params_oficiales}
        if verbose:
            print(f"Parámetros DH equivalentes de '{self.origen}' vs oficiales del FR5 (mm):")
            for k in params_oficiales:
                marca = "  ← DIFIERE" if abs(dif[k]) > 1.0 else ""
                print(f"   {k}: {1000 * p[k]:8.1f}   oficial {1000 * params_oficiales[k]:8.1f}   "
                      f"dif {dif[k]:+6.1f}{marca}")
        return dif

    # ── IK analítica (brazo tipo UR, 8 soluciones) ───────────────────────────
    def ik_analitica(self, T_tool_mundo, q_ref=None, incluir_fuera_de_limites=False, con_rama=False):
        """
        Todas las configuraciones q (lista de arrays de 6) que ponen el marco de herramienta en
        T_tool_mundo. Hasta 8; se descartan las que salen de los límites articulares (salvo
        incluir_fuera_de_limites=True, para diagnóstico). Cada q viene normalizada al equivalente
        (±2π) más cercano a q_ref (o al centro del rango). Con la muñeca en singularidad
        (ejes 4 y 6 alineados) joint6 toma el valor de q_ref.
        con_rama=True → lista de (q, rama) con rama = (hombro, muñeca, codo) ∈ {0, 1}³: índice de
        la solución de θ1 (hombro izq/der), θ5 (muñeca normal/volteada) y θ3 (codo arriba/abajo).
        Algoritmo: Hawkins (2013), "Analytic IK for the Universal Robots UR-5/UR-10 arms",
        aplicado a los parámetros DH equivalentes de ESTA cadena.
        """
        dh = self.dh_equivalente()
        p = dh["params"]
        d1, a2, a3, d4, d5, d6 = p["d1"], p["a2"], p["a3"], p["d4"], p["d5"], p["d6"]
        s = dh["signos"]
        q_ref = np.asarray(q_ref, dtype=float) if q_ref is not None else self.limites.mean(axis=1)
        # objetivo en el marco DH 0 y hasta la brida DH (sin el giro de +90° del MJCF ni el tool)
        T06 = inv_T(self.T_base @ dh["T_0"]) @ np.asarray(T_tool_mundo, dtype=float) \
            @ inv_T(dh["T_6tool"]) @ T_de(rot_eje([0, 0, 1], -math.pi / 2))
        P06 = T06[:3, 3]
        P05 = P06 - d6 * T06[:3, 2]
        r = math.hypot(P05[0], P05[1])
        soluciones = []
        if r < 1e-9 or abs(d4 / r) > 1.0:
            return soluciones                                # fuera de alcance para theta1
        T60 = inv_T(T06)
        psi = math.atan2(P05[1], P05[0])
        phi = math.acos(min(1.0, max(-1.0, d4 / r)))
        for i1, th1 in enumerate((psi + phi + math.pi / 2, psi - phi + math.pi / 2)):
            c1, s1 = math.cos(th1), math.sin(th1)
            c5 = (P06[0] * s1 - P06[1] * c1 - d4) / d6
            if abs(c5) > 1.0 + 1e-9:
                continue
            c5 = min(1.0, max(-1.0, c5))
            for i5, th5 in enumerate((math.acos(c5), -math.acos(c5))):
                s5 = math.sin(th5)
                if abs(s5) < 1e-6:                            # singularidad de muñeca
                    th6 = float(q_ref[5]) * s[5]
                else:
                    th6 = math.atan2((-T60[1, 0] * s1 + T60[1, 1] * c1) / s5,
                                     (T60[0, 0] * s1 - T60[0, 1] * c1) / s5)
                T01 = _dh_T(th1, 0.0, d1, math.pi / 2)
                T45 = _dh_T(th5, 0.0, d5, -math.pi / 2)
                T56 = _dh_T(th6, 0.0, d6, 0.0)
                T14 = inv_T(T01) @ T06 @ inv_T(T45 @ T56)
                P14 = T14[:3, 3]
                # En el marco 1 (z1 = eje de joint2, horizontal) el brazo y el antebrazo (a2, a3)
                # viven en el plano x1-y1; la componente z1 de P14 es el desplazamiento d4.
                P14xy = math.hypot(P14[0], P14[1])
                c3 = (P14xy ** 2 - a2 ** 2 - a3 ** 2) / (2.0 * a2 * a3)
                if abs(c3) > 1.0 + 1e-9:
                    continue
                c3 = min(1.0, max(-1.0, c3))
                for i3, th3 in enumerate((math.acos(c3), -math.acos(c3))):
                    s3 = math.sin(th3)
                    # P14xy = a2·(c2, s2) + a3·(c23, s23) = (k1 c2 − k2 s2, k1 s2 + k2 c2)
                    th2 = math.atan2(P14[1], P14[0]) - math.atan2(a3 * s3, a2 + a3 * c3)
                    T12 = _dh_T(th2, a2, 0.0, 0.0)
                    T23 = _dh_T(th3, a3, 0.0, 0.0)
                    T34 = inv_T(T23) @ inv_T(T12) @ T14
                    th4 = math.atan2(T34[1, 0], T34[0, 0])
                    th = np.array([th1, th2, th3, th4, th5, th6]) * s
                    q = np.array([_equivalente(v, lo, hi, ref)
                                  for v, (lo, hi), ref in zip(th, self.limites, q_ref)])
                    dentro = bool(np.all(q >= self.limites[:, 0] - 1e-9)
                                  and np.all(q <= self.limites[:, 1] + 1e-9))
                    if dentro or incluir_fuera_de_limites:
                        soluciones.append((q, (i1, i5, i3)) if con_rama else q)
        return soluciones

    # ── IK numérica (mínimos cuadrados amortiguados, con límites) ────────────
    def ik_numerica(self, q0, p_obj, R_obj=None, offset_tool=OFFSET_MUNECA_ORCA,
                    w_pos=1.0, w_ori=0.3, n_iter=30, lam=0.05, paso_max=0.25,
                    q_post=None, w_post=0.0, eje_libre=None, mascara=None):
        """
        Refina q0 para que el punto 'offset_tool' llegue a p_obj y (opcional) la orientación de
        la herramienta a R_obj. 'eje_libre' (3, en el marco de la herramienta) deja libre el giro
        alrededor de ese eje (p. ej. [0,1,0] = pronosupinación libre). 'q_post' con w_post>0 atrae
        hacia una postura de referencia en el espacio nulo/como tarea secundaria débil. 'mascara'
        (6 bool) deja joints fijos. Devuelve (q, error_pos, error_ori).
        """
        q = np.clip(np.asarray(q0, dtype=float).copy(), self.limites[:, 0], self.limites[:, 1])
        lo, hi = self.limites[:, 0], self.limites[:, 1]
        libres = np.ones(6, dtype=bool) if mascara is None else np.asarray(mascara, dtype=bool)
        p_obj = np.asarray(p_obj, dtype=float)

        def errores(qq, marcos):
            Tt = marcos[6]
            p = Tt[:3, :3] @ offset_tool + Tt[:3, 3]
            e_p = p_obj - p
            e_o = np.zeros(3)
            if R_obj is not None:
                Rt = Tt[:3, :3]
                dR = R_obj @ Rt.T
                ang = math.acos(min(1.0, max(-1.0, (np.trace(dR) - 1.0) / 2.0)))
                if ang > 1e-9:
                    e_o = ang / (2.0 * math.sin(ang)) * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
                if eje_libre is not None:
                    ej = Rt @ np.asarray(eje_libre, dtype=float)
                    e_o = e_o - float(e_o @ ej) * ej
            return e_p, e_o

        marcos = self.fk_marcos(q)
        e_p, e_o = errores(q, marcos)
        err = math.hypot(w_pos * np.linalg.norm(e_p), w_ori * np.linalg.norm(e_o))
        for _ in range(n_iter):
            Jl, Ja = self.jacobiana(q, offset_tool, marcos)
            filas, rhs = [w_pos * Jl], [w_pos * e_p]
            if R_obj is not None:
                Ja_ = Ja
                if eje_libre is not None:
                    ej = marcos[6][:3, :3] @ np.asarray(eje_libre, dtype=float)
                    Ja_ = Ja - np.outer(ej, ej @ Ja)
                filas.append(w_ori * Ja_)
                rhs.append(w_ori * e_o)
            if q_post is not None and w_post > 0:
                filas.append(w_post * np.eye(6))
                rhs.append(w_post * (np.asarray(q_post) - q))
            J = np.vstack(filas)
            e = np.concatenate(rhs)
            act = libres.copy()
            for _k in range(6):
                Jm = J * act
                dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + lam ** 2 * np.eye(J.shape[0]), e)
                dq[~act] = 0.0
                fuera = act & (((q + dq) < lo) | ((q + dq) > hi))
                if not fuera.any():
                    break
                act &= ~fuera
            mx = float(np.abs(dq).max())
            if mx > paso_max:
                dq *= paso_max / mx
            # búsqueda lineal
            alfa, ok = 1.0, False
            for _ in range(5):
                qn = np.clip(q + alfa * dq, lo, hi)
                mn = self.fk_marcos(qn)
                e_pn, e_on = errores(qn, mn)
                errn = math.hypot(w_pos * np.linalg.norm(e_pn), w_ori * np.linalg.norm(e_on))
                if q_post is not None and w_post > 0:
                    errn = math.hypot(errn, w_post * np.linalg.norm(np.asarray(q_post) - qn))
                    err_cmp = math.hypot(err, w_post * np.linalg.norm(np.asarray(q_post) - q))
                else:
                    err_cmp = err
                if errn <= err_cmp + 1e-9:
                    q, marcos, e_p, e_o, ok = qn, mn, e_pn, e_on, True
                    err = math.hypot(w_pos * np.linalg.norm(e_p), w_ori * np.linalg.norm(e_o))
                    break
                alfa *= 0.5
            if not ok or (np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_o) < 1e-3):
                break
        return q, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_o))


def _dh_T(theta, a, d, alpha):
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def _equivalente(v, lo, hi, ref):
    """De v, v±2π, v±4π el que cae en [lo, hi] más cerca de ref (o v recortado si ninguno)."""
    cands = [v + k * 2.0 * math.pi for k in (-2, -1, 0, 1, 2)]
    dentro = [c for c in cands if lo - 1e-9 <= c <= hi + 1e-9]
    if dentro:
        return float(min(dentro, key=lambda c: abs(c - ref)))
    return float(min(cands, key=lambda c: abs(c - ref)))


def cargar_cadena(ruta_mjcf=None, verbose=True):
    """Cadena del MJCF de config.RUTA_XML (o la ruta dada); si no se puede leer, la DH oficial."""
    if ruta_mjcf is None:
        try:
            from config import RUTA_XML as ruta_mjcf
        except Exception:
            ruta_mjcf = None
    if ruta_mjcf and os.path.exists(ruta_mjcf):
        c = CadenaFR5.desde_mjcf(ruta_mjcf)
        if verbose:
            print(f"🦾 Cadena FR5 leída de {os.path.basename(ruta_mjcf)} "
                  f"(base {'INCLINADA' if not np.allclose(c.T_base[:3, :3], np.eye(3)) else 'vertical'})")
        return c
    if verbose:
        print("⚠️  Sin MJCF: cadena FR5 con los parámetros DH oficiales")
    return CadenaFR5.desde_dh()
