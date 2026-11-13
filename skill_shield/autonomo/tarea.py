"""
Tarea de la herramienta (paso 10, 1ª versión, 11-nov): COGER → USAR → DEJAR → VOLVER, con el
FR5 y la OrcaHand reales y SIEMPRE con el pedal de la web (hombre muerto).

Todavía no hay herramienta de verdad, solo los marcadores ArUco del catálogo, así que el robot
«finge»:
  · COGER: pone la palma delante del marcador de la herramienta, mirándolo, a
    AUTONOMO_TAREA_DIST_AGARRE; cierra la mano (postura AUTONOMO_TAREA_MANO_AGARRE) y se retira.
  · USAR: un destornillador virtual sale del puño por el lado del índice (AUTONOMO_TAREA_PUNTA y
    AUTONOMO_TAREA_EJE, marco de mano_mount). La punta se lleva delante del marcador del punto de
    uso, apuntando hacia él, y AUTONOMO_TAREA_USO_VECES veces: acerca → gira la mano alrededor de
    la varilla → aleja → deshace el giro (como una persona atornillando). Al girar, la punta no
    se mueve.
  · DEJAR: palma delante del marcador del punto de dejada, abre la mano y se retira.
  · VOLVER: a la postura en la que estaba al planificar.

PLAN (Planificador, en un hilo aparte: el gemelo y la web siguen funcionando mientras tanto).
IK analítica del FR5. A cada sitio se llega de frente si se puede y, si la muñeca no da, de lado
(hasta AUTONOMO_TAREA_INCLINACION_MAX_DEG); de cada sitio se guardan varias formas de llegar y se
prueban combinaciones hasta que también los tramos entre sitios están libres. Tramos RECTOS para acercar y alejar; tramos ARTICULARES para ir de un sitio
a otro (si uno choca, se prueba pasando por la postura de inicio). Cada postura del camino (cada
AUTONOMO_TAREA_PASO_DEG) se comprueba:
  · límites articulares (los que fr5_servo puede mandar, con los blandos de la controladora);
  · cápsulas del brazo contra sí mismo, el soporte, el suelo, la cámara y TELEOP_FR5_OBSTACULOS;
  · la mano, con sus mallas, contra el brazo (giro del J4 incluido), el soporte y el entorno
    (choque_mano.ChoqueManoBrazo, sin dejar que se cierre ningún hueco de la postura de inicio);
  · el mapa de obstáculos de la cámara, menos una esfera alrededor de cada marcador
    (AUTONOMO_TAREA_EXCLUIR_RADIO: el soporte o la hoja del marcador no cuentan).
RODEAR (paso 8, 11-nov): si el tramo directo entre dos sitios choca, y pasando por la postura de
inicio también, se busca un rodeo en el espacio de joints (RRT-Connect: dos árboles que crecen
desde cada punta con posturas al azar hasta tocarse) y se acorta con atajos rectos. Con tiempo
acotado (AUTONOMO_TAREA_RODEO_S por tramo, AUTONOMO_TAREA_RODEO_TOTAL_S por plan). La mano se
comprueba contra el mapa con los puntos de sus mallas, no con la cápsula gruesa.
Lo que NO hace todavía: personas (para eso están el pedal y la seta) ni lo desconocido del mapa.

PELOTAS (11-nov, planificar_pelota): «coger, apretar un poco y dejar». La mano llega desde arriba
(o algo inclinada), baja hasta rodear la pelota, cierra, aprieta un poco, afloja y comprueba con
los táctiles que la tiene (si los táctiles no funcionan, se avisa y sigue); sube, la lleva al punto
de dejada, la deja apoyada sobre el marcador, sube y vuelve. Sin punto de uso.

EJECUCIÓN (Ejecutor + Tarea.paso, llamado desde el bucle de modo.py): cada tramo con perfil
trapezoidal a AUTONOMO_TAREA_VEL_DEG_S (el joint más rápido). Solo avanza con el pedal pisado
y el FR5 enviando; al soltarlo, fr5_servo hace ServoMoveEnd y la tarea espera donde está. Se
aborta si la controladora da error o si el robot real se separa más de
AUTONOMO_TAREA_DESVIO_MAX_DEG de lo mandado. La mano (ManoActiva) se conecta con par la primera
vez que se pisa el pedal tras «Ejecutar» (orca_core la lleva entonces a su postura neutra).
"""
import math
import threading
import time
from collections import Counter

import numpy as np

from skill_shield.teleop_fr5.cinematica import OFFSET_PALMA_ORCA, T_de

# Postura de la mano por defecto (índices de ctrl 0..16 de la mano, rad, convención del sim;
# lo no indicado = 0). Abierta = la de reposo (POSICION_STANDBY); agarre = puño de fuerza, como
# el de la demo del taladro pero con el índice cerrado igual que los demás.
MANO_ABIERTA = {0: 0.0, 13: -0.0979, 14: 0.641, 15: 0.0, 16: 0.175}
MANO_AGARRE = {0: 0.0, 2: 1.25, 3: 1.10, 5: 1.25, 6: 1.10, 8: 1.25, 9: 1.10, 11: 1.25, 12: 1.10,
               13: 0.35, 14: 0.45, 15: 0.70, 16: 0.90}
# Pelota cogida desde arriba: dedos abiertos y algo separados → rodear la pelota → apretar un poco más.
MANO_PELOTA_ABIERTA = {0: 0.0, 1: -0.20, 4: -0.10, 7: 0.0, 10: 0.15, 13: 0.0, 14: 0.9, 15: 0.0, 16: 0.0}
MANO_PELOTA_CERRADA = {0: 0.0, 1: -0.15, 2: 0.85, 3: 0.75, 4: -0.08, 5: 0.85, 6: 0.75, 7: 0.0, 8: 0.85, 9: 0.75,
                       10: 0.10, 11: 0.85, 12: 0.75, 13: 0.25, 14: 0.9, 15: 0.55, 16: 0.65}
MANO_PELOTA_APRETAR = {k: (v + 0.15 if k in (2, 3, 5, 6, 8, 9, 11, 12, 15, 16) else v)
                       for k, v in MANO_PELOTA_CERRADA.items()}
# Pinza (13-nov): pulgar contra índice y medio; anular y meñique recogidos para que no estorben.
# ABIERTA: yemas a ~13 cm. CERRADA: la yema del pulgar toca la del índice (buscado con el modelo de
# la mano). Para una pelota se cierra hasta tocarla (interpolando entre las dos) y un poco más para
# apretar (AUTONOMO_TAREA_PINZA_APRIETE).
MANO_PINZA_ABIERTA = {0: 0.0, 2: 1.4, 3: 1.3, 5: 1.4, 6: 1.3, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0,
                      13: 0.094, 14: 0.282, 15: -0.25, 16: -0.20}
MANO_PINZA_CERRADA = {0: 0.0, 2: 1.4, 3: 1.3, 5: 1.4, 6: 1.3, 7: 0.0, 8: 0.98, 9: 0.733, 10: 0.0, 11: 0.98, 12: 0.733,
                      13: 0.094, 14: 0.282, 15: 0.393, 16: 0.323}
# Destornillador virtual (marco de mano_mount): cogido como lo coge una persona, con el mango en
# el puño y la varilla saliendo por el lado del índice (+Z), punta a 9 cm de la mano.
PUNTA = (-0.03, 0.15, 0.14)
EJE = (0.0, 0.0, 1.0)
PESOS_IK = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5])     # qué joints «cuesta» más mover
NOMBRES = {"base": "la base", "brazo": "el brazo", "antebrazo": "el antebrazo", "muneca": "la muñeca",
           "mano": "la mano", "suelo": "el suelo", "camara_cabeza": "la cámara de cabeza"}


class ErrorPlan(Exception):
    pass


def _rot_eje(eje, ang):
    x, y, z = _unit(eje)
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def _alinear(a, b):
    """Giro mínimo que lleva el vector unitario a sobre b."""
    a, b = _unit(a), _unit(b)
    v, c = np.cross(a, b), float(a @ b)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        u, _ = _base_perp(a)
        return _rot_eje(u, math.pi)
    return _rot_eje(v, math.atan2(np.linalg.norm(v), c))


def _tipo_razon(m):
    """Agrupa los motivos para el mensaje («J5 se sale…» → la muñeca no se dobla tanto)."""
    if m.startswith(("J4", "J5", "J6")):
        return "la muñeca tendría que doblarse más de lo que permiten sus límites (J4-J6)"
    if m.startswith(("J1", "J2", "J3")):
        return "el brazo tendría que salirse de sus límites (J1-J3)"
    if "bajaría de la mesa" in m:
        return "la mano bajaría más que la pelota (se metería en la mesa)"
    if "mapa" in m or "apoyo" in m:
        return ("la mano tocaría la mesa u otro obstáculo que ve la cámara" if m.startswith(("la mano", "la pelota"))
                else "el brazo tocaría un obstáculo que ve la cámara")
    if m.startswith("la mano queda"):
        return "la mano queda demasiado cerca del propio brazo o del soporte"
    return "chocaría (" + m.split(" (")[0] + ")"


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _base_perp(z):
    """Dos vectores unitarios perpendiculares a z (y entre sí)."""
    z = _unit(z)
    a = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _unit(np.cross(a, z))
    return u, np.cross(z, u)


def _g(cfg, clave, defecto):
    v = cfg.get(clave)
    return defecto if v is None else v


# ══════════════════════════════════════════════════════════════════════════════
# PLAN
# ══════════════════════════════════════════════════════════════════════════════

class Fase:
    """Un tramo del plan: brazo (qs, posturas seguidas, N x 6) o mano (postura de destino)."""

    def __init__(self, nombre, qs=None, mano=None, dur=None, comprobar=None):
        self.nombre = nombre
        self.qs = None if qs is None else np.asarray(qs, float).reshape(-1, 6)
        self.mano = mano
        self.dur = dur
        self.comprobar = comprobar          # p. ej. "agarre": la Tarea mira los táctiles al empezarla
        self.mano_chk = (None, None)        # (postura de la mano, carga) con la que se comprobó el tramo

    def __repr__(self):
        que = (f"mano {self.mano}" if self.mano else f"{len(self.qs)} posturas" if self.qs is not None
               else f"comprobar {self.comprobar}")
        return f"Fase({self.nombre!r}, {que})"


class Plan:
    def __init__(self, q0, fases, nombre="coger, usar y dejar"):
        self.q0 = np.asarray(q0, float)
        self.fases = list(fases)
        self.nombre = nombre
        self.pelota = None                  # (pelotas) lo necesario para afinar la bajada al ejecutar
        self.posturas = {}                  # posturas de la mano propias de este plan (la pinza)

    @property
    def q_final(self):
        for f in reversed(self.fases):
            if f.qs is not None:
                return f.qs[-1]
        return self.q0


class Planificador:
    """cadena: CadenaFR5 · cfg: dict con las claves de config · T_mundo_cam: calibración (para
    meter la cámara de cabeza como obstáculo). modelo / choque: para las pruebas."""

    def __init__(self, cadena, cfg, T_mundo_cam=None, modelo=None, choque=None):
        from skill_shield.autonomo.mapa import modelo_colision
        from skill_shield.hardware.fr5_servo import PARAMETROS_POR_DEFECTO, limites_envio_rad
        self.c = cadena
        g = dict(cfg)
        self.modelo = modelo or modelo_colision(cadena, g, T_mundo_cam)
        if choque is None:
            from skill_shield.autonomo.choque_mano import desde_config
            choque = desde_config(cadena, g, entorno=self.modelo.entorno)
        self.choque = choque
        p = dict(PARAMETROS_POR_DEFECTO)
        p.update({k: g[k] for k in p if g.get(k) is not None})
        lim = limites_envio_rad(p, g.get("FR5_LIMITES_BLANDOS_DEG"))
        m = math.radians(float(_g(g, "AUTONOMO_TAREA_MARGEN_LIMITES_DEG", 2.0)))
        self.lo = np.maximum(lim[:, 0], cadena.limites[:, 0]) + m
        self.hi = np.minimum(lim[:, 1], cadena.limites[:, 1]) - m
        self.paso = math.radians(float(_g(g, "AUTONOMO_TAREA_PASO_DEG", 2.0)))
        self.d_agarre = float(_g(g, "AUTONOMO_TAREA_DIST_AGARRE", 0.10))
        self.d_previa = float(_g(g, "AUTONOMO_TAREA_DIST_PREVIA", 0.20))
        self.punta = np.asarray(_g(g, "AUTONOMO_TAREA_PUNTA", PUNTA), float)
        self.eje = _unit(_g(g, "AUTONOMO_TAREA_EJE", EJE))       # hacia dónde apunta el destornillador
        self.uso_previa = float(_g(g, "AUTONOMO_TAREA_USO_PREVIA", 0.12))
        self.uso_cerca = float(_g(g, "AUTONOMO_TAREA_USO_CERCA", 0.02))
        self.uso_lejos = float(_g(g, "AUTONOMO_TAREA_USO_LEJOS", 0.06))
        self.uso_giro = math.radians(float(_g(g, "AUTONOMO_TAREA_USO_GIRO_DEG", 30.0)))
        self.uso_veces = int(_g(g, "AUTONOMO_TAREA_USO_VECES", 3))
        self.n_vueltas = int(_g(g, "AUTONOMO_TAREA_VUELTAS_PROBADAS", 12))
        self.margen_mapa = float(_g(g, "AUTONOMO_TAREA_MARGEN_MAPA", 0.03))
        self.salto_max = math.radians(8.0)        # entre dos posturas seguidas de un tramo recto
        self.inc_max = float(_g(g, "AUTONOMO_TAREA_INCLINACION_MAX_DEG", 60.0))
        self.inc_paso = float(_g(g, "AUTONOMO_TAREA_INCLINACION_PASO_DEG", 20.0))
        self.n_azimut = int(_g(g, "AUTONOMO_TAREA_DIRECCIONES", 8))
        self.inclinaciones = []                   # inclinación usada en cada sitio del último plan
        self.n_opciones = int(_g(g, "AUTONOMO_TAREA_OPCIONES", 3))     # formas de llegar a cada sitio
        self.n_probar = 8                          # candidatas que se prueban por inclinación
        self._arbol, self._res = None, 0.02
        # la mano contra el mapa: con los puntos de sus mallas (marco de wrist3), no con su cápsula
        P = getattr(self.choque, "mano", None)
        self._mano_pts0 = None if P is None else np.asarray(P, float)[::max(1, len(P) // 400)]
        self._mano_pts = self._mano_pts0          # la que se usa ahora (cambia con la postura de la fase)
        self._carga = None                         # puntos de lo que lleva la mano (marco de wrist3)
        self._ruta_xml = g.get("RUTA_XML")
        self._posturas = {k: _postura_cfg(g, c, d) for k, c, d in (
            ("pelota_abierta", "AUTONOMO_TAREA_MANO_PELOTA_ABIERTA", MANO_PELOTA_ABIERTA),
            ("pelota_cerrada", "AUTONOMO_TAREA_MANO_PELOTA_CERRADA", MANO_PELOTA_CERRADA))}
        self._nubes = {}
        self._C = np.zeros((0, 3))                # vóxeles ocupados (ya sin lo excluido)
        self._arbol_suave = None                  # (ya no se usa: ver zonas_apoyo)
        self._C_quitado = np.zeros((0, 3))        # vóxeles quitados por «excluir» (la pelota y lo de debajo)
        self._suelo = None                        # (árbol 2D de las columnas quitadas, z mínima de cada una)
        self._choques = []                        # vóxeles del mapa con los que ha chocado la mano (para explicarlo)
        self.margen_suave = float(_g(g, "AUTONOMO_TAREA_MARGEN_APOYO", 0.005))
        # las cápsulas del brazo ya son más gordas que el brazo de verdad: con el mapa basta menos margen
        self.margen_mapa_brazo = float(_g(g, "AUTONOMO_TAREA_MARGEN_MAPA_BRAZO", 0.015))
        # rodear obstáculos (paso 8)
        self.rodeo = bool(_g(g, "AUTONOMO_TAREA_RODEAR", True))
        self.rodeo_s = float(_g(g, "AUTONOMO_TAREA_RODEO_S", 6.0))           # por tramo
        self.rodeo_total_s = float(_g(g, "AUTONOMO_TAREA_RODEO_TOTAL_S", 25.0))  # por plan
        self._rodeo_resto = self.rodeo_total_s
        self._permitir_rodeo = False
        self.rodeos = 0                                                          # cuántos tramos rodean
        self._rng = np.random.default_rng(int(_g(g, "AUTONOMO_TAREA_SEMILLA", 0)))
        # pelotas
        self.pel_holgura = float(_g(g, "AUTONOMO_TAREA_PELOTA_HOLGURA_PALMA", 0.03))
        self.pel_subida = float(_g(g, "AUTONOMO_TAREA_PELOTA_SUBIDA", 0.12))
        self.zona_apoyo = float(_g(g, "AUTONOMO_TAREA_PELOTA_ZONA_APOYO", 0.20))   # + radio, en horizontal
        self.apoyo_grueso = float(_g(g, "AUTONOMO_TAREA_PELOTA_APOYO_GRUESO", 0.06))  # capa de mesa con ruido
        self.grosor_suelo = float(_g(g, "AUTONOMO_TAREA_PELOTA_GROSOR_SUELO", 0.08))  # grosor de ese «suelo»
        self.pel_inc_max = float(_g(g, "AUTONOMO_TAREA_PELOTA_INCLINACION_MAX_DEG", 90.0))  # 90 = de lado
        self.pel_dedos = float(_g(g, "AUTONOMO_TAREA_PELOTA_DEDOS_EXTRA", 0.035))     # 0 = sin «con los dedos»
        self.margen_propio = float(_g(g, "AUTONOMO_TAREA_MARGEN_PROPIO", 0.05))     # cubos pegados al robot: él
        self.margen_mapa_malla = float(_g(g, "AUTONOMO_TAREA_MARGEN_MAPA_MALLA", 0.02))
        # puntos de las mallas del brazo (marco de su eslabón), para mirarlo contra el mapa
        self._brazo_pts = []
        arboles = getattr(self.choque, "arboles", None) or {}
        if arboles:
            from skill_shield.autonomo.choque_mano import reducir
            nombres = {"upperarm_Link": "el brazo", "forearm_Link": "el antebrazo", "wrist1_Link": "la muñeca"}
            self._brazo_pts = [(marco, reducir(np.asarray(a.data), 0.02), nombres[k])
                               for k, (marco, a) in arboles.items() if k in nombres]
        self.quitados_propios = 0
        self.afinar_s = float(_g(g, "AUTONOMO_TAREA_PELOTA_AFINAR_S", 2.0))   # 0 = sin parar a afinar
        self.pref_derecho = float(_g(g, "AUTONOMO_TAREA_PELOTA_PREFERIR_DERECHO", 0.3))  # coste por rad de inclinación
        # formas de coger la pelota, en orden de preferencia (13-nov: primero en pinza)
        self.pel_modos = tuple(_g(g, "AUTONOMO_TAREA_PELOTA_MODOS", ("pinza", "rodeándola", "con los dedos")))
        self.pinza_ab = _postura_cfg(g, "AUTONOMO_TAREA_MANO_PINZA_ABIERTA", MANO_PINZA_ABIERTA)
        self.pinza_ce = _postura_cfg(g, "AUTONOMO_TAREA_MANO_PINZA_CERRADA", MANO_PINZA_CERRADA)
        ap = _g(g, "AUTONOMO_TAREA_PINZA_APRIETE", (0.15, 0.30))   # cuánto más se cierra tras tocarla (0..1)
        self.pinza_apriete = (float(ap[0]), float(ap[1]))
        self.lejos_hombro = float(_g(g, "AUTONOMO_TAREA_LEJOS_HOMBRO", 1.0))
        self.alcance_palma = float(_g(g, "AUTONOMO_TAREA_ALCANCE_PALMA_M", 1.14))   # máx. hombro → centro de la palma
        self.lejos_camara = float(_g(g, "AUTONOMO_TAREA_LEJOS_CAMARA", 1.0))
        self.k_ruido = float(_g(g, "AUTONOMO_TAREA_RUIDO_PROF_K", 0.02))     # m de ruido por m² de distancia
        self.T_cam = None if T_mundo_cam is None else np.asarray(T_mundo_cam, float)
        self.dejar_de_pie = False
        self.modo_pelota = None                   # cómo se cogió la pelota en el último plan

    # ── mapa ──
    def poner_mapa(self, centros, res, excluir=()):
        """centros: vóxeles ocupados (N x 3). excluir: [(centro, radio)] que no cuentan."""
        C = np.asarray(centros if centros is not None else [], float).reshape(-1, 3)
        quitado = []
        for c, r in excluir:
            dentro = np.linalg.norm(C - np.asarray(c, float), axis=1) <= float(r)
            quitado.append(C[dentro])
            C = C[~dentro]
        self._C = C
        self._C_quitado = np.vstack(quitado) if quitado else np.zeros((0, 3))
        self._res = float(res or 0.02)
        self.zonas_apoyo(())
        return len(C)

    def zonas_apoyo(self, zonas):
        """zonas: [(centro de la pelota, radio horizontal, radio de la pelota)]. Es el sitio por el que
        TIENE que pasar la mano para cogerla (o dejarla): alrededor de la pelota y por debajo de
        su parte de arriba. Ahí solo puede haber la pelota y la mesa en la que está, y es justo
        donde la profundidad tiene más ruido (bordes de la pelota, mesa vista en ángulo y de
        lejos: sale como una capa de 4-6 cm de grueso). Así que ahí los vóxeles NO cuentan y, en
        su lugar, nada de la mano (ni la pelota que lleva) puede bajar de la parte de abajo de la
        pelota (+ AUTONOMO_TAREA_MARGEN_APOYO): un suelo plano donde está la mesa de verdad.
        Lo que haya más alto que la pelota (una caja al lado, una bolsa…) sigue contando, y lo
        que esté muy por debajo también (AUTONOMO_TAREA_PELOTA_APOYO_GRUESO + el ruido): si la
        pelota está en alto (en una caja, en la mano de alguien), la mesa de abajo y el brazo de
        quien la sujeta son obstáculos de verdad y no ponen «suelo» a la altura de la pelota."""
        from scipy.spatial import cKDTree
        C, Q = self._C, self._C_quitado
        fuera = np.zeros(len(C), bool)
        suelo_C, suelo_Q = np.full(len(C), np.inf), np.full(len(Q), np.inf)
        for c, rh, rb in zonas:
            c = np.asarray(c, float)
            # hasta un vóxel por encima de la pelota (su parte de arriba cae en el vóxel de encima) +
            # el ruido de la profundidad a esa distancia (crece con la distancia al cuadrado)
            techo = c[2] + float(rb) + self._res + self.ruido_prof(c)
            fondo = c[2] - float(rb) - self.apoyo_grueso - self._res - self.ruido_prof(c)
            z_min = float(c[2] - rb + self.margen_suave)
            en = ((np.linalg.norm(C[:, :2] - c[:2], axis=1) < float(rh)) & (C[:, 2] < techo)
                  & (C[:, 2] > fondo))
            fuera |= en
            suelo_C[en] = np.minimum(suelo_C[en], z_min)      # si se solapan, el suelo más bajo
            enq = (np.linalg.norm(Q[:, :2] - c[:2], axis=1) < float(rh)) & (Q[:, 2] > fondo)
            suelo_Q[enq] = np.minimum(suelo_Q[enq], z_min)
        self._arbol = cKDTree(C[~fuera]) if (~fuera).any() else None
        # el suelo solo está donde la cámara vio algo (la mesa, o la pelota apoyada en ella):
        # donde no hay nada no se inventa una mesa
        # y siempre, justo debajo de cada pelota (lo que ocupa), lo que la sostiene: aunque la cámara no
        # lo vea (la tapa la propia pelota, o se quitó su halo del mapa), los dedos no pasan por debajo
        D = []
        for c, rh, rb in zonas:
            c = np.asarray(c, float)
            g_ = np.arange(-float(rb), float(rb) + 1e-9, 0.5 * self._res)
            X, Y = np.meshgrid(g_, g_)
            k = X ** 2 + Y ** 2 <= float(rb) ** 2
            D.append(np.column_stack([X[k] + c[0], Y[k] + c[1], np.full(k.sum(), c[2] - rb + self.margen_suave)]))
        D = np.vstack(D) if D else np.zeros((0, 3))
        S = np.vstack([C[fuera], Q[np.isfinite(suelo_Q)], np.column_stack([D[:, :2], np.zeros(len(D))])])
        z = np.r_[suelo_C[fuera], suelo_Q[np.isfinite(suelo_Q)], D[:, 2]]
        self._suelo = (cKDTree(S[:, :2]), z) if len(S) else None
        self._arbol_suave = None

    def ruido_prof(self, p):
        """Ruido (m) de la profundidad estéreo en el punto p: k · distancia² a la cámara."""
        if self.T_cam is None:
            return 0.0
        d = float(np.linalg.norm(np.asarray(p, float) - self.T_cam[:3, 3]))
        return self.k_ruido * d * d

    def _nube_postura(self, clave):
        """Puntos de la mano (marco de wrist3) con los dedos en la postura «clave» (None = a 0)."""
        if clave is None or self._ruta_xml is None or self._mano_pts0 is None:
            return self._mano_pts0
        if clave not in self._nubes:
            from skill_shield.autonomo.choque_mano import nube_mano
            P = nube_mano(self._ruta_xml, tuple(sorted(self._posturas[clave].items())))
            self._nubes[clave] = P[::max(1, len(P) // 400)]
        return self._nubes[clave]

    def con_mano(self, clave=None, carga=None):
        """Contexto: comprobar las posturas con la mano en la postura «clave» y llevando «carga»
        (puntos en el marco de wrist3, p. ej. la pelota cogida)."""
        pl = self

        class _Ctx:
            def __enter__(self):
                self.antes = (pl._mano_pts, pl._carga)
                pl._mano_pts, pl._carga = pl._nube_postura(clave), carga
                return pl

            def __exit__(self, *a):
                pl._mano_pts, pl._carga = self.antes
        return _Ctx()

    # ── ¿vale esta postura? ──
    def motivo(self, q):
        """None si la postura q vale; si no, por qué (texto)."""
        q = np.asarray(q, float)
        fuera = np.maximum(self.lo - q, q - self.hi)
        if np.any(fuera > 0):
            k = int(np.argmax(fuera))
            return f"J{k + 1} se sale de su recorrido ({math.degrees(q[k]):.0f}°)"
        marcos = self.c.fk_marcos(q)
        for a, b, pen in self.modelo.colisiones(q, marcos):
            if "mano" in (a, b):               # la mano la mira choque_mano, con sus mallas
                continue
            return f"{NOMBRES.get(a, a)} choca con {NOMBRES.get(b, b)} ({pen * 100:.1f} cm)"
        m = self.choque.motivo(q)
        if m:
            return m
        if self._suelo is not None:
            arbol2d, z_suelo = self._suelo
            Tb = marcos[5]
            for pts, que, tol in ((self._mano_pts, "la mano", 0.0),
                                  (self._carga, "la pelota que lleva", 2 * self.margen_suave)):   # apoyada: sí toca
                if pts is None:
                    continue
                P = pts @ Tb[:3, :3].T + Tb[:3, 3] + np.array([0.0, 0.0, tol])
                d, i = arbol2d.query(P[:, :2], distance_upper_bound=self._res)
                sobre = np.isfinite(d)
                # el «suelo» es una capa (la mesa o lo que sostiene la pelota), no infinito hacia
                # abajo: muy por debajo (al llevar la pelota a una dejada más baja) no cuenta
                zs = z_suelo[i[sobre]] if sobre.any() else None
                if sobre.any() and np.any((P[sobre, 2] < zs) & (P[sobre, 2] > zs - self.grosor_suelo)):
                    return f"{que} bajaría de la mesa (por debajo de la pelota)"
        for arbol, margen, m_brazo, donde in (
                (self._arbol, self.margen_mapa, self.margen_mapa_brazo, "un obstáculo del mapa"),
                (self._arbol_suave, self.margen_suave, self.margen_suave, "la superficie de apoyo")):
            if arbol is None:
                continue
            umbral = margen + 0.5 * self._res
            Tb = marcos[5]
            for pts, que in ((self._mano_pts, "la mano"), (self._carga, "la pelota que lleva")):
                if pts is None:
                    continue
                d, i = arbol.query(pts @ Tb[:3, :3].T + Tb[:3, 3])
                if float(np.min(d)) < umbral:
                    if arbol is self._arbol and len(self._choques) < 5000:
                        self._choques.append(arbol.data[i[int(np.argmin(d))]])
                    return f"{que} toca {donde}"
            if self._brazo_pts:
                # el brazo con los puntos de sus MALLAS (12-nov): la cápsula «brazo» va por la línea
                # entre joints, pero el tramo del brazo del FR5 va desplazado ~14 cm de ella
                for marco, pts, nombre in self._brazo_pts:
                    T = marcos[marco]
                    d, _ = arbol.query(pts @ T[:3, :3].T + T[:3, 3])
                    if float(np.min(d)) < self.margen_mapa_malla + 0.5 * self._res:
                        return f"{nombre} toca {donde}"
            for cap in self.modelo.capsulas(q, marcos):
                if cap.nombre == "base" or (cap.nombre == "mano" and self._mano_pts is not None):
                    continue
                if self._brazo_pts and cap.nombre in ("brazo", "antebrazo"):
                    continue                              # ya mirados con sus mallas
                n = max(2, int(math.ceil(np.linalg.norm(cap.b - cap.a) / 0.02)) + 1)
                pts = cap.a + np.linspace(0.0, 1.0, n)[:, None] * (cap.b - cap.a)
                d, _ = arbol.query(pts)
                if float(np.min(d)) < cap.r + m_brazo + 0.5 * self._res:
                    return f"{NOMBRES.get(cap.nombre, cap.nombre)} toca {donde}"
        return None

    # ── IK ──
    def _coste(self, q, q_ref):
        return float(np.sum(PESOS_IK * np.abs(np.asarray(q) - np.asarray(q_ref))))

    def ik(self, T_tool, q_ref, n=2, razones=None):
        """Hasta n soluciones VÁLIDAS (dentro de límites, sin choques), de la que menos mueve a la
        que más. razones (Counter): se apunta por qué no vale cada una de las demás."""
        sols = sorted(self.c.ik_analitica(T_tool, q_ref, incluir_fuera_de_limites=True),
                      key=lambda q: self._coste(q, q_ref))
        out = []
        for q in sols:
            m = self.motivo(q)
            if m is None:
                out.append(np.asarray(q, float))
                if len(out) >= n:
                    break
            elif razones is not None:
                razones[_tipo_razon(m)] += 1
        if not sols and razones is not None:
            razones["el brazo no llega"] += 1
        return out

    def _ik_cercana(self, T_tool, q_prev):
        sols = self.c.ik_analitica(T_tool, q_prev, incluir_fuera_de_limites=True)   # el límite lo dice motivo()
        if not sols:
            return None
        return min(sols, key=lambda q: float(np.max(np.abs(np.asarray(q) - q_prev))))

    # ── tramos ──
    def recta(self, q0, p_loc, destino, nombre="tramo recto", paso_m=0.01):
        """Lleva el punto p_loc (marco de mano_mount) en línea recta hasta destino (mundo), con la
        orientación de la mano FIJA. Lista de posturas (la primera es q0)."""
        q0 = np.asarray(q0, float)
        Tt = self.c.fk_marcos(q0)[6]
        R = Tt[:3, :3]
        p0 = R @ p_loc + Tt[:3, 3]
        destino = np.asarray(destino, float)
        n = max(1, int(math.ceil(np.linalg.norm(destino - p0) / paso_m)))
        qs = [q0]
        for k in range(1, n + 1):
            p = p0 + (destino - p0) * (k / n)
            q = self._ik_cercana(T_de(R, p - R @ p_loc), qs[-1])
            if q is None:
                raise ErrorPlan(f"{nombre}: el robot no llega ({np.round(p, 3).tolist()} m)")
            if float(np.max(np.abs(q - qs[-1]))) > self.salto_max:
                raise ErrorPlan(f"{nombre}: no se puede seguir la recta sin un salto (el brazo se "
                                "estiraría del todo o la muñeca pasaría por una singularidad)")
            m = self.motivo(q)
            if m:
                raise ErrorPlan(f"{nombre}: {m}")
            qs.append(np.asarray(q, float))
        return np.array(qs)

    def articular(self, q0, q1, nombre="tramo"):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = max(1, int(math.ceil(float(np.max(np.abs(q1 - q0))) / self.paso)))
        qs = q0 + (q1 - q0) * np.linspace(0.0, 1.0, n + 1)[:, None]
        for q in qs[1:]:
            m = self.motivo(q)
            if m:
                raise ErrorPlan(f"{nombre}: {m}")
        return qs

    def conectar(self, q0, q1, q_via=None, nombre="tramo"):
        """Articular de q0 a q1; si choca, pasando por q_via; si también, rodeando (paso 8)."""
        try:
            return self.articular(q0, q1, nombre)
        except ErrorPlan as e:
            if q_via is not None and not (np.allclose(q_via, q0) or np.allclose(q_via, q1)):
                try:
                    a = self.articular(q0, q_via, nombre)
                    return np.vstack([a, self.articular(q_via, q1, nombre)[1:]])
                except ErrorPlan:
                    pass
            if not (self.rodeo and self._permitir_rodeo) or self._rodeo_resto <= 0:
                raise e
            qs = self.rodear(q0, q1, min(self.rodeo_s, self._rodeo_resto))
            if qs is None:
                raise ErrorPlan(f"{nombre}: el camino directo choca ({str(e).split(': ', 1)[-1]}) y no "
                                f"encuentro un rodeo")
            self.rodeos += 1
            return qs

    # ── rodear obstáculos (paso 8): RRT-Connect en el espacio de joints + atajos ──
    def _libre_tramo(self, qa, qb):
        n = max(1, int(math.ceil(float(np.max(np.abs(qb - qa))) / self.paso)))
        for k in range(1, n + 1):
            if self.motivo(qa + (qb - qa) * (k / n)) is not None:
                return False
        return True

    def rodear(self, qa, qb, t_max):
        """Camino libre de qa a qb (posturas cada AUTONOMO_TAREA_PASO_DEG) o None si no lo
        encuentra en t_max s. Dos árboles (desde cada punta) que crecen hacia posturas al azar
        hasta tocarse; luego se acorta con atajos rectos."""
        t0 = time.monotonic()
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        eps = math.radians(10.0)
        arboles = [([qa], [-1]), ([qb], [-1])]

        def cercano(nodos, q):
            return int(np.argmin(np.max(np.abs(np.asarray(nodos) - q), axis=1)))

        def extender(arbol, q):
            nodos, padres = arbol
            i = cercano(nodos, q)
            d = q - nodos[i]
            m = float(np.max(np.abs(d)))
            nuevo = q if m <= eps else nodos[i] + d * (eps / m)
            if not self._libre_tramo(nodos[i], nuevo):
                return "atascado", None
            nodos.append(nuevo)
            padres.append(i)
            return ("llegado" if m <= eps else "avanza"), len(nodos) - 1

        def camino(arbol, i):
            nodos, padres = arbol
            out = []
            while i >= 0:
                out.append(nodos[i])
                i = padres[i]
            return out[::-1]

        a, b = 0, 1
        while time.monotonic() - t0 < t_max:
            q_r = qb if self._rng.random() < 0.1 and a == 0 else (
                qa if self._rng.random() < 0.1 else self.lo + (self.hi - self.lo) * self._rng.random(6))
            r, ia = extender(arboles[a], q_r)
            if r != "atascado":
                q_nuevo = arboles[a][0][ia]
                while True:                                  # el otro árbol va a por él
                    r2, ib = extender(arboles[b], q_nuevo)
                    if r2 != "avanza":
                        break
                if r2 == "llegado":
                    pa, pb = camino(arboles[a], ia), camino(arboles[b], ib)
                    ruta = pa + pb[::-1][1:] if a == 0 else pb + pa[::-1][1:]
                    return self._acortar(ruta, t0, t_max)
            a, b = b, a
        self._rodeo_resto -= time.monotonic() - t0
        return None

    def _acortar(self, ruta, t0, t_max):
        ruta = [np.asarray(q, float) for q in ruta]
        for _ in range(60):
            if len(ruta) <= 2 or time.monotonic() - t0 > t_max + 2.0:
                break
            i, j = sorted(self._rng.choice(len(ruta), 2, replace=False))
            if j - i >= 2 and self._libre_tramo(ruta[i], ruta[j]):
                ruta = ruta[:i + 1] + ruta[j:]
        qs = [ruta[0]]
        for qa_, qb_ in zip(ruta[:-1], ruta[1:]):
            n = max(1, int(math.ceil(float(np.max(np.abs(qb_ - qa_))) / self.paso)))
            qs += [qa_ + (qb_ - qa_) * (k / n) for k in range(1, n + 1)]
        self._rodeo_resto -= time.monotonic() - t0
        return np.array(qs)

    # ── poses ──
    def T_palma(self, T_m, ang, d, a=None, punto=None):
        """mano_mount con la palma a d del marcador en la dirección a (por defecto su normal),
        mirándolo; dedos girados ang alrededor de a. Con 'punto' (en mano_mount), es ese punto
        (p. ej. el de la pinza) el que queda a d del marcador, con la mano igual de orientada."""
        p_m = T_m[:3, 3]
        a = _unit(T_m[:3, 2] if a is None else a)
        u, w = _base_perp(a)
        f = math.cos(ang) * u + math.sin(ang) * w
        R = np.column_stack([a, f, np.cross(a, f)])      # +X de mano_mount = −normal de la palma
        return T_de(R, p_m + a * d - R @ (OFFSET_PALMA_ORCA if punto is None else np.asarray(punto, float)))

    def T_punta(self, T_m, ang, d, a=None):
        """mano_mount con la punta del destornillador a d del marcador en la dirección a,
        apuntando a él; la mano girada ang alrededor del eje del destornillador."""
        p_m = T_m[:3, 3]
        a = _unit(T_m[:3, 2] if a is None else a)
        R = _rot_eje(-a, ang) @ _alinear(self.eje, -a)
        return T_de(R, p_m + a * d - R @ self.punta)

    def girar(self, q0, ang, nombre="girar", paso_rad=math.radians(3.0)):
        """Gira la mano ang alrededor del eje del destornillador, con la punta quieta."""
        q0 = np.asarray(q0, float)
        Tt = self.c.fk_marcos(q0)[6]
        R0 = Tt[:3, :3]
        punta = R0 @ self.punta + Tt[:3, 3]
        eje = R0 @ self.eje
        n = max(1, int(math.ceil(abs(ang) / paso_rad)))
        qs = [q0]
        for k in range(1, n + 1):
            R = _rot_eje(eje, ang * k / n) @ R0
            q = self._ik_cercana(T_de(R, punta - R @ self.punta), qs[-1])
            if q is None:
                raise ErrorPlan(f"{nombre}: el robot no llega")
            if float(np.max(np.abs(q - qs[-1]))) > self.salto_max:
                raise ErrorPlan(f"{nombre}: no se puede girar sin un salto de la muñeca")
            m = self.motivo(q)
            if m:
                raise ErrorPlan(f"{nombre}: {m}")
            qs.append(np.asarray(q, float))
        return np.array(qs)

    def _vueltas(self):
        return [2 * math.pi * k / self.n_vueltas for k in range(self.n_vueltas)]

    def _direcciones(self, z):
        """Direcciones de acercamiento por grupos de inclinación respecto a la normal del marcador:
        [(grados, [direcciones])], de la normal (0°) hasta inclinacion_max. No hace falta llegar
        de frente: si la muñeca no da para tanto, se llega algo de lado."""
        z = _unit(z)
        u, w = _base_perp(z)
        grupos = [(0.0, [z])]
        inc = self.inc_paso
        while inc <= self.inc_max + 1e-9:
            r = math.radians(inc)
            grupos.append((inc, [math.cos(r) * z + math.sin(r) * (math.cos(az) * u + math.sin(az) * w)
                                 for az in np.linspace(0, 2 * math.pi, self.n_azimut, endpoint=False)]))
            inc += self.inc_paso
        return grupos

    def _opciones(self, T_m, q_ref, T_pose, intentar, todas=False):
        """Hasta n_opciones formas distintas de llegar (q de llegada, resultado de intentar,
        inclinación), de menos a más inclinadas y, dentro de cada inclinación, de la que menos
        mueve el brazo a la que más. intentar(q_llegada, a) da el tramo recto o el ciclo de uso
        (ErrorPlan si no se puede). Varias, por si el tramo para ir hasta allí choca."""
        razones, fallos = Counter(), Counter()
        opciones = []
        grupos = self._direcciones(T_m[:3, 2])
        if todas:
            # todas las inclinaciones a la vez (pelotas, 12-nov): la mejor es la que menos mueve el
            # brazo, con un poco de preferencia por llegar derecho; de arriba, de lado o como sea
            cands = []
            for inc, dirs in grupos:
                for a in dirs:
                    for ang in self._vueltas():
                        for q_l in self.ik(T_pose(a, ang), q_ref, 2, razones):
                            cands.append((self._coste(q_l, q_ref) + self.pref_derecho * math.radians(inc),
                                          q_l, a, inc))
            cands.sort(key=lambda c: c[0])
            probados = 0
            for _, q_l, a, inc in cands:
                if any(float(np.max(np.abs(q_l - o[0]))) < math.radians(15) for o in opciones):
                    continue
                if probados >= self.n_probar * 3:
                    break
                probados += 1
                try:
                    opciones.append((q_l, intentar(q_l, a), inc))
                except ErrorPlan as e:
                    fallos[str(e)] += 1
                    continue
                if len(opciones) >= self.n_opciones:
                    break
            return opciones, (razones, fallos)
        for g_i, (inc, dirs) in enumerate(grupos):
            cands = []
            for a in dirs:
                for ang in self._vueltas():
                    for q_l in self.ik(T_pose(a, ang), q_ref, 2, razones):
                        cands.append((self._coste(q_l, q_ref), q_l, a))
            cands.sort(key=lambda c: c[0])
            probados = 0
            for _, q_l, a in cands:
                if any(float(np.max(np.abs(q_l - o[0]))) < math.radians(15) for o in opciones):
                    continue                              # casi igual a otra que ya está
                if probados >= self.n_probar:
                    break
                probados += 1
                try:
                    opciones.append((q_l, intentar(q_l, a), inc))
                except ErrorPlan as e:
                    fallos[str(e)] += 1
                    continue
                if len(opciones) >= self.n_opciones:
                    return opciones, (razones, fallos)
            if opciones and g_i + 1 < len(grupos) and inc > 0:
                return opciones, (razones, fallos)        # ya hay: no hace falta inclinarse más
        return opciones, (razones, fallos)

    def _no_encuentro(self, que, info):
        razones, fallos = info
        txt = f"no encuentro cómo {que}"
        if razones:
            # el motivo más «avanzado» (motivo() mira en orden: límites → choques → mano → mapa):
            # si alguna postura solo fallaba por el mapa, eso es lo que hay que contar
            orden = ("cámara", "la mano queda", "chocaría", "J1-J3", "J4-J6")
            r = min(razones, key=lambda k: (next((i for i, o in enumerate(orden) if o in k), 9), -razones[k]))
            txt += f": casi siempre {r}"
            if r.startswith("la muñeca"):
                txt += (" (el marcador está demasiado cerca del robot o mirándolo de frente: aléjalo "
                        "o gíralo un poco)")
        if fallos:
            txt += f" · {fallos.most_common(1)[0][0]}"
        return ErrorPlan(txt)

    def opciones_palma(self, T_m, q_ref, que):
        """[(postura delante del marcador a d_previa, tramo recto hasta d_agarre, inclinación)]."""
        p_m = T_m[:3, 3]
        ops, info = self._opciones(
            T_m, q_ref, lambda a, ang: self.T_palma(T_m, ang, self.d_previa, a),
            lambda q_l, a: self.recta(q_l, OFFSET_PALMA_ORCA, p_m + _unit(a) * self.d_agarre, f"acercar a {que}"))
        if not ops:
            raise self._con_aviso(self._no_encuentro(f"poner la mano delante de {que}", info), p_m,
                                  "acércalo al robot")
        return ops

    def opciones_uso(self, T_m, q_ref):
        """[(postura delante del punto de uso, fases del ciclo de uso, inclinación)]."""
        p_m = T_m[:3, 3]

        def ciclo(q_l, a):
            a = _unit(a)
            return self._ciclo_uso(q_l, p_m + a * self.uso_cerca, p_m + a * self.uso_lejos, p_m + a * self.uso_previa)
        ops, info = self._opciones(T_m, q_ref, lambda a, ang: self.T_punta(T_m, ang, self.uso_previa, a), ciclo)
        if not ops:
            raise self._con_aviso(self._no_encuentro("usar la herramienta en el punto de uso", info), p_m,
                                  "acércalo al robot")
        return ops

    def _ciclo_uso(self, q_l, cerca, lejos, previa):
        fases = []
        qs = self.recta(q_l, self.punta, cerca, "acercar la punta")
        fases.append(Fase("Acercar la punta", qs))
        q = qs[-1]
        try:
            signo = 1.0
            self.girar(q, self.uso_giro)
        except ErrorPlan:
            signo = -1.0                                   # hacia el otro lado (límites, choques…)
        for i in range(self.uso_veces):
            n = f" ({i + 1}/{self.uso_veces})"
            qs = self.girar(q, signo * self.uso_giro)
            fases.append(Fase("Girar" + n, qs))
            qs = self.recta(qs[-1], self.punta, lejos, "alejar la punta")
            fases.append(Fase("Alejar" + n, qs))
            qs = self.girar(qs[-1], -signo * self.uso_giro, "deshacer el giro")
            fases.append(Fase("Deshacer el giro" + n, qs))
            q = qs[-1]
            if i < self.uso_veces - 1:
                qs = self.recta(q, self.punta, cerca, "acercar la punta")
                fases.append(Fase("Acercar la punta", qs))
                q = qs[-1]
        fases.append(Fase("Retirarse del punto de uso", self.recta(q, self.punta, previa, "retirarse")))
        return fases

    # ── plan completo ──
    def _combinar(self, q0, sitios, mano_vuelta=None):
        """sitios: [(nombre del tramo para llegar, [(q_llegada, q_salida, datos, inclinación)])
        [, (postura de la mano, carga) en ese tramo]]. mano_vuelta: lo mismo para la vuelta.
        Prueba combinaciones (una opción por sitio) hasta que todos los tramos entre ellos y la
        vuelta a q0 están libres. Primero sin rodeos (rápido) y, si no hay, permitiéndolos.
        Devuelve ([opción elegida por sitio], [tramo para llegar a cada sitio], tramo de vuelta)."""
        fallos = Counter()
        for rodeo in (False, True):
            if rodeo and not self.rodeo:
                break
            self._permitir_rodeo = rodeo
            cache = {}

            def tramo(qa, qb, nombre, mano=None):
                clave = (qa.tobytes(), qb.tobytes(), nombre)
                if clave not in cache:
                    try:
                        with self.con_mano(*(mano or (None, None))):
                            cache[clave] = self.conectar(qa, qb, q0, nombre)
                    except ErrorPlan as e:
                        cache[clave] = e
                r = cache[clave]
                if isinstance(r, ErrorPlan):
                    fallos[str(r)] += 1
                    return None
                return r

            def buscar(k, q_actual, elegidas, tramos):
                if k == len(sitios):
                    vuelta = tramo(q_actual, q0, "volver al inicio", mano_vuelta)
                    return None if vuelta is None else (elegidas, tramos, vuelta)
                nombre, opciones = sitios[k][:2]
                mano = sitios[k][2] if len(sitios[k]) > 2 else None
                for op in opciones:
                    ir = tramo(q_actual, op[0], nombre, mano)
                    if ir is None:
                        continue
                    r = buscar(k + 1, op[1], elegidas + [op], tramos + [ir])
                    if r is not None:
                        return r
                return None
            r = buscar(0, q0, [], [])
            self._permitir_rodeo = False
            if r is not None:
                return r
        motivo = fallos.most_common(1)[0][0] if fallos else "?"
        raise ErrorPlan(f"sé llegar a cada sitio, pero no encuentro cómo ir de uno a otro: {motivo}")

    def _empezar(self, q0):
        q0 = np.asarray(q0, float)
        self.inclinaciones, self.rodeos = [], 0
        self._choques = []
        self._rodeo_resto = self.rodeo_total_s
        self._mano_pts, self._carga = self._mano_pts0, None
        # donde está ahora el robot no puede haber nada: si quedan cubos ahí (el brazo visto con
        # retraso o con la calibración algo desajustada), son el propio brazo
        if len(self._C):
            caps = self.modelo.capsulas(q0)
            A = np.array([c.a for c in caps]); B = np.array([c.b for c in caps])
            R = np.array([c.r for c in caps]) + self.margen_propio
            lejos = np.ones(len(self._C), bool)
            for a, b, rr in zip(A, B, R):
                ab = b - a
                t = np.clip((self._C - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0)
                lejos &= np.linalg.norm(self._C - (a + t[:, None] * ab), axis=1) > rr
            if self._brazo_pts:
                from scipy.spatial import cKDTree
                marcos = self.c.fk_marcos(q0)
                M = np.vstack([P @ marcos[k][:3, :3].T + marcos[k][:3, 3] for k, P, _ in self._brazo_pts])
                d, _ = cKDTree(M).query(self._C, distance_upper_bound=self.margen_propio + 0.5 * self._res)
                lejos &= ~np.isfinite(d)
            self.quitados_propios = int((~lejos).sum())
            self._C = self._C[lejos]
        self.zonas_apoyo(())
        self.choque.fijar_inicio(q0)
        m = self.motivo(q0)
        if m:
            raise ErrorPlan(f"la postura actual del robot no pasa las comprobaciones: {m}")
        return q0

    def planificar(self, q0, T_herr, T_uso, T_dejada):
        q0 = self._empezar(q0)
        ops_h = [(q, q, qs, inc) for q, qs, inc in self.opciones_palma(T_herr, q0, "la herramienta")]
        ops_u = [(q, ciclo[-1].qs[-1], ciclo, inc) for q, ciclo, inc in self.opciones_uso(T_uso, q0)]
        ops_d = [(q, q, qs, inc) for q, qs, inc in self.opciones_palma(T_dejada, q0, "el punto de dejada")]
        (h, u, d), (ir_h, ir_u, ir_d), vuelta = self._combinar(
            q0, [("ir a la herramienta", ops_h), ("ir al punto de uso", ops_u), ("ir al punto de dejada", ops_d)])
        self.inclinaciones = [h[3], u[3], d[3]]
        fases = [Fase("Abrir la mano", mano="abierta"),
                 Fase("Ir a la herramienta", ir_h),
                 Fase("Acercar la mano", h[2]),
                 Fase("Cerrar la mano (coger)", mano="agarre"),
                 Fase("Retirarse con la herramienta", h[2][::-1]),
                 Fase("Ir al punto de uso", ir_u)] + list(u[2]) + [
                 Fase("Ir al punto de dejada", ir_d),
                 Fase("Acercar la mano", d[2]),
                 Fase("Abrir la mano (dejar)", mano="abierta"),
                 Fase("Retirarse", d[2][::-1]),
                 Fase("Volver al inicio", vuelta)]
        return Plan(q0, fases)

    def _marco_arriba(self, p):
        """Marco con origen en p y z hacia arriba (la mano llega desde arriba)."""
        T = np.eye(4)
        u, w = _base_perp(np.array([0.0, 0.0, 1.0]))
        T[:3, :3] = np.column_stack([u, w, [0.0, 0.0, 1.0]])
        T[:3, 3] = p
        return T

    def _aviso_lejos(self, p, accion="acércala"):
        """Si el sitio está lejos (del hombro o de la cámara), lo dice: suele ser la causa."""
        avisos = []
        m = self.c.fk_marcos(np.zeros(6))
        d_h = float(np.linalg.norm(np.asarray(p) - m[1][:3, 3]))
        d_b = float(np.linalg.norm(np.asarray(p) - m[0][:3, 3]))
        if d_h > self.lejos_hombro:
            donde = f"está a {d_h:.2f} m del hombro (eje del J2; {d_b:.2f} m desde la base del brazo)"
            if d_h < self.alcance_palma:
                avisos.append(donde + ", al límite del alcance (con el brazo casi estirado la muñeca apenas "
                              "puede girar)")
            else:
                avisos.append(donde + f", FUERA DEL ALCANCE: la palma llega como mucho a {self.alcance_palma:.2f} m "
                              f"del centro del hombro con el brazo estirado del todo (las puntas de los dedos, "
                              f"{self.alcance_palma + 0.12:.2f} m)")
        if self.T_cam is not None:
            d_c = float(np.linalg.norm(np.asarray(p) - self.T_cam[:3, 3]))
            if d_c > self.lejos_camara:
                avisos.append(f"está a {d_c:.2f} m de la cámara, donde la profundidad tiene ruido de varios cm")
        return (" · OJO: " + " y ".join(avisos) + f": {accion}") if avisos else ""

    def _con_aviso(self, texto, p, accion="acércala"):
        """El error con el aviso de distancia; si está fuera del alcance, eso lo primero (es la causa)."""
        a = self._aviso_lejos(p, accion)
        if "FUERA DEL ALCANCE" in a:
            return ErrorPlan(a[len(" · OJO: "):] + " · (detalle: " + str(texto) + ")")
        return ErrorPlan(str(texto) + a)

    def _que_estorba(self, c, r):
        """Describe lo del mapa con lo que ha chocado la mano cerca de la pelota (a < 35 cm):
        a qué distancia está y cuánto se levanta sobre la mesa. Vacío si no ha chocado con nada."""
        if not self._choques:
            return ""
        V = np.asarray(self._choques, float)
        V = V[np.linalg.norm(V[:, :2] - c[:2], axis=1) < 0.35]
        if not len(V):
            return ""
        dh = np.linalg.norm(V[:, :2] - c[:2], axis=1)
        # lo alto del objeto: los vóxeles del mapa en las columnas de los choques
        C = self._C
        cerca = np.zeros(len(C), bool)
        for v in np.unique(np.round(V[:, :2] / self._res).astype(int), axis=0) * self._res:
            cerca |= np.linalg.norm(C[:, :2] - v, axis=1) < 1.5 * self._res
        alto = (float(C[cerca, 2].max()) if cerca.any() else float(V[:, 2].max())) + 0.5 * self._res - (c[2] - r)
        return (f" · lo que estorba (el mapa lo ve a unos {100 * float(np.median(dh)):.0f} cm de la pelota, "
                f"de unos {100 * max(alto, 0.0):.0f} cm de alto) no deja sitio a la mano abierta: "
                f"separa la pelota de eso o quítalo")

    def _pelota_en_mano(self, r, extra=0.0, centro=None):
        """Puntos de la pelota cogida, en el marco de wrist3 (centro a r + holgura (+ extra, si se
        coge con los dedos) delante de la palma; o en 'centro', en mano_mount: la pinza)."""
        m = self.c.fk_marcos(np.zeros(6))
        T_w_mm = np.linalg.inv(m[5]) @ m[6]                     # wrist3 → mano_mount (fijo)
        c = (OFFSET_PALMA_ORCA + np.array([-(r + self.pel_holgura + extra), 0.0, 0.0]) if centro is None
             else np.asarray(centro, float))                     # la palma mira a −X
        k = np.arange(120) + 0.5                                # puntos repartidos por la esfera
        phi, th = np.arccos(1 - 2 * k / 120), math.pi * (1 + 5 ** 0.5) * k
        S = c + r * np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])
        return S @ T_w_mm[:3, :3].T + T_w_mm[:3, 3]

    def planificar_pelota(self, q0, T_pelota, diametro, T_dejada):
        """Coger una pelota, apretarla un poco y dejarla en el punto de dejada. Se prueban, por
        orden, varias formas de cogerla y, para cada una, de llegar:
          · rodeándola con la mano (la palma pegada a la pelota) y, si no se puede,
          · con los dedos (la palma AUTONOMO_TAREA_PELOTA_DEDOS_EXTRA más lejos: la cogen las
            yemas, deja más sitio alrededor);
          · desde arriba primero y, si no, cada vez más inclinada hasta de lado
            (AUTONOMO_TAREA_PELOTA_INCLINACION_MAX_DEG; 90 = la palma en horizontal).
        Cada tramo se comprueba con la mano en la postura que tiene en ese momento (abierta al ir
        y acercarse; cerrada y con la pelota al retirarse, llevarla y dejarla), y alrededor de la
        pelota y del sitio de dejada la mesa se trata aparte (zonas_apoyo)."""
        todos = {"pinza": ("en pinza", 0.0), "rodeándola": ("rodeándola", 0.0),
                 "con los dedos": ("con los dedos", self.pel_dedos)}
        modos = []
        for m in self.pel_modos:
            if m == "con los dedos" and self.pel_dedos <= 0:
                continue
            if m == "pinza" and self.pinza(diametro) is None:
                continue                                  # no cabe entre los dedos
            if m in todos:
                modos.append((m,) + todos[m])
        if not modos:
            modos = [("rodeándola", "rodeándola", 0.0)]
        errores = []
        inc_antes = self.inc_max
        self.inc_max = self.pel_inc_max
        try:
            for clave, nombre, extra in modos:
                try:
                    plan = self._plan_pelota(q0, T_pelota, diametro, T_dejada, extra,
                                             pinza=self.pinza(diametro) if clave == "pinza" else None)
                    plan.nombre = f"coger ({nombre}), apretar y dejar la pelota"
                    self.modo_pelota = nombre
                    return plan
                except ErrorPlan as e:
                    errores.append((nombre, str(e)))
        finally:
            self.inc_max = inc_antes
        if len(errores) == 1:
            raise ErrorPlan(errores[0][1])
        # el de la primera forma (la normal) y, resumido, que la otra tampoco
        raise ErrorPlan(errores[0][1] + " · tampoco " + " ni ".join(n for n, _ in errores[1:]))

    def pinza(self, diametro):
        """(13-nov) La pinza para una pelota de este diámetro, con el modelo de la mano: se cierra
        de MANO_PINZA_ABIERTA a MANO_PINZA_CERRADA (s de 0 a 1) hasta que las yemas de pulgar e
        índice quedan a 'diámetro'. Devuelve dict(s, punto (centro de la pelota en mano_mount,
        entre las dos yemas), contacto/cerrada/apretar (posturas)) o None si no cabe."""
        if self._ruta_xml is None:
            return None
        clave = round(float(diametro), 3)
        cache = self.__dict__.setdefault("_pinzas", {})
        if clave in cache:
            return cache[clave]
        from skill_shield.autonomo.choque_mano import yemas_mano
        m = self.c.fk_marcos(np.zeros(6))
        T_mm_w = np.linalg.inv(m[6]) @ m[5]                     # wrist3 → mano_mount

        def postura(s_):
            ks = set(self.pinza_ab) | set(self.pinza_ce)
            return {k: self.pinza_ab.get(k, 0.0) + s_ * (self.pinza_ce.get(k, 0.0) - self.pinza_ab.get(k, 0.0))
                    for k in ks}

        ss = np.linspace(0.0, 1.0, 21)
        gaps, mids = [], []
        for s_ in ss:
            y = yemas_mano(self._ruta_xml, tuple(sorted((k, round(v, 4)) for k, v in postura(s_).items())), 0.004)
            if "pulgar" not in y or "indice" not in y:
                cache[clave] = None
                return None
            t, i = (T_mm_w[:3, :3] @ y[d] + T_mm_w[:3, 3] for d in ("pulgar", "indice"))
            gaps.append(float(np.linalg.norm(t - i)))
            mids.append((t + i) / 2)
        gaps = np.array(gaps)
        objetivo = float(diametro)
        if objetivo > gaps[0] - 0.015 or objetivo < 0.02:      # más de lo que abre (con holgura) o diminuta
            cache[clave] = None
            return None
        k = int(np.nonzero(gaps <= objetivo)[0][0])              # primera s donde ya la toca
        u = (gaps[k - 1] - objetivo) / max(gaps[k - 1] - gaps[k], 1e-9)
        s_c = float(ss[k - 1] + u * (ss[k] - ss[k - 1]))
        punto = mids[k - 1] + u * (mids[k] - mids[k - 1])
        res = dict(s=s_c, punto=punto, contacto=postura(s_c),
                   cerrada=postura(min(1.0, s_c + self.pinza_apriete[0])),
                   apretar=postura(min(1.0, s_c + self.pinza_apriete[1])))
        cache[clave] = res
        return res

    def _plan_pelota(self, q0, T_pelota, diametro, T_dejada, extra=0.0, pinza=None):
        q0 = self._empezar(q0)
        r = float(diametro) / 2.0
        d_palma = r + self.pel_holgura + extra         # centro de la pelota ↔ centro de la palma
        punto = OFFSET_PALMA_ORCA                      # el punto de la mano que va a la pelota
        if pinza is not None:                          # pinza: el centro entre las yemas, a la pelota
            d_palma, punto = 0.0, np.asarray(pinza["punto"], float)
            self._posturas["pinza_abierta"] = dict(self.pinza_ab)
            self._posturas["pinza_contacto"] = dict(pinza["contacto"])
            self._nubes = {k: v for k, v in self._nubes.items() if not str(k).startswith("pinza")}
        k_ab, k_ce, k_ap = (("pinza_abierta", "pinza_cerrada", "pinza_apretar") if pinza is not None
                            else ("pelota_abierta", "pelota_cerrada", "pelota_apretar"))
        c = np.asarray(T_pelota, float)[:3, 3]
        # dejar: si el marcador está tumbado, la pelota encima; si está de pie (en una hoja), se
        # suelta delante de él y cae. Siempre llegando desde arriba.
        Tm_d = np.asarray(T_dejada, float)
        n_d = _unit(Tm_d[:3, 2])
        self.dejar_de_pie = bool(n_d[2] < math.cos(math.radians(45)))
        c_d = Tm_d[:3, 3] + (n_d * (r + 0.03) if self.dejar_de_pie else np.array([0.0, 0.0, r + 0.01]))
        T_p, T_d = self._marco_arriba(c), self._marco_arriba(c_d)
        # (la pelota se suelta 1 cm por encima del marcador de dejada: su «suelo» es el marcador)
        self.zonas_apoyo([(c, r + self.zona_apoyo, r), (c_d, r + self.zona_apoyo, r + 0.01)])
        if pinza is not None:
            abierta, cerrada = ("pinza_abierta", None), ("pinza_contacto", self._pelota_en_mano(r, centro=punto))
        else:
            abierta, cerrada = ("pelota_abierta", None), ("pelota_cerrada", self._pelota_en_mano(r, extra))

        def coger(q_l, a):
            centro = c + _unit(a) * d_palma
            with self.con_mano(*abierta):
                qs = self.recta(q_l, punto, centro, "bajar a la pelota")
            with self.con_mano(*cerrada):                # al subir ya la lleva
                for q in qs[::-1]:
                    m = self.motivo(q)
                    if m:
                        raise ErrorPlan(f"subir con la pelota: {m}")
            return qs

        def dejar(q_l, a):
            centro = c_d + _unit(a) * d_palma
            with self.con_mano(*cerrada):
                qs = self.recta(q_l, punto, centro, "bajar a dejarla")
            with self.con_mano(*abierta):                # ya la ha soltado
                for q in qs[::-1]:
                    m = self.motivo(q)
                    if m:
                        raise ErrorPlan(f"subir tras dejarla: {m}")
            return qs
        with self.con_mano(*abierta):
            ops_p, info = self._opciones(T_p, q0, lambda a, ang: self.T_palma(T_p, ang, d_palma + self.pel_subida, a,
                                                                          punto),
                                         coger, todas=True)
        if not ops_p:
            raise self._con_aviso(str(self._no_encuentro("llegar a la pelota (ni desde arriba ni de lado)", info))
                                  + self._que_estorba(c, r), c)
        with self.con_mano(*cerrada):
            ops_d, info = self._opciones(T_d, q0, lambda a, ang: self.T_palma(T_d, ang, d_palma + self.pel_subida, a,
                                                                          punto),
                                         dejar, todas=True)
        if not ops_d:
            raise self._con_aviso(self._no_encuentro("llevar la pelota al punto de dejada", info), c_d,
                                  "acerca el punto de dejada")
        self._choques = []                                # solo los del camino
        try:
            (p, d), (ir_p, ir_d), vuelta = self._combinar(
                q0, [("ir a la pelota", [(q, q, qs, inc) for q, qs, inc in ops_p], abierta),
                     ("ir al punto de dejada", [(q, q, qs, inc) for q, qs, inc in ops_d], cerrada)],
                mano_vuelta=abierta)
        except ErrorPlan as e:
            raise self._con_aviso(str(e) + self._que_estorba(c, r), c) from None
        self.inclinaciones = [p[3], d[3]]
        f_ir, f_bajar, f_subir = (Fase("Ir encima de la pelota", ir_p), Fase("Bajar a la pelota", p[2]),
                                  Fase("Subir con la pelota", p[2][::-1]))
        f_llevar, f_dejar, f_alejar, f_volver = (
            Fase("Ir al punto de dejada", ir_d),
            Fase("Bajar a dejarla" if not self.dejar_de_pie else "Bajar delante del marcador", d[2]),
            Fase("Subir", d[2][::-1]), Fase("Volver al inicio", vuelta))
        # parada encima de la pelota para afinar con la cámara (12-nov): con el brazo quieto y la
        # mano al lado de la pelota, la cámara corrige la calibración con el brazo y vuelve a medir
        # la pelota; si se ha movido, se rehace la bajada (Tarea._afinar)
        afinar = [Fase("Afinar con la cámara", dur=self.afinar_s, comprobar="afinar")] if self.afinar_s > 0 else []
        fases = ([Fase("Abrir la mano", mano=k_ab), f_ir] + afinar +
                 [f_bajar,
                  Fase("Cerrar la mano", mano=k_ce),
                  Fase("Apretar un poco", mano=k_ap),
                  Fase("Aflojar", mano=k_ce),
                  Fase("Comprobar que la tengo", dur=0.3, comprobar="agarre"),
                  f_subir, f_llevar, f_dejar,
                  Fase("Abrir la mano (dejar)", mano=k_ab),
                  f_alejar, f_volver,
                  Fase("Mano en reposo", mano="abierta")])
        for f, chk in ((f_ir, abierta), (f_bajar, abierta), (f_subir, cerrada), (f_llevar, cerrada),
                       (f_dejar, cerrada), (f_alejar, abierta), (f_volver, abierta)):
            f.mano_chk = chk
        plan = Plan(q0, fases, nombre="coger, apretar y dejar la pelota")
        plan.pelota = dict(c=c.copy(), r=r, d_palma=d_palma, extra=extra, punto=np.asarray(punto, float).copy(),
                           i_bajar=fases.index(f_bajar), i_subir=fases.index(f_subir),
                           c_dejada=c_d.copy(), pinza=pinza is not None)
        if pinza is not None:                              # posturas que la Tarea manda a la mano
            plan.posturas = dict(pinza_abierta=dict(self.pinza_ab), pinza_cerrada=dict(pinza["cerrada"]),
                                 pinza_apretar=dict(pinza["apretar"]))
        return plan

    def rehacer_bajada(self, plan, c_nueva):
        """(12-nov) La bajada a la pelota rehecha hacia c_nueva (medida otra vez desde encima),
        desde la misma postura de encima y con la misma dirección de llegada. ErrorPlan si no se
        puede. La subida es la misma al revés."""
        pz = plan.pelota
        fb, fs = plan.fases[pz["i_bajar"]], plan.fases[pz["i_subir"]]
        Tt = self.c.fk_marcos(fb.qs[-1])[6]
        a = _unit(Tt[:3, 0])                                # dirección de llegada (+X de mano_mount)
        punto = np.asarray(pz.get("punto", OFFSET_PALMA_ORCA), float)
        centro = np.asarray(c_nueva, float) + a * pz["d_palma"]
        r = pz["r"]                                        # la zona de agarre, donde está ahora la pelota
        self.zonas_apoyo([(np.asarray(c_nueva, float), r + self.zona_apoyo, r),
                          (pz["c_dejada"], r + self.zona_apoyo, r + 0.01)])
        with self.con_mano(*fb.mano_chk):
            qs = self.recta(fb.qs[0], punto, centro, "bajar a la pelota")
        with self.con_mano(*fs.mano_chk):
            for q in qs[::-1]:
                m = self.motivo(q)
                if m:
                    raise ErrorPlan(f"subir con la pelota: {m}")
        return qs

    def valida(self, fase, q):
        """motivo(q) con la mano como estaba en ese tramo del plan (para comprobar un plan hecho)."""
        with self.con_mano(*fase.mano_chk):
            return self.motivo(q)

    def plan_volver(self, q_actual, q_inicio):
        self.choque.fijar_inicio(np.asarray(q_actual, float))
        self._rodeo_resto, self._permitir_rodeo, self.rodeos = self.rodeo_total_s, True, 0
        try:
            qs = self.conectar(np.asarray(q_actual, float), np.asarray(q_inicio, float), None, "volver")
        finally:
            self._permitir_rodeo = False
        return Plan(q_actual, [Fase("Volver al inicio", qs)], nombre="volver al inicio")


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN (en el tiempo)
# ══════════════════════════════════════════════════════════════════════════════

def _trapecio(L, v, a):
    """(duración, s(t)) para recorrer L con velocidad máx v y aceleración a, parando al final."""
    if L <= 1e-9:
        return 0.0, lambda t: L
    if L <= v * v / a:                                     # no llega a velocidad máxima
        T = 2.0 * math.sqrt(L / a)
        return T, lambda t: (0.5 * a * t * t if t <= T / 2 else L - 0.5 * a * (T - t) ** 2) if t < T else L
    ta = v / a
    T = L / v + ta

    def s(t):
        if t >= T:
            return L
        if t <= ta:
            return 0.5 * a * t * t
        if t >= T - ta:
            return L - 0.5 * a * (T - t) ** 2
        return 0.5 * a * ta * ta + v * (t - ta)
    return T, s


def _minjerk(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


class Ejecutor:
    """Recorre el plan en el tiempo. avanzar(dt) solo cuando se puede mover; q() y mano() dan el
    comando de ESTE instante (si no se avanza, el mismo: el robot espera donde está)."""

    def __init__(self, plan, posturas, vel_rad_s, acel_rad_s2, mano_ini=None, t_mano=(1.0, 1.5)):
        self.plan = plan
        self.posturas = dict(posturas, **(getattr(plan, "posturas", None) or {}))
        self.v, self.a = float(vel_rad_s), float(acel_rad_s2)
        self.t_abrir, self.t_cerrar = t_mano
        self.i, self.t = 0, 0.0
        self._q = plan.q0.copy()
        self._mano = dict(mano_ini if mano_ini is not None else posturas["abierta"])
        self._mano_ini = dict(self._mano)
        self._prep = [self._preparar(f) for f in plan.fases]
        self.duracion = sum(p[0] for p in self._prep)

    def _preparar(self, f):
        if f.qs is not None:
            d = np.max(np.abs(np.diff(f.qs, axis=0)), axis=1) if len(f.qs) > 1 else np.zeros(0)
            s_acum = np.r_[0.0, np.cumsum(d)]
            T, s = _trapecio(float(s_acum[-1]), self.v, self.a)
            return T, s_acum, s
        T = f.dur if f.dur is not None else (self.t_cerrar if f.mano in ("agarre", "pelota_cerrada")
                                             else self.t_abrir)
        return float(T), None, None

    @property
    def terminado(self):
        return self.i >= len(self.plan.fases)

    @property
    def fase(self):
        return None if self.terminado else self.plan.fases[self.i]

    def progreso(self):
        hecho = sum(p[0] for p in self._prep[:self.i]) + (self.t if not self.terminado else 0.0)
        return min(1.0, hecho / self.duracion) if self.duracion > 0 else 1.0

    def avanzar(self, dt):
        while dt > 0 and not self.terminado:
            T, s_acum, s = self._prep[self.i]
            f = self.plan.fases[self.i]
            usar = min(dt, max(T - self.t, 0.0))
            fin = self.t + usar >= T - 1e-9
            dt = dt - usar if fin else 0.0           # (sin restos de coma flotante que no avanzan)
            if fin and f.comprobar == "afinar":
                dt = 0.0                             # se para aquí: la Tarea puede rehacer el siguiente tramo
            t = self.t = T if fin else self.t + usar
            if f.qs is not None:
                si = s(t)
                k = int(np.clip(np.searchsorted(s_acum, si, side="right") - 1, 0, max(len(f.qs) - 2, 0)))
                if len(f.qs) == 1 or s_acum[k + 1] <= s_acum[k]:
                    self._q = f.qs[min(k + 1, len(f.qs) - 1)].copy() if t >= T else f.qs[k].copy()
                else:
                    u = (si - s_acum[k]) / (s_acum[k + 1] - s_acum[k])
                    self._q = f.qs[k] + (f.qs[k + 1] - f.qs[k]) * min(max(u, 0.0), 1.0)
            elif f.mano is not None:
                u = _minjerk(t / T) if T > 0 else 1.0
                dest = self.posturas[f.mano]
                claves = set(dest) | set(self._mano_ini)
                self._mano = {k: self._mano_ini.get(k, 0.0) + (dest.get(k, 0.0) - self._mano_ini.get(k, 0.0)) * u
                              for k in claves}
            if self.t >= T:
                if f.qs is not None:
                    self._q = f.qs[-1].copy()
                self.i += 1
                self.t = 0.0
                self._mano_ini = dict(self._mano)

    def q(self):
        return self._q.copy()

    def mano(self):
        return dict(self._mano)

    def cambiar_tramo(self, i, qs):
        """Sustituye las posturas de un tramo que aún no ha empezado (o justo empieza)."""
        f = self.plan.fases[i]
        f.qs = np.asarray(qs, float).reshape(-1, 6)
        self._prep[i] = self._preparar(f)
        self.duracion = sum(p[0] for p in self._prep)


# ══════════════════════════════════════════════════════════════════════════════
# MANO CON PAR
# ══════════════════════════════════════════════════════════════════════════════

class ManoActiva:
    """La OrcaHand con par (OrcaHardwareWrapper), para la tarea. Misma cara que MonitorMano
    (estado() y postura()) para que modo.py la use igual. Al conectar, orca_core la lleva a su
    postura neutra; al cerrar el modo, también (desconectar())."""

    def __init__(self, puerto="/dev/ttyUSB0", fabrica=None):
        self.puerto = puerto
        self._fabrica = fabrica
        self.w = None
        self.lista = False
        self.error = None
        self._postura, self._t_postura, self._t_leida = None, 0.0, 0.0

    def conectar(self):
        try:
            if self._fabrica is not None:
                self.w = self._fabrica()
            else:
                from orca_hardware import OrcaHardwareWrapper
                self.w = OrcaHardwareWrapper(puerto=self.puerto)
            if not self.w.conectar():
                self.error = "no se pudo conectar la mano con par (mira la terminal)"
                return False
            self.lista = True
            return True
        except Exception as e:
            self.error = f"error al conectar la mano: {e}"
            return False

    def enviar(self, postura_ctrl):
        """postura_ctrl: {índice de ctrl de la mano 0..16: rad}."""
        if not self.lista:
            return False
        w = self.w
        arr = [float(postura_ctrl.get(w.JOINT_A_CTRL[j], 0.0)) for j in w._joints]
        return w.enviar_posiciones(arr)

    def postura(self, max_edad_s=1.0):
        """{joint de orca_core: rad (sim)} real, leída como mucho cada 0,1 s."""
        if not self.lista:
            return None
        ahora = time.time()
        if ahora - self._t_leida >= 0.1:
            self._t_leida = ahora
            try:
                e = self.w.estado_real()
                q = self.w._desde_sdk({j: v[1] for j, v in e.items()})
                if q is not None and np.all(np.isfinite(q)):
                    self._postura, self._t_postura = dict(zip(self.w._joints, (float(v) for v in q))), ahora
            except Exception:
                pass
        if self._postura is None or ahora - self._t_postura > max_edad_s:
            return None
        return dict(self._postura)

    def estado(self):
        if self.error:
            return dict(estado="error", texto=self.error, puerto=self.puerto)
        if not self.lista:
            return dict(estado="aviso", texto="conectando con par…", puerto=self.puerto)
        n = len(getattr(self.w, "_joints", []) or [])
        return dict(estado="ok", texto="con par · la mueve la tarea", motores_ok=n, motores_total=n,
                    par_activo=n, puerto=self.puerto, postura=self._postura is not None)

    def cerrar(self):
        if self.w is not None and self.lista:
            self.w.desconectar()
        self.lista = False


# ══════════════════════════════════════════════════════════════════════════════
# LA TAREA (estados, órdenes de la web y FR5)
# ══════════════════════════════════════════════════════════════════════════════

ESTADOS = ("libre", "planificando", "lista", "ejecutando", "pausada", "terminada", "error")


def _postura_cfg(cfg, clave, defecto):
    v = cfg.get(clave)
    return {int(k): float(x) for k, x in (v if v is not None else defecto).items()}


class Tarea:
    """La lleva modo.py: orden(mando) con cada mando de la web y paso(...) en cada vuelta."""

    def __init__(self, cfg, raiz, fr5, simulacro, monitor_mano=None, fabrica_planificador=None,
                 fabrica_mano=None, tactil=None):
        self.cfg = dict(cfg)
        self.raiz = raiz
        self.fr5 = fr5
        self.simulacro = bool(simulacro)
        self.monitor = monitor_mano
        self.mover_mano = bool(_g(self.cfg, "AUTONOMO_TAREA_MOVER_MANO", True))
        self.posturas = dict(
            abierta=_postura_cfg(self.cfg, "AUTONOMO_TAREA_MANO_ABIERTA", MANO_ABIERTA),
            agarre=_postura_cfg(self.cfg, "AUTONOMO_TAREA_MANO_AGARRE", MANO_AGARRE),
            pelota_abierta=_postura_cfg(self.cfg, "AUTONOMO_TAREA_MANO_PELOTA_ABIERTA", MANO_PELOTA_ABIERTA),
            pelota_cerrada=_postura_cfg(self.cfg, "AUTONOMO_TAREA_MANO_PELOTA_CERRADA", MANO_PELOTA_CERRADA),
            pelota_apretar=_postura_cfg(self.cfg, "AUTONOMO_TAREA_MANO_PELOTA_APRETAR", MANO_PELOTA_APRETAR))
        self.tactil = tactil                    # función → estado_tactil() (dedos con su fuerza), o None
        self.tactil_n = float(_g(self.cfg, "AUTONOMO_TAREA_PELOTA_TACTIL_N", 0.3))
        self.tactil_dedos = int(_g(self.cfg, "AUTONOMO_TAREA_PELOTA_TACTIL_DEDOS", 2))
        self._comprobada = None
        self.vel = math.radians(float(_g(self.cfg, "AUTONOMO_TAREA_VEL_DEG_S", 15.0)))
        self.acel = math.radians(float(_g(self.cfg, "AUTONOMO_TAREA_ACEL_DEG_S2", 30.0)))
        self.desvio_max = math.radians(float(_g(self.cfg, "AUTONOMO_TAREA_DESVIO_MAX_DEG", 8.0)))
        self.excluir_r = float(_g(self.cfg, "AUTONOMO_TAREA_EXCLUIR_RADIO", 0.15))
        self.usar_mapa = bool(_g(self.cfg, "AUTONOMO_TAREA_USAR_MAPA", True))
        self._fabrica_plan = fabrica_planificador
        self._fabrica_mano = fabrica_mano
        self._planificador = None
        self.estado, self.texto = "libre", "Elige la herramienta y pulsa «Planificar»."
        self.plan, self.ejec = None, None
        self._q_inicio = None
        self._hilo = None
        self._resultado = None
        self._gen = 0
        self._auto = False
        self._orden_vista = None
        self._enviando = False
        self._t_desvio = 0.0
        self.mano_activa = None
        self._hilo_mano = None
        self._pelota_id = None
        self.afinado_cm = None
        self.afinar_min = float(_g(self.cfg, "AUTONOMO_TAREA_AFINAR_MIN_M", 0.005))
        self.afinar_max = float(_g(self.cfg, "AUTONOMO_TAREA_AFINAR_MAX_M", 0.10))

    # ── órdenes de la web ──
    def orden(self, mando, contexto):
        """mando: el JSON de la web. contexto: dict(q_real, herramientas, objetivo_id, mapa) o una
        función que lo devuelve (solo se llama si hay una orden nueva)."""
        t = (mando or {}).get("tarea") or {}
        oid = t.get("id")
        if oid is None or oid == self._orden_vista:
            return
        self._orden_vista = oid
        contexto = contexto() if callable(contexto) else contexto
        a = t.get("accion")
        if a == "planificar":
            self._pedir_plan(contexto, volver=False)
        elif a == "ejecutar":
            self._ejecutar(contexto.get("q_real"))
        elif a == "cancelar":
            self._cancelar("Cancelada: el robot se queda donde está.")
        elif a == "volver":
            self._pedir_plan(contexto, volver=True)

    def _puede_planificar(self):
        return self.estado in ("libre", "lista", "terminada", "error", "pausada")

    def _pedir_plan(self, ctx, volver):
        if not self._puede_planificar():
            return
        if self.estado == "pausada":
            self._cancelar("")
        q = ctx.get("q_real")
        if q is None:
            return self._error("No hay pose real del FR5: hace falta para planificar.")
        if volver:
            if self._q_inicio is None:
                return self._error("No hay postura de inicio: primero planifica la tarea.")
            datos = dict(volver=True, q=np.asarray(q, float), q_inicio=self._q_inicio.copy())
        else:
            lista = [h for h in (ctx.get("herramientas") or []) if h.get("T_mundo") is not None]
            elegida = ctx.get("objetivo_id")
            herrs = [h for h in lista if h.get("tipo") in ("herramienta", "pelota")]
            herr = next((h for h in herrs if h.get("id") == elegida), None)
            if herr is None and len(herrs) == 1 and elegida is None:
                herr = herrs[0]
            uso = next((h for h in lista if h.get("tipo") == "punto_uso"), None)
            dej = next((h for h in lista if h.get("tipo") == "punto_dejada"), None)
            pelota = herr is not None and herr.get("tipo") == "pelota"
            self._pelota_id = herr.get("id") if pelota else None
            necesarios = [("la herramienta o la pelota (elígela en la lista)", herr)] + \
                ([] if pelota else [("el punto de uso", uso)]) + [("el punto de dejada", dej)]
            falta = [n for n, h in necesarios if h is None]
            if falta:
                return self._error("No veo " + ", ".join(falta) + " con la calibración cámara→robot.")
            sitios = [h for _, h in necesarios]
            inestable = [h["nombre"] for h in sitios if not h.get("estable")]
            if inestable:
                return self._error("Todavía no son estables: " + ", ".join(inestable) + ". Espera un momento.")
            datos = dict(volver=False, q=np.asarray(q, float), pelota=pelota,
                         T=[np.asarray(h["T_mundo"], float) for h in sitios],
                         nombres=[h["nombre"] for h in sitios],
                         # lo que NO cuenta como obstáculo: la pelota justo ella (+3 cm: vóxeles y ruido); de los
                         # marcadores, su hoja y su soporte (AUTONOMO_TAREA_EXCLUIR_RADIO)
                         excluir=[(np.asarray(h["T_mundo"], float)[:3, 3],
                                   float(h.get("diametro_m") or 0.07) / 2 + 0.03 if h.get("tipo") == "pelota"
                                   else self.excluir_r) for h in sitios],
                         diametro=float(herr.get("diametro_m") or 0.07) if pelota else None)
        datos["mapa"] = ctx.get("mapa") if self.usar_mapa else None
        self.estado, self.texto = "planificando", "Planificando…"
        self.plan, self.ejec, self._resultado = None, None, None
        self._auto = volver
        self._gen += 1
        datos["gen"] = self._gen
        self._hilo = threading.Thread(target=self._planificar, args=(datos,), daemon=True, name="Tarea-plan")
        self._hilo.start()

    def _planificar(self, d):
        t0 = time.monotonic()
        try:
            if self._planificador is None:
                self._planificador = (self._fabrica_plan() if self._fabrica_plan is not None
                                      else self._crear_planificador())
            pl = self._planificador
            res, centros = d["mapa"] if d.get("mapa") else (0.02, None)
            n_vox = pl.poner_mapa(centros, res, d.get("excluir", ()))
            if d["volver"]:
                plan = pl.plan_volver(d["q"], d["q_inicio"])
            elif d.get("pelota"):
                plan = pl.planificar_pelota(d["q"], d["T"][0], d["diametro"], d["T"][1])
            else:
                plan = pl.planificar(d["q"], *d["T"])
            res = ("ok", plan, time.monotonic() - t0, n_vox)
        except ErrorPlan as e:
            res = ("error", str(e) + self._guardar_fallo(d, str(e)), time.monotonic() - t0, 0)
        except Exception as e:                          # fallo de programa: que se vea
            import traceback
            traceback.print_exc()
            res = ("error", f"fallo al planificar: {e}" + self._guardar_fallo(d, repr(e)), time.monotonic() - t0, 0)
        if d["gen"] == self._gen:                       # si entretanto se canceló o se pidió otro, no vale
            self._resultado = res

    def _guardar_fallo(self, d, error):
        """Guarda lo que se usó para planificar (pose, sitios, mapa) en diagnosticos/, para poder
        reproducir el fallo fuera del robot. Devuelve el texto a añadir al mensaje."""
        if not _g(self.cfg, "AUTONOMO_TAREA_GUARDAR_FALLOS", True):
            return ""
        try:
            import json
            import os
            carpeta = os.path.join(self.raiz, "diagnosticos")
            os.makedirs(carpeta, exist_ok=True)
            ruta = os.path.join(carpeta, time.strftime("plan_fallido_%Y%m%d_%H%M%S.npz"))
            res, centros = d["mapa"] if d.get("mapa") else (0.02, None)
            np.savez_compressed(
                ruta, q=d["q"], T=np.array(d.get("T", []), float).reshape(-1, 4, 4),
                q_inicio=d.get("q_inicio", np.full(6, np.nan)),
                mapa=np.zeros((0, 3)) if centros is None else np.asarray(centros, float), res=float(res),
                excluir=np.array([[*np.asarray(c, float), r] for c, r in d.get("excluir", ())]).reshape(-1, 4),
                info=json.dumps(dict(error=error, nombres=d.get("nombres"), pelota=bool(d.get("pelota")),
                                     diametro=d.get("diametro"), volver=bool(d.get("volver")))))
            print(f"💾 Tarea: datos del plan fallido en {ruta}")
            return f" (guardado en diagnosticos/{os.path.basename(ruta)})"
        except Exception as e:
            print(f"⚠️  Tarea: no se pudo guardar el plan fallido ({e})")
            return ""

    def _crear_planificador(self):
        import os
        from skill_shield.teleop_fr5.cinematica import cargar_cadena
        T = None
        ruta = self.cfg.get("CALIB_ROBOT_ARCHIVO", "camara_robot.json")
        ruta = ruta if os.path.isabs(ruta) else os.path.join(self.raiz, ruta)
        if os.path.exists(ruta):
            from skill_shield.autonomo.calibracion import cargar
            T = cargar(ruta)["T_mundo_cam"]
        return Planificador(cargar_cadena(self.cfg.get("RUTA_XML"), verbose=False), self.cfg, T)

    def _ejecutar(self, q_real):
        if self.estado == "pausada":
            return                                      # ya está en marcha: solo falta el pedal
        if self.estado != "lista" or self.plan is None:
            return self._error("No hay plan listo: pulsa «Planificar».") if self.estado != "planificando" else None
        if q_real is None:
            return self._error("No hay pose real del FR5.")
        desv = float(np.max(np.abs(np.asarray(q_real) - self.plan.q0)))
        if desv > math.radians(3.0):
            return self._error(f"El robot se ha movido {math.degrees(desv):.0f}° desde que se planificó: "
                               "vuelve a planificar.")
        mano0 = None
        fuente = self.mano_activa if self.mano_activa is not None and self.mano_activa.lista else self.monitor
        if fuente is not None and hasattr(fuente, "postura"):
            p = fuente.postura()
            if p:
                from orca_hardware import OrcaHardwareWrapper
                mano0 = {OrcaHardwareWrapper.JOINT_A_CTRL[j]: v for j, v in p.items()
                         if j in OrcaHardwareWrapper.JOINT_A_CTRL}
        self._comprobada = None
        self.ejec = Ejecutor(self.plan, self.posturas, self.vel, self.acel, mano0,
                             (float(_g(self.cfg, "AUTONOMO_TAREA_T_ABRIR_S", 1.0)),
                              float(_g(self.cfg, "AUTONOMO_TAREA_T_CERRAR_S", 1.5))))
        self.estado, self.texto = "pausada", "Pisa el pedal para empezar."
        self._t_desvio = 0.0

    def _cancelar(self, texto):
        self._parar_fr5()
        self.ejec = None
        self.plan = None
        self._gen += 1                                  # un plan que esté calculándose ya no vale
        self._resultado = None
        self._auto = False
        self.estado, self.texto = "libre", texto or "Elige la herramienta y pulsa «Planificar»."

    def _error(self, texto):
        self._parar_fr5()
        self.ejec = None
        self.estado, self.texto = "error", texto
        print(f"⚠️  Tarea: {texto}")

    def _parar_fr5(self):
        if self._enviando and self.fr5 is not None:
            self.fr5.habilitar(False)
        self._enviando = False

    # ── cada vuelta del bucle ──
    def paso(self, dt, q_real, pedal, contexto=None):
        """Devuelve la q que debe enseñar el gemelo en simulacro (o None: la pose real).
        contexto: como en orden() (para afinar la pelota al llegar encima de ella)."""
        if self.estado == "planificando" and self._resultado is not None:
            r, x, seg, n_vox = self._resultado
            self._resultado = None
            if r == "ok":
                self.plan = x
                if x.nombre != "volver al inicio":
                    self._q_inicio = x.q0.copy()
                dur = Ejecutor(x, self.posturas, self.vel, self.acel).duracion
                self.estado = "lista"
                inc = [i for i in (getattr(self._planificador, "inclinaciones", None) or []) if i]
                self.texto = (f"Plan listo: {len(x.fases)} tramos, unos {dur:.0f} s "
                              f"(calculado en {seg:.1f} s, {n_vox} cubos del mapa)"
                              + (f"; llega de lado, hasta {max(inc):.0f}° de la perpendicular" if inc else "")
                              + (f"; rodea obstáculos en {nr} tramo{'s' if nr > 1 else ''}"
                                 if (nr := getattr(self._planificador, "rodeos", 0)) else "")
                              + (f"; la coge {self._planificador.modo_pelota}"
                                 if getattr(self._planificador, "modo_pelota", None) and x.pelota is not None else "")
                              + ("; el punto de dejada está de pie: la soltará delante del marcador"
                                 if x.nombre.endswith("pelota") and getattr(self._planificador, "dejar_de_pie", False)
                                 else "")
                              + ". Pulsa «Ejecutar».")
                print(f"🗺️  Tarea: {self.texto}")
                if self._auto:
                    self._auto = False
                    self._ejecutar(q_real)
            else:
                self._error(x)
        if self.estado not in ("ejecutando", "pausada") or self.ejec is None:
            return None
        fr5 = self.fr5
        if fr5 is None or (not self.simulacro and not fr5.conectado):
            self._error("El FR5 no está conectado o tiene un error: tarea abortada.")
            return None
        if fr5.en_error:
            self._error("La controladora del FR5 dio error (¿colisión?): tarea abortada. "
                        "Reinicia desde la web y vuelve a planificar.")
            return None
        if pedal:
            if self.mover_mano and not self._mano_lista():
                self.estado, self.texto = "ejecutando", "Conectando la mano con par…"
                return self._q_gemelo()
            if not self._enviando:
                fr5.publicar(self.ejec.q())
                fr5.habilitar(True)                     # ServoMoveStart desde la pose real
                self._enviando = True
            from skill_shield.hardware.fr5_servo import ENVIANDO
            listo = self.simulacro or fr5.estado()["estado"] == ENVIANDO
            f_act = self.ejec.fase
            if listo and f_act is not None and f_act.comprobar and self._comprobada != self.ejec.i:
                self._comprobada = self.ejec.i
                fallo = self._comprobar(f_act.comprobar)
                if fallo:
                    self._error(fallo)
                    return None
            if listo:
                i0 = self.ejec.i
                self.ejec.avanzar(dt)
                if (self.ejec.i > i0 and self.plan.fases[i0].comprobar == "afinar"):
                    fallo = self._afinar(contexto)
                    if fallo:
                        self._error(fallo)
                        return None
                if not self.simulacro and q_real is not None:
                    if float(np.max(np.abs(np.asarray(q_real) - self.ejec.q()))) > self.desvio_max:
                        self._t_desvio += dt
                        if self._t_desvio > 0.5:
                            self._error(f"El robot no sigue lo mandado (más de "
                                        f"{math.degrees(self.desvio_max):.0f}°): tarea abortada.")
                            return None
                    else:
                        self._t_desvio = 0.0
            fr5.publicar(self.ejec.q())
            self.estado = "ejecutando"
            f = self.ejec.fase
            self.texto = (f.nombre if f is not None else "Terminando…") if listo else "Arrancando el FR5…"
        else:
            if self._enviando:
                fr5.habilitar(False)                    # ServoMoveEnd: se para donde está
                self._enviando = False
            self.estado = "pausada"
            f = self.ejec.fase
            self.texto = "Pedal suelto: en pausa" + (f" · {f.nombre}" if f is not None else "")
        if self.mano_activa is not None and self.mano_activa.lista:
            self.mano_activa.enviar(self.ejec.mano())
        if self.ejec.terminado:
            self._parar_fr5()
            nombre = self.plan.nombre
            self.estado, self.texto = "terminada", f"Hecho: {nombre}."
            print(f"✅ Tarea: {self.texto}")
            q = self.ejec.q()
            self.ejec = None
            return q if self.simulacro else None
        return self._q_gemelo()

    def _comprobar(self, que):
        """None si todo bien; texto del fallo si no. Sin táctiles (o en simulacro) no se para."""
        if que != "agarre" or self.simulacro or self.tactil is None:
            return None
        try:
            e = self.tactil() or {}
        except Exception:
            e = {}
        if e.get("estado") != "ok" or not e.get("dedos"):
            print("⚠️  Tarea: sin táctiles, no se comprueba si se ha cogido la pelota.")
            return None
        n = sum(1 for d in e["dedos"] if (d.get("fz") or 0.0) >= self.tactil_n)
        if n < self.tactil_dedos:
            return (f"No noto la pelota en los dedos (solo {n} con más de {self.tactil_n:g} N): tarea parada. "
                    "La mano se queda como está.")
        print(f"✋ Tarea: pelota cogida ({n} dedos la notan).")
        return None

    def _afinar(self, contexto):
        """(12-nov) Encima de la pelota y con el brazo quieto: la cámara ha corregido la calibración
        con el brazo (que ahora está justo al lado de la pelota) y ha vuelto a medir la pelota. Si
        está en otro sitio que al planificar, se rehace la bajada hacia donde está de verdad.
        None si todo bien (o si no se puede medir: se sigue con el plan); texto si hay que parar."""
        pz = getattr(self.plan, "pelota", None)
        if pz is None or self._planificador is None:
            return None
        try:
            ctx = contexto() if callable(contexto) else (contexto or {})
        except Exception:
            ctx = {}
        pelotas = [h for h in (ctx.get("herramientas") or []) if h.get("tipo") == "pelota" and h.get("T_mundo")
                   is not None and (h.get("edad_s") or 0.0) <= 0.5]
        if not pelotas:
            print("⚠️  Tarea: al afinar no veo la pelota (¿la tapa la mano?): sigo con el plan.")
            return None
        h = next((h for h in pelotas if h.get("id") == self._pelota_id), None)
        if h is None:                                   # perdió el seguimiento: la más cercana
            h = min(pelotas, key=lambda h: np.linalg.norm(np.asarray(h["T_mundo"])[:3, 3] - pz["c"]))
        c2 = np.asarray(h["T_mundo"], float)[:3, 3]
        n = float(np.linalg.norm(c2 - pz["c"]))
        if n < self.afinar_min:
            print(f"🎯 Tarea: afinado: la pelota está donde se planificó ({n * 100:.1f} cm).")
            return None
        if n > self.afinar_max:
            return (f"La pelota está a {n * 100:.0f} cm de donde estaba al planificar (¿se ha movido?): "
                    "tarea parada. Vuelve a planificar.")
        try:
            qs = self._planificador.rehacer_bajada(self.plan, c2)
        except ErrorPlan as e:
            return f"Al afinar (la pelota está {n * 100:.1f} cm desplazada) no puedo rehacer la bajada: {e}"
        self.ejec.cambiar_tramo(pz["i_bajar"], qs)
        self.ejec.cambiar_tramo(pz["i_subir"], qs[::-1])
        pz["c"] = c2
        self.afinado_cm = round(n * 100, 1)
        print(f"🎯 Tarea: afinado con la cámara: la pelota estaba {n * 100:.1f} cm desplazada → bajada corregida.")
        return None

    def _q_gemelo(self):
        return self.ejec.q() if (self.simulacro and self.ejec is not None) else None

    def _mano_lista(self):
        if self.mano_activa is not None and self.mano_activa.lista:
            return True
        if self.mano_activa is not None and self.mano_activa.error:
            self._error(self.mano_activa.error)
            return False
        if self._hilo_mano is None:
            if self.monitor is not None:
                self.monitor.cerrar()                   # suelta el puerto de la mano
            puerto = getattr(self.monitor, "puerto", None) or self.cfg.get("PUERTO_HARDWARE", "/dev/ttyUSB0")
            self.mano_activa = (self._fabrica_mano() if self._fabrica_mano is not None
                                else ManoActiva(puerto))
            print("🖐️  Tarea: conectando la OrcaHand con par (irá a su postura neutra)…")
            self._hilo_mano = threading.Thread(target=self.mano_activa.conectar, daemon=True, name="Tarea-mano")
            self._hilo_mano.start()
        return False

    # ── para la web ──
    def estado_web(self):
        e = self.ejec
        f = e.fase if e is not None else None
        return dict(estado=self.estado, texto=self.texto, simulacro=self.simulacro,
                    fase=f.nombre if f is not None else None,
                    i=(e.i + 1) if e is not None and not e.terminado else None,
                    n=len(self.plan.fases) if self.plan is not None else None,
                    progreso=round(e.progreso(), 3) if e is not None else (1.0 if self.estado == "terminada" else 0.0),
                    vel_deg_s=round(math.degrees(self.vel), 1),
                    puede=dict(planificar=self._puede_planificar(),
                               ejecutar=self.estado == "lista",
                               cancelar=self.estado in ("planificando", "lista", "ejecutando", "pausada"),
                               volver=self._q_inicio is not None and self._puede_planificar()))

    @property
    def mueve(self):
        return self.estado == "ejecutando" and self._enviando

    def cerrar(self):
        self._parar_fr5()
        if self.mano_activa is not None:
            self.mano_activa.cerrar()
