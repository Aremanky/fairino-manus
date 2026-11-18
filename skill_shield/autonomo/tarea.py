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
from collections import Counter, deque

import numpy as np

from skill_shield.teleop_fr5.cinematica import OFFSET_PALMA_ORCA, T_de
# Ejecutor (tiempos de cada tramo, cierre con los táctiles) y ManoActiva: en ejecucion.py (18-nov).
# Se importan aquí también para que el código de fuera (agarre_asistido, calibración, pruebas) siga igual.
from skill_shield.autonomo.ejecucion import (DEDOS_CTRL, Ejecutor, ManoActiva, _minjerk, _trapecio,  # noqa: F401
                                             tiempos_crucero)
from skill_shield.autonomo import mundo_tarea as MT

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


def zona_excluida(h, T_mundo_cam, g):
    """Lo que NO cuenta como obstáculo para el sitio h (herramienta, pelota o marcador de la lista de la cámara):
    [(centro, radio)]. Marcadores: sus esquinas (AUTONOMO_TAREA_EXCLUIR_RADIO). La pelota, donde la ve la cámara
    (sin la corrección: es lo que hay en el mapa): (18-nov, tarde) TODOS sus cubos, que caen en una cápsula a lo
    largo del rayo de la cámara (pelotas.zona_propia, la misma que se quita del mapa), como esferas seguidas. Antes,
    r + 1 cm alrededor del centro: con el centro desviado por el estéreo, su cara salía como obstáculo («lo que
    estorba, a unos 3 cm de la pelota, de unos 8 cm de alto») y no había forma de bajar a cogerla."""
    from skill_shield.autonomo.pelotas import esferas_capsula, params_zona, zona_propia
    c = np.asarray(h["T_mundo"], float)[:3, 3]
    if h.get("tipo") != "pelota":
        return [(c, float(_g(g, "AUTONOMO_TAREA_EXCLUIR_RADIO", 0.09)))]
    r = float(h.get("diametro_m") or 0.07) / 2
    a, b, radio = zona_propia(c, r, T_mundo_cam, margen=float(_g(g, "AUTONOMO_TAREA_EXCLUIR_PELOTA_MARGEN", 0.0)),
                              **params_zona(g))
    return esferas_capsula(a, b, radio)


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
    if "bajaría de la mesa" in m or "la mano tocaría la mesa" in m:
        return "la mano bajaría más que la pelota (se metería en la mesa)"
    if "tocaría la mesa" in m:
        return ("la pelota que lleva se metería en la mesa" if m.startswith("la pelota")
                else "el brazo tocaría la mesa")
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
        # (18-nov) huecos de la mano FIJOS para la tarea (choque_mano.desde_config_tarea); con
        # AUTONOMO_TAREA_HUECOS_DESDE_INICIO = True, las reglas de la calibración (hueco que no se cierra
        # respecto a la postura de inicio), como antes
        self.huecos_desde_inicio = bool(_g(g, "AUTONOMO_TAREA_HUECOS_DESDE_INICIO", False))
        if choque is None:
            from skill_shield.autonomo.choque_mano import desde_config, desde_config_tarea
            choque = (desde_config(cadena, g, entorno=self.modelo.entorno) if self.huecos_desde_inicio
                      else desde_config_tarea(cadena, g, entorno=self.modelo.entorno))
        self.choque = choque
        p = dict(PARAMETROS_POR_DEFECTO)
        p.update({k: g[k] for k in p if g.get(k) is not None})
        lim = limites_envio_rad(p, g.get("FR5_LIMITES_BLANDOS_DEG"))
        m = math.radians(float(_g(g, "AUTONOMO_TAREA_MARGEN_LIMITES_DEG", 2.0)))
        self.lo = np.maximum(lim[:, 0], cadena.limites[:, 0]) + m
        self.hi = np.minimum(lim[:, 1], cadena.limites[:, 1]) - m
        self.paso = math.radians(float(_g(g, "AUTONOMO_TAREA_PASO_DEG", 2.0)))
        self.d_agarre = float(_g(g, "AUTONOMO_TAREA_DIST_AGARRE", 0.10))
        self.d_previa = float(_g(g, "AUTONOMO_TAREA_DIST_PREVIA", 0.14))   # = config.py
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
        self.n_opciones_gpu = int(_g(g, "AUTONOMO_TAREA_OPCIONES_GPU", 8))   # (18-nov) con cuRobo (goalset), más
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
        # (17-nov) donde está apoyada la pelota, el suelo cubre TODA la zona (no solo donde la cámara vio la
        # mesa: la tapa la propia mano, la pelota o el halo) y vale también para el BRAZO, hasta esta profundidad
        self.grosor_suelo_zona = float(_g(g, "AUTONOMO_TAREA_PELOTA_GROSOR_SUELO_ZONA", 0.30))
        # (17-nov) TODA la mano (el pulgar también) a esta distancia como mínimo por encima de la mesa: la
        # altura de la pelota, la calibración y la mano real frente a su malla tienen 1-2 cm de error, y con
        # 5 mm el pulgar de la pinza (que coge la pelota por debajo de su centro) tocaba la mesa
        self.holgura_mesa_mano = float(_g(g, "AUTONOMO_TAREA_HOLGURA_MESA_MANO", 0.02))
        # (17-nov) comprobación CONTINUA entre dos posturas seguidas de un tramo (mano, lo que lleva y brazo):
        # si lo que se mueve un punto puede ser más que su holgura al mapa, se mira también entre medias
        self.continuo = bool(_g(g, "AUTONOMO_TAREA_CONTINUO", True))
        self.subdiv_max = int(_g(g, "AUTONOMO_TAREA_SUBDIV_MAX", 4))          # 2° / 2⁴ = 0,125° como mucho
        # (17-nov) lo que la cámara NO ha visto cuenta como coste: de dos caminos válidos, el que menos pasa
        # por ahí (no se prohíbe: no sabemos qué hay, pero mejor no meter la mano)
        self.peso_desconocido = float(_g(g, "AUTONOMO_TAREA_PESO_DESCONOCIDO", 2.0))
        self.alternativas = int(_g(g, "AUTONOMO_TAREA_ALTERNATIVAS", 1))      # planes completos que se comparan (1 = el primero)
        self.alternativas_gpu = int(_g(g, "AUTONOMO_TAREA_ALTERNATIVAS_GPU", 4))  # lo mismo, con cuRobo
        self._visto = None
        # (17-nov) comprobación del mapa EN LOTE con un mapa de distancias (GPU si hay; comprobador_lote.py)
        from skill_shield.autonomo.comprobador_lote import crear_motor
        self.comprobar_lote = bool(_g(g, "AUTONOMO_TAREA_COMPROBAR_LOTE", True))
        self._lote_torch, self._lote_dev = crear_motor(str(_g(g, "AUTONOMO_TAREA_COMPROBAR_MOTOR", "auto")))
        self._md, self._md_arbol, self._libre_cache, self._libre_ctx = None, None, {}, None
        self.lote_aciertos = self.lote_dudas = 0
        self.lote_tam = max(1, int(_g(g, "AUTONOMO_TAREA_COMPROBAR_LOTE_TAM", 24)))
        # (17-nov) cuRobo (GPU): propone los tramos articulares; los de aquí los verifican.
        # (18-nov) se crea al FINAL de __init__ (necesita la pinza para meter la pelota cogida en sus esferas)
        self.curobo, self.tramos_curobo, self.curobo_rechazos = None, 0, 0
        # (18-nov) fallos de la GPU: ya no la apagan para toda la sesión; se avisa y se reintenta en el siguiente plan
        self._curobo_pausado, self.curobo_fallos = None, 0
        self._lote_pausado, self.lote_fallos = None, 0
        self.gpu_fallos_max = int(_g(g, "AUTONOMO_TAREA_GPU_FALLOS_MAX", 3))
        self.avisos = []                          # avisos del último plan (se enseñan en la web)
        self.rrt_con_curobo = bool(_g(g, "AUTONOMO_TAREA_RRT_CON_CUROBO", False))
        self._discos = []                         # [(centro xy, radio, z del suelo, grosor, envolvente, r pelota)]
        self._mesa = None                         # (17-nov) (z, vértices xy) de la mesa medida con un plano
        self._superficies = []                    # (17-nov) mesas del mapa [(z, vértices xy)]
        self._zonas_actuales = ()
        self.zona_extra = float(_g(g, "AUTONOMO_TAREA_PELOTA_ZONA_MESA_EXTRA", 0.15))   # dónde se busca la mesa
        # (18-nov, tarde) quitado el relleno de columnas hasta el suelo bajo cada cubo (AUTONOMO_TAREA_MAPA_MACIZO,
        # ya apagado desde el 17-nov): los cubos del mapa son solo lo que mide la cámara. Las MESAS detectadas como
        # plano siguen siendo losas (AUTONOMO_TAREA_MESA_HASTA_SUELO: hasta dónde llega su fondo)
        self.mesa_hasta_suelo = bool(_g(g, "AUTONOMO_TAREA_MESA_HASTA_SUELO", True))
        self.grupo_min = int(_g(g, "AUTONOMO_TAREA_MAPA_GRUPO_MIN", 12))   # grupos de cubos más pequeños: ruido
        ent = getattr(self.modelo, "entorno", None)
        self.z_suelo_mapa = getattr(ent, "z_suelo", None)
        if self.z_suelo_mapa is None:
            self.z_suelo_mapa = g.get("SOPORTE_SUELO_Z")
        # (17-nov) ir de un sitio a otro POR ARRIBA: subir en vertical hasta pasar por encima de lo que
        # haya en el camino, trasladarse a esa altura y bajar en vertical (como se hace en pick-and-place)
        self.por_alto = bool(_g(g, "AUTONOMO_TAREA_POR_ALTO", True))
        self.alto_margen = float(_g(g, "AUTONOMO_TAREA_ALTO_MARGEN", 0.10))    # m sobre lo más alto del pasillo
        self.alto_max = float(_g(g, "AUTONOMO_TAREA_ALTO_MAX", 0.35))          # m que puede subir sobre el sitio más alto
        self.alto_pasillo = float(_g(g, "AUTONOMO_TAREA_ALTO_PASILLO", 0.15))  # m de ancho extra del pasillo
        self.tramos_por_alto = 0
        self.apoyo_tol = float(_g(g, "AUTONOMO_TAREA_PELOTA_APOYO_TOL", 0.04))  # m que puede corregir la mesa medida
        # (17-nov) si el brazo le tapa a la cámara la zona del objetivo, primero se aparta para mirar
        self.mirar_radio = float(_g(g, "AUTONOMO_TAREA_MIRAR_RADIO", 0.15))    # m alrededor del objetivo que hay que ver
        self.mirar_s = float(_g(g, "AUTONOMO_TAREA_MIRAR_S", 2.0))             # s quieto mirando antes de replanificar
        pm = _g(g, "AUTONOMO_TAREA_POSTURA_MIRAR_DEG", None)                  # postura fija para mirar (o None: busca)
        self.postura_mirar = None if pm is None else np.radians(np.asarray(pm, float))
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
        self.pel_modos = tuple(_g(g, "AUTONOMO_TAREA_PELOTA_MODOS", ("pinza",)))
        # (17-nov) la pinza, desde ARRIBA y con pulgar e índice a los lados de la pelota (no por debajo)
        self.pinza_arriba = bool(_g(g, "AUTONOMO_TAREA_PINZA_DESDE_ARRIBA", True))
        self.pinza_inc_max = float(_g(g, "AUTONOMO_TAREA_PINZA_INCLINACION_MAX_DEG", 20.0))
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
        # (17-nov) escena FIJA (escena_fija.py): mesas, cajonera… escaneadas una vez y guardadas como cajas.
        # Su superficie entra como cubos limpios (el mapa en vivo ya no los pone) y sus tapas como «suelo»
        # para mano y brazo; siguen ahí aunque la mano o el brazo las tapen a la cámara
        from skill_shield.autonomo.escena_fija import EscenaFija
        self.escena = EscenaFija(g)
        self._fijo_version, self._fijo_pts, self._tapas_fijas = None, np.zeros((0, 3)), []
        self._superficies_vivas = []
        self.apoyo_fijo_tol = float(_g(g, "AUTONOMO_ESCENA_FIJA_APOYO_TOL", 0.05))
        self._z_fijo_usado = None
        # (18-nov) MUNDO SIMPLE (mundo_tarea.py): mesas = losas (escena fija, planos del mapa, mesa medida),
        # el resto = cubos del mapa sin la capa de la mesa, y tres márgenes. "completo" = como antes.
        self.mundo_simple = str(_g(g, "AUTONOMO_TAREA_MUNDO", "simple")).lower() == "simple"
        self._losas = []
        self.holgura_mesa_brazo = float(_g(g, "AUTONOMO_TAREA_HOLGURA_MESA_BRAZO", 0.02))
        self.quitados_mesa = 0
        if self.mundo_simple:
            mb = float(_g(g, "AUTONOMO_TAREA_MARGEN_BRAZO", 0.02))
            self.margen_mapa_brazo = self.margen_mapa_malla = mb
        # (18-nov) la pelota se coloca deslizándola por el RAYO de la cámara hasta apoyarla en la mesa
        self.pelota_por_rayo = bool(_g(g, "AUTONOMO_TAREA_PELOTA_POR_RAYO", True))
        self._T_cam_medida = None                  # cámara→mundo con la que se midió la pelota (o T_cam)
        # (18-nov) al elegir la postura de encima de la pelota, que el brazo NO le tape la pelota a la cámara
        # (ahí se para a afinar): coste por la parte tapada y, por encima de TAPADA_MAX, al final de la lista
        self.peso_tapada = float(_g(g, "AUTONOMO_TAREA_PESO_TAPADA", 2.0))
        self.tapada_max = float(_g(g, "AUTONOMO_TAREA_TAPADA_MAX_PLAN", 0.25))
        self._mirar_obj = None                     # (centro, radio) del objeto que hay que ver al afinar
        # (18-nov) pinza: si la punta de los dedos baja por debajo del punto de pinza más de lo que deja la
        # holgura con la mesa, se coge la pelota algo por encima de su centro (hasta esta fracción del radio)
        self.pinza_elevar_max = float(_g(g, "AUTONOMO_TAREA_PINZA_ELEVAR_MAX_FRAC", 0.35))
        self.lote_ik = max(16, int(_g(g, "AUTONOMO_TAREA_LOTE_IK", 512)))
        # (18-nov) tramos rectos a velocidad CARTESIANA de la palma (m/s), más despacio en la llegada
        self.vel_cart = float(_g(g, "AUTONOMO_TAREA_VEL_CART_M_S", 0.05))
        self.lento_m = float(_g(g, "AUTONOMO_TAREA_LLEGADA_LENTA_M", 0.03))
        self.vel_lenta = float(_g(g, "AUTONOMO_TAREA_VEL_LLEGADA_M_S", 0.02))
        # (18-nov) cierre de la mano con los TÁCTILES (cada dedo hasta que toca) y bajada vigilada
        self.cierre_tactil = bool(_g(g, "AUTONOMO_TAREA_CIERRE_TACTIL", True))
        self.cierre_n = float(_g(g, "AUTONOMO_TAREA_CIERRE_TACTIL_N", 0.2))
        self.cierre_s = float(_g(g, "AUTONOMO_TAREA_CIERRE_TACTIL_S", 2.5))
        self._cargar_fijo()
        if str(_g(g, "AUTONOMO_TAREA_PLANIFICADOR", "propio")).lower() == "curobo":
            self.curobo = self._crear_curobo(g)

    def _cargar_fijo(self):
        """(17-nov) Recarga la escena fija si ha cambiado (otro escaneo, o el archivo editado a mano)."""
        self.escena.actualizar()
        if self._fijo_version == (self.escena.version, self._res):
            return
        self._fijo_version = (self.escena.version, self._res)
        from skill_shield.autonomo.escena_fija import superficie
        # (18-nov) mundo simple: las cajas fijas son LOSAS (sus tapas, macizas hacia abajo), no cubos
        P = np.zeros((0, 3)) if self.mundo_simple else superficie(self.escena.cajas, self._res)
        V = np.asarray(self.escena.voxeles, float).reshape(-1, 3)
        self._fijo_pts = np.vstack([P, V]) if len(V) else P
        self._tapas_fijas = [(float(z), np.asarray(p, float)) for z, p in self.escena.tapas()]

    def _apoyo_fijo(self, p, r):
        """(17-nov) Altura de la tapa FIJA sobre la que está apoyada una pelota (centro p, radio r), o None."""
        from skill_shield.autonomo.tsdf import dentro_poligono
        p = np.asarray(p, float)
        mejor = None
        for z, poli in self._tapas_fijas:
            dz = z + r - p[2]
            if abs(dz) <= self.apoyo_fijo_tol and dentro_poligono(poli, p[None, :2])[0]:
                if mejor is None or abs(dz) < abs(mejor + r - p[2]):
                    mejor = z
        return mejor

    # ── mapa ──
    def poner_mapa(self, centros, res, excluir=()):
        """centros: vóxeles ocupados (N x 3). excluir: [(centro, radio)] que no cuentan."""
        C = self._sin_grupos_sueltos(np.asarray(centros if centros is not None else [], float).reshape(-1, 3),
                                     float(res or 0.02))
        self._res = float(res or 0.02)
        self._cargar_fijo()                       # (17-nov) + la escena fija, con su forma limpia
        if len(self._fijo_pts):
            C = np.vstack([C, self._fijo_pts])
        self._superficies = list(self._superficies_vivas) + list(self._tapas_fijas)
        quitado = []
        for c, r in excluir:
            dentro = np.linalg.norm(C - np.asarray(c, float), axis=1) <= float(r)
            quitado.append(C[dentro])
            C = C[~dentro]
        self._C = C
        self._C_quitado = np.vstack(quitado) if quitado else np.zeros((0, 3))
        self._res = float(res or 0.02)
        self._mesa = None                         # (17-nov) la mesa medida (z, envolvente): la pone el plan
        self.zonas_apoyo(())
        return len(C)

    def _sin_grupos_sueltos(self, C, res):
        """(17-nov) Quita los grupos de cubos (vecinos en 26 direcciones) de menos de grupo_min cubos: ruido
        del estéreo flotando. Lo que está apoyado en algo (en la mesa, en una caja) va unido a su grupo."""
        if self.grupo_min <= 1 or len(C) < 2:
            return C
        from scipy.ndimage import label
        idx = np.round((C - C.min(0)) / res).astype(int)
        g = np.zeros(tuple(idx.max(0) + 1), bool)
        g[tuple(idx.T)] = True
        etq, _ = label(g, structure=np.ones((3, 3, 3), bool))
        e = etq[tuple(idx.T)]
        tam = np.bincount(e)
        return C[tam[e] >= self.grupo_min]

    def zonas_apoyo(self, zonas):
        """(18-nov) Con el mundo simple: mesas = losas y, de los cubos del mapa, fuera la capa de la mesa.
        Solo las zonas (pelota o dejada) que NO están apoyadas en ninguna losa conocida se tratan como
        antes (_zonas_apoyo_completo). Con AUTONOMO_TAREA_MUNDO = "completo", todo como antes."""
        if not self.mundo_simple:
            self._losas = []
            return self._zonas_apoyo_completo(zonas)
        losas = MT.losas_de(list(self._superficies) + ([self._mesa] if self._mesa is not None else []),
                            lambda z: z - self._grosor_mesa(z))
        sueltas = []
        for zona in zonas:
            c, rb = np.asarray(zona[0], float), float(zona[2])
            if MT.losa_bajo(losas, c - [0.0, 0.0, rb], self.apoyo_tol) is None:
                sueltas.append(zona)              # no se sabe en qué está apoyada: como antes
        self._zonas_apoyo_completo(sueltas, con_superficies=False)
        self._zonas_actuales = tuple(zonas)
        self._losas = losas
        self.quitados_mesa = 0
        if losas and self._arbol is not None:
            from scipy.spatial import cKDTree
            C = self._arbol.data
            # la capa: lo que la cámara ve de la propia mesa sale hasta 1 cubo + el ruido del estéreo por encima
            capa = (self._res + self.k_ruido * np.sum((C - self.T_cam[:3, 3]) ** 2, axis=1)
                    if self.T_cam is not None else 2 * self._res)
            C2, self.quitados_mesa = MT.quitar_capa_mesa(C, losas, capa, self._res)
            if self.quitados_mesa:
                self._arbol = cKDTree(C2) if len(C2) else None

    def _zonas_apoyo_completo(self, zonas, con_superficies=True):
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
        # (17-nov) zona «entera» (la pelota apoyada en algo, o la dejada tumbada): el suelo cubre todo el
        # disco y vale para mano Y brazo hasta grosor_suelo_zona. Si no (la dejada de pie, en una hoja: la
        # pelota se suelta en el aire), como antes: solo donde la cámara vio algo y solo para la mano.
        self._discos = []
        normales = []
        for zona in zonas:
            c, rh, rb = zona[:3]
            entera = bool(zona[3]) if len(zona) > 3 else True
            c = np.asarray(c, float)
            # hasta un vóxel por encima de la pelota (su parte de arriba cae en el vóxel de encima) +
            # el ruido de la profundidad a esa distancia (crece con la distancia al cuadrado)
            techo = c[2] + float(rb) + self._res + self.ruido_prof(c)
            fondo = c[2] - float(rb) - self.apoyo_grueso - self._res - self.ruido_prof(c)
            z_min = float(c[2] - rb + self.margen_suave)
            if entera:
                # la mesa es la ENVOLVENTE (en planta) de lo que la cámara ve a la altura de la pelota
                # alrededor de ella, buscando algo más lejos que la zona: así se tapa el hueco que deja la
                # mano (o lo olvidado) pero no se inventa mesa más allá de su borde
                lejos = float(rh) + self.zona_extra
                en_m = ((np.linalg.norm(C[:, :2] - c[:2], axis=1) < lejos) & (C[:, 2] < techo) & (C[:, 2] > fondo))
                en_q = (np.linalg.norm(Q[:, :2] - c[:2], axis=1) < lejos) & (Q[:, 2] > fondo) & (Q[:, 2] < techo)
                self._discos.append((c[:2].copy(), float(rh), z_min, self.grosor_suelo_zona,
                                     _envolvente(np.vstack([C[en_m, :2], Q[en_q, :2]]), self._res),
                                     float(rb) + self._res))
            else:
                normales.append((c, rh, rb))
            en = ((np.linalg.norm(C[:, :2] - c[:2], axis=1) < float(rh)) & (C[:, 2] < techo)
                  & (C[:, 2] > fondo))
            fuera |= en
            if entera:
                continue                                      # su suelo es el disco entero
            suelo_C[en] = np.minimum(suelo_C[en], z_min)      # si se solapan, el suelo más bajo
            enq = (np.linalg.norm(Q[:, :2] - c[:2], axis=1) < float(rh)) & (Q[:, 2] > fondo)
            suelo_Q[enq] = np.minimum(suelo_Q[enq], z_min)
        self._zonas_actuales = tuple(zonas)
        self._version_mundo = getattr(self, "_version_mundo", 0) + 1
        for z_m, poli in (self._superficies if con_superficies else ()):   # (17-nov) mesas del mapa
            env = _envolvente(poli, 0.0)
            if env is not None:
                self._discos.append((poli.mean(0), np.inf, float(z_m) + self.margen_suave,
                                     self._grosor_mesa(z_m), env, 0.0))
        if self._mesa is not None and con_superficies:
            # (17-nov) la mesa MEDIDA (plano de la profundidad alrededor del objeto, pelotas.py): es suelo en
            # toda su envolvente, para mano y brazo, aunque no haya cubos (tapada, olvidada, hueco del mapa)
            z_m, poli = self._mesa
            env = _envolvente(np.asarray(poli, float), 0.0)
            if env is not None:
                self._discos.append((np.asarray(poli, float).mean(0), np.inf, float(z_m) + self.margen_suave,
                                     self._grosor_mesa(z_m), env, 0.0))
        self._arbol = cKDTree(C[~fuera]) if (~fuera).any() else None
        # el suelo solo está donde la cámara vio algo (la mesa, o la pelota apoyada en ella):
        # donde no hay nada no se inventa una mesa
        # y siempre, justo debajo de cada pelota (lo que ocupa), lo que la sostiene: aunque la cámara no
        # lo vea (la tapa la propia pelota, o se quitó su halo del mapa), los dedos no pasan por debajo
        D = []
        for c, rh, rb in normales:
            c = np.asarray(c, float)
            g_ = np.arange(-float(rb), float(rb) + 1e-9, 0.5 * self._res)
            X, Y = np.meshgrid(g_, g_)
            k = X ** 2 + Y ** 2 <= float(rb) ** 2
            D.append(np.column_stack([X[k] + c[0], Y[k] + c[1], np.full(k.sum(), c[2] - rb + self.margen_suave)]))
        D = np.vstack(D) if D else np.zeros((0, 3))
        con_suelo = fuera & np.isfinite(suelo_C)
        S = np.vstack([C[con_suelo], Q[np.isfinite(suelo_Q)], np.column_stack([D[:, :2], np.zeros(len(D))])])
        z = np.r_[suelo_C[con_suelo], suelo_Q[np.isfinite(suelo_Q)], D[:, 2]]
        self._suelo = (cKDTree(S[:, :2]), z) if len(S) else None
        self._arbol_suave = None

    def _grosor_mesa(self, z_m):
        """Una mesa detectada como plano es maciza hasta el suelo (una cajonera, un mueble); si no, la capa
        de siempre."""
        if self.mesa_hasta_suelo and self.z_suelo_mapa is not None:
            return max(float(z_m) - float(self.z_suelo_mapa), self.grosor_suelo_zona)
        return self.grosor_suelo_zona

    # ── (17-nov) comprobación del mapa en lote ──
    def _mapa_dist(self):
        if not self.comprobar_lote or self._arbol is None or self._arbol_suave is not None:
            return None
        if self._md_arbol is not self._arbol:            # (se guarda el árbol mismo, no su id: los id se reusan)
            from skill_shield.autonomo.comprobador_lote import MapaDistancias
            self._md = MapaDistancias(self._arbol.data, self._res)
            self._md_arbol = self._arbol
        self._ctx_lote()
        return self._md

    def _ctx_lote(self):
        ctx = (self._arbol, self._mano_pts, self._carga, getattr(self.choque, "exigido", None))
        if self._libre_ctx is None or any(a is not b for a, b in zip(ctx, self._libre_ctx)):
            self._libre_cache, self._libre_ctx = {}, ctx  # otro mapa, otra mano, carga o inicio: se olvida

    def _choque_lote(self):
        return self.comprobar_lote and hasattr(self.choque, "cumple_lote")

    def _puntos_lote(self, Qs, marcos_lista=None):
        """(N, M, 3) puntos del robot (lo mismo que mira motivo() contra el mapa) y (M,) sus umbrales."""
        h = 0.5 * self._res
        grupos, U = [], []
        ms = marcos_lista if marcos_lista is not None else [self.c.fk_marcos(q) for q in Qs]
        R5 = np.stack([m[5][:3, :3] for m in ms])
        t5 = np.stack([m[5][:3, 3] for m in ms])
        for pts, u in ((self._mano_pts, self.margen_mapa + h), (self._carga, self.margen_mapa + h)):
            if pts is not None and len(pts):
                grupos.append(np.matmul(R5, pts.T).transpose(0, 2, 1) + t5[:, None, :])
                U.append(np.full(len(pts), u))
        for marco, pts, _ in self._brazo_pts:
            R = np.stack([m[marco][:3, :3] for m in ms])
            t = np.stack([m[marco][:3, 3] for m in ms])
            grupos.append(np.matmul(R, pts.T).transpose(0, 2, 1) + t[:, None, :])
            U.append(np.full(len(pts), self.margen_mapa_malla + h))
        caps = [self.modelo.capsulas(q, m) for q, m in zip(Qs, ms)]
        for k, c0 in enumerate(caps[0]):
            if c0.nombre == "base" or (c0.nombre == "mano" and self._mano_pts is not None) or \
                    (self._brazo_pts and c0.nombre in ("brazo", "antebrazo")):
                continue
            n = max(2, int(math.ceil(np.linalg.norm(c0.b - c0.a) / 0.02)) + 1)
            s_ = np.linspace(0.0, 1.0, n)
            A = np.stack([cs[k].a for cs in caps])
            B = np.stack([cs[k].b for cs in caps])
            grupos.append(A[:, None, :] + s_[None, :, None] * (B - A)[:, None, :])
            U.append(np.full(n, c0.r + self.margen_mapa_brazo + h))
        return np.concatenate(grupos, axis=1), np.concatenate(U)

    @staticmethod
    def _clave_libre(q):
        return np.asarray(q, float).tobytes()

    def prefiltrar(self, Qs):
        """Mira de una vez (mapa de distancias en la GPU si hay + árbol solo para lo que queda cerca) qué
        posturas de Qs chocan con el mapa y cuáles con la mano contra el brazo. Resultado EXACTO (el mismo
        que postura a postura): motivo() lo usa y solo repite la cuenta en las que chocan, para el texto."""
        Qs = np.asarray(Qs, float).reshape(-1, 6)
        md = self._mapa_dist()
        ch = self._choque_lote()
        if not len(Qs) or (md is None and not ch):
            return
        self._ctx_lote()
        from skill_shield.autonomo.comprobador_lote import exacto_en_lote
        ms = [self.c.fk_marcos(q) for q in Qs]
        def calcular(torch, dev):
            lm = lc = None
            if md is not None:
                P, U = self._puntos_lote(Qs, ms)
                lm = exacto_en_lote(md, self._arbol, P, U, torch, dev)
            if ch:
                lc = self.choque.cumple_lote(ms, torch=torch, dev=dev)
            return lm, lc
        try:
            lm, lc = calcular(self._lote_torch, self._lote_dev)
        except Exception as e:                           # la GPU nunca debe tumbar el plan
            if self._lote_torch is None:
                raise
            self.lote_fallos += 1
            self._lote_pausado = (self._lote_torch, self._lote_dev)
            self._lote_torch = self._lote_dev = None
            txt = f"la comprobación en la GPU falló ({e}): este plan va con numpy"
            print(f"⚠️  Tarea: {txt} (fallo {self.lote_fallos}/{self.gpu_fallos_max}).")
            self.avisos.append(txt)
            lm, lc = calcular(None, None)
        if len(self._libre_cache) > 200000:
            self._libre_cache = {}
        for k, q in enumerate(Qs):
            self._libre_cache[self._clave_libre(q)] = (None if lm is None else bool(lm[k]),
                                                       None if lc is None else bool(lc[k]))

    def motivos(self, qs):
        """motivo() de cada postura de qs, en orden, comprobándolas EN LOTES (de lote_tam): para en
        cuanto una choca, así no se gasta en las de detrás (un tramo que choca suele hacerlo pronto)."""
        qs = np.asarray(qs, float).reshape(-1, 6)
        for k0 in range(0, len(qs), self.lote_tam):
            trozo = qs[k0:k0 + self.lote_tam]
            self.prefiltrar(trozo)
            for q in trozo:
                yield q, self.motivo(q)

    def _de_cache(self, q, cual):
        r = self._libre_cache.get(self._clave_libre(q))
        return None if r is None else r[cual]

    def _contar(self, r):
        if r:
            self.lote_aciertos += 1
        else:
            self.lote_dudas += 1
        return r

    def _mapa_libre(self, q, marcos=None):
        md = self._mapa_dist()
        if md is None:
            return False
        r = self._de_cache(q, 0)
        if r is None:
            from skill_shield.autonomo.comprobador_lote import exacto_en_lote
            P, U = self._puntos_lote(np.asarray(q, float)[None], None if marcos is None else [marcos])
            r = bool(exacto_en_lote(md, self._arbol, P, U)[0])   # una sola: numpy (la GPU no compensa)
        return self._contar(r)

    def _choque_libre(self, q, marcos):
        if not self._choque_lote():
            return False
        self._ctx_lote()
        r = self._de_cache(q, 1)
        if r is None:                                    # no estaba en un lote: la cuenta exacta, rápida
            r = self.choque.ok_rapido(marcos)
        return self._contar(r)

    def _bajo_suelo(self, P, solo_discos=False):
        """¿Algún punto de P (mundo) por debajo del «suelo» de las zonas de apoyo?"""
        return bool(self._mascara_suelo(P, solo_discos).any())

    def _mascara_suelo(self, P, solo_discos=False):
        """Máscara de los puntos de P (mundo) que quedan por debajo del «suelo» (dentro de su capa)."""
        P = np.asarray(P, float)
        out = np.zeros(len(P), bool)
        if self._discos:
            fl, gr = np.full(len(P), np.inf), np.zeros(len(P))
            for cxy, rh, zs, g, env, rb in self._discos:
                d = np.linalg.norm(P[:, :2] - cxy, axis=1)
                en = (d < rh) & (zs < fl)                       # si se solapan, el más bajo
                if en.any():
                    sobre_mesa = d[en] < rb                     # justo debajo de la pelota, siempre
                    if env is not None:
                        sobre_mesa |= env.find_simplex(P[en, :2]) >= 0
                    en[np.nonzero(en)[0][~sobre_mesa]] = False
                fl[en], gr[en] = zs, g
            en = np.isfinite(fl)
            out[en] |= (P[en, 2] < fl[en]) & (P[en, 2] > fl[en] - gr[en])
        if self._suelo is not None and not solo_discos:
            arbol2d, z_suelo = self._suelo
            d, i = arbol2d.query(P[:, :2], distance_upper_bound=self._res)
            sobre = np.isfinite(d)
            # el «suelo» es una capa (la mesa o lo que sostiene la pelota), no infinito hacia
            # abajo: muy por debajo (al llevar la pelota a una dejada más baja) no cuenta
            if sobre.any():
                zs = z_suelo[i[sobre]]
                out[sobre] |= (P[sobre, 2] < zs) & (P[sobre, 2] > zs - self.grosor_suelo)
        return out

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
        m = None if self._choque_libre(q, marcos) else self.choque.motivo(q)
        if m:
            return m
        if self._losas:                                  # (18-nov) mundo simple: mesas = losas
            m = self._motivo_losas(q, marcos)
            if m:
                return m
        if self._suelo is not None or self._discos:
            Tb = marcos[5]
            for pts, que, tol in ((self._mano_pts, "la mano", -self.holgura_mesa_mano),
                                  (self._carga, "la pelota que lleva", 2 * self.margen_suave)):   # apoyada: sí toca
                if pts is None:
                    continue
                if self._bajo_suelo(pts @ Tb[:3, :3].T + Tb[:3, 3] + np.array([0.0, 0.0, tol])):
                    return f"{que} bajaría de la mesa (por debajo de la pelota)"
            if self._discos:
                # (17-nov) el brazo tampoco: antes solo se miraba la mano y el antebrazo podía meterse
                # en la cajonera dentro de la zona de la pelota (ahí los cubos de la mesa no cuentan)
                for marco, pts, nombre in self._brazo_pts:
                    T = marcos[marco]
                    if self._bajo_suelo(pts @ T[:3, :3].T + T[:3, 3], solo_discos=True):
                        return f"{nombre} se metería en la mesa (por debajo de la pelota)"
                for cap in self.modelo.capsulas(q, marcos):
                    if cap.nombre in ("base", "mano") or (self._brazo_pts and cap.nombre in ("brazo", "antebrazo")):
                        continue
                    n = max(2, int(math.ceil(np.linalg.norm(cap.b - cap.a) / 0.02)) + 1)
                    pts = cap.a + np.linspace(0.0, 1.0, n)[:, None] * (cap.b - cap.a) - np.array([0.0, 0.0, cap.r])
                    if self._bajo_suelo(pts, solo_discos=True):
                        return f"{NOMBRES.get(cap.nombre, cap.nombre)} se metería en la mesa (por debajo de la pelota)"
        if self._mapa_libre(q, marcos):                  # (17-nov) ya comprobado (en lote, exacto): no choca
            return None
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
                # (17-nov) con tope: solo importa si algo está más cerca que el umbral (~8 veces más rápido)
                d, i = arbol.query(pts @ Tb[:3, :3].T + Tb[:3, 3], distance_upper_bound=umbral)
                if float(np.min(d)) < umbral:
                    if arbol is self._arbol and len(self._choques) < 5000:
                        self._choques.append(arbol.data[i[int(np.argmin(d))]])
                    return f"{que} toca {donde}"
            if self._brazo_pts:
                # el brazo con los puntos de sus MALLAS (12-nov): la cápsula «brazo» va por la línea
                # entre joints, pero el tramo del brazo del FR5 va desplazado ~14 cm de ella
                for marco, pts, nombre in self._brazo_pts:
                    T = marcos[marco]
                    u = self.margen_mapa_malla + 0.5 * self._res
                    d, _ = arbol.query(pts @ T[:3, :3].T + T[:3, 3], distance_upper_bound=u)
                    if float(np.min(d)) < u:
                        return f"{nombre} toca {donde}"
            for cap in self.modelo.capsulas(q, marcos):
                if cap.nombre == "base" or (cap.nombre == "mano" and self._mano_pts is not None):
                    continue
                if self._brazo_pts and cap.nombre in ("brazo", "antebrazo"):
                    continue                              # ya mirados con sus mallas
                n = max(2, int(math.ceil(np.linalg.norm(cap.b - cap.a) / 0.02)) + 1)
                pts = cap.a + np.linspace(0.0, 1.0, n)[:, None] * (cap.b - cap.a)
                u = cap.r + m_brazo + 0.5 * self._res
                d, _ = arbol.query(pts, distance_upper_bound=u)
                if float(np.min(d)) < u:
                    return f"{NOMBRES.get(cap.nombre, cap.nombre)} toca {donde}"
        return None

    def _motivo_losas(self, q, marcos):
        """(18-nov) Mano, lo que lleva y brazo contra las mesas (losas), cada uno con su holgura."""
        Tb = marcos[5]
        for pts, que, h in ((self._mano_pts, "la mano", self.holgura_mesa_mano),
                            (self._carga, "la pelota que lleva", -2 * self.margen_suave)):   # apoyada: sí toca
            if pts is not None and len(pts) and \
                    MT.mascara_losas(self._losas, pts @ Tb[:3, :3].T + Tb[:3, 3], h).any():
                return f"{que} tocaría la mesa (holgura {100 * max(h, 0.0):.0f} cm)"
        for marco, pts, nombre in self._brazo_pts:
            T = marcos[marco]
            if MT.mascara_losas(self._losas, pts @ T[:3, :3].T + T[:3, 3], self.holgura_mesa_brazo).any():
                return f"{nombre} tocaría la mesa"
        for cap in self.modelo.capsulas(q, marcos):
            if cap.nombre in ("base", "mano") or (self._brazo_pts and cap.nombre in ("brazo", "antebrazo")):
                continue
            n = max(2, int(math.ceil(np.linalg.norm(cap.b - cap.a) / 0.02)) + 1)
            pts = cap.a + np.linspace(0.0, 1.0, n)[:, None] * (cap.b - cap.a)
            if MT.mascara_losas(self._losas, pts, cap.r + self.holgura_mesa_brazo).any():
                return f"{NOMBRES.get(cap.nombre, cap.nombre)} tocaría la mesa"
        return None

    def _mascara_losas(self, P, holgura=0.0):
        """Para cuRobo (mundo.py): puntos dentro de alguna mesa."""
        return MT.mascara_losas(self._losas, P, holgura)

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

    def ik_lote(self, poses, q_ref, n=2, razones=None):
        """(18-nov) ik() de muchas poses a la vez: primero TODAS las soluciones de la IK analítica, se
        descartan las que se salen de los límites (sin calcular nada más) y las demás se comprueban contra
        el mapa y la mano↔brazo EN LOTE (GPU si hay, comprobador_lote) antes de mirar una a una con motivo(),
        que ya encuentra el resultado hecho. Mismo resultado que llamar a ik() con cada pose."""
        sols = [sorted(self.c.ik_analitica(T, q_ref, incluir_fuera_de_limites=True),
                       key=lambda q: self._coste(q, q_ref)) for T in poses]
        planas = [np.asarray(q, float) for ss in sols for q in ss]
        if planas:
            Q = np.array(planas)
            dentro = np.all((Q >= self.lo) & (Q <= self.hi), axis=1)
            Q = Q[dentro]
            for k0 in range(0, len(Q), self.lote_ik):
                self.prefiltrar(Q[k0:k0 + self.lote_ik])
        out = []
        for ss in sols:
            val = []
            for q in ss:
                m = self.motivo(q)
                if m is None:
                    val.append(np.asarray(q, float))
                    if len(val) >= n:
                        break
                elif razones is not None:
                    razones[_tipo_razon(m)] += 1
            if not ss and razones is not None:
                razones["el brazo no llega"] += 1
            out.append(val)
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
        for k in range(1, n + 1):                        # primero la IK de toda la recta…
            p = p0 + (destino - p0) * (k / n)
            q = self._ik_cercana(T_de(R, p - R @ p_loc), qs[-1])
            if q is None:
                raise ErrorPlan(f"{nombre}: el robot no llega ({np.round(p, 3).tolist()} m)")
            if float(np.max(np.abs(q - qs[-1]))) > self.salto_max:
                raise ErrorPlan(f"{nombre}: no se puede seguir la recta sin un salto (el brazo se "
                                "estiraría del todo o la muñeca pasaría por una singularidad)")
            qs.append(np.asarray(q, float))
        for _, m in self.motivos(qs[1:]):                # …luego las comprobaciones, en lotes (GPU)
            if m:
                raise ErrorPlan(f"{nombre}: {m}")
        qs = np.array(qs)
        self.comprobar_continuo(qs, nombre)
        return qs

    def _puntos_holgura(self, q):
        """Puntos (mundo) de mano, lo que lleva y brazo, y la distancia mínima que cada uno tiene que
        guardar con los cubos del mapa (la misma que mira motivo())."""
        marcos = self.c.fk_marcos(np.asarray(q, float))
        Tb, h = marcos[5], 0.5 * self._res
        P, U = [], []
        for pts, u in ((self._mano_pts, self.margen_mapa + h), (self._carga, self.margen_mapa + h)):
            if pts is not None and len(pts):
                p = pts[::3] @ Tb[:3, :3].T + Tb[:3, 3]
                P.append(p); U.append(np.full(len(p), u))
        for marco, pts, _ in self._brazo_pts:
            p = pts[::2] @ marcos[marco][:3, :3].T + marcos[marco][:3, 3]
            P.append(p); U.append(np.full(len(p), self.margen_mapa_malla + h))
        for cap in self.modelo.capsulas(q, marcos):
            if cap.nombre == "base" or (cap.nombre == "mano" and self._mano_pts is not None) or \
                    (self._brazo_pts and cap.nombre in ("brazo", "antebrazo")):
                continue
            n = max(2, int(math.ceil(np.linalg.norm(cap.b - cap.a) / 0.03)) + 1)
            p = cap.a + np.linspace(0.0, 1.0, n)[:, None] * (cap.b - cap.a)
            P.append(p); U.append(np.full(n, cap.r + self.margen_mapa_brazo + h))
        return (np.vstack(P), np.concatenate(U)) if P else (np.zeros((0, 3)), np.zeros(0))

    def _holgura(self, q):
        P, U = self._puntos_holgura(q)
        if self._arbol is None or not len(P):
            return np.inf, P
        md = self._mapa_dist()
        if md is not None:
            # (17-nov) con el mapa de distancias: cada punto tiene su holgura entre est y est + 2h; la mínima
            # solo puede estar entre los que tienen est ≤ mín + 2h → búsqueda exacta solo de esos pocos
            est = md.leer(P) - md.h - U
            e0 = float(est.min())
            if e0 >= 0.25:
                return e0, P
            cerca = est <= e0 + 2.0 * md.h
            d, _ = self._arbol.query(P[cerca], distance_upper_bound=float(U[cerca].max()) + 0.25)
            return float(np.min(np.minimum(d, 10.0) - U[cerca])), P
        d, _ = self._arbol.query(P, distance_upper_bound=float(U.max()) + 0.25)
        return float(np.min(np.minimum(d, 10.0) - U)), P

    def comprobar_continuo(self, qs, nombre="tramo"):
        """(17-nov) Entre cada dos posturas seguidas de qs (que ya valen): si algún punto se mueve más
        de lo que le sobra de holgura, se mira la postura de en medio (y así, partiendo en dos, hasta
        subdiv_max veces). ErrorPlan si algo choca entre medias."""
        if not self.continuo or self._arbol is None or len(qs) < 2:
            return
        hs = [self._holgura(q) for q in qs]

        def entre(qa, qb, ha, hb, prof):
            delta = float(np.max(np.linalg.norm(hb[1] - ha[1], axis=1))) if len(ha[1]) else 0.0
            if min(ha[0], hb[0]) >= 0.5 * delta or prof >= self.subdiv_max:
                return
            qm = 0.5 * (qa + qb)
            m = self.motivo(qm)
            if m:
                raise ErrorPlan(f"{nombre}: {m} (entre dos posturas del tramo)")
            hm = self._holgura(qm)
            entre(qa, qm, ha, hm, prof + 1)
            entre(qm, qb, hm, hb, prof + 1)
        for k in range(len(qs) - 1):
            entre(np.asarray(qs[k], float), np.asarray(qs[k + 1], float), hs[k], hs[k + 1], 0)

    def articular(self, q0, q1, nombre="tramo"):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = max(1, int(math.ceil(float(np.max(np.abs(q1 - q0))) / self.paso)))
        qs = q0 + (q1 - q0) * np.linspace(0.0, 1.0, n + 1)[:, None]
        for q, m in self.motivos(qs[1:]):
            if m:
                raise ErrorPlan(f"{nombre}: {m}")
        self.comprobar_continuo(qs, nombre)
        return qs

    def _palma(self, q):
        T = self.c.fk_marcos(np.asarray(q, float))[6]
        return T[:3, :3] @ OFFSET_PALMA_ORCA + T[:3, 3]

    def tramo_por_alto(self, q0, q1, nombre="tramo"):
        """(17-nov) De q0 a q1 POR ARRIBA: la palma sube en vertical (orientación fija) hasta una altura
        que pasa por encima de todo lo del mapa en el pasillo entre las dos, se traslada (articular)
        y baja en vertical. Así el brazo no barre la mesa en arco, que es lo que pasaba yendo directo."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        p0, p1 = self._palma(q0), self._palma(q1)
        z_sitio = max(p0[2], p1[2])
        # cada punta sube SOLO hasta pasar por encima de lo que hay en el pasillo (no hasta la altura de la
        # otra punta: subir 40 cm desde encima de la pelota dobla la muñeca contra el antebrazo)
        z = min(p0[2], p1[2])
        if self._arbol is not None and len(self._arbol.data):
            V = self._arbol.data
            # el pasillo: lo que está (en planta) a menos de alto_pasillo del segmento entre las dos palmas
            ab = p1[:2] - p0[:2]
            t = np.clip((V[:, :2] - p0[:2]) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0)
            en = np.linalg.norm(V[:, :2] - (p0[:2] + t[:, None] * ab), axis=1) < self.alto_pasillo
            if en.any():
                z = max(z, float(V[en, 2].max()) + 0.5 * self._res + self.alto_margen)
        if z > z_sitio + self.alto_max:
            raise ErrorPlan(f"{nombre}: por arriba habría que subir {100 * (z - z_sitio):.0f} cm")
        a = self.recta(q0, OFFSET_PALMA_ORCA, p0 + [0.0, 0.0, z - p0[2]], f"{nombre} (subir)") \
            if z - p0[2] > 0.01 else q0[None]
        c = self.recta(q1, OFFSET_PALMA_ORCA, p1 + [0.0, 0.0, z - p1[2]], f"{nombre} (bajar)") \
            if z - p1[2] > 0.01 else q1[None]
        b = self.articular(a[-1], c[-1], nombre)
        return np.vstack([a, b[1:], c[::-1][1:]])

    def _crear_curobo(self, g):
        import os
        try:
            from skill_shield.autonomo.curobo_fr5.planificador import PlanificadorCuRobo
            z0 = getattr(self.modelo.entorno, "z_suelo", None)
            z0 = g.get("SOPORTE_SUELO_Z", -0.975) if z0 is None else z0
            lim = g.get("AUTONOMO_MAPA_LIMITES") or ((-1.0, 0.6), (-1.6, 0.4), (z0, 0.9))
            ruta = g.get("AUTONOMO_CUROBO_ROBOT", "curobo_fr5/fr5_orca.yml")
            if not os.path.isabs(ruta):
                ruta = os.path.join(os.path.dirname(os.path.abspath(g.get("RUTA_XML", "."))), "..", "..", ruta)
            # (18-nov) cuRobo con la MISMA geometría y márgenes que las comprobaciones de aquí: el mundo
            # inflado con el margen del brazo, las esferas de la mano con lo que le falta hasta el de la mano,
            # y la pelota cogida como esferas de mano_mount. Así lo que propone casi siempre pasa motivo().
            return PlanificadorCuRobo(os.path.normpath(ruta), self.modelo.entorno, lim,
                                      float(g.get("AUTONOMO_MAPA_RES", 0.02)),
                                      margen_entorno=self.margen_mapa_malla,
                                      margen_mundo=self.margen_mapa_malla,
                                      margen_mano_extra=max(0.0, self.margen_mapa - self.margen_mapa_malla),
                                      esferas_mano=self._esferas_carga(g),
                                      max_goalset=int(g.get("AUTONOMO_CUROBO_MAX_GOALSET", 16)),
                                      intentos=int(g.get("AUTONOMO_CUROBO_INTENTOS", 4)),
                                      semillas=int(g.get("AUTONOMO_CUROBO_SEMILLAS", 8)),
                                      desconocido_extra=g.get("AUTONOMO_CUROBO_DESCONOCIDO_EXTRA", 0.02),
                                      activacion=g.get("AUTONOMO_CUROBO_ACTIVACION"),
                                      cuda_graph=bool(g.get("AUTONOMO_CUROBO_CUDA_GRAPH", False)))
        except Exception as e:
            global _AVISO_CUROBO
            self._curobo_no_disponible = str(e)          # (18-nov) se dice en la web en cada plan
            if _AVISO_CUROBO != str(e):                   # una vez (no en cada planificador que se cree)
                _AVISO_CUROBO = str(e)
                print(f"⚠️  Tarea: cuRobo no disponible ({e}): planifico con el planificador propio.")
            return None

    def _esferas_carga(self, g):
        """(18-nov) [(centro en mano_mount, radio)] de la pelota cogida en cada forma de cogerla que está
        activa (AUTONOMO_TAREA_PELOTA_MODOS), para añadirla a las esferas de la mano de cuRobo."""
        r = 0.5 * float(_g(g, "AUTONOMO_CUROBO_PELOTA_DIAMETRO", 0.07))
        out = []
        for m in self.pel_modos:
            try:
                if m == "pinza":
                    pz = self.pinza(2 * r)
                    if pz is not None:
                        out.append((np.asarray(pz["punto"], float), r))
                elif m in ("rodeándola", "con los dedos"):
                    extra = self.pel_dedos if m == "con los dedos" else 0.0
                    out.append((OFFSET_PALMA_ORCA + np.array([-(r + self.pel_holgura + extra), 0.0, 0.0]), r))
            except Exception as e:
                print(f"⚠️  Tarea: no puedo poner la pelota cogida en cuRobo ({m}: {e}).")
        return out

    def _goalset(self, q0, opciones, nombre, mano=None):
        """(18-nov) cuRobo elige en la GPU, de una vez, la mejor de las opciones de un sitio (su pose de la mano) y el
        camino desde q0 (plan_goalset). Se acepta solo si acaba en la MISMA postura que la opción (cuRobo podría
        llegar a la misma pose con otra configuración del brazo, y la bajada está calculada desde la de la opción)
        y si el camino pasa todas las comprobaciones de aquí. (índice, posturas) o None."""
        cu = self.curobo
        if cu is None or not hasattr(cu, "plan_goalset") or len(opciones) < 2:
            return None
        Ts = [self.c.fk_marcos(np.asarray(op[0], float))[6] for op in opciones]
        try:
            with self.con_mano(*(mano or (None, None))):
                cu.actualizar_mundo(self, clave=(id(self), getattr(self, "_version_mundo", 0),
                                                 id(self._arbol), id(self._visto)))
                r = cu.plan_goalset(q0, Ts)
                if r is None:
                    return None
                idx, qs = r
                q_op = np.asarray(opciones[idx][0], float)
                if float(np.max(np.abs(qs[-1] - q_op))) > math.radians(3.0):
                    self.curobo_rechazos += 1
                    return None                          # otra configuración del brazo: no empalma con la bajada
                qs = np.vstack([qs[:-1], q_op[None]])
                qs = self._densificar(qs)
                for q in qs[1:]:
                    m = self.motivo(q)
                    if m:
                        self.curobo_rechazos += 1
                        return None
                self.comprobar_continuo(qs, nombre)
        except ErrorPlan:
            self.curobo_rechazos += 1
            return None
        except Exception as e:                           # la GPU nunca debe tumbar el plan
            self._pausar_curobo(e)
            return None
        self.tramos_curobo += 1
        self.goalsets_curobo = getattr(self, "goalsets_curobo", 0) + 1
        return idx, qs

    def _densificar(self, qs):
        """Posturas intermedias para que ningún joint salte más de self.paso entre dos seguidas."""
        out = [np.asarray(qs[0], float)]
        for a, b in zip(qs[:-1], qs[1:]):
            n = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / self.paso)))
            out += [a + (b - a) * (k / n) for k in range(1, n + 1)]
        return np.array(out)

    def tramo_curobo(self, q0, q1, nombre="tramo"):
        """(17-nov) El tramo que propone cuRobo, VERIFICADO aquí postura a postura (mano con su postura
        y su carga, suelo de las mesas, J4…) y entre medias. ErrorPlan si no hay o no vale."""
        try:
            self.curobo.actualizar_mundo(self, clave=(id(self), getattr(self, "_version_mundo", 0),
                                                      id(self._arbol), id(self._visto)))
            qs = self.curobo.plan(q0, q1)
        except Exception as e:                           # la GPU nunca debe tumbar el plan
            self._pausar_curobo(e)
            raise ErrorPlan(f"{nombre}: cuRobo falló")
        if qs is None:
            raise ErrorPlan(f"{nombre}: cuRobo no encuentra camino")
        qs = self._densificar(qs)
        for q in qs[1:]:
            m = self.motivo(q)
            if m:
                self.curobo_rechazos += 1
                raise ErrorPlan(f"{nombre}: el camino de cuRobo no pasa las comprobaciones ({m})")
        self.comprobar_continuo(qs, nombre)
        return qs

    def conectar(self, q0, q1, q_via=None, nombre="tramo"):
        """cuRobo (si está); si no, por arriba (si se puede); si no, articular de q0 a q1; si choca,
        pasando por q_via; si también, rodeando (paso 8)."""
        if self.curobo is not None:
            try:
                qs = self.tramo_curobo(q0, q1, nombre)
                self.tramos_curobo += 1
                return qs
            except ErrorPlan:
                pass
        alto = None
        if self.por_alto:
            try:
                alto = self.tramo_por_alto(q0, q1, nombre)
            except ErrorPlan:
                alto = None
        if alto is not None and self._visto is not None:
            # (17-nov) por arriba es lo preferido, salvo que pase claramente más por lo que la cámara no ha
            # visto que el camino directo
            try:
                directo = self.articular(q0, q1, nombre)
                if self.desconocido(alto) > self.desconocido(directo) + 0.10:
                    return directo
            except ErrorPlan:
                pass
        if alto is not None:
            self.tramos_por_alto += 1
            return alto
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
            if self.curobo is not None and not self.rrt_con_curobo:
                # (18-nov) con cuRobo en la GPU, el RRT en CPU (6-25 s) no aporta: cuRobo ya ha buscado rodeos
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
        for _, m in self.motivos(qa + (qb - qa) * (np.arange(1, n + 1) / n)[:, None]):
            if m is not None:
                return False
        try:
            self.comprobar_continuo(qa + (qb - qa) * np.linspace(0.0, 1.0, n + 1)[:, None])
        except ErrorPlan:
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
        qs = np.array(qs)
        self.comprobar_continuo(qs, nombre)
        return qs

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
        n_op = self.n_opciones_gpu if (self.curobo is not None and hasattr(self.curobo, "plan_goalset")) \
            else self.n_opciones
        if todas:
            # todas las inclinaciones a la vez (pelotas, 12-nov): la mejor es la que menos mueve el
            # brazo, con un poco de preferencia por llegar derecho; de arriba, de lado o como sea.
            # (18-nov) todas las IK en lote; y, si hay que afinar con la cámara al llegar, coste por la
            # parte del objeto que el brazo le tape (por encima de tapada_max, al final de la lista)
            claves = [(inc, a, ang) for inc, dirs in grupos for a in dirs for ang in self._vueltas()]
            cands = []
            for (inc, a, ang), qls in zip(claves, self.ik_lote([T_pose(a, ang) for inc, a, ang in claves],
                                                                 q_ref, 2, razones)):
                for q_l in qls:
                    cands.append((self._coste(q_l, q_ref) + self.pref_derecho * math.radians(inc)
                                  + self._coste_tapada(q_l), q_l, a, inc))
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
                if len(opciones) >= n_op:
                    break
            return opciones, (razones, fallos)
        for g_i, (inc, dirs) in enumerate(grupos):
            cands = []
            claves = [(a, ang) for a in dirs for ang in self._vueltas()]
            for (a, ang), qls in zip(claves, self.ik_lote([T_pose(a, ang) for a, ang in claves], q_ref, 2, razones)):
                for q_l in qls:
                    cands.append((self._coste(q_l, q_ref) + self._coste_tapada(q_l), q_l, a))
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
                if len(opciones) >= n_op:
                    return opciones, (razones, fallos)
            if opciones and g_i + 1 < len(grupos) and inc > 0:
                return opciones, (razones, fallos)        # ya hay: no hace falta inclinarse más
        return opciones, (razones, fallos)

    def _coste_tapada(self, q):
        """(18-nov) Coste de que el brazo, en la postura q, le tape a la cámara el objeto que hay que mirar
        (self._mirar_obj): peso_tapada × parte tapada; si pasa de tapada_max, se va al final de la lista."""
        if self._mirar_obj is None or self.T_cam is None or self.peso_tapada <= 0:
            return 0.0
        f = self.fraccion_tapada(q, *self._mirar_obj)
        if f is None:
            return 0.0
        return self.peso_tapada * f + (100.0 if f > self.tapada_max else 0.0)

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

            soluciones = []
            # (17-nov) con cuRobo (GPU) cada tramo cuesta milisegundos: se buscan varios planes completos y se
            # elige el mejor (longitud y lo no visto); sin él, el primero (comparar en CPU es lento)
            n_max = max(1, self.alternativas_gpu if self.curobo is not None else
                        (self.alternativas if self._visto is not None else 1))
            t_lim = time.monotonic() + 8.0                  # s como mucho buscando alternativas

            def buscar(k, q_actual, elegidas, tramos):
                if len(soluciones) >= n_max or (soluciones and time.monotonic() > t_lim):
                    return
                if k == len(sitios):
                    vuelta = tramo(q_actual, q0, "volver al inicio", mano_vuelta)
                    if vuelta is not None:
                        soluciones.append((elegidas, tramos, vuelta))
                    return
                nombre, opciones = sitios[k][:2]
                mano = sitios[k][2] if len(sitios[k]) > 2 else None
                # (18-nov) con cuRobo: primero UNA llamada a la GPU con todas las opciones del sitio a la vez
                # (goalset): elige la mejor y su camino; esa opción se prueba la primera
                g = self._goalset(q_actual, opciones, nombre, mano) if not soluciones else None
                if g is not None:
                    idx, qs_g = g
                    cache[(q_actual.tobytes(), np.asarray(opciones[idx][0], float).tobytes(), nombre)] = qs_g
                    opciones = [opciones[idx]] + [o for i_, o in enumerate(opciones) if i_ != idx]
                for op in opciones:
                    if len(soluciones) >= n_max:
                        return
                    ir = tramo(q_actual, op[0], nombre, mano)
                    if ir is None:
                        continue
                    buscar(k + 1, op[1], elegidas + [op], tramos + [ir])
            buscar(0, q0, [], [])
            self._permitir_rodeo = False
            if soluciones:
                # (17-nov) de los planes completos encontrados, el de menor coste (longitud y lo no visto);
                # sin «lo visto» se queda con el primero, como antes
                if len(soluciones) == 1:
                    return soluciones[0]
                return min(soluciones, key=lambda r_: sum(self.coste_tramo(t) for t in r_[1] + [r_[2]]))
        motivo = fallos.most_common(1)[0][0] if fallos else "?"
        raise ErrorPlan(f"sé llegar a cada sitio, pero no encuentro cómo ir de uno a otro: {motivo}")

    def _empezar(self, q0):
        q0 = np.asarray(q0, float)
        self.inclinaciones, self.rodeos, self.tramos_por_alto = [], 0, 0
        self.tramos_curobo, self.curobo_rechazos, self.goalsets_curobo = 0, 0, 0
        self._choques = []
        self.avisos = []
        self._reactivar_gpu()
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
        if self._superficies:
            # (17-nov) una mesa del mapa bajo la que está AHORA el robot no puede ser maciza ahí (o está mal
            # medida, o es un estante con hueco debajo): no se usa como suelo en este plan
            P = self._puntos_robot(q0)
            quedan = []
            for z_m, poli in self._superficies:
                from skill_shield.autonomo.tsdf import dentro_poligono
                en = dentro_poligono(poli, P[:, :2])
                debajo = en & (P[:, 2] < z_m + self.margen_suave) & (P[:, 2] > z_m - self.grosor_suelo_zona)
                if debajo.any():
                    print(f"ℹ️  Tarea: el robot está ahora debajo de la mesa a z={z_m:.2f} m del mapa: no la uso de suelo.")
                else:
                    quedan.append((z_m, poli))
            if len(quedan) != len(self._superficies):
                self._superficies = quedan
                self.zonas_apoyo(())
        if self.huecos_desde_inicio:
            self.choque.fijar_inicio(q0)
        m = self.motivo(q0)
        if m:
            raise ErrorPlan(f"la postura actual del robot no pasa las comprobaciones: {m}")
        return q0

    # ── (18-nov) la GPU: un fallo ya no la apaga para toda la sesión ──
    def _reactivar_gpu(self):
        """Al empezar cada plan: si cuRobo o la comprobación en lote se pausaron por un fallo, se vuelven a
        probar (hasta gpu_fallos_max fallos; después se quedan apagados y la web lo dice)."""
        if getattr(self, "_curobo_no_disponible", None):
            self.avisos.append(f"cuRobo no disponible ({self._curobo_no_disponible}): planifico en la CPU")
        if self.curobo is None and self._curobo_pausado is not None:
            if self.curobo_fallos < self.gpu_fallos_max:
                self.curobo, self._curobo_pausado = self._curobo_pausado, None
            else:
                self.avisos.append(f"cuRobo apagado tras {self.curobo_fallos} fallos (planifico en la CPU)")
        if self._lote_torch is None and self._lote_pausado is not None:
            if self.lote_fallos < self.gpu_fallos_max:
                (self._lote_torch, self._lote_dev), self._lote_pausado = self._lote_pausado, None
            else:
                self.avisos.append(f"comprobación en la GPU apagada tras {self.lote_fallos} fallos (va en la CPU)")

    def _pausar_curobo(self, e):
        self.curobo_fallos += 1
        self._curobo_pausado, self.curobo = self.curobo, None
        txt = f"cuRobo falló ({e}): este plan va con el planificador propio"
        print(f"⚠️  Tarea: {txt} (fallo {self.curobo_fallos}/{self.gpu_fallos_max}).")
        self.avisos.append(txt)

    def _puntos_robot(self, q, marcos=None):
        """Puntos (mundo) de la mano y del brazo en la postura q (para ver qué tapa a la cámara)."""
        marcos = self.c.fk_marcos(np.asarray(q, float)) if marcos is None else marcos
        Tb = marcos[5]
        P = [] if self._mano_pts0 is None else [self._mano_pts0 @ Tb[:3, :3].T + Tb[:3, 3]]
        for marco, pts, _ in self._brazo_pts:
            P.append(pts @ marcos[marco][:3, :3].T + marcos[marco][:3, 3])
        for cap in self.modelo.capsulas(q, marcos):
            if cap.nombre != "base":
                P.append(cap.a + np.linspace(0.0, 1.0, 8)[:, None] * (cap.b - cap.a))
        return np.vstack(P) if P else np.zeros((0, 3))

    def tapa_la_vista(self, q, c, radio=None):
        """(17-nov) ¿El robot en la postura q está entre la cámara y la zona del objetivo (centro c,
        radio)? Entonces la cámara no ve la mesa alrededor del objetivo y el mapa tiene un hueco."""
        if self.T_cam is None:
            return False
        radio = self.mirar_radio if radio is None else float(radio)
        cam, c = self.T_cam[:3, 3], np.asarray(c, float)
        w = c - cam
        L = float(np.linalg.norm(w))
        if L < 1e-6:
            return False
        u = w / L
        V = self._puntos_robot(q) - cam
        a = V @ u
        perp = np.linalg.norm(V - a[:, None] * u, axis=1)
        delante = (a > 0.10) & (a < L - 0.5 * radio)
        return bool(np.any(delante & (perp < radio * a / L)))

    def plan_mirar(self, q0, c):
        """(17-nov) Si el brazo le tapa a la cámara el objetivo (centro c), un plan para apartarse a una
        postura desde la que se ve (AUTONOMO_TAREA_POSTURA_MIRAR_DEG o, si no hay, la primera que
        sirva moviendo la mano de lado, hacia la cámara o hacia arriba) y quedarse quieto mirando
        AUTONOMO_TAREA_MIRAR_S. None si no tapa (o si no encuentra adónde apartarse: se avisa)."""
        q0 = self._empezar(q0)
        c = np.asarray(c, float)
        if not self.tapa_la_vista(q0, c):
            return None
        qs = None
        if self.postura_mirar is not None:
            try:
                qs = self.conectar(q0, self.postura_mirar, None, "apartarse para mirar")
                if self.tapa_la_vista(qs[-1], c):
                    print("⚠️  Tarea: desde AUTONOMO_TAREA_POSTURA_MIRAR_DEG el brazo también tapa el objetivo.")
                    qs = None
            except ErrorPlan as e:
                print(f"⚠️  Tarea: no puedo ir a AUTONOMO_TAREA_POSTURA_MIRAR_DEG ({e}).")
        if qs is None:
            p = self._palma(q0)
            w = c - self.T_cam[:3, 3]
            hor = _unit(np.array([w[0], w[1], 0.0]))
            lado = np.cross(hor, [0.0, 0.0, 1.0])
            diag = [_unit(lado - hor), _unit(-lado - hor)]
            for k in (0.15, 0.25, 0.35, 0.45):
                for d in (lado, -lado, -hor, *diag, np.zeros(3)):
                    dest = p + d * k + np.array([0.0, 0.0, 0.08 + (k if not d.any() else 0.0)])
                    try:
                        cand = self.recta(q0, OFFSET_PALMA_ORCA, dest, "apartarse para mirar")
                    except ErrorPlan:
                        continue
                    if not self.tapa_la_vista(cand[-1], c):
                        qs = cand
                        break
                if qs is not None:
                    break
        if qs is None:
            print("⚠️  Tarea: el brazo tapa el objetivo a la cámara y no encuentro adónde apartarlo: "
                  "planifico con el mapa que hay.")
            return None
        plan = Plan(q0, [Fase("Apartarse para que la cámara vea la mesa", qs),
                         Fase("Mirar la mesa", dur=self.mirar_s)], nombre="apartarse para mirar")
        plan.replanificar = True
        return plan

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
        for f in fases:                                   # (18-nov) tramos rectos a velocidad de la palma
            if f.qs is not None and f.nombre.startswith(("Acercar", "Retirarse", "Alejar")):
                self._recta_lenta(f, llegada=f.nombre.startswith("Acercar"))
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

    def poner_visto(self, visto):
        """(17-nov) Qué cubos ha visto la cámara: dict(res, origen, n, conocido) (mapa.desempaquetar_visto)
        o None (entonces no hay coste por lo desconocido)."""
        self._visto = visto if visto and visto.get("conocido") is not None else None
        # (17-nov) las mesas que ha encontrado el mapa (planos): suelo para mano y brazo, como la medida
        self._superficies_vivas = [(float(z), np.asarray(p, float)) for z, p in ((visto or {}).get("superficies") or [])
                                   if len(p) >= 3]
        self._superficies = self._superficies_vivas + list(self._tapas_fijas)    # (17-nov) + las tapas fijas
        self.zonas_apoyo(self._zonas_actuales)

    def desconocido(self, qs, cada=3):
        """Fracción (0-1) de los puntos del robot que, a lo largo de qs, caen en cubos que la cámara
        no ha visto. Lo de fuera de la rejilla no cuenta (no se sabe ni se puede saber)."""
        v = self._visto
        if v is None or qs is None or len(qs) == 0:
            return 0.0
        n_dentro = n_desc = 0
        o, res, N, K = np.asarray(v["origen"], float), float(v["res"]), np.asarray(v["n"]), v["conocido"]
        for q in list(qs[::cada]) + [qs[-1]]:
            P, _ = self._puntos_holgura(q)
            idx = np.floor((P - o) / res).astype(int)
            ok = np.all((idx >= 0) & (idx < N), axis=1)
            n_dentro += int(ok.sum())
            n_desc += int((~K[idx[ok, 0], idx[ok, 1], idx[ok, 2]]).sum())
        return n_desc / n_dentro if n_dentro else 0.0

    def coste_tramo(self, qs):
        """Longitud del tramo (rad, el joint que más se mueve en cada paso) × (1 + peso · desconocido)."""
        if qs is None or len(qs) < 2:
            return 0.0
        L = float(np.sum(np.max(np.abs(np.diff(qs, axis=0)), axis=1)))
        return L * (1.0 + self.peso_desconocido * self.desconocido(qs))

    def fraccion_tapada(self, q, c, r, n=9):
        """(18-nov) Qué parte de la pelota (centro c, radio r) tapa el propio robot, en la postura q, a la
        cámara de cabeza: 0 = se ve entera, 1 = no se ve nada. None si no hay calibración.
        Con la cámara detrás del robot, al pararse encima de la pelota el ANTEBRAZO suele quedar justo entre
        la cámara y la pelota: lo que «ve» entonces el detector (el borde que asoma, con la profundidad del
        brazo mezclada) no es la pelota, y no hay que fiarse de esa medida."""
        if self.T_cam is None:
            return None
        o = np.asarray(self.T_cam, float)[:3, 3]
        c = np.asarray(c, float)
        v = c - o
        dist = float(np.linalg.norm(v))
        if dist <= float(r) + 1e-6:
            return None
        v /= dist
        u, w = _base_perp(v)
        g = np.linspace(-1.0, 1.0, int(n))
        X, Y = np.meshgrid(g, g)
        k = X ** 2 + Y ** 2 <= 1.0
        # el disco de la pelota que ve la cámara, a la altura de su cara delantera
        F = c + float(r) * (X[k][:, None] * u + Y[k][:, None] * w) - float(r) * v
        D = F - o                                         # rayos cámara → pelota (N x 3)
        L2 = np.einsum("ij,ij->i", D, D)
        q = np.asarray(q, float)
        tapado = np.zeros(len(F), bool)
        for cap in self.modelo.capsulas(q, self.c.fk_marcos(q)):
            if cap.nombre == "base":
                continue
            m = max(2, int(math.ceil(np.linalg.norm(cap.b - cap.a) / 0.01)) + 1)
            A = cap.a + np.linspace(0.0, 1.0, m)[:, None] * (cap.b - cap.a)          # eje de la cápsula (M x 3)
            t = np.clip(np.einsum("mk,nk->nm", A - o, D) / L2[:, None], 0.0, 1.0)    # punto más cercano del rayo
            d = np.linalg.norm(o + t[..., None] * D[:, None, :] - A[None], axis=-1)
            tapado |= (d < cap.r).any(axis=1)
        return float(tapado.mean())

    def poner_mesa(self, apoyo_z, apoyo_poligono):
        """(17-nov) La mesa medida con un plano (altura y envolvente en planta) o None."""
        self._mesa = (None if apoyo_z is None or not apoyo_poligono or len(apoyo_poligono) < 3
                      else (float(apoyo_z), [list(map(float, v)) for v in apoyo_poligono]))

    def apoyar(self, T_pelota, diametro, apoyo_z, T_mundo_cam=None):
        """(17-nov) La pelota está APOYADA en la mesa: si se ha medido la altura de la mesa (plano
        ajustado a la profundidad alrededor de la pelota, pelotas.py) y cuadra, su centro está a un radio
        por encima. Es más fiable que la altura que da el estéreo en la propia pelota (sus bordes).
        (18-nov) Se DESLIZA por el rayo de la cámara (no en vertical): el error del estéreo va a lo largo del
        rayo y, con la cámara mirando casi en horizontal, es sobre todo horizontal («la mano baja más lejos»).
        Subirla o bajarla dejaba ese error entero. T_mundo_cam: la cámara con la que se midió (por defecto, la
        de la última medida o la de la calibración)."""
        T = np.array(T_pelota, float)
        self._apoyo_ok = False
        r = float(diametro) / 2.0
        # (17-nov) si está sobre una tapa de la escena FIJA, su altura (escaneada con calma) manda
        self._cargar_fijo()
        z_fijo = self._apoyo_fijo(T[:3, 3], r) if self._tapas_fijas else None
        self._z_fijo_usado = z_fijo               # (la tapa fija ya es «suelo»: no hace falta la mesa medida)
        if z_fijo is not None:
            if apoyo_z is None or abs(float(apoyo_z) - z_fijo) > 0.005:
                print(f"🧱 Tarea: la pelota está sobre una tapa de la escena fija: apoyo a z = {z_fijo:+.3f} m"
                      + ("" if apoyo_z is None else f" (la medida en vivo daba {float(apoyo_z):+.3f})"))
            apoyo_z = z_fijo
        if apoyo_z is None:
            return T
        dz = float(apoyo_z) + r - T[2, 3]
        if abs(dz) <= (max(self.apoyo_tol, self.apoyo_fijo_tol) if z_fijo is not None else self.apoyo_tol):
            c = self.por_rayo(T[:3, 3], float(apoyo_z) + r, T_mundo_cam)
            T[:3, 3] = c if c is not None else T[:3, 3] + [0.0, 0.0, dz]
            self._apoyo_ok = True
        else:
            print(f"⚠️  Tarea: la mesa medida está {100 * dz:+.1f} cm de donde debería apoyar la pelota: no la uso.")
        return T

    def por_rayo(self, c, z, T_mundo_cam=None, max_desliz=0.12):
        """(18-nov) El punto del rayo cámara→c que está a la altura z (o None: sin cámara, rayo casi
        horizontal o haría falta deslizarlo más de max_desliz)."""
        if not self.pelota_por_rayo:
            return None
        Tc = T_mundo_cam if T_mundo_cam is not None else (
            self._T_cam_medida if self._T_cam_medida is not None else self.T_cam)
        if Tc is None:
            return None
        o, c = np.asarray(Tc, float)[:3, 3], np.asarray(c, float)
        d = c - o
        n = float(np.linalg.norm(d))
        if n < 1e-6 or abs(d[2]) / n < 0.1:               # casi horizontal: deslizar amplifica el error
            return None
        p = o + d * ((float(z) - o[2]) / d[2])
        return p if float(np.linalg.norm(p - c)) <= max_desliz else None

    def planificar_pelota(self, q0, T_pelota, diametro, T_dejada, apoyo_z=None, apoyo_poligono=None,
                          T_mundo_cam=None):
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
        self._T_cam_medida = None if T_mundo_cam is None else np.asarray(T_mundo_cam, float)
        T_pelota = self.apoyar(T_pelota, diametro, apoyo_z)
        self.poner_mesa(apoyo_z if self._apoyo_ok and self._z_fijo_usado is None else None, apoyo_poligono)
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
                self.inc_max = self.pinza_inc_max if (clave == "pinza" and self.pinza_arriba) else self.pel_inc_max
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
        gaps, mids, lineas = [], [], []
        for s_ in ss:
            y = yemas_mano(self._ruta_xml, tuple(sorted((k, round(v, 4)) for k, v in postura(s_).items())), 0.004)
            if "pulgar" not in y or "indice" not in y:
                cache[clave] = None
                return None
            t, i = (T_mm_w[:3, :3] @ y[d] + T_mm_w[:3, 3] for d in ("pulgar", "indice"))
            gaps.append(float(np.linalg.norm(t - i)))
            mids.append((t + i) / 2)
            lineas.append(i - t)
        gaps = np.array(gaps)
        objetivo = float(diametro)
        if objetivo > gaps[0] - 0.015 or objetivo < 0.02:      # más de lo que abre (con holgura) o diminuta
            cache[clave] = None
            return None
        k = int(np.nonzero(gaps <= objetivo)[0][0])              # primera s donde ya la toca
        u = (gaps[k - 1] - objetivo) / max(gaps[k - 1] - gaps[k], 1e-9)
        s_c = float(ss[k - 1] + u * (ss[k] - ss[k - 1]))
        punto = mids[k - 1] + u * (mids[k] - mids[k - 1])
        linea = lineas[k - 1] + u * (lineas[k] - lineas[k - 1])    # pulgar → índice, en mano_mount
        res = dict(s=s_c, punto=punto, linea=linea, contacto=postura(s_c),
                   cerrada=postura(min(1.0, s_c + self.pinza_apriete[0])),
                   apretar=postura(min(1.0, s_c + self.pinza_apriete[1])))
        cache[clave] = res
        return res

    def T_pinza_arriba(self, T_m, ang, d, a, punto, linea):
        """(17-nov) Pinza DESDE ARRIBA: la mano como en T_palma (llegando por a, girada ang) y luego
        inclinada lo justo para que la línea pulgar-índice quede HORIZONTAL: los dos dedos cogen la
        pelota por los lados, a la altura de su centro (antes el pulgar iba por debajo, rozando la mesa)."""
        T = self.T_palma(T_m, ang, d, a, punto)
        R = T[:3, :3]
        lw = R @ np.asarray(linea, float)
        lw = lw / max(float(np.linalg.norm(lw)), 1e-9)
        k = np.cross(lw, [0.0, 0.0, 1.0])
        if float(np.linalg.norm(k)) > 1e-6 and abs(lw[2]) > 1e-6:
            th = math.asin(max(-1.0, min(1.0, float(lw[2]))))
            Rk = min((_rot_eje(k, sg * th) for sg in (1.0, -1.0)), key=lambda M: abs(float((M @ lw)[2])))
            R = Rk @ R
        p_m = np.asarray(T_m, float)[:3, 3]
        return T_de(R, p_m + R[:, 0] * d - R @ np.asarray(punto, float))

    def _pose_pelota(self, T_m, d, punto, pinza):
        if pinza is not None and self.pinza_arriba and pinza.get("linea") is not None:
            return lambda a, ang: self.T_pinza_arriba(T_m, ang, d, a, punto, pinza["linea"])
        return lambda a, ang: self.T_palma(T_m, ang, d, a, punto)

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
        self.zonas_apoyo([(c, r + self.zona_apoyo, r), (c_d, r + self.zona_apoyo, r + 0.01, not self.dejar_de_pie)])
        elevar = self._elevar_pinza(pinza, r) if pinza is not None else 0.0
        subir = np.array([0.0, 0.0, elevar])
        if pinza is not None:
            abierta, cerrada = ("pinza_abierta", None), ("pinza_contacto",
                                                          self._pelota_en_mano(r, centro=self._bola_en_pinza(pinza, elevar)))
        else:
            abierta, cerrada = ("pelota_abierta", None), ("pelota_cerrada", self._pelota_en_mano(r, extra))

        def coger(q_l, a):
            centro = c + _unit(a) * d_palma + subir
            with self.con_mano(*abierta):
                qs = self.recta(q_l, punto, centro, "bajar a la pelota")
            with self.con_mano(*cerrada):                # al subir ya la lleva
                for q in qs[::-1]:
                    m = self.motivo(q)
                    if m:
                        raise ErrorPlan(f"subir con la pelota: {m}")
            return qs

        def dejar(q_l, a):
            centro = c_d + _unit(a) * d_palma + subir
            with self.con_mano(*cerrada):
                qs = self.recta(q_l, punto, centro, "bajar a dejarla")
            with self.con_mano(*abierta):                # ya la ha soltado
                for q in qs[::-1]:
                    m = self.motivo(q)
                    if m:
                        raise ErrorPlan(f"subir tras dejarla: {m}")
            return qs
        # (18-nov) encima de la pelota se para a afinar con la cámara: que el brazo no se la tape
        self._mirar_obj = (c, r) if self.afinar_s > 0 else None
        try:
            with self.con_mano(*abierta):
                ops_p, info = self._opciones(T_p, q0, self._pose_pelota(T_p, d_palma + self.pel_subida, punto, pinza),
                                             coger, todas=True)
        finally:
            self._mirar_obj = None
        if not ops_p:
            raise self._con_aviso(str(self._no_encuentro("llegar a la pelota (ni desde arriba ni de lado)", info))
                                  + self._que_estorba(c, r) + self._aviso_pinza(), c)
        inc_coger, self.inc_max = self.inc_max, self.pel_inc_max   # (17-nov) «desde arriba» es para COGERLA; para
        try:                                                       # soltarla (ya la lleva) vale cualquier inclinación
            with self.con_mano(*cerrada):
                ops_d, info = self._opciones(T_d, q0, self._pose_pelota(T_d, d_palma + self.pel_subida, punto, pinza),
                                             dejar, todas=True)
                if not ops_d and pinza is not None and self.pinza_arriba:
                    # soltarla con la pinza nivelada no se puede: con la mano en su orientación normal (la holgura
                    # de toda la mano con la mesa se sigue comprobando)
                    ops_d, info = self._opciones(T_d, q0, self._pose_pelota(T_d, d_palma + self.pel_subida, punto, None),
                                                 dejar, todas=True)
        finally:
            self.inc_max = inc_coger
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
        # (18-nov) los tramos rectos, a velocidad de la palma; las llegadas, despacio en los últimos cm
        for f in (f_bajar, f_dejar):
            self._recta_lenta(f, llegada=True)
        for f in (f_subir, f_alejar):
            self._recta_lenta(f, llegada=False)
        f_bajar.vigilar_tactil = True            # (18-nov) si un dedo toca algo al bajar: parar y reintentar
        # parada encima de la pelota para afinar con la cámara (12-nov): con el brazo quieto y la
        # mano al lado de la pelota, la cámara corrige la calibración con el brazo y vuelve a medir
        # la pelota; si se ha movido, se rehace la bajada (Tarea._afinar)
        afinar = [Fase("Afinar con la cámara", dur=self.afinar_s, comprobar="afinar")] if self.afinar_s > 0 else []
        f_cerrar = Fase("Cerrar la mano", mano=k_ce)
        if self.cierre_tactil:
            # (18-nov) con táctiles: cada dedo se cierra hasta que su yema nota la pelota (como mucho hasta
            # k_max) y desde ahí se aprieta y se afloja; sin táctiles, a la postura del modelo (como antes)
            k_max = "pinza_max" if pinza is not None else k_ap
            ap = self.pinza_apriete if pinza is not None else (0.0, 0.15)
            f_cerrar.cierre_tactil = dict(destino=k_max, apretar=k_ap, aflojar=k_ce, apriete=ap[1], aflojado=ap[0],
                                          umbral_n=self.cierre_n, dur=self.cierre_s)
        fases = ([Fase("Abrir la mano", mano=k_ab), f_ir] + afinar +
                 [f_bajar,
                  f_cerrar,
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
                           c_dejada=c_d.copy(), pinza=pinza is not None, elevar=float(elevar),
                           k_abierta=k_ab)
        if pinza is not None:                              # posturas que la Tarea manda a la mano
            plan.posturas = dict(pinza_abierta=dict(self.pinza_ab), pinza_cerrada=dict(pinza["cerrada"]),
                                 pinza_apretar=dict(pinza["apretar"]), pinza_max=dict(self.pinza_ce))
        return plan

    def _recta_lenta(self, f, llegada):
        """(18-nov) Tramo recto: a vel_cart de la palma y, si es una llegada, a vel_lenta en los últimos lento_m."""
        f.vel_cart = self.vel_cart
        if llegada:
            f.lento_m, f.vel_lenta = self.lento_m, self.vel_lenta

    def _caida_pinza(self, pinza):
        """(18-nov) Cuánto baja la mano (abierta o en contacto) por debajo del punto de pinza con la mano
        llegando desde arriba (m)."""
        try:
            m = self.c.fk_marcos(np.zeros(6))
            T_mm_w = np.linalg.inv(m[6]) @ m[5]                 # wrist3 → mano_mount
            T = (self.T_pinza_arriba(np.eye(4), 0.0, 0.0, np.array([0.0, 0.0, 1.0]), pinza["punto"], pinza["linea"])
                 if self.pinza_arriba and pinza.get("linea") is not None
                 else self.T_palma(np.eye(4), 0.0, 0.0, np.array([0.0, 0.0, 1.0]), pinza["punto"]))
            R, punto = T[:3, :3], np.asarray(pinza["punto"], float)
            caida = 0.0
            for clave in ("pinza_abierta", "pinza_contacto"):
                P = self._nube_postura(clave)
                if P is None:
                    continue
                Pmm = P @ T_mm_w[:3, :3].T + T_mm_w[:3, 3]
                caida = max(caida, float(-(((Pmm - punto) @ R.T)[:, 2]).min()))
            return max(caida, 0.0)
        except Exception as e:
            print(f"⚠️  Tarea: no puedo medir cuánto bajan los dedos de la pinza ({e}).")
            return 0.0

    def _elevar_pinza(self, pinza, r):
        """(18-nov) La pinza coge la pelota a la altura de su centro; si la mano baja por debajo del punto de
        pinza más de lo que deja la holgura con la mesa (holgura_mesa_mano), se coge algo más arriba (como
        mucho pinza_elevar_max × r: más arriba, la pelota se escurre hacia abajo al levantarla)."""
        self._info_pinza = None
        caida = self._caida_pinza(pinza)
        hace_falta = caida + self.holgura_mesa_mano + 0.003 - r
        if hace_falta <= 0:
            return 0.0
        elevar = min(hace_falta, self.pinza_elevar_max * r)
        self._info_pinza = (caida, hace_falta, elevar)
        txt = (f"la pinza coge la pelota {100 * elevar:.1f} cm por encima de su centro (la mano baja "
               f"{100 * caida:.1f} cm por debajo del punto de pinza y hay que dejar {100 * self.holgura_mesa_mano:.0f} cm "
               f"con la mesa)")
        if hace_falta > elevar + 1e-4:
            txt += f"; harían falta {100 * hace_falta:.1f} cm (máx. {100 * elevar:.1f}): probablemente no salga"
        print(f"ℹ️  Tarea: {txt}.")
        self.avisos.append(txt)
        return elevar

    def _aviso_pinza(self):
        i = getattr(self, "_info_pinza", None)
        if not i or i[1] <= i[2] + 1e-4:
            return ""
        return (f" · la pelota es demasiado pequeña para esta pinza desde arriba: la mano baja {100 * i[0]:.1f} cm por "
                f"debajo del punto de pinza (ajustad AUTONOMO_TAREA_MANO_PINZA_* o AUTONOMO_TAREA_HOLGURA_MESA_MANO)")

    def _bola_en_pinza(self, pinza, elevar):
        """(18-nov) Centro de la pelota cogida en mano_mount: el punto de pinza, bajado 'elevar' en vertical
        (con la mano llegando desde arriba)."""
        punto = np.asarray(pinza["punto"], float)
        if elevar <= 0:
            return punto
        T = (self.T_pinza_arriba(np.eye(4), 0.0, 0.0, np.array([0.0, 0.0, 1.0]), punto, pinza["linea"])
             if self.pinza_arriba and pinza.get("linea") is not None
             else self.T_palma(np.eye(4), 0.0, 0.0, np.array([0.0, 0.0, 1.0]), punto))
        return punto + T[:3, :3].T @ np.array([0.0, 0.0, -float(elevar)])

    def rehacer_bajada(self, plan, c_nueva):
        """(12-nov) La bajada a la pelota rehecha hacia c_nueva (medida otra vez desde encima),
        desde la misma postura de encima y con la misma dirección de llegada. ErrorPlan si no se
        puede. La subida es la misma al revés."""
        pz = plan.pelota
        fb, fs = plan.fases[pz["i_bajar"]], plan.fases[pz["i_subir"]]
        Tt = self.c.fk_marcos(fb.qs[-1])[6]
        a = _unit(Tt[:3, 0])                                # dirección de llegada (+X de mano_mount)
        punto = np.asarray(pz.get("punto", OFFSET_PALMA_ORCA), float)
        centro = np.asarray(c_nueva, float) + a * pz["d_palma"] + [0.0, 0.0, float(pz.get("elevar", 0.0))]
        r = pz["r"]                                        # la zona de agarre, donde está ahora la pelota
        self.zonas_apoyo([(np.asarray(c_nueva, float), r + self.zona_apoyo, r),
                          (pz["c_dejada"], r + self.zona_apoyo, r + 0.01, not self.dejar_de_pie)])
        with self.con_mano(*fb.mano_chk):
            qs = self.recta(fb.qs[0], punto, centro, "bajar a la pelota")
        with self.con_mano(*fs.mano_chk):
            for q in qs[::-1]:
                m = self.motivo(q)
                if m:
                    raise ErrorPlan(f"subir con la pelota: {m}")
        return qs

    # ── 16-nov: solo hasta la postura de AGARRE (modo GRABACION · agarre aprendido) ──
    def planificar_previa(self, q0, det):
        """Lleva la mano a la postura de agarre del objeto `det` (una entrada de la lista de la
        cámara: T_mundo, tipo, diametro_m/caja) —la MISMA a la que llega la Tarea justo antes de
        bajar a cogerlo— y se para ahí. Es donde empieza una demo del modo GRABACION (y, más
        adelante, el agarre aprendido). Pelota: encima (o inclinada), con la bajada comprobada,
        de cualquiera de las formas de cogerla. Herramienta: delante, a AUTONOMO_TAREA_DIST_PREVIA.
        Plan: [Abrir la mano, Ir encima de la pelota / Ir a la herramienta]."""
        T = np.asarray(det["T_mundo"], float)
        if det.get("tipo") == "pelota":
            caja = det.get("caja")
            diam = float(det.get("diametro_m") or (2.0 * float(caja[0]) if caja else 0.07))
            return self._previa_pelota(q0, T, diam, det.get("apoyo_z"), det.get("apoyo_poligono"),
                                       det.get("T_mundo_cam"))
        q0 = self._empezar(q0)
        ops = [(q, q, qs, inc) for q, qs, inc in self.opciones_palma(T, q0, "la herramienta")]
        (h,), (ir,), _vuelta = self._combinar(q0, [("ir a la herramienta", ops)])
        self.inclinaciones = [h[3]]
        return Plan(q0, [Fase("Abrir la mano", mano="abierta"), Fase("Ir a la herramienta", ir)],
                    nombre="llevar la mano a la herramienta")

    def _previa_pelota(self, q0, T_pelota, diametro, apoyo_z=None, apoyo_poligono=None, T_mundo_cam=None):
        """Como planificar_pelota (mismas formas de cogerla, en el mismo orden), hasta encima."""
        self._T_cam_medida = None if T_mundo_cam is None else np.asarray(T_mundo_cam, float)
        T_pelota = self.apoyar(T_pelota, diametro, apoyo_z)
        self.poner_mesa(apoyo_z if self._apoyo_ok and self._z_fijo_usado is None else None, apoyo_poligono)
        todos = {"pinza": ("en pinza", 0.0), "rodeándola": ("rodeándola", 0.0),
                 "con los dedos": ("con los dedos", self.pel_dedos)}
        modos = []
        for m in self.pel_modos:
            if m == "con los dedos" and self.pel_dedos <= 0:
                continue
            if m == "pinza" and self.pinza(diametro) is None:
                continue
            if m in todos:
                modos.append((m,) + todos[m])
        if not modos:
            modos = [("rodeándola", "rodeándola", 0.0)]
        errores = []
        inc_antes = self.inc_max
        self.inc_max = self.pel_inc_max
        try:
            for clave, nombre, extra in modos:
                self.inc_max = self.pinza_inc_max if (clave == "pinza" and self.pinza_arriba) else self.pel_inc_max
                try:
                    plan = self._previa_pelota_modo(q0, T_pelota, diametro, extra,
                                                    pinza=self.pinza(diametro) if clave == "pinza" else None)
                    plan.nombre = f"llevar la mano a la pelota ({nombre})"
                    self.modo_pelota = nombre
                    return plan
                except ErrorPlan as e:
                    errores.append((nombre, str(e)))
        finally:
            self.inc_max = inc_antes
        if len(errores) == 1:
            raise ErrorPlan(errores[0][1])
        raise ErrorPlan(errores[0][1] + " · tampoco " + " ni ".join(n for n, _ in errores[1:]))

    def _previa_pelota_modo(self, q0, T_pelota, diametro, extra=0.0, pinza=None):
        """La primera parte de _plan_pelota: ir encima, comprobando que desde ahí se puede bajar a
        cogerla y subir con ella (si no, esa postura de encima no vale)."""
        q0 = self._empezar(q0)
        r = float(diametro) / 2.0
        d_palma = r + self.pel_holgura + extra
        punto = OFFSET_PALMA_ORCA
        if pinza is not None:
            d_palma, punto = 0.0, np.asarray(pinza["punto"], float)
            self._posturas["pinza_abierta"] = dict(self.pinza_ab)
            self._posturas["pinza_contacto"] = dict(pinza["contacto"])
            self._nubes = {k: v for k, v in self._nubes.items() if not str(k).startswith("pinza")}
        k_ab = "pinza_abierta" if pinza is not None else "pelota_abierta"
        c = np.asarray(T_pelota, float)[:3, 3]
        T_p = self._marco_arriba(c)
        self.zonas_apoyo([(c, r + self.zona_apoyo, r)])
        elevar = self._elevar_pinza(pinza, r) if pinza is not None else 0.0
        if pinza is not None:
            abierta, cerrada = ("pinza_abierta", None), ("pinza_contacto",
                                                          self._pelota_en_mano(r, centro=self._bola_en_pinza(pinza, elevar)))
        else:
            abierta, cerrada = ("pelota_abierta", None), ("pelota_cerrada", self._pelota_en_mano(r, extra))

        def coger(q_l, a):
            centro = c + _unit(a) * d_palma + [0.0, 0.0, elevar]
            with self.con_mano(*abierta):
                qs = self.recta(q_l, punto, centro, "bajar a la pelota")
            with self.con_mano(*cerrada):
                for q in qs[::-1]:
                    m = self.motivo(q)
                    if m:
                        raise ErrorPlan(f"subir con la pelota: {m}")
            return qs

        self._mirar_obj = (c, r) if self.afinar_s > 0 else None
        try:
            with self.con_mano(*abierta):
                ops_p, info = self._opciones(T_p, q0, self._pose_pelota(T_p, d_palma + self.pel_subida, punto, pinza),
                                             coger, todas=True)
        finally:
            self._mirar_obj = None
        if not ops_p:
            raise self._con_aviso(str(self._no_encuentro("llegar a la pelota (ni desde arriba ni de lado)", info))
                                  + self._que_estorba(c, r), c)
        self._choques = []
        try:
            (p,), (ir_p,), _vuelta = self._combinar(
                q0, [("ir a la pelota", [(q, q, qs, inc) for q, qs, inc in ops_p], abierta)], mano_vuelta=abierta)
        except ErrorPlan as e:
            raise self._con_aviso(str(e) + self._que_estorba(c, r), c) from None
        self.inclinaciones = [p[3]]
        f_ir = Fase("Ir encima de la pelota", ir_p)
        f_ir.mano_chk = abierta
        plan = Plan(q0, [Fase("Abrir la mano", mano=k_ab), f_ir], nombre="llevar la mano a la pelota")
        plan.pelota = dict(c=c.copy(), r=r, d_palma=d_palma, extra=extra, punto=np.asarray(punto, float).copy(),
                           bajada=np.asarray(p[2], float).copy(), elevar=float(elevar))
        if pinza is not None:
            plan.posturas = dict(pinza_abierta=dict(self.pinza_ab))
        return plan

    def valida(self, fase, q):
        """motivo(q) con la mano como estaba en ese tramo del plan (para comprobar un plan hecho)."""
        with self.con_mano(*fase.mano_chk):
            return self.motivo(q)

    def plan_volver(self, q_actual, q_inicio):
        self.avisos = []
        self._reactivar_gpu()
        if self.huecos_desde_inicio:
            self.choque.fijar_inicio(np.asarray(q_actual, float))
        self._rodeo_resto, self._permitir_rodeo, self.rodeos = self.rodeo_total_s, True, 0
        try:
            qs = self.conectar(np.asarray(q_actual, float), np.asarray(q_inicio, float), None, "volver")
        finally:
            self._permitir_rodeo = False
        return Plan(q_actual, [Fase("Volver al inicio", qs)], nombre="volver al inicio")


# ══════════════════════════════════════════════════════════════════════════════
# LA TAREA (estados, órdenes de la web y FR5)
# ══════════════════════════════════════════════════════════════════════════════

ESTADOS = ("libre", "planificando", "lista", "ejecutando", "pausada", "terminada", "error")


_AVISO_CUROBO = None


def _envolvente(P2, res):
    """Triangulación (scipy Delaunay) de la envolvente convexa de los puntos 2D P2, un poco
    ampliada (medio cubo), o None si no hay bastantes."""
    P2 = np.asarray(P2, float).reshape(-1, 2)
    if len(P2) < 3:
        return None
    from scipy.spatial import Delaunay
    m = P2.mean(0)
    v = P2 - m
    n = np.linalg.norm(v, axis=1, keepdims=True)
    P2 = P2 + v / np.maximum(n, 1e-9) * (0.5 * res)
    try:
        return Delaunay(P2)
    except Exception:                                   # todos en línea
        return None


def _acepta(f, nombre):
    """¿La función f acepta el argumento 'nombre'? (planificadores de prueba con firmas antiguas)"""
    import inspect
    try:
        return nombre in inspect.signature(f).parameters
    except (TypeError, ValueError):
        return False


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
        # (18-nov) desvío del robot real respecto a lo mandado: en joints y en la PALMA (lo que importa con 2-3
        # cm de margen), comparando con lo mandado en la última ventana (el robot va algo por detrás)
        self.desvio_max = math.radians(float(_g(self.cfg, "AUTONOMO_TAREA_DESVIO_MAX_DEG", 4.0)))
        self.desvio_max_m = float(_g(self.cfg, "AUTONOMO_TAREA_DESVIO_MAX_M", 0.02))
        self.desvio_ventana_s = float(_g(self.cfg, "AUTONOMO_TAREA_DESVIO_VENTANA_S", 0.5))
        self.desvio_tiempo_s = float(_g(self.cfg, "AUTONOMO_TAREA_DESVIO_TIEMPO_S", 0.3))
        self._hist_cmd, self._t_hist = deque(), 0.0
        # (18-nov) reintentos del agarre (no la notan los táctiles, o un dedo toca algo al bajar)
        self.reintentos_max = int(_g(self.cfg, "AUTONOMO_TAREA_REINTENTOS", 2))
        self._reintentos = 0
        self._conservar_inicio = False
        self.bajada_tactil_n = float(_g(self.cfg, "AUTONOMO_TAREA_BAJADA_TACTIL_N", 0.5))
        self._fase_vigilada, self._base_bajada = None, None
        self.cierre_tactil = bool(_g(self.cfg, "AUTONOMO_TAREA_CIERRE_TACTIL", True))
        # (18-nov) tramos articulares con TOPP-RA (si está instalado: lo instala ./arrancar_nivel6.sh)
        self.toppra = bool(_g(self.cfg, "AUTONOMO_TAREA_TOPPRA", True))
        self.toppra_tol = math.radians(float(_g(self.cfg, "AUTONOMO_TAREA_TOPPRA_TOL_DEG", 0.5)))
        # (18-nov) el afinado (rehacer la bajada) en un hilo: el bucle de 60 Hz no se queda parado
        self.afinar_en_hilo = bool(_g(self.cfg, "AUTONOMO_TAREA_AFINAR_EN_HILO", True))
        self._hilo_afinar, self._res_afinar = None, None
        # (18-nov) la autocorrección de la calibración, congelada mientras hay una tarea (modo.py la avisa)
        self.congelar_autocalib_cfg = bool(_g(self.cfg, "AUTONOMO_AUTOCALIB_CONGELAR_EN_TAREA", True))
        # (18-nov) corrección fija de dónde se coge la pelota (derecha, lejos, arriba) vista desde la cámara
        self.corr_pelota = np.asarray(_g(self.cfg, "AUTONOMO_TAREA_PELOTA_CORRECCION_M", (0.0, 0.0, 0.0)), float)
        self.usar_mapa = bool(_g(self.cfg, "AUTONOMO_TAREA_USAR_MAPA", True))
        self.mirar = bool(_g(self.cfg, "AUTONOMO_TAREA_MIRAR", True))
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
        self.afinar_tapada_max = float(_g(self.cfg, "AUTONOMO_TAREA_AFINAR_TAPADA_MAX", 0.25))
        self.afinado_tapada = None                     # (18-nov) parte de la pelota tapada por el brazo al afinar
        self._T_cam_plan = None                         # (18-nov) cámara→mundo al planificar (ver _recolocar)
        self.afinado_recolocado_cm = None
        # (17-nov) el planificador (mallas de la mano, pinza, cuRobo en la GPU) se prepara al arrancar, en otro
        # hilo: así el primer «Planificar» no paga esos segundos
        self._lock_pl = threading.Lock()
        if fabrica_planificador is None and _g(self.cfg, "AUTONOMO_TAREA_PRECALENTAR", True):
            threading.Thread(target=self._precalentar, daemon=True, name="Tarea-precalentar").start()

    def _precalentar(self):
        try:
            t0 = time.monotonic()
            with self._lock_pl:
                if self._planificador is None:
                    self._planificador = self._crear_planificador()
                pl = self._planificador
                for d in (0.065, 0.07):
                    pl.pinza(d)
                for k in ("pelota_abierta", "pelota_cerrada"):
                    pl._nube_postura(k)
                if pl._choque_lote():
                    pl.choque._rejillas()                # mapas de distancias de los eslabones (~1 s)
            gpu = [t for t, si in (("cuRobo", getattr(pl, "curobo", None) is not None),
                                   ("comprobaciones", getattr(pl, "_lote_torch", None) is not None)) if si]
            print(f"🔥 Tarea: planificador listo en {time.monotonic() - t0:.1f} s"
                  + (f" ({' y '.join(gpu)} en la GPU)" if gpu else " (todo en la CPU)"))
        except Exception as e:
            print(f"⚠️  Tarea: no se pudo preparar el planificador por adelantado ({e}): se hará al planificar.")

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
            self._pedir_plan(contexto, volver=False)              # (un «Planificar» nuevo: sin reintentos)
        elif a == "ejecutar":
            self._ejecutar(contexto.get("q_real"))
        elif a == "cancelar":
            self._cancelar("Cancelada: el robot se queda donde está.")
        elif a == "volver":
            self._pedir_plan(contexto, volver=True)

    def _puede_planificar(self):
        return self.estado in ("libre", "lista", "terminada", "error", "pausada")

    def _pedir_plan(self, ctx, volver, mirar=True, reintento=False):
        """reintento (18-nov): lo pide la propia tarea tras subir para reintentar el agarre (o tras apartarse
        a mirar durante un reintento): se ejecuta solo (con el pedal) y conserva la postura de inicio."""
        if not self._puede_planificar():
            return
        if self.estado == "pausada":
            self._cancelar("")
        if not reintento:
            self._reintentos, self._conservar_inicio = 0, False
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
            # (18-nov) cámara→mundo con la que se midió la pelota (con la autocorrección de ese momento): al
            # afinar, si no se ve, se recoloca con la de entonces (ver _recolocar)
            tc = ctx.get("T_mundo_cam")
            self._T_cam_plan = np.asarray(tc, float) if tc is not None else None
            necesarios = [("la herramienta o la pelota (elígela en la lista)", herr)] + \
                ([] if pelota else [("el punto de uso", uso)]) + [("el punto de dejada", dej)]
            falta = [n for n, h in necesarios if h is None]
            if falta:
                return self._error("No veo " + ", ".join(falta) + " con la calibración cámara→robot.")
            sitios = [h for _, h in necesarios]
            T_sitios = [np.asarray(h["T_mundo"], float) for h in sitios]
            if pelota:
                T_sitios[0] = self._corregir_pelota(T_sitios[0], tc)
            inestable = [h["nombre"] for h in sitios if not h.get("estable")]
            if inestable:
                return self._error("Todavía no son estables: " + ", ".join(inestable) + ". Espera un momento.")
            datos = dict(volver=False, q=np.asarray(q, float), pelota=pelota,
                         T=T_sitios,
                         nombres=[h["nombre"] for h in sitios],
                         # lo que NO cuenta como obstáculo: solo el objeto (zona_excluida)
                         excluir=[z for h in sitios for z in zona_excluida(h, tc, self.cfg)],
                         diametro=float(herr.get("diametro_m") or 0.07) if pelota else None,
                         # (17-nov) altura de la mesa medida con un plano alrededor de la pelota (pelotas.py)
                         apoyo_z=herr.get("apoyo_z") if pelota else None,
                         apoyo_poligono=herr.get("apoyo_poligono") if pelota else None,
                         # (17-nov) si el brazo tapa el objetivo, primero apartarse y mirar (no en simulacro:
                         # ahí el robot real no se mueve y la cámara seguiría viendo lo mismo)
                         mirar=bool(mirar and not self.simulacro and self.mirar))
        datos["mapa"] = ctx.get("mapa") if self.usar_mapa else None
        datos["visto"] = ctx.get("visto") if self.usar_mapa else None
        if not volver:
            datos["T_mundo_cam"] = self._T_cam_plan
        self.estado, self.texto = "planificando", "Planificando…"
        self.plan, self.ejec, self._resultado = None, None, None
        self._auto = volver or reintento
        self._gen += 1
        datos["gen"] = self._gen
        self._hilo = threading.Thread(target=self._planificar, args=(datos,), daemon=True, name="Tarea-plan")
        self._hilo.start()

    def _planificar(self, d):
        t0 = time.monotonic()
        try:
            with self._lock_pl:                         # (si se está preparando al arrancar, se espera)
                if self._planificador is None:
                    self._planificador = (self._fabrica_plan() if self._fabrica_plan is not None
                                          else self._crear_planificador())
            pl = self._planificador
            res, centros = d["mapa"] if d.get("mapa") else (0.02, None)
            n_vox = pl.poner_mapa(centros, res, d.get("excluir", ()))
            if hasattr(pl, "poner_visto"):
                pl.poner_visto(d.get("visto"))
            plan = (pl.plan_mirar(d["q"], d["T"][0][:3, 3])
                    if d.get("mirar") and not d["volver"] and hasattr(pl, "plan_mirar") else None)
            if plan is not None:
                pass                                    # primero apartarse; al acabar se vuelve a planificar
            elif d["volver"]:
                plan = pl.plan_volver(d["q"], d["q_inicio"])
            elif d.get("pelota"):
                kw = ({"apoyo_z": d["apoyo_z"], "apoyo_poligono": d.get("apoyo_poligono")}
                      if d.get("apoyo_z") is not None else {})
                if d.get("T_mundo_cam") is not None and _acepta(pl.planificar_pelota, "T_mundo_cam"):
                    # (18-nov) para deslizarla por el rayo de esa cámara
                    kw["T_mundo_cam"] = d["T_mundo_cam"]
                plan = pl.planificar_pelota(d["q"], d["T"][0], d["diametro"], d["T"][1], **kw)
            else:
                plan = pl.planificar(d["q"], *d["T"])
            res = ("ok", plan, time.monotonic() - t0, n_vox)
            if _g(self.cfg, "AUTONOMO_TAREA_GUARDAR_PLANES", True):    # (17-nov) para revisar un plan raro
                self._guardar_fallo(d, "", plan=plan)
        except ErrorPlan as e:
            res = ("error", str(e) + self._guardar_fallo(d, str(e)), time.monotonic() - t0, 0)
        except Exception as e:                          # fallo de programa: que se vea
            import traceback
            traceback.print_exc()
            res = ("error", f"fallo al planificar: {e}" + self._guardar_fallo(d, repr(e)), time.monotonic() - t0, 0)
        if d["gen"] == self._gen:                       # si entretanto se canceló o se pidió otro, no vale
            self._resultado = res

    def _guardar_fallo(self, d, error, plan=None):
        """Guarda lo que se usó para planificar (pose, sitios, mapa) en diagnosticos/, para poder
        reproducir el fallo fuera del robot. Devuelve el texto a añadir al mensaje.
        plan (17-nov): un plan que SÍ ha salido → plan_ok_*.npz, con sus posturas (se guardan los
        AUTONOMO_TAREA_PLANES_GUARDADOS últimos), para revisar uno que no convence."""
        if plan is None and not _g(self.cfg, "AUTONOMO_TAREA_GUARDAR_FALLOS", True):
            return ""
        try:
            import json
            import os
            carpeta = os.path.join(self.raiz, "diagnosticos")
            os.makedirs(carpeta, exist_ok=True)
            ruta = os.path.join(carpeta, time.strftime(("plan_ok" if plan is not None else "plan_fallido")
                                                       + "_%Y%m%d_%H%M%S.npz"))
            res, centros = d["mapa"] if d.get("mapa") else (0.02, None)
            extra = {}
            if plan is not None:
                fs = [f for f in plan.fases if f.qs is not None]
                extra = dict(plan_qs=np.vstack([f.qs for f in fs]) if fs else np.zeros((0, 6)),
                             plan_n=np.array([len(f.qs) for f in fs], int),
                             plan_fases=np.array([f.nombre for f in fs]))
            np.savez_compressed(
                ruta, q=d["q"], T=np.array(d.get("T", []), float).reshape(-1, 4, 4),
                q_inicio=d.get("q_inicio", np.full(6, np.nan)),
                mapa=np.zeros((0, 3)) if centros is None else np.asarray(centros, float), res=float(res),
                excluir=np.array([[*np.asarray(c, float), r] for c, r in d.get("excluir", ())]).reshape(-1, 4),
                info=json.dumps(dict(error=error, nombres=d.get("nombres"), pelota=bool(d.get("pelota")),
                                     diametro=d.get("diametro"), volver=bool(d.get("volver")),
                                     # (18-nov) para reproducirlo igual en diagnosticos/banco_planes.py
                                     apoyo_z=d.get("apoyo_z"), apoyo_poligono=d.get("apoyo_poligono"),
                                     T_mundo_cam=None if d.get("T_mundo_cam") is None
                                     else np.asarray(d["T_mundo_cam"], float).tolist())), **extra)
            if plan is not None:
                import glob
                viejos = sorted(glob.glob(os.path.join(carpeta, "plan_ok_*.npz")))
                for v in viejos[:-int(_g(self.cfg, "AUTONOMO_TAREA_PLANES_GUARDADOS", 10))]:
                    os.remove(v)
                print(f"💾 Tarea: plan guardado en {ruta}")
                return ""
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

    def _fk_palma(self):
        """q → centro de la palma (mundo), del planificador de verdad (None con planificadores de prueba)."""
        return getattr(self._planificador, "_palma", None)

    def _fuerzas_dedos(self):
        """(18-nov) {dedo: fuerza normal (N) o None} de los táctiles, o None si no hay (o en simulacro)."""
        if self.tactil is None or self.simulacro:
            return None
        try:
            e = self.tactil() or {}
        except Exception:
            return None
        if e.get("estado") != "ok" or not e.get("dedos"):
            return None
        return {d.get("dedo"): d.get("fz") for d in e["dedos"]}

    def _nuevo_ejecutor(self, plan, mano0=None):
        return Ejecutor(plan, self.posturas, self.vel, self.acel, mano0,
                        (float(_g(self.cfg, "AUTONOMO_TAREA_T_ABRIR_S", 1.0)),
                         float(_g(self.cfg, "AUTONOMO_TAREA_T_CERRAR_S", 1.5))),
                        fk_palma=self._fk_palma(),
                        contactos=self._fuerzas_dedos if self.cierre_tactil else None,
                        toppra=self.toppra, toppra_tol_rad=self.toppra_tol)

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
        self._fase_vigilada = None
        self._hist_cmd.clear()
        self.ejec = self._nuevo_ejecutor(self.plan, mano0)
        self.estado, self.texto = "pausada", "Pisa el pedal para empezar."
        self._t_desvio = 0.0

    def _cancelar(self, texto):
        self._parar_fr5()
        self.ejec = None
        self.plan = None
        self._gen += 1                                  # un plan que esté calculándose ya no vale
        self._resultado = None
        self._auto = False
        self._hilo_afinar = None
        self._conservar_inicio = False
        self.estado, self.texto = "libre", texto or "Elige la herramienta y pulsa «Planificar»."

    def _error(self, texto):
        self._parar_fr5()
        self.ejec = None
        self._hilo_afinar = None
        self._conservar_inicio = False
        self.estado, self.texto = "error", texto
        print(f"⚠️  Tarea: {texto}")

    def _parar_fr5(self):
        if self._enviando and self.fr5 is not None:
            self.fr5.habilitar(False)
        self._enviando = False

    @property
    def afinando(self):
        """(18-nov) True mientras se rehace la bajada en su hilo (el robot espera quieto)."""
        return self._hilo_afinar is not None

    @property
    def congelar_autocalib(self):
        """(18-nov) Lo mira modo.py: mientras hay una tarea (de planificar a terminar), la cámara no aplica
        la autocorrección de la calibración (la pelota y el robot no se «mueven» a mitad de tarea)."""
        return self.congelar_autocalib_cfg and self.estado in ("planificando", "lista", "ejecutando", "pausada")

    # ── cada vuelta del bucle ──
    def paso(self, dt, q_real, pedal, contexto=None):
        """Devuelve la q que debe enseñar el gemelo en simulacro (o None: la pose real).
        contexto: como en orden() (para afinar la pelota al llegar encima de ella)."""
        if self.estado == "planificando" and self._resultado is not None:
            r, x, seg, n_vox = self._resultado
            self._resultado = None
            if r == "ok":
                self.plan = x
                apartarse = bool(getattr(x, "replanificar", False))
                if x.nombre != "volver al inicio" and not apartarse and not self._conservar_inicio:
                    self._q_inicio = x.q0.copy()
                dur = Ejecutor(x, self.posturas, self.vel, self.acel, fk_palma=self._fk_palma(),
                               toppra=self.toppra, toppra_tol_rad=self.toppra_tol).duracion
                self.estado = "lista"
                avisos = list(getattr(self._planificador, "avisos", None) or [])
                if apartarse:
                    self.texto = (f"El brazo le tapa a la cámara la mesa alrededor del objetivo: primero me aparto "
                                  f"para mirar (unos {dur:.0f} s) y después planifico la tarea con el mapa al día. "
                                  "Pulsa «Ejecutar».")
                    print(f"🗺️  Tarea: {self.texto}")
                    if self._auto:                      # (18-nov) durante un reintento, sigue solo
                        self._auto = False
                        self._ejecutar(q_real)
                    return None
                inc = [i for i in (getattr(self._planificador, "inclinaciones", None) or []) if i]
                self.texto = (f"Plan listo: {len(x.fases)} tramos, unos {dur:.0f} s "
                              f"(calculado en {seg:.1f} s, {n_vox} cubos del mapa)"
                              + (f"; llega de lado, hasta {max(inc):.0f}° de la perpendicular" if inc else "")
                              + (f"; rodea obstáculos en {nr} tramo{'s' if nr > 1 else ''}"
                                 if (nr := getattr(self._planificador, "rodeos", 0)) else "")
                              + (f"; cuRobo (GPU) en {nc} tramo{'s' if nc > 1 else ''}"
                                 if (nc := getattr(self._planificador, "tramos_curobo", 0)) else "")
                              + (f" ({ng} eligiendo la mejor de varias poses a la vez)"
                                 if (ng := getattr(self._planificador, "goalsets_curobo", 0)) else "")
                              + (f" ({nrc} propuesto{'s' if nrc > 1 else ''} por cuRobo no pasaron las comprobaciones)"
                                 if (nrc := getattr(self._planificador, "curobo_rechazos", 0)) else "")
                              + (f"; la coge {self._planificador.modo_pelota}"
                                 if getattr(self._planificador, "modo_pelota", None) and x.pelota is not None else "")
                              + ("; el punto de dejada está de pie: la soltará delante del marcador"
                                 if x.nombre.endswith("pelota") and getattr(self._planificador, "dejar_de_pie", False)
                                 else "")
                              + (f"; reintento {self._reintentos} de {self.reintentos_max}"
                                 if self._conservar_inicio and self._reintentos else "")
                              + ". " + ("" if not avisos else "⚠️ " + " · ".join(avisos) + ". ")
                              + ("Sigo (pisa el pedal)." if self._auto else "Pulsa «Ejecutar»."))
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
        # (18-nov) ¿ha terminado el afinado que iba en su hilo? Se aplica aquí, en el bucle principal
        if self._hilo_afinar is not None and not self._hilo_afinar.is_alive():
            hilo, res = self._hilo_afinar, self._res_afinar
            self._hilo_afinar, self._res_afinar = None, None
            if getattr(hilo, "ejec", None) is self.ejec:
                fallo = self._aplicar_afinado(res)
                if fallo:
                    self._error(fallo)
                    return None
        if pedal:
            if self.mover_mano and not self._mano_lista():
                self.estado, self.texto = "ejecutando", "Conectando la mano con par…"
                return self._q_gemelo()
            if not self._enviando:
                fr5.publicar(self.ejec.q())
                fr5.habilitar(True)                     # ServoMoveStart desde la pose real
                self._enviando = True
                self._hist_cmd.clear()
                self._t_desvio = 0.0
            from skill_shield.hardware.fr5_servo import ENVIANDO
            listo = self.simulacro or fr5.estado()["estado"] == ENVIANDO
            f_act = self.ejec.fase
            if listo and f_act is not None and f_act.comprobar and self._comprobada != self.ejec.i:
                self._comprobada = self.ejec.i
                fallo = self._comprobar(f_act.comprobar)
                if fallo:
                    if f_act.comprobar == "agarre":     # (18-nov) no la tiene: abrir, subir y reintentar
                        return self._reintentar(fallo)
                    self._error(fallo)
                    return None
            if listo and f_act is not None and getattr(f_act, "vigilar_tactil", False):
                toque = self._toque_al_bajar()          # (18-nov) bajada vigilada con los táctiles
                if toque:
                    return self._reintentar(toque)
            if listo and self._hilo_afinar is None:
                i0 = self.ejec.i
                self.ejec.avanzar(dt)
                if (self.ejec.i > i0 and self.plan.fases[i0].comprobar == "afinar"):
                    fallo = self._lanzar_afinar(contexto)
                    if fallo:
                        self._error(fallo)
                        return None
                if not self.simulacro and q_real is not None:
                    fallo = self._desvio(q_real, dt)
                    if fallo:
                        self._error(fallo)
                        return None
            fr5.publicar(self.ejec.q())
            self.estado = "ejecutando"
            f = self.ejec.fase
            if self._hilo_afinar is not None:
                self.texto = "Afinando con la cámara (el robot espera quieto)…"
            else:
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
        if self.ejec.terminado and getattr(self.plan, "replanificar", False):
            # (17-nov) ya se ha apartado y ha mirado: ahora sí, el plan de la tarea con el mapa al día.
            # (18-nov) también al acabar de subir para reintentar el agarre
            reintento = bool(getattr(self.plan, "reintento", False))
            self._parar_fr5()
            q = self.ejec.q()
            self.ejec, self.plan = None, None
            self.estado, self.texto = "terminada", "Ya veo la mesa: planificando la tarea…"
            try:
                ctx = contexto() if callable(contexto) else contexto
            except Exception:
                ctx = None
            if ctx:
                self._pedir_plan(ctx, volver=False, mirar=reintento, reintento=self._conservar_inicio)
            else:
                self.texto = "Ya veo la mesa: pulsa «Planificar»."
            return q if self.simulacro else None
        if self.ejec.terminado:
            self._parar_fr5()
            nombre = self.plan.nombre
            info = getattr(self.ejec, "cierre_info", None)
            self.estado, self.texto = "terminada", f"Hecho: {nombre}." + (
                f" Cerró con los táctiles ({', '.join(info['tocados']) or 'ningún dedo notó la pelota'})." if info else "")
            self._conservar_inicio = False
            print(f"✅ Tarea: {self.texto}")
            q = self.ejec.q()
            self.ejec = None
            return q if self.simulacro else None
        return self._q_gemelo()

    def _desvio(self, q_real, dt):
        """(18-nov) Texto si el robot real no sigue lo mandado; None si va bien. Se compara con lo mandado en
        la última ventana (desvio_ventana_s: el robot va algo por detrás) en joints y en la palma."""
        self._t_hist += dt
        qc = self.ejec.q()
        fk = self._fk_palma()
        self._hist_cmd.append((self._t_hist, qc, None if fk is None else fk(qc)))
        while self._hist_cmd and self._t_hist - self._hist_cmd[0][0] > self.desvio_ventana_s:
            self._hist_cmd.popleft()
        qr = np.asarray(q_real, float)
        Q = np.array([h[1] for h in self._hist_cmd])
        dj = float(np.min(np.max(np.abs(Q - qr), axis=1)))
        dm = 0.0
        if fk is not None:
            P = np.array([h[2] for h in self._hist_cmd])
            dm = float(np.min(np.linalg.norm(P - fk(qr), axis=1)))
        if dj > self.desvio_max or dm > self.desvio_max_m:
            self._t_desvio += dt
            if self._t_desvio > self.desvio_tiempo_s:
                return (f"El robot no sigue lo mandado ({math.degrees(dj):.1f}° en joints, {100 * dm:.1f} cm en la "
                        f"palma; máx. {math.degrees(self.desvio_max):.0f}° y {100 * self.desvio_max_m:.0f} cm): "
                        "tarea abortada.")
        else:
            self._t_desvio = 0.0
        return None

    def _toque_al_bajar(self):
        """(18-nov) Bajando a coger: si algún dedo nota más de bajada_tactil_n que al empezar la bajada, ha tocado
        algo (la pelota descolocada, la mesa): texto, o None."""
        fz = self._fuerzas_dedos()
        if not fz:
            return None
        if self._fase_vigilada != self.ejec.i:
            self._fase_vigilada, self._base_bajada = self.ejec.i, dict(fz)
            return None
        toque = [d for d, v in fz.items() if v is not None and self._base_bajada.get(d) is not None
                 and float(v) - float(self._base_bajada[d]) >= self.bajada_tactil_n]
        return f"Al bajar, {' y '.join(toque)} ha tocado algo" if toque else None

    def _reintentar(self, motivo):
        """(18-nov) El agarre no ha salido (los táctiles no notan la pelota, o un dedo tocó algo al bajar): en vez
        de parar, se abre la mano, se sube por donde bajó y se vuelve a planificar con lo que ve ahora la cámara
        (hasta reintentos_max veces). Sigue haciendo falta el pedal en todo momento."""
        pz = getattr(self.plan, "pelota", None)
        if pz is None or "i_bajar" not in pz or self._reintentos >= self.reintentos_max:
            self._error(motivo + (": tarea parada." if self._reintentos == 0 else
                                  f" (tras {self._reintentos} reintentos): tarea parada.") + " La mano se queda como está.")
            return None
        self._reintentos += 1
        q = self.ejec.q()
        bajada = np.asarray(self.plan.fases[pz["i_bajar"]].qs, float)
        k = int(np.argmin(np.max(np.abs(bajada - q), axis=1)))
        subida = np.vstack([q[None], bajada[:k + 1][::-1]])
        if len(subida) < 2:
            subida = np.vstack([q, q])
        f_subir = Fase("Subir para reintentar", subida)
        if hasattr(self._planificador, "_recta_lenta"):
            self._planificador._recta_lenta(f_subir, llegada=False)
        plan = Plan(q, [Fase("Abrir la mano", mano=pz.get("k_abierta", "abierta")), f_subir],
                    nombre="subir para reintentar")
        plan.posturas = dict(getattr(self.plan, "posturas", None) or {})
        plan.replanificar, plan.reintento = True, True
        mano0 = self.ejec.mano()
        self.plan = plan
        self.ejec = self._nuevo_ejecutor(plan, mano0)
        self._comprobada, self._fase_vigilada = None, None
        self._hist_cmd.clear()
        self._conservar_inicio = True
        self.texto = f"{motivo}: abro, subo y lo vuelvo a intentar ({self._reintentos}/{self.reintentos_max})."
        print(f"🔁 Tarea: {self.texto}")
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
            return f"No noto la pelota en los dedos (solo {n} con más de {self.tactil_n:g} N)"
        print(f"✋ Tarea: pelota cogida ({n} dedos la notan).")
        return None

    # ── afinar (12-nov; 18-nov: en un hilo) ──
    def _lanzar_afinar(self, contexto):
        """Al acabar la parada de «Afinar con la cámara». Con afinar_en_hilo, el cálculo va en otro hilo y el
        robot espera quieto (paso() lo aplica al acabar); si no, aquí mismo. Texto si hay que parar."""
        if not self.afinar_en_hilo:
            return self._aplicar_afinado(self._calcular_afinado(contexto))
        self._res_afinar = None

        def _hilo():
            try:
                self._res_afinar = self._calcular_afinado(contexto)
            except Exception as e:                       # un fallo al afinar no para la tarea: sigue con el plan
                import traceback
                traceback.print_exc()
                print(f"⚠️  Tarea: fallo al afinar ({e}): sigo con el plan.")
                self._res_afinar = None
        h = threading.Thread(target=_hilo, daemon=True, name="Tarea-afinar")
        h.ejec = self.ejec                               # si entretanto se cancela, el resultado no vale
        self._hilo_afinar = h
        h.start()
        return None

    def _aplicar_afinado(self, res):
        """res de _calcular_afinado: None (sigue con el plan), ("error", texto) o ("bajada", qs, c2, n, recolocado)."""
        if res is None or self.ejec is None or self.plan is None:
            return None
        if res[0] == "error":
            return res[1]
        _, qs, c2, n, recolocado = res
        pz = self.plan.pelota
        self.ejec.cambiar_tramo(pz["i_bajar"], qs)
        self.ejec.cambiar_tramo(pz["i_subir"], qs[::-1])
        dc = np.asarray(c2, float) - pz["c"]
        pz["c"] = np.asarray(c2, float)
        self.afinado_cm = round(n * 100, 1)
        if recolocado:
            print(f"🎯 Tarea: afinado sin verla: con la corrección de la cámara junto al brazo, la pelota está "
                  f"{n * 100:.1f} cm desplazada (x, y, z: {', '.join(f'{100 * v:+.1f}' for v in dc)} cm) → bajada corregida.")
        else:
            print(f"🎯 Tarea: afinado con la cámara: la pelota estaba {n * 100:.1f} cm desplazada → bajada corregida.")
        return None

    def _calcular_afinado(self, contexto):
        """(12-nov) Encima de la pelota y con el brazo quieto: la cámara ha vuelto a medir la pelota. Si está
        en otro sitio que al planificar, se rehace la bajada hacia donde está de verdad.
        Devuelve None (sigue con el plan), ("error", texto) o ("bajada", qs, c2, n, recolocado). No toca el
        Ejecutor (puede ir en otro hilo)."""
        pz = getattr(self.plan, "pelota", None)
        if pz is None or self._planificador is None:
            return None
        # (18-nov) si el propio brazo tapa la pelota a la cámara (con la cámara detrás del robot, el antebrazo
        # suele quedar justo delante al pararse encima), lo que mide ahora no vale: se sigue con el plan
        self.afinado_tapada = None
        self.afinado_recolocado_cm = None
        try:
            ctx = contexto() if callable(contexto) else (contexto or {})
        except Exception:
            ctx = {}
        ctx = ctx or {}
        q_aqui = self.ejec.q() if self.ejec is not None else None
        f = None
        if q_aqui is not None and hasattr(self._planificador, "fraccion_tapada"):
            try:
                f = self._planificador.fraccion_tapada(q_aqui, pz["c"], pz["r"])
            except Exception as e:
                print(f"⚠️  Tarea: no puedo mirar si el brazo tapa la pelota: {e}")
        if f is not None:
            self.afinado_tapada = round(f, 2)
            if f > self.afinar_tapada_max:
                print(f"⚠️  Tarea: al afinar, el brazo tapa el {100 * f:.0f} % de la pelota a la cámara: no me fío "
                      "de lo que mide ahora.")
                return self._recolocar(pz, ctx)
        pelotas = [h for h in (ctx.get("herramientas") or []) if h.get("tipo") == "pelota" and h.get("T_mundo")
                   is not None and (h.get("edad_s") or 0.0) <= 0.5]
        h = next((h for h in pelotas if h.get("id") == self._pelota_id), None)
        if h is None and pelotas:                       # perdió el seguimiento: la más cercana…
            h = min(pelotas, key=lambda h: np.linalg.norm(np.asarray(h["T_mundo"])[:3, 3] - pz["c"]))
            if np.linalg.norm(np.asarray(h["T_mundo"])[:3, 3] - pz["c"]) > self.afinar_max:
                h = None                                # (18-nov) …si está cerca: lejos es OTRA cosa roja
        if h is None:
            print("⚠️  Tarea: al afinar no veo la pelota (¿la tapa la mano?).")
            return self._recolocar(pz, ctx)
        T_h = self._corregir_pelota(np.asarray(h["T_mundo"], float), ctx.get("T_mundo_cam"))
        with self._lock_pl:
            c2 = self._planificador.apoyar(T_h, 2 * pz["r"], h.get("apoyo_z"),
                                           **({"T_mundo_cam": ctx["T_mundo_cam"]} if ctx.get("T_mundo_cam") is not None
                                              and _acepta(self._planificador.apoyar, "T_mundo_cam") else {}))[:3, 3]
        n = float(np.linalg.norm(c2 - pz["c"]))
        if n < self.afinar_min:
            print(f"🎯 Tarea: afinado: la pelota está donde se planificó ({n * 100:.1f} cm).")
            return None
        if n > self.afinar_max:
            return ("error", f"La pelota está a {n * 100:.0f} cm de donde estaba al planificar (¿se ha movido?): "
                             "tarea parada. Vuelve a planificar.")
        try:
            with self._lock_pl:
                qs = self._planificador.rehacer_bajada(self.plan, c2)
        except ErrorPlan as e:
            return ("error", f"Al afinar (la pelota está {n * 100:.1f} cm desplazada) no puedo rehacer la bajada: {e}")
        return ("bajada", qs, c2, n, False)

    def _corregir_pelota(self, T, T_mundo_cam=None):
        """(18-nov) Corrección FIJA de dónde se coge la pelota, AUTONOMO_TAREA_PELOTA_CORRECCION_M = (derecha, lejos,
        arriba) en m, tal como se ve en la imagen de la cámara de cabeza (la de la web): «derecha» = hacia la
        derecha de la imagen, en horizontal; «lejos» = alejándose de la cámara, en horizontal; «arriba» = vertical.
        Si la mano la coge 1,5 cm a la DERECHA (vista en la cámara), pon derecha = -0.015."""
        T = np.array(T, float)
        if not np.any(self.corr_pelota):
            return T
        Tc = T_mundo_cam if T_mundo_cam is not None else (
            getattr(self._planificador, "T_cam", None))
        if Tc is None:
            print("⚠️  Tarea: sin calibración de la cámara no puedo aplicar AUTONOMO_TAREA_PELOTA_CORRECCION_M.")
            return T
        R = np.asarray(Tc, float)[:3, :3]
        der = np.array([R[0, 0], R[1, 0], 0.0])           # x de la cámara (derecha de la imagen), en horizontal
        der /= max(float(np.linalg.norm(der)), 1e-9)
        lej = np.array([R[0, 2], R[1, 2], 0.0])           # z de la cámara (hacia donde mira), en horizontal
        lej -= der * float(lej @ der)
        lej /= max(float(np.linalg.norm(lej)), 1e-9)
        d, l, a = (float(v) for v in self.corr_pelota)
        T[:3, 3] += d * der + l * lej + np.array([0.0, 0.0, a])
        return T

    def _recolocar(self, pz, ctx):
        """(18-nov) La pelota no se ve bien ahora (la tapa el brazo): no se puede volver a medir, pero SÍ volver a
        situar lo que se midió al planificar. Con el brazo parado al lado de la pelota, la cámara corrige su
        calibración alineando la profundidad del brazo con su modelo (autocalib): esa corrección, hecha a la
        distancia de la pelota, recoge tanto el error de la calibración en esa zona como el de la profundidad del
        estéreo (que a ~0,9 m mide algún cm de más). La pelota medida al planificar, pasada a la cámara con la
        transformación de ENTONCES y vuelta al mundo con la de AHORA, queda donde está de verdad.
        (Con la autocorrección congelada durante la tarea, AUTONOMO_AUTOCALIB_CONGELAR_EN_TAREA, no cambia: se
        sigue con el plan.)
        Devuelve None (se sigue con la bajada del plan) o ("bajada", qs, c2, n, True); nunca para la tarea."""
        T_ant, T_now = self._T_cam_plan, ctx.get("T_mundo_cam") if ctx else None
        if T_ant is None or T_now is None:
            print("   Sigo con el plan (la pelota, donde se vio al planificar).")
            return None
        D = np.asarray(T_now, float) @ np.linalg.inv(T_ant)     # lo que ha cambiado la corrección desde el plan
        c2 = D[:3, :3] @ pz["c"] + D[:3, 3]
        n = float(np.linalg.norm(c2 - pz["c"]))
        self.afinado_recolocado_cm = round(n * 100, 1)
        if n < self.afinar_min:
            print(f"   La corrección de la cámara apenas ha cambiado desde el plan ({n * 100:.1f} cm): sigo con el plan.")
            return None
        if n > self.afinar_max:
            print(f"⚠️  La corrección de la cámara ha cambiado {n * 100:.0f} cm desde el plan (demasiado: no me fío): "
                  "sigo con el plan.")
            return None
        try:
            with self._lock_pl:
                qs = self._planificador.rehacer_bajada(self.plan, c2)
        except ErrorPlan as e:
            print(f"⚠️  Con la corrección de la cámara la pelota estaría {n * 100:.1f} cm desplazada, pero no puedo "
                  f"rehacer la bajada ({e}): sigo con el plan.")
            return None
        return ("bajada", qs, c2, n, True)

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
        avisos = list(getattr(self._planificador, "avisos", None) or []) if self.estado != "libre" else []
        return dict(estado=self.estado, texto=self.texto, simulacro=self.simulacro, avisos=avisos,
                    reintentos=self._reintentos if self._conservar_inicio else 0,
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
