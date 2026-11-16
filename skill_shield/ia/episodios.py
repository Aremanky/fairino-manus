"""
Datos de la IA (16-nov): episodios grabados con [G] → lo que VE y lo que HACE la red. Sin torch.

Lo usan el entrenamiento (act.py), la reproducción (politica.py), la evaluación (evaluar.py) y
las pruebas. La idea clave: las MISMAS funciones construyen las entradas al entrenar (un episodio
entero) y al reproducir (un instante), así la red ve exactamente lo mismo en los dos casos.

Formatos de episodio (episodios/<tarea>/episodio_XXXX.csv + .json):
  1 · solo mano, sin cabecera: 17 qpos + 17 ctrl (+ t)                                (antiguo)
  2 · cabecera t, t_reloj, q:<joint>…, a:<joint>…   (6 del FR5 + 17 de la OrcaHand, simulados)
  3 · formato 2 + lo que hay en el mundo (grabador.py + fuentes.py). NaN = no había dato:
        br:<joint>             postura REAL del FR5 (rad, convenio del MJCF)
        hr:<joint>, hc:<joint> postura REAL de la OrcaHand (rad, convenio del sim) y su corriente
                               (fracción de max_current)
        tac:<dedo>:fx|fy|fz    fuerza de cada yema (N), por dedo FÍSICO (thumb = pulgar…: ya
                               corregido con ORCA_TACTIL_SENSOR_DE_DEDO)
        obj:id, obj:x…obj:qz, obj:tam   pose en el mundo (marco del MJCF) del objeto «objetivo»
                               (el más cercano a la palma al empezar) y su tamaño (m)
      y en el .json: exito (True salvo que se marque con [M]), objeto (nombre, tipo, clase)…

Qué VE la red (la «receta», que se guarda dentro del .pth):
  estado  : postura de la mano (17) [+ brazo (6) en modo "joints"] + táctiles (5 + 1 bandera)
  entorno : el objeto VISTO DESDE LA MANO (posición 3 + giro 6D) + tamaño + clase (one-hot).
            La pose del objeto se CONGELA al empezar, porque la mano lo tapa al acercarse: al
            entrenar, la de las primeras detecciones del episodio; al reproducir, la del arranque.
Qué HACE la red (un bloque de CHUNK fotogramas):
  "joints"         : joints del brazo + de la mano (absolutos), como antes.
  "mano_en_objeto" : dónde poner mano_mount EN EL MARCO DEL OBJETO (3 + giro 6D) + mano (17). La
                     IK analítica lo pasa a joints al reproducir. Si el objeto está en otro sitio,
                     el gesto se traslada con él: generaliza con muchas menos demos.
Las pelotas usan su centro con los ejes del mundo (una esfera no tiene giro que signifique nada).
"""
import csv
import json
import os
import re

import numpy as np

try:
    import config as _cfg
except Exception:                       # pruebas sueltas sin config.py
    _cfg = None

from skill_shield.teleop_fr5.cinematica import T_de, quat_a_mat

FORMATO_ACTUAL = 3
DEDOS = ("thumb", "index", "middle", "ring", "pinky")      # mismo orden que hardware/tactil_real.py
EJES_TACTIL = ("fx", "fy", "fz")
COLS_OBJ = ("id", "x", "y", "z", "qw", "qx", "qy", "qz", "tam")
TIPOS_COGIBLES = ("pelota", "herramienta")
TACTIL_ESCALA_N = 5.0          # las fuerzas entran a la red divididas por esto (≈ 0…4)
SIGMA_MIN = 1e-2               # desviación mínima al normalizar (un joint casi quieto no se dispara)
GRUPOS_FIJOS = ("tactil", "tactil_ok", "obj_clase")   # ya vienen en su escala: no se normalizan

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _p(nombre, defecto):
    return getattr(_cfg, nombre, defecto) if _cfg is not None else defecto


class SinObjeto(ValueError):
    """La receta necesita el objeto y no lo hay (episodio sin detecciones o arranque sin verlo)."""


# ══════════════════════════════════════════════════════════════════════════════
# CARPETAS Y NOMBRES
# ══════════════════════════════════════════════════════════════════════════════

def _tiene_episodios(carpeta):
    if not os.path.isdir(carpeta):
        return False
    for t in os.listdir(carpeta):
        r = os.path.join(carpeta, t)
        if os.path.isdir(r) and any(f.startswith("episodio_") and f.endswith(".csv") for f in os.listdir(r)):
            return True
    return False


def directorios(avisar=True):
    """(carpeta de episodios, carpeta de modelos). 16-nov: FUERA del repo (los datos crecen mucho,
    y más cuando lleven imagen). Orden: variable de entorno SKILL_SHIELD_DATOS → IA_DATOS_DIR de
    config.py → ~/skill_shield_datos. Si ahí todavía no hay episodios y en el repo quedan los de
    antes (episodios/), se siguen usando los del repo (y se avisa): nada se rompe al actualizar."""
    base = (os.environ.get("SKILL_SHIELD_DATOS") or _p("IA_DATOS_DIR", None)
            or os.path.join(os.path.expanduser("~"), "skill_shield_datos"))
    base = os.path.abspath(os.path.expanduser(str(base)))
    ep, mo = os.path.join(base, "episodios"), os.path.join(base, "modelos")
    viejo_ep, viejo_mo = os.path.join(_RAIZ, "episodios"), os.path.join(_RAIZ, "modelos")
    if base != _RAIZ and not _tiene_episodios(ep) and _tiene_episodios(viejo_ep):
        if avisar:
            print(f"ℹ️  IA: uso los episodios que hay dentro del repo ({viejo_ep}). Para sacarlos "
                  f"(recomendado): mkdir -p {base} && mv episodios modelos {base}/ && "
                  "git rm -r --cached episodios modelos")
        return viejo_ep, viejo_mo
    return ep, mo


def nombre_seguro(texto: str) -> str:
    """Nombre de tarea escrito por el usuario → nombre de carpeta/archivo seguro."""
    limpio = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(texto).strip())
    return limpio.lower() or "tarea_sin_nombre"


def listar_episodios(carpeta):
    if not os.path.isdir(carpeta):
        return []
    return sorted(os.path.join(carpeta, f) for f in os.listdir(carpeta)
                  if f.startswith("episodio_") and f.endswith(".csv"))


def listar_tareas(episodios_dir):
    """Subcarpetas de tareas con al menos 1 episodio."""
    if not os.path.isdir(episodios_dir):
        return []
    return [n for n in sorted(os.listdir(episodios_dir))
            if os.path.isdir(os.path.join(episodios_dir, n)) and listar_episodios(os.path.join(episodios_dir, n))]


def listar_modelos(modelos_dir):
    """Nombres (sin .pth) de los modelos entrenados."""
    if not os.path.isdir(modelos_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(modelos_dir) if f.endswith(".pth"))


# ══════════════════════════════════════════════════════════════════════════════
# LEER EPISODIOS
# ══════════════════════════════════════════════════════════════════════════════

def _flotante(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def leer_meta(ruta_csv):
    r = ruta_csv[:-4] + ".json"
    if not os.path.exists(r):
        return {}
    try:
        with open(r, encoding="utf-8") as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def _n_brazo(nombres, meta):
    if meta.get("n_brazo") is not None:
        return int(meta["n_brazo"])
    brazo = set(_p("ARM_JOINTS", [f"joint{k}" for k in range(1, 7)]))
    n = 0
    for nm in nombres:
        if nm not in brazo:
            break
        n += 1
    return n


def leer_episodio(ruta):
    """Lee un episodio de cualquier formato → dict(ruta, formato, fps, t, nombres, n_brazo, q, a,
    br, hr, hc, tac (T, 5, 3), obj (T, 9), meta) con None en lo que no haya, o None si no se puede."""
    try:
        with open(ruta, newline="") as f:
            filas = [r for r in csv.reader(f) if r]
    except OSError:
        return None
    if not filas:
        return None
    meta = leer_meta(ruta)
    if filas[0][0] == "t":                                           # ── formatos 2 y 3
        cab = filas[0]
        datos = np.array([[_flotante(v) for v in r] for r in filas[1:] if len(r) == len(cab)], dtype=float)
        if datos.size == 0:
            return None
        datos = datos.reshape(-1, len(cab))
        col = {c: i for i, c in enumerate(cab)}

        def grupo(prefijo):
            idx = [i for i, c in enumerate(cab) if c.startswith(prefijo)]
            return [cab[i][len(prefijo):] for i in idx], (datos[:, idx] if idx else None)

        nombres, q = grupo("q:")
        nombres_a, a = grupo("a:")
        if q is None or a is None or nombres != nombres_a:
            return None
        br, hr, hc = grupo("br:")[1], grupo("hr:")[1], grupo("hc:")[1]
        tac = None
        if any(c.startswith("tac:") for c in cab):
            tac = np.full((len(datos), len(DEDOS), 3), np.nan)
            for k, d in enumerate(DEDOS):
                for j, eje in enumerate(EJES_TACTIL):
                    i = col.get(f"tac:{d}:{eje}")
                    if i is not None:
                        tac[:, k, j] = datos[:, i]
        obj = None
        if any(c.startswith("obj:") for c in cab):
            obj = np.full((len(datos), len(COLS_OBJ)), np.nan)
            for j, c in enumerate(COLS_OBJ):
                i = col.get(f"obj:{c}")
                if i is not None:
                    obj[:, j] = datos[:, i]
        formato = 3 if any(x is not None for x in (br, hr, hc, tac, obj)) else 2
        fps = meta.get("fps")
        return dict(ruta=ruta, formato=formato, fps=float(fps) if fps else None, t=datos[:, 0],
                    nombres=nombres, n_brazo=_n_brazo(nombres, meta), q=q, a=a, br=br, hr=hr, hc=hc,
                    tac=tac, obj=obj, meta=meta)
    n = int(_p("N_ACT", 17))                                         # ── formato 1 (solo mano)
    q, a, t = [], [], []
    for r in filas:
        if len(r) in (2 * n, 2 * n + 1):
            v = [_flotante(x) for x in r]
            q.append(v[:n])
            a.append(v[n:2 * n])
            if len(r) == 2 * n + 1:
                t.append(v[-1])
    if not q:
        return None
    return dict(ruta=ruta, formato=1, fps=None, t=np.array(t) if len(t) == len(q) else None,
                nombres=None, n_brazo=0, q=np.array(q), a=np.array(a), br=None, hr=None, hc=None,
                tac=None, obj=None, meta=meta)


def exito(ep):
    """False si el episodio se marcó como fallido con [M] (no se usa para entrenar)."""
    return bool(ep.get("meta", {}).get("exito", True))


def compatibles(episodios):
    """(episodios que se pueden mezclar en un modelo, avisos). Formato 1 (solo mano) no se mezcla
    con 2/3 (brazo + mano): si hay de los dos se usan los nuevos. Tampoco otras articulaciones."""
    eps = [e for e in episodios if e is not None]
    avisos = []
    if any(e["formato"] >= 2 for e in eps):
        viejos = sum(1 for e in eps if e["formato"] == 1)
        eps = [e for e in eps if e["formato"] >= 2]
        ref = eps[0]["nombres"]
        distintos = sum(1 for e in eps if e["nombres"] != ref)
        eps = [e for e in eps if e["nombres"] == ref]
        if viejos:
            avisos.append(f"{viejos} episodio(s) del formato antiguo (solo mano) ignorados: este "
                          "modelo es de brazo + mano.")
        if distintos:
            avisos.append(f"{distintos} episodio(s) con otras articulaciones ignorados.")
    return eps, avisos


def fps_de(episodios):
    """Ritmo de grabación: el del .json (formatos 2/3) o la mediana de los tiempos (formato 1)."""
    for e in episodios:
        if e.get("fps"):
            return float(e["fps"])
    dts = []
    for e in episodios:
        t = e.get("t")
        if t is not None and len(t) > 1:
            dts.extend(np.diff(t).tolist())
    if dts:
        dt = float(np.median(dts))           # mediana: robusta a alguna pausa
        if 0.01 <= dt <= 0.5:
            return 1.0 / dt
    return 15.0


# ══════════════════════════════════════════════════════════════════════════════
# GIROS Y OBJETOS
# ══════════════════════════════════════════════════════════════════════════════

def mat_a_quat(R):
    """Matriz 3x3 → cuaternión (w, x, y, z) (método de Shepperd)."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def rot6d(R):
    """Giro → 6 números (las dos primeras columnas). Continuo, a diferencia de Euler/cuaterniones:
    mejor para una red. Admite (..., 3, 3)."""
    R = np.asarray(R, float)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def R_de_rot6d(v):
    """Inversa de rot6d (Gram-Schmidt): siempre da una rotación válida aunque v no sea exacto."""
    v = np.asarray(v, float)
    a1, a2 = v[:3], v[3:6]
    b1 = a1 / max(np.linalg.norm(a1), 1e-9)
    a2 = a2 - b1 * float(b1 @ a2)
    b2 = a2 / max(np.linalg.norm(a2), 1e-9)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def _inv_T_lote(T):
    """Inversa de un lote de transformaciones (N, 4, 4)."""
    Rt = np.swapaxes(T[:, :3, :3], 1, 2)
    out = np.zeros_like(T)
    out[:, :3, :3] = Rt
    out[:, :3, 3] = -np.einsum("nij,nj->ni", Rt, T[:, :3, 3])
    out[:, 3, 3] = 1.0
    return out


def clase_de(nombre):
    """«Pelota roja 2» → «pelota roja»: la clase, sin el número que se añade si hay varias iguales."""
    return re.sub(r"\s+\d+$", "", str(nombre or "objeto").strip()).lower() or "objeto"


def objeto_desde_deteccion(d):
    """Una entrada de la lista de la cámara (pelotas.py, herramientas.py) o del simulador →
    dict(T 4x4 en el mundo, tam, nombre, clase, tipo, id, clave, estable), o None sin pose."""
    T = d.get("T_mundo")
    if T is None:
        return None
    T = np.asarray(T, float).reshape(4, 4)
    if not np.all(np.isfinite(T)):
        return None
    tam = d.get("tam")
    if tam is None:
        tam = d.get("diametro_m")
    if tam is None and d.get("caja") is not None:
        try:
            tam = 2.0 * float(max(d["caja"]))
        except (TypeError, ValueError):
            tam = None
    tipo, nombre = str(d.get("tipo", "herramienta")), str(d.get("nombre", "objeto"))
    return dict(T=T, tam=float(tam) if tam is not None else float("nan"), nombre=nombre,
                clase=clase_de(nombre), tipo=tipo, id=d.get("id"), clave=f"{tipo}:{d.get('id')}",
                estable=bool(d.get("estable", True)))


def elegir_objeto(lista, palma=None, radio=None, tipos=TIPOS_COGIBLES):
    """El objeto «objetivo»: el que se puede coger más cercano a la palma (y a menos de 'radio' m,
    si se da). Prefiere los estables. None si no hay ninguno."""
    cands = []
    for d in lista or []:
        if str(d.get("tipo")) not in tipos:
            continue
        o = objeto_desde_deteccion(d)
        if o is None:
            continue
        dist = float(np.linalg.norm(o["T"][:3, 3] - np.asarray(palma, float))) if palma is not None else 0.0
        if radio is not None and palma is not None and dist > radio:
            continue
        cands.append((not o["estable"], dist, len(cands), o))
    if not cands:
        return None
    return min(cands, key=lambda c: c[:3])[3]


def objeto_congelado(ep, n_primeros=5):
    """Pose del objeto que se usa en TODO el episodio: la mediana de sus primeras detecciones
    (luego la mano lo tapa). None si el episodio no lo tiene."""
    obj = ep.get("obj")
    if obj is None:
        return None
    ok = np.all(np.isfinite(obj[:, 1:8]), axis=1)
    idx = np.nonzero(ok)[0][:n_primeros]
    if len(idx) == 0:
        return None
    p = np.median(obj[idx, 1:4], axis=0)
    R = quat_a_mat(obj[idx[0], 4:8])
    tams = obj[idx, 8]
    tams = tams[np.isfinite(tams)]
    info = dict(ep.get("meta", {}).get("objeto") or {})
    nombre = info.get("nombre", "objeto")
    return dict(T=T_de(R, p), tam=float(np.median(tams)) if len(tams) else float("nan"), nombre=nombre,
                clase=info.get("clase") or clase_de(nombre), tipo=info.get("tipo", "herramienta"),
                id=info.get("id"), clave=info.get("clave"), estable=True, fila=int(idx[0]))


def marco_objeto(objeto):
    """Marco en el que se miden la mano y las acciones: el del objeto, salvo las pelotas (su
    centro con los ejes del mundo, z hacia arriba)."""
    T = np.array(objeto["T"], float)
    if objeto.get("tipo") == "pelota":
        T[:3, :3] = np.eye(3)
    return T


# ══════════════════════════════════════════════════════════════════════════════
# RECETA: QUÉ VE Y QUÉ HACE LA RED
# ══════════════════════════════════════════════════════════════════════════════

def disposicion(grupos):
    """[(nombre, inicio, fin)] a partir de [(nombre, n)]."""
    out, i = [], 0
    for g, n in grupos:
        out.append((g, i, i + int(n)))
        i += int(n)
    return out


def dimension(grupos):
    return sum(int(n) for _, n in grupos)


def _frac_filas(arr):
    """Fracción de filas completas (sin NaN) de un array (T, …) o 0 si no hay."""
    if arr is None or len(arr) == 0:
        return 0.0
    a = np.asarray(arr, float).reshape(len(arr), -1)
    return float(np.mean(np.all(np.isfinite(a), axis=1)))


def receta_para(episodios, tareas, accion="auto", ventana=None, chunk=30, con_tactil=None,
                con_objeto=None, incluir_q_brazo=None):
    """Decide qué ve y qué hace la red a partir de lo que HAY en los episodios (ya compatibles).
    con_tactil / con_objeto: None = si están en los datos, False = no usarlos, True = obligatorio."""
    eps = list(episodios)
    if not eps:
        raise ValueError("no hay episodios")
    ep0 = eps[0]
    n_b = int(ep0["n_brazo"])
    n_m = int(ep0["q"].shape[1]) - n_b
    nombres = ep0["nombres"]
    objs = [objeto_congelado(e) for e in eps]
    hay_objeto = n_b == 6 and sum(o is not None for o in objs) >= max(1.0, 0.5 * len(eps))
    if con_objeto is True and not hay_objeto:
        raise ValueError("se pidió usar el objeto, pero menos de la mitad de los episodios lo tienen "
                         "(¿se grabó con la cámara de cabeza y calibrada?)")
    hay_objeto = hay_objeto and con_objeto is not False
    frac_tac = float(np.mean([_frac_filas(None if e.get("tac") is None else e["tac"][:, :, 2]) for e in eps]))
    if con_tactil is True and frac_tac == 0.0:
        raise ValueError("se pidió usar los táctiles, pero ningún episodio los tiene")
    hay_tactil = frac_tac >= 0.1 and con_tactil is not False
    usar_mano_real = float(np.mean([_frac_filas(e.get("hr")) for e in eps])) >= 0.5
    usar_brazo_real = n_b > 0 and float(np.mean([_frac_filas(e.get("br")) for e in eps])) >= 0.5
    if accion in (None, "auto"):
        accion = "mano_en_objeto" if hay_objeto else "joints"
    if accion not in ("joints", "mano_en_objeto"):
        raise ValueError(f"acción desconocida: {accion}")
    if accion == "mano_en_objeto" and not hay_objeto:
        raise ValueError("la acción 'mano_en_objeto' necesita episodios con objeto")
    if incluir_q_brazo is None:          # con el objeto, la postura absoluta del brazo sobra: la red
        incluir_q_brazo = accion == "joints"   # se fijaría en DÓNDE está la mesa y no generalizaría
    estado = []
    if n_b and incluir_q_brazo:
        estado.append(["q_brazo", n_b])
    estado.append(["q_mano", n_m])
    if hay_tactil:
        estado += [["tactil", len(DEDOS)], ["tactil_ok", 1]]
    entorno, clases, tam_defecto = [], [], 0.0
    if hay_objeto:
        validos = [o for o in objs if o is not None]
        clases = sorted({o["clase"] for o in validos})
        tams = [o["tam"] for o in validos if np.isfinite(o["tam"])]
        tam_defecto = float(np.median(tams)) if tams else 0.0
        entorno = [["obj_pos", 3], ["obj_rot", 6], ["obj_tam", 1], ["obj_clase", len(clases)]]
    if ventana is None:                  # ciega: el historial es lo único que distingue «abriendo» de
        ventana = 1 if (hay_objeto or hay_tactil) else 8    # «cerrando»; viendo el mundo, sobra
    return dict(version=1, formato=max(int(e["formato"]) for e in eps),
                nombres_brazo=list(nombres[:n_b]) if nombres else [],
                nombres_mano=list(nombres[n_b:]) if nombres else None, n_brazo=n_b, n_mano=n_m,
                accion=accion, estado=estado, entorno=entorno, clases=clases, tareas=list(tareas),
                ventana=int(ventana), chunk=int(chunk), fps=fps_de(eps),
                usar_mano_real=bool(usar_mano_real), usar_brazo_real=bool(usar_brazo_real),
                tactil_escala=TACTIL_ESCALA_N, tam_defecto=tam_defecto,
                xml=os.path.basename(str(_p("RUTA_XML", ""))))


def describir(receta):
    """Una línea para la terminal / el HUD."""
    ve = ", ".join(f"{g}({n})" for g, n in receta["estado"] + receta["entorno"])
    return (f"acción {receta['accion']} · ve {ve} · historial {receta['ventana']} · bloque {receta['chunk']}"
            + (f" · clases {receta['clases']}" if receta.get("clases") else ""))


# ══════════════════════════════════════════════════════════════════════════════
# DATOS CRUDOS (episodio entero o un instante) → CARACTERÍSTICAS
# ══════════════════════════════════════════════════════════════════════════════

def _preferir(real, sim):
    """Fila a fila: la lectura real si está completa; si no, la simulada."""
    ok = np.all(np.isfinite(real), axis=1)
    out = np.array(sim, float, copy=True)
    out[ok] = real[ok]
    return out


def crudo_de_episodio(ep, receta):
    """Lo que necesitan las características, fotograma a fotograma, sacado de un episodio."""
    n_b = int(ep["n_brazo"])
    q, a = ep["q"], ep["a"]
    q_b, q_m = q[:, :n_b], q[:, n_b:]
    if receta.get("usar_brazo_real") and ep.get("br") is not None and n_b:
        q_b = _preferir(ep["br"], q_b)
    if receta.get("usar_mano_real") and ep.get("hr") is not None:
        q_m = _preferir(ep["hr"], q_m)
    return dict(T=len(q), q_brazo=q_b, q_mano=q_m, tactil=ep.get("tac"),
                objeto=objeto_congelado(ep) if receta.get("entorno") else None,
                a_brazo=a[:, :n_b], a_mano=a[:, n_b:])


def tactil_array(t):
    """dict {dedo: [fx, fy, fz]} (TactilReal.fuerzas()) o array (5, 3) → array (5, 3), NaN si falta."""
    out = np.full((len(DEDOS), 3), np.nan)
    if t is None:
        return out
    if isinstance(t, dict):
        for k, d in enumerate(DEDOS):
            v = t.get(d)
            if v is not None and len(v) >= 3:
                out[k] = [float(x) for x in v[:3]]
        return out
    arr = np.asarray(t, float)
    return arr if arr.shape == out.shape else out


def crudo_de_obs(receta, obs, objeto=None):
    """Lo mismo que crudo_de_episodio, para UN instante en vivo. obs: dict con q_brazo (6, sim),
    q_mano (17, sim) y, si los hay, q_brazo_real, q_mano_real, tactil."""
    q_b, q_m = obs.get("q_brazo"), obs.get("q_mano")
    if receta.get("usar_brazo_real") and obs.get("q_brazo_real") is not None:
        q_b = obs["q_brazo_real"]
    if receta.get("usar_mano_real") and obs.get("q_mano_real") is not None:
        q_m = obs["q_mano_real"]
    q_b = np.zeros(0) if q_b is None else np.asarray(q_b, float).ravel()
    q_m = np.asarray(q_m, float).ravel()
    return dict(T=1, q_brazo=q_b[None, :], q_mano=q_m[None, :], tactil=tactil_array(obs.get("tactil"))[None],
                objeto=objeto, a_brazo=None, a_mano=None)


def _T_mano(cadena, q_brazo):
    """mano_mount en el mundo para cada fila de q_brazo (T, 6) → (T, 4, 4)."""
    return np.stack([cadena.fk_marcos(q)[6] for q in np.asarray(q_brazo, float)])


def caracteristicas(receta, crudo, cadena=None):
    """→ (estado (T, Ds), entorno (T, De) o None, accion (T, Da) o None). El MISMO código al entrenar
    (un episodio) y al reproducir (un instante). Lanza SinObjeto si la receta lo necesita y no hay."""
    T = int(crudo["T"])
    est = []
    tac = crudo.get("tactil")
    for g, n in receta["estado"]:
        if g == "q_brazo":
            est.append(np.asarray(crudo["q_brazo"], float).reshape(T, -1))
        elif g == "q_mano":
            est.append(np.asarray(crudo["q_mano"], float).reshape(T, -1))
        elif g == "tactil":
            fz = np.zeros((T, len(DEDOS))) if tac is None else np.abs(np.asarray(tac, float)[:, :, 2])
            est.append(np.where(np.isfinite(fz), fz, 0.0) / float(receta.get("tactil_escala", TACTIL_ESCALA_N)))
        elif g == "tactil_ok":
            ok = (np.zeros((T, 1)) if tac is None
                  else np.all(np.isfinite(np.asarray(tac, float)[:, :, 2]), axis=1, keepdims=True).astype(float))
            est.append(ok)
        else:
            raise ValueError(f"grupo de estado desconocido: {g}")
        if est[-1].shape[1] != int(n):
            raise ValueError(f"'{g}' tiene {est[-1].shape[1]} valores y la receta espera {n}")
    estado = np.concatenate(est, axis=1) if est else np.zeros((T, 0))

    entorno = None
    if receta.get("entorno"):
        obj = crudo.get("objeto")
        if obj is None:
            raise SinObjeto("la red necesita ver el objeto y no hay ninguno")
        if cadena is None:
            raise ValueError("hace falta la cinemática del FR5 para ver el objeto desde la mano")
        T_om = _inv_T_lote(_T_mano(cadena, crudo["q_brazo"])) @ marco_objeto(obj)   # objeto en la mano
        partes = []
        for g, n in receta["entorno"]:
            if g == "obj_pos":
                partes.append(T_om[:, :3, 3])
            elif g == "obj_rot":
                partes.append(rot6d(T_om[:, :3, :3]))
            elif g == "obj_tam":
                tam = obj.get("tam")
                if tam is None or not np.isfinite(tam):
                    tam = receta.get("tam_defecto", 0.0)
                partes.append(np.full((T, 1), float(tam)))
            elif g == "obj_clase":
                v = np.zeros((T, int(n)))
                clases = list(receta.get("clases") or [])
                if obj.get("clase") in clases:
                    v[:, clases.index(obj["clase"])] = 1.0
                partes.append(v)
            else:
                raise ValueError(f"grupo de entorno desconocido: {g}")
        entorno = np.concatenate(partes, axis=1)

    accion = acciones_de(receta, crudo, cadena) if crudo.get("a_mano") is not None else None
    return estado, entorno, accion


def acciones_de(receta, crudo, cadena=None):
    """Acciones del episodio en la forma que aprende la red (ver cabecera)."""
    a_b, a_m = np.asarray(crudo["a_brazo"], float), np.asarray(crudo["a_mano"], float)
    if receta["accion"] == "joints":
        return np.concatenate([a_b, a_m], axis=1) if receta.get("nombres_brazo") else a_m.copy()
    obj = crudo.get("objeto")
    if obj is None:
        raise SinObjeto("la acción 'mano_en_objeto' necesita el objeto")
    Tt = _inv_T_lote(np.repeat(marco_objeto(obj)[None], len(a_b), axis=0)) @ _T_mano(cadena, a_b)
    return np.concatenate([Tt[:, :3, 3], rot6d(Tt[:, :3, :3]), a_m], axis=1)


def accion_a_joints(receta, a, objeto=None, q_brazo=None, cadena=None):
    """Acción de la red (un instante, en unidades reales) → (brazo (6) o None, mano (n), error o None).
    En modo 'mano_en_objeto' resuelve la IK analítica con la solución más cercana a q_brazo."""
    a = np.asarray(a, float).ravel()
    n_m = int(receta.get("n_mano") or len(receta.get("nombres_mano") or []) or int(_p("N_ACT", 17)))
    if receta["accion"] == "joints":
        nb = len(receta.get("nombres_brazo") or [])
        return (a[:nb] if nb else None), a[nb:nb + n_m], None
    mano = a[9:9 + n_m]
    if objeto is None or cadena is None:
        return None, mano, "falta el objeto o la cinemática del FR5"
    T = marco_objeto(objeto) @ T_de(R_de_rot6d(a[3:9]), a[:3])
    q_ref = None if q_brazo is None else np.asarray(q_brazo, float).ravel()
    sols = cadena.ik_analitica(T, q_ref=q_ref)
    if not sols:
        return None, mano, "la red pide una pose de la mano fuera del alcance del brazo (IK)"
    q = sols[0] if q_ref is None else min(sols, key=lambda s: float(np.max(np.abs(np.asarray(s) - q_ref))))
    return np.asarray(q, float), mano, None


# ══════════════════════════════════════════════════════════════════════════════
# CONJUNTO DE ENTRENAMIENTO: ventanas con relleno, sin copiar datos
# ══════════════════════════════════════════════════════════════════════════════

class Conjunto:
    """Los episodios ya convertidos (estado, entorno, acción) y, para CADA fotograma, los índices
    de su ventana. Una muestra por fotograma, también los primeros y los últimos:
      · historial: los H fotogramas hasta t; antes del principio se repite el PRIMERO (lo mismo
        que hace la reproducción al arrancar: antes nunca lo veía al entrenar).
      · bloque: las C acciones desde t; pasado el final se repite la ÚLTIMA (pad = True), así la
        red aprende a quedarse quieta al terminar en vez de seguir moviéndose.
      · progreso: t / (T − 1) de cada acción del bloque (1 al final): la red aprende cuándo ha
        terminado, y la reproducción puede parar sola."""

    def __init__(self, receta, episodios, tarea_de=None, cadena=None):
        H, C = int(receta["ventana"]), int(receta["chunk"])
        tarea_de = list(tarea_de) if tarea_de is not None else [0] * len(episodios)
        S, E, A, oi, ai_, pad, prog, ep_id, tarea = [], [], [], [], [], [], [], [], []
        self.rutas, self.longitudes, self.tarea_ep, self.saltados = [], [], [], []
        self.episodios = []                    # los dicts de leer_episodio que se usan, por índice
        base = 0
        for ep, ti in zip(episodios, tarea_de):
            try:
                s, e, a = caracteristicas(receta, crudo_de_episodio(ep, receta), cadena)
            except SinObjeto:
                self.saltados.append((ep["ruta"], "sin objeto"))
                continue
            except ValueError as err:
                self.saltados.append((ep["ruta"], str(err)))
                continue
            T = len(s)
            if T < 2:
                self.saltados.append((ep["ruta"], "demasiado corto"))
                continue
            if not (np.all(np.isfinite(s)) and np.all(np.isfinite(a)) and (e is None or np.all(np.isfinite(e)))):
                self.saltados.append((ep["ruta"], "valores no válidos (NaN)"))
                continue
            i_ep = len(self.rutas)
            t = np.arange(T)
            o = np.clip(t[:, None] + np.arange(-H + 1, 1)[None, :], 0, T - 1)
            ai = t[:, None] + np.arange(C)[None, :]
            pad.append(ai > T - 1)
            prog.append(np.clip(ai, 0, T - 1) / float(T - 1))
            ai = np.clip(ai, 0, T - 1)
            S.append(s)
            A.append(a)
            if e is not None:
                E.append(e)
            oi.append(o + base)
            ai_.append(ai + base)
            ep_id.append(np.full(T, i_ep))
            tarea.append(np.full(T, int(ti)))
            self.rutas.append(ep["ruta"])
            self.episodios.append(ep)
            self.longitudes.append(T)
            self.tarea_ep.append(int(ti))
            base += T
        if not S:
            raise ValueError("ningún episodio válido para entrenar"
                             + (f" ({self.saltados[0][1]})" if self.saltados else ""))
        self.receta = receta
        self.S = np.concatenate(S).astype(np.float32)
        self.E = np.concatenate(E).astype(np.float32) if E else None
        self.A = np.concatenate(A).astype(np.float32)
        self.obs_idx = np.concatenate(oi).astype(np.int64)
        self.act_idx = np.concatenate(ai_).astype(np.int64)
        self.pad = np.concatenate(pad)
        self.prog = np.concatenate(prog).astype(np.float32)
        self.ep_id = np.concatenate(ep_id)
        self.tarea = np.concatenate(tarea).astype(np.int64)

    @property
    def n(self):
        return len(self.S)

    @property
    def n_episodios(self):
        return len(self.rutas)

    def dividir(self, frac_val=0.2, semilla=0):
        """(índices train, índices val, episodios de val). Por EPISODIO y dentro de cada tarea: un
        episodio entero va a train o a val, nunca repartido. Con 1 episodio en una tarea, a train."""
        rng = np.random.default_rng(semilla)
        val = []
        for ti in sorted(set(self.tarea_ep)):
            eps = [i for i, t in enumerate(self.tarea_ep) if t == ti]
            if len(eps) < 2:
                continue
            n_val = min(max(1, int(round(len(eps) * frac_val))), len(eps) - 1)
            val += [int(i) for i in rng.permutation(eps)[:n_val]]
        es_val = np.isin(self.ep_id, val)
        return np.nonzero(~es_val)[0], np.nonzero(es_val)[0], sorted(val)

    def normalizacion(self, idx=None):
        """Media y desviación (por dimensión) de estado, entorno y acción, con los fotogramas idx."""
        idx = np.arange(self.n) if idx is None else np.asarray(idx)

        def ms(X, grupos):
            mu = X.mean(axis=0).astype(np.float64)
            sd = np.maximum(X.std(axis=0), SIGMA_MIN).astype(np.float64)
            for g, i0, i1 in disposicion(grupos or []):
                if g in GRUPOS_FIJOS:
                    mu[i0:i1], sd[i0:i1] = 0.0, 1.0
            return mu.astype(np.float32), sd.astype(np.float32)

        return dict(estado=ms(self.S[idx], self.receta["estado"]),
                    entorno=ms(self.E[idx], self.receta["entorno"]) if self.E is not None else None,
                    accion=ms(self.A[idx], None))

    def lote(self, idx):
        """Muestras idx (sin normalizar): x (B, H, Ds), e (B, De) o None, y (B, C, Da), pad, prog, tarea."""
        idx = np.asarray(idx)
        return dict(x=self.S[self.obs_idx[idx]], e=None if self.E is None else self.E[idx],
                    y=self.A[self.act_idx[idx]], pad=self.pad[idx], prog=self.prog[idx], tarea=self.tarea[idx])
