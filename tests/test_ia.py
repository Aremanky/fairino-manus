"""Pruebas de la IA (16-nov), sin hardware: grabación formato 3, datos, receta, ventanas, política,
filtro de seguridad, ensayo en lazo cerrado y, si hay torch, entrenar → cargar → reproducir.

    python -m pytest tests/test_ia.py -q
"""
import csv
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import config
from skill_shield.ia import episodios as E
from skill_shield.ia.evaluar import ensayo, seguimiento
from skill_shield.ia.fuentes import FuentesGrabacion, LectorManoReal
from skill_shield.ia.grabador import GrabadorEpisodio
from skill_shield.ia.politica import FiltroSeguridad, Paso, Politica
from skill_shield.teleop_fr5.cinematica import OFFSET_PALMA_ORCA, CadenaFR5

CADENA = CadenaFR5.desde_mjcf(config.RUTA_XML)
BRAZO = list(config.ARM_JOINTS)
MANO = [f"mano_{i}" for i in range(17)]
Q0 = np.array([0.0, -1.5708, 2.0, 0.0, 1.5708, 0.0])        # alcanzable, mano mirando a la mesa


# ══════════════════════════════════════════════════════════════════════════════
# episodios sintéticos grabados con el grabador DE VERDAD (y fuentes falsas)
# ══════════════════════════════════════════════════════════════════════════════

class _ManoFalsa:
    def __init__(self):
        self.valor = None

    def ultima(self):
        return self.valor


def _trayectoria(T0, palma_final, T):
    """Joints del brazo que llevan mano_mount en recta (mismo giro) hasta que la palma llega a
    palma_final, resolviendo la IK en cada fotograma."""
    R0 = T0[:3, :3]
    p_ini, p_fin = T0[:3, 3], palma_final - R0 @ OFFSET_PALMA_ORCA
    sols = CADENA.ik_analitica(T0, q_ref=Q0)
    q = min(sols, key=lambda s: np.max(np.abs(s - Q0)))
    out = []
    for k in range(T):
        Tk = T0.copy()
        Tk[:3, 3] = p_ini + (p_fin - p_ini) * k / (T - 1)
        s = CADENA.ik_analitica(Tk, q_ref=q)
        assert s, "trayectoria de prueba fuera de alcance"
        q = min(s, key=lambda x: np.max(np.abs(x - q)))
        out.append(q.copy())
    return np.array(out)


def grabar_episodio(carpeta, desplaz=(0.0, 0.0, 0.0), T=40, tactil=True, mano_real=True, objeto=True,
                    tapar_desde=None, exito=True, tarea="coger_pelota"):
    """Graba con GrabadorEpisodio un agarre de pelota sintético: la mano baja en recta hasta la
    pelota y cierra los dedos. desplaz traslada a la vez la pelota y la postura de partida."""
    d = np.asarray(desplaz, float)
    T0 = CADENA.fk_marcos(Q0)[6].copy()
    T0[:3, 3] += d
    palma0 = T0[:3, :3] @ OFFSET_PALMA_ORCA + T0[:3, 3]
    pelota = palma0 + np.array([0.0, 0.0, -0.10])
    q_b = _trayectoria(T0, pelota + np.array([0.0, 0.0, 0.035]), T)
    cierre = np.clip((np.arange(T) - T * 0.5) / (T * 0.4), 0.0, 1.0)
    a_m = np.outer(cierre, np.linspace(0.3, 1.2, 17))
    estado = dict(k=0)
    mano = _ManoFalsa()

    def obj_lista():
        k = estado["k"]
        if not objeto or (tapar_desde is not None and k >= tapar_desde):
            return []                                       # la mano tapa la pelota
        Tp = np.eye(4)
        Tp[:3, 3] = pelota
        Tlejos = np.eye(4)
        Tlejos[:3, 3] = pelota + np.array([0.6, 0.4, 0.0])
        return [dict(id=7, nombre="Pelota roja", tipo="pelota", T_mundo=Tlejos.tolist(), diametro_m=0.07),
                dict(id=3, nombre="Pelota de tenis", tipo="pelota", T_mundo=Tp.tolist(), diametro_m=0.067,
                     estable=True),
                dict(id=21, nombre="Punto de uso", tipo="punto_uso", T_mundo=Tp.tolist())]

    def tac():
        k = estado["k"]
        fz = 3.0 * cierre[k]
        return {dd: [0.0, 0.0, -fz] for dd in E.DEDOS}

    fuentes = FuentesGrabacion(BRAZO, MANO, cadena=CADENA, brazo_real=lambda: q_b[estado["k"]],
                               mano_real=mano if mano_real else None, tactil=tac if tactil else None,
                               objetos=obj_lista, radio_objeto=0.35)
    g = GrabadorEpisodio(BRAZO + MANO, list(range(23)), list(range(23)), fps=30.0, recortar=False,
                         n_brazo=6, fuentes=fuentes)
    datos = SimpleNamespace(time=0.0, qpos=np.zeros(23), ctrl=np.zeros(23))
    g.iniciar(tarea, datos)
    for k in range(T):
        estado["k"] = k
        datos.time = k / 30.0
        datos.qpos[:6] = q_b[k]
        datos.ctrl[:6] = q_b[min(k + 1, T - 1)]             # el comando va un fotograma por delante
        datos.qpos[6:] = a_m[k - 1] if k else a_m[0]
        datos.ctrl[6:] = a_m[k]
        mano.valor = (datos.qpos[6:].copy(), np.full(17, 0.2))
        g.muestrear(datos)
    ruta, n = g.parar(carpeta, exito=exito)
    g.esperar()
    assert n == T
    return ruta, g


@pytest.fixture
def datos_pelota(tmp_path):
    """6 episodios de coger la pelota en sitios distintos (formato 3 completo)."""
    carpeta = tmp_path / "episodios" / "coger_pelota"
    rutas = []
    for d in [(0, 0, 0), (0.03, 0, 0), (-0.03, 0.02, 0), (0, -0.04, 0), (0.02, 0.03, 0.01), (-0.02, -0.02, 0)]:
        rutas.append(grabar_episodio(str(carpeta), desplaz=d, tapar_desde=25)[0])
    return tmp_path, rutas


# ══════════════════════════════════════════════════════════════════════════════
# grabación y lectura
# ══════════════════════════════════════════════════════════════════════════════

def test_grabador_formato3_columnas_meta_y_objeto_tapado(tmp_path):
    ruta, g = grabar_episodio(str(tmp_path / "t"), tapar_desde=20)
    ep = E.leer_episodio(ruta)
    assert ep["formato"] == 3 and ep["n_brazo"] == 6 and ep["q"].shape == (40, 23)
    assert ep["br"].shape == (40, 6) and ep["hr"].shape == (40, 17) and ep["hc"].shape == (40, 17)
    assert ep["tac"].shape == (40, 5, 3) and np.all(np.isfinite(ep["tac"]))
    # el objetivo es la pelota cercana (no la lejana ni el punto de uso), y desaparece al taparla
    assert np.all(ep["obj"][:20, 0] == 3) and np.all(np.isnan(ep["obj"][20:, 1]))
    meta = ep["meta"]
    assert meta["formato"] == 3 and meta["exito"] is True and meta["n_brazo"] == 6
    assert meta["objeto"]["nombre"] == "Pelota de tenis" and meta["objeto"]["clase"] == "pelota de tenis"
    assert meta["fuentes"]["obj"] == pytest.approx(0.5) and meta["fuentes"]["tac"] == 1.0
    # el objeto congelado = el de las primeras detecciones, con su tamaño
    o = E.objeto_congelado(ep)
    assert o["tipo"] == "pelota" and o["tam"] == pytest.approx(0.067)
    # [M]: fallido y otra vez bueno
    assert g.marcar_ultimo() == (ruta, False)
    assert not E.exito(E.leer_episodio(ruta))
    assert g.marcar_ultimo() == (ruta, True)


def test_grabador_sin_fuentes_sigue_siendo_formato2(tmp_path):
    g = GrabadorEpisodio(BRAZO + MANO, list(range(23)), list(range(23)), recortar=True)
    datos = SimpleNamespace(time=0.0, qpos=np.zeros(23), ctrl=np.zeros(23))
    g.iniciar("t", datos)
    for k in range(60):
        datos.time = k / 30.0
        datos.ctrl[0] = 0.5 if 20 <= k < 40 else 0.0      # se mueve solo en el medio
        g.muestrear(datos)
    ruta, n = g.parar(str(tmp_path))
    g.esperar()
    ep = E.leer_episodio(ruta)
    assert ep["formato"] == 2 and ep["br"] is None and ep["obj"] is None
    assert n < 60                                            # recortó lo quieto (con 0,5 s de margen)


def test_leer_formato1_antiguo_y_compatibles(tmp_path):
    r1 = tmp_path / "episodio_0001.csv"
    with open(r1, "w", newline="") as f:
        w = csv.writer(f)
        for k in range(20):
            w.writerow([0.01 * k] * 17 + [0.02 * k] * 17 + [k / 15.0])
    e1 = E.leer_episodio(str(r1))
    assert e1["formato"] == 1 and e1["n_brazo"] == 0 and e1["q"].shape == (20, 17)
    assert E.fps_de([e1]) == pytest.approx(15.0)
    ruta2, _ = grabar_episodio(str(tmp_path / "nuevo"))
    usados, avisos = E.compatibles([e1, E.leer_episodio(ruta2)])
    assert len(usados) == 1 and usados[0]["formato"] == 3 and "formato antiguo" in avisos[0]


def test_lector_mano_real_en_orden_de_ctrl():
    mapa = {"wrist": 0, "pinky_abd": 1, "pinky_mcp": 2, "pinky_pip": 3, "ring_abd": 4, "ring_mcp": 5,
            "ring_pip": 6, "middle_abd": 7, "middle_mcp": 8, "middle_pip": 9, "index_abd": 10,
            "index_mcp": 11, "index_pip": 12, "thumb_cmc": 13, "thumb_abd": 14, "thumb_mcp": 15, "thumb_dip": 16}

    class W:
        conectado = True
        JOINT_A_CTRL = mapa
        _joints = sorted(mapa)                              # orden de motor ≠ orden de ctrl

        def estado_real(self):
            return {j: (None, float(k), 0.1 * k) for k, j in enumerate(self._joints)}

        def _desde_sdk(self, grados):
            return np.radians([grados[j] for j in self._joints])

    lec = LectorManoReal(W())
    assert lec.leer_una_vez()
    q, frac = lec.ultima()
    for k, j in enumerate(W._joints):
        assert q[mapa[j]] == pytest.approx(np.radians(k)) and frac[mapa[j]] == pytest.approx(0.1 * k)


def test_fuentes_de_objetos_simulador_camara_y_teleop(capsys):
    from skill_shield.autonomo.canal import Canal
    from skill_shield.ia.fuentes import (FuenteObjetosCamara, FuenteObjetosSim, ObjetosCombinados,
                                         crear_fuentes_teleop)
    # simulador: pose exacta de los bodies, solo con la mesa visible
    datos = SimpleNamespace(xpos=np.array([[0, 0, 0], [0.4, -0.1, -0.3]]), xmat=np.tile(np.eye(3).ravel(), (2, 1)))
    visible = dict(v=False)
    sim = FuenteObjetosSim(datos, {"taller_destornillador": 1}, visible=lambda: visible["v"])
    assert sim() == [] and "escondida" in sim.estado()
    visible["v"] = True
    (o,) = sim()
    assert o["nombre"] == "destornillador" and o["tipo"] == "herramienta"
    assert np.allclose(np.asarray(o["T_mundo"])[:3, 3], [0.4, -0.1, -0.3])
    # cámara: lo que publica el proceso de la cámara de cabeza en el canal
    canal = Canal("skill_ia_objetos_test", capacidad=4096, crear=True)
    try:
        cam = FuenteObjetosCamara(canal)
        assert cam() == [] and cam.estado() == "esperando a la cámara de cabeza"
        canal.escribir(json.dumps(dict(lista=[dict(id=3, nombre="Pelota de tenis", tipo="pelota",
                                                   T_mundo=np.eye(4).tolist())], calibrada=True)).encode())
        cam._t_leido = 0.0
        assert cam()[0]["nombre"] == "Pelota de tenis" and cam.estado() == "ok"
        assert ObjetosCombinados([cam, sim])()[0]["nombre"] == "Pelota de tenis"   # la real primero
        cam.max_edad_s, cam._t_leido = -1.0, 0.0                                   # dato viejo → nada
        assert ObjetosCombinados([cam, sim])()[0]["nombre"] == "destornillador"
    finally:
        canal.cerrar()
    # las fuentes de la app: sin hardware → sin cámara; con la cámara ocupada → avisa y no la lanza
    f = crear_fuentes_teleop(dict(RUTA_XML=config.RUTA_XML), "/tmp", BRAZO, MANO, objetos_sim=sim)
    assert f.cadena is not None and f.objetos is not None and f.mano_real is None
    assert "objeto ✗" in f.resumen(Q0) or "objeto ✓" in f.resumen(Q0)
    obs = f.observacion(Q0, np.zeros(17))
    assert obs["q_brazo_real"] is None and obs["tactil"] is None
    f2 = crear_fuentes_teleop(dict(RUTA_XML=config.RUTA_XML), "/tmp", BRAZO, MANO, hardware=True, camara_libre=False)
    assert f2.objetos is None and "MULTICAMARA_ACTIVA = False" in capsys.readouterr().out
    assert "objeto ✗ (ni cámara" in f2.resumen(Q0)
    # táctiles: nombres de SENSOR de orca_core → dedos FÍSICOS con el mapa del cableado
    sensores = {s: [0.0, 0.0, float(k)] for k, s in enumerate(("thumb", "index", "middle", "ring", "pinky"))}
    f3 = crear_fuentes_teleop(dict(RUTA_XML=config.RUTA_XML, ORCA_TACTIL_SENSOR_DE_DEDO=config.ORCA_TACTIL_SENSOR_DE_DEDO),
                              "/tmp", BRAZO, MANO, tactil=lambda: sensores)
    t = f3.leer_tactil()
    assert t["thumb"] == sensores[config.ORCA_TACTIL_SENSOR_DE_DEDO["pulgar"]]     # pulgar = sensor «index»
    assert t["middle"] == sensores[config.ORCA_TACTIL_SENSOR_DE_DEDO["corazon"]]


def test_elegir_objeto_cercano_estable_y_solo_cogibles():
    def det(id_, p, tipo="pelota", estable=True):
        T = np.eye(4)
        T[:3, 3] = p
        return dict(id=id_, nombre=f"cosa {id_}", tipo=tipo, T_mundo=T.tolist(), estable=estable)
    lista = [det(1, [0.5, 0, 0]), det(2, [0.1, 0, 0], tipo="punto_uso"), det(3, [0.2, 0, 0], estable=False),
             det(4, [0.3, 0, 0])]
    o = E.elegir_objeto(lista, palma=[0, 0, 0], radio=0.35)
    assert o["id"] == 4                                     # el estable más cercano dentro del radio
    assert E.elegir_objeto(lista, palma=[0, 0, 0], radio=0.1) is None
    assert E.clase_de("Pelota roja 2") == "pelota roja"


def test_directorios_fuera_del_repo_y_compatibilidad(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "episodios" / "t").mkdir(parents=True)
    (repo / "episodios" / "t" / "episodio_0001.csv").write_text("t\n")
    monkeypatch.setattr(E, "_RAIZ", str(repo))
    nuevo = tmp_path / "datos"
    monkeypatch.setenv("SKILL_SHIELD_DATOS", str(nuevo))
    assert E.directorios(avisar=False)[0] == str(repo / "episodios")      # aún no se han movido
    (nuevo / "episodios" / "t").mkdir(parents=True)
    (nuevo / "episodios" / "t" / "episodio_0001.csv").write_text("t\n")
    assert E.directorios(avisar=False) == (str(nuevo / "episodios"), str(nuevo / "modelos"))


# ══════════════════════════════════════════════════════════════════════════════
# receta, características y ventanas
# ══════════════════════════════════════════════════════════════════════════════

def test_receta_ve_el_mundo_y_acciones_respecto_al_objeto(datos_pelota):
    _, rutas = datos_pelota
    eps = [E.leer_episodio(r) for r in rutas]
    r = E.receta_para(eps, ["coger_pelota"])
    assert r["accion"] == "mano_en_objeto" and r["ventana"] == 1
    assert [g for g, _ in r["estado"]] == ["q_mano", "tactil", "tactil_ok"]     # sin el brazo absoluto
    assert [g for g, _ in r["entorno"]] == ["obj_pos", "obj_rot", "obj_tam", "obj_clase"]
    assert r["clases"] == ["pelota de tenis"] and r["usar_mano_real"] and r["usar_brazo_real"]
    ciega = E.receta_para(eps, ["coger_pelota"], con_objeto=False, con_tactil=False)
    assert ciega["accion"] == "joints" and ciega["ventana"] == 8 and ciega["estado"][0][0] == "q_brazo"


def test_objeto_visto_desde_la_mano_e_ik_de_ida_y_vuelta(datos_pelota):
    _, rutas = datos_pelota
    ep = E.leer_episodio(rutas[0])
    r = E.receta_para([ep], ["coger_pelota"])
    crudo = E.crudo_de_episodio(ep, r)
    s, e, a = E.caracteristicas(r, crudo, CADENA)
    assert s.shape == (40, 17 + 6) and e.shape == (40, 3 + 6 + 1 + 1) and a.shape == (40, 9 + 17)
    # la pelota en el marco de la mano: comprobación directa en un fotograma
    k = 10
    Tm = CADENA.fk_marcos(crudo["q_brazo"][k])[6]
    p_obj = crudo["objeto"]["T"][:3, 3]
    assert np.allclose(e[k, :3], Tm[:3, :3].T @ (p_obj - Tm[:3, 3]), atol=1e-9)
    # la acción (mano en el marco de la pelota) vuelve a los MISMOS joints con la IK
    for k in (0, 15, 39):
        brazo, mano, err = E.accion_a_joints(r, a[k], crudo["objeto"], crudo["q_brazo"][k], CADENA)
        assert err is None
        assert np.allclose(brazo, crudo["a_brazo"][k], atol=1e-6) and np.allclose(mano, crudo["a_mano"][k])


def test_si_se_mueve_la_pelota_el_gesto_es_el_mismo(tmp_path):
    """La clave para generalizar con pocas demos: pelota y mano trasladadas → entradas y acciones
    de la red IDÉNTICAS (aunque los joints del brazo sean otros)."""
    ra, _ = grabar_episodio(str(tmp_path / "a"))
    rb, _ = grabar_episodio(str(tmp_path / "b"), desplaz=(0.05, -0.04, 0.02))
    ea, eb = E.leer_episodio(ra), E.leer_episodio(rb)
    r = E.receta_para([ea, eb], ["coger_pelota"])
    sa, na, aa = E.caracteristicas(r, E.crudo_de_episodio(ea, r), CADENA)
    sb, nb, ab = E.caracteristicas(r, E.crudo_de_episodio(eb, r), CADENA)
    assert not np.allclose(ea["q"][:, :6], eb["q"][:, :6], atol=1e-3)      # el brazo va distinto
    assert np.allclose(sa, sb, atol=1e-6) and np.allclose(na, nb, atol=1e-6) and np.allclose(aa, ab, atol=1e-6)


def test_ventanas_con_relleno_al_principio_y_al_final(datos_pelota):
    _, rutas = datos_pelota
    eps = [E.leer_episodio(r) for r in rutas[:2]]
    r = E.receta_para(eps, ["coger_pelota"], ventana=3, chunk=5)
    cj = E.Conjunto(r, eps, [0, 0], CADENA)
    assert cj.n == 80 and cj.n_episodios == 2                 # una muestra por fotograma
    b = cj.lote([0, 39, 40])
    assert np.allclose(b["x"][0][0], b["x"][0][2])             # historial del 1º: repite el primero
    assert np.allclose(b["x"][2][0], cj.S[40])                 # el 2º episodio no mira al 1º
    assert b["pad"][1].tolist() == [False, True, True, True, True]
    assert np.allclose(b["y"][1][1:], b["y"][1][0])             # tras el final, la última acción
    assert b["prog"][1].tolist() == [1.0] * 5 and b["prog"][0][0] == 0.0
    tr, va, eps_va = cj.dividir(0.5, semilla=1)
    assert len(eps_va) == 1 and set(cj.ep_id[va]) == set(eps_va) and not set(cj.ep_id[tr]) & set(eps_va)
    norm = cj.normalizacion(tr)
    i0, i1 = [(a, b_) for g, a, b_ in E.disposicion(r["estado"]) if g == "tactil"][0]
    assert np.all(norm["estado"][0][i0:i1] == 0) and np.all(norm["estado"][1][i0:i1] == 1)


def test_episodio_sin_objeto_se_salta_si_la_receta_lo_necesita(tmp_path, datos_pelota):
    _, rutas = datos_pelota
    eps = [E.leer_episodio(r) for r in rutas]
    sin = E.leer_episodio(grabar_episodio(str(tmp_path / "x"), objeto=False)[0])
    r = E.receta_para(eps, ["coger_pelota"])
    cj = E.Conjunto(r, eps + [sin], None, CADENA)
    assert cj.n_episodios == 6 and cj.saltados[0][1] == "sin objeto"


# ══════════════════════════════════════════════════════════════════════════════
# política, filtro y ensayo (con una red «oráculo», sin torch)
# ══════════════════════════════════════════════════════════════════════════════

def _red_oraculo(receta, ep):
    """Una «red» que devuelve las acciones de la propia demo (y su progreso), paso a paso."""
    crudo = E.crudo_de_episodio(ep, receta)
    _, _, A = E.caracteristicas(receta, crudo, CADENA)
    T, C = len(A), receta["chunk"]
    k = dict(n=0)

    def predecir(hist, entorno=None, tarea=0):
        assert hist.shape == (receta["ventana"], E.dimension(receta["estado"]))
        assert (entorno is None) == (not receta["entorno"])
        i = np.clip(k["n"] + np.arange(C), 0, T - 1)
        k["n"] += 1
        return A[i], i / (T - 1)

    return dict(receta=receta, predecir=predecir, tareas=receta["tareas"])


def test_politica_necesita_el_objeto_y_ensambla(datos_pelota):
    _, rutas = datos_pelota
    ep = E.leer_episodio(rutas[0])
    r = E.receta_para([ep], ["coger_pelota"])
    pol = Politica(_red_oraculo(r, ep), cadena=CADENA)
    crudo = E.crudo_de_episodio(ep, r)
    obs = dict(q_brazo=crudo["q_brazo"][0], q_mano=crudo["q_mano"][0], tactil=crudo["tactil"][0])
    assert not pol.reiniciar(obs) and "objeto" in pol.error
    assert pol.reiniciar(dict(obs, objeto=crudo["objeto"]))
    p = pol.paso(obs)
    assert p.error is None and np.allclose(p.brazo, crudo["a_brazo"][0], atol=1e-6)
    assert p.progreso == pytest.approx(0.0) and 0.08 < p.dist_objeto < 0.11     # palma ~10 cm encima
    with pytest.raises(ValueError):
        Politica(_red_oraculo(r, ep), tarea="otra_tarea", cadena=CADENA)
    rota = dict(_red_oraculo(r, ep), predecir=lambda h, e, t: 1 / 0)   # la red falla → parada, no cuelgue
    pol = Politica(rota, cadena=CADENA)
    pol.reiniciar(dict(obs, objeto=crudo["objeto"]))
    p = pol.paso(obs)
    assert p.error.startswith("fallo de la red") and FiltroSeguridad().aplicar(p, np.zeros(6)).parar


def test_ensamblado_temporal_promedia_bloques():
    receta = dict(ventana=1, chunk=3, estado=[["q_mano", 2]], entorno=[], accion="joints", nombres_brazo=[],
                  nombres_mano=["a", "b"], n_mano=2, tareas=["t"], fps=30)
    salidas = iter([np.zeros((3, 2)), np.ones((3, 2))])
    red = dict(receta=receta, predecir=lambda h, e, t: (next(salidas), None))
    pol = Politica(red, peso_exp=0.0)                       # pesos iguales
    obs = dict(q_mano=[0.0, 0.0])
    pol.reiniciar(obs)
    assert np.allclose(pol.paso(obs).mano, 0.0)
    assert np.allclose(pol.paso(obs).mano, 0.5)              # media del bloque nuevo y el anterior


def test_filtro_de_seguridad():
    f = FiltroSeguridad(lim_brazo=[(-1, 1)] * 6, lim_mano=[(0, 1)] * 2, v_max_brazo=1.0, dt=0.1,
                        salto_max=0.3, max_dist_objeto=0.3, libre=lambda q: q[0] < 0.5)
    ant = np.zeros(6)
    c = f.aplicar(Paso(np.full(6, 0.05), np.array([2.0, -1.0]), 0.5, None, None, 0.1), ant, np.zeros(2))
    assert not c.parar and np.allclose(c.mano, [1.0, 0.0]) and np.allclose(c.brazo, 0.05)
    c = f.aplicar(Paso(np.full(6, 0.25), None, None, None, None, None), ant)
    assert np.allclose(c.brazo, 0.1)                         # velocidad: 1 rad/s · 0,1 s
    assert f.aplicar(Paso(np.full(6, 0.5), None, None, None, None, None), ant).parar           # salto
    assert f.aplicar(Paso(np.full(6, np.nan), None, None, None, None, None), ant).parar        # NaN
    assert f.aplicar(Paso(None, None, None, None, "sin IK", None), ant).parar                  # error
    assert f.aplicar(Paso(ant, None, None, None, None, 0.5), ant).parar                        # lejos
    cerca = np.full(6, 0.45)
    c = f.aplicar(Paso(np.full(6, 0.55), None, None, None, None, None), cerca)
    assert np.allclose(c.brazo, cerca) and f.n_choques == 1 and c.motivo.startswith("choque")


def test_ensayo_en_lazo_cerrado_con_red_perfecta(datos_pelota):
    _, rutas = datos_pelota
    ep = E.leer_episodio(rutas[1])
    r = E.receta_para([ep], ["coger_pelota"])
    res = ensayo(_red_oraculo(r, ep), ep, cadena=CADENA, lim_brazo=config.LIMITES_CTRL_BRAZO)
    assert res["ok"] and res["terminado"], res
    assert res["err_palma_cm"] < 0.5 and res["err_dedos_rad"] < 0.05 and res["desvio_max_cm"] < 0.5
    assert seguimiento(np.arange(10.0)[:, None], np.arange(1.0, 11.0)[:, None]) == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# con torch: entrenar → cargar → reproducir (en el PC del robot)
# ══════════════════════════════════════════════════════════════════════════════

def test_entrenar_cargar_y_reproducir(datos_pelota):
    pytest.importorskip("torch")
    from skill_shield.ia import act
    raiz, rutas = datos_pelota
    ruta = act.entrenar(["coger_pelota"], episodios_dir=str(raiz / "episodios"), modelos_dir=str(raiz / "modelos"),
                        pasos=60, lote=16, eval_cada=20, cpu=True)
    assert ruta and os.path.exists(ruta)
    informe = json.load(open(os.path.join(raiz, "modelos", "coger_pelota", "informe.json"), encoding="utf-8"))
    assert informe["checkpoints"] and informe["episodios_val"]          # se ensayaron los checkpoints
    red = act.cargar_ia(ruta)
    assert red["receta"]["accion"] == "mano_en_objeto" and not red["legado"]
    ep = E.leer_episodio(rutas[0])
    crudo = E.crudo_de_episodio(ep, red["receta"])
    pol = Politica(red, cadena=CADENA)
    obs = dict(q_brazo=crudo["q_brazo"][0], q_mano=crudo["q_mano"][0], tactil=None)
    assert pol.reiniciar(dict(obs, objeto=crudo["objeto"]))
    p = pol.paso(obs)                                        # sin táctiles: también funciona
    assert p.mano.shape == (17,) and np.all(np.isfinite(p.mano)) and 0.0 <= p.progreso <= 1.0
    assert p.error is not None or p.brazo.shape == (6,)      # (sin entrenar apenas, la IK puede no llegar)


def test_entrenar_multitarea_y_ciega(tmp_path):
    pytest.importorskip("torch")
    from skill_shield.ia import act
    for t, d in (("a", (0, 0, 0)), ("a", (0.02, 0, 0)), ("b", (0, 0.02, 0)), ("b", (-0.02, 0, 0))):
        grabar_episodio(str(tmp_path / "episodios" / t), desplaz=d, tactil=False, objeto=False, tarea=t)
    ruta = act.entrenar(["a", "b"], modelo="ab", episodios_dir=str(tmp_path / "episodios"),
                        modelos_dir=str(tmp_path / "modelos"), pasos=40, lote=16, eval_cada=20, cpu=True)
    red = act.cargar_ia(ruta)
    assert red["tareas"] == ["a", "b"] and red["receta"]["accion"] == "joints" and red["ventana"] == 8
    pol = Politica(red, tarea="b")
    obs = dict(q_brazo=Q0, q_mano=np.zeros(17))
    assert pol.reiniciar(obs)
    p = pol.paso(obs)
    assert p.brazo.shape == (6,) and p.mano.shape == (17,) and p.error is None


def test_modelo_antiguo_se_sigue_reproduciendo():
    torch = pytest.importorskip("torch")
    from skill_shield.ia import act
    viejo = act.ACTModel(n_joints=23, ventana=8, chunk=30, dim_accion=23)    # formato 2 de antes
    nombres = BRAZO + MANO
    chk = dict(state_dict=viejo.state_dict(), mu_x=np.zeros(23), sigma_x=np.ones(23), mu_y=np.zeros(23),
               sigma_y=np.ones(23), ventana=8, chunk=30, dt_objetivo=1 / 30, formato=2,
               col_estado=[f"q:{n}" for n in nombres], col_accion=[f"a:{n}" for n in nombres],
               dim_estado=23, dim_accion=23)
    red = act.red_desde_paquete(chk, nombre="viejo")
    assert red["legado"] and red["receta"]["nombres_brazo"] == BRAZO and red["ventana"] == 8
    pol = Politica(red)
    obs = dict(q_brazo=Q0, q_mano=np.zeros(17), q_mano_real=np.ones(17))   # (la real no la usa)
    pol.reiniciar(obs)
    p = pol.paso(obs)
    assert p.brazo.shape == (6,) and p.mano.shape == (17,) and p.progreso is None
    assert isinstance(torch.__version__, str)
