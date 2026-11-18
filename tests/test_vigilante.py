"""
(18-nov) Vigilante de los procesos del modo autónomo: si la cámara, la web o el gemelo se caen o se cuelgan,
se reinician solos; la web reiniciada reconoce las sesiones abiertas y no borra el mapa.

    python -m pytest tests/test_vigilante.py -q
"""
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from skill_shield.autonomo.vigilante import Vigilado, Vigilante, comprobar_http   # noqa: E402

RAPIDO = dict(gracia_s=0.5, sin_datos_s=0.6, espera_ini_s=0.2, espera_max_s=0.4, estable_s=60.0)


def _hijo(codigo):
    return [sys.executable, "-c", codigo]


def _latido(ruta, cada=0.05, durante=60):
    """Proceso que da señales de vida (toca un archivo) durante 'durante' s."""
    return _hijo(f"import time,pathlib\np=pathlib.Path({str(ruta)!r})\nt=time.time()\n"
                 f"while time.time()-t<{durante}:\n    p.write_text(str(time.time()))\n    time.sleep({cada})")


def _senal(ruta):
    def f():
        try:
            return os.path.getmtime(ruta)
        except OSError:
            return None
    return f


def _esperar(cond, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_si_se_cae_se_reinicia(tmp_path):
    v = Vigilado("prueba", _hijo("import time; time.sleep(0.3)"), str(tmp_path), **RAPIDO)
    vg = Vigilante([v], cada=0.05).arrancar()
    try:
        assert _esperar(lambda: v.reinicios >= 2)
        assert "se ha cerrado" in v.motivo
    finally:
        vg.parar()
    assert v.proc is None


def test_si_se_cuelga_se_cierra_y_se_relanza(tmp_path):
    ruta = tmp_path / "latido"
    # da señales 1 s y luego se queda vivo pero callado (colgado)
    cmd = _hijo(f"import time,pathlib\np=pathlib.Path({str(ruta)!r})\nt=time.time()\n"
                "while time.time()-t<1.0:\n    p.write_text('x'); time.sleep(0.05)\ntime.sleep(60)")
    v = Vigilado("prueba", cmd, str(tmp_path), senal=_senal(ruta), **RAPIDO)
    vg = Vigilante([v], cada=0.05).arrancar()
    try:
        p0 = v.proc
        assert _esperar(lambda: v.reinicios >= 1)
        assert "colgado" in v.motivo and p0.poll() is not None      # el colgado, cerrado de verdad
    finally:
        vg.parar()


def test_con_senales_no_se_toca(tmp_path):
    ruta = tmp_path / "latido"
    v = Vigilado("prueba", _latido(ruta), str(tmp_path), senal=_senal(ruta), **RAPIDO)
    vg = Vigilante([v], cada=0.05).arrancar()
    try:
        time.sleep(2.0)
        assert v.reinicios == 0 and v.estado()["vivo"]
    finally:
        vg.parar()


def test_la_senal_del_proceso_anterior_no_vale(tmp_path):
    """Tras un reinicio, la última señal es la del proceso que se cayó: si el nuevo no da ninguna, colgado."""
    ruta = tmp_path / "latido"
    ruta.write_text("viejo")
    v = Vigilado("prueba", _hijo("import time; time.sleep(60)"), str(tmp_path), senal=_senal(ruta), **RAPIDO)
    time.sleep(0.05)
    v.lanzar()
    try:
        assert v.problema() is None                                  # arrancando
        time.sleep(0.7)
        assert "colgado" in (v.problema() or "")
    finally:
        v.cerrar()


def test_espera_cada_vez_mas_si_no_arranca(tmp_path):
    v = Vigilado("prueba", _hijo("raise SystemExit(1)"), str(tmp_path),
                 **dict(RAPIDO, espera_ini_s=0.3, espera_max_s=10.0))
    vg = Vigilante([v], cada=0.02).arrancar()
    try:
        time.sleep(1.5)                     # 0,3 + 0,6 + … : en 1,5 s caben 2 reintentos, no 20
        assert 1 <= v.reinicios <= 3
    finally:
        vg.parar()


def test_comprobar_http_sin_servidor():
    assert "no responde" in comprobar_http("http://127.0.0.1:9/api/estado", timeout=0.3)()


def test_extra_mal_un_rato_es_colgado(tmp_path):
    estado = {"m": None}
    v = Vigilado("prueba", _hijo("import time; time.sleep(60)"), str(tmp_path), extra=lambda: estado["m"], **RAPIDO)
    v.lanzar()
    try:
        time.sleep(0.6)
        assert v.problema() is None
        estado["m"] = "no responde"
        assert v.problema() is None                                  # un fallo suelto no basta
        time.sleep(0.7)
        assert "no responde" in v.problema()
    finally:
        v.cerrar()


# ── la web reiniciada ──
def test_web_reiniciada_reconoce_la_sesion(monkeypatch):
    from skill_shield.autonomo.web import ServicioWeb
    monkeypatch.setenv("SKILL_SHIELD_WEB_SECRETO", "abc123")
    a = ServicioWeb("x_img", "x_info", "x_estado", "x_mando", clave="clave")
    tok = a.entrar("clave")
    assert tok and a.autorizado(tok)
    b = ServicioWeb("x_img", "x_info", "x_estado", "x_mando", clave="clave")      # la web nueva
    assert b.autorizado(tok)
    assert not b.autorizado(tok[:-1] + ("0" if tok[-1] != "0" else "1"))          # firma mala
    monkeypatch.setenv("SKILL_SHIELD_WEB_SECRETO", "otro")
    c = ServicioWeb("x_img", "x_info", "x_estado", "x_mando", clave="clave")      # otra sesión del modo
    assert not c.autorizado(tok)


def test_web_reiniciada_no_borra_el_mapa():
    from skill_shield.autonomo.mapa import ConstructorMapa
    import json

    class Mando:
        def __init__(self):
            self.m = None

        def leer(self):
            return json.dumps(self.m).encode(), time.time()
    borrados = []
    cm = ConstructorMapa.__new__(ConstructorMapa)       # solo la parte de los mandos
    cm.canal_mando, cm._borrado_visto, cm._fijo_vistos = Mando(), None, {}
    cm.mapa = type("M", (), {"borrar": lambda self: borrados.append(1)})()
    cm.correccion = type("C", (), {"borrar": lambda self: None, "t": __import__("numpy").zeros(3)})()
    cm.canal_mando.m = dict(mapa_borrar_id="aaaa-m0")
    cm._mandos()
    cm.canal_mando.m = dict(mapa_borrar_id="aaaa-m1")   # «Borrar mapa»
    cm._mandos()
    assert len(borrados) == 1
    cm.canal_mando.m = dict(mapa_borrar_id="bbbb-m0")   # la web se ha reiniciado
    cm._mandos()
    assert len(borrados) == 1
    cm.canal_mando.m = dict(mapa_borrar_id="bbbb-m1")   # y en la nueva, «Borrar mapa»
    cm._mandos()
    assert len(borrados) == 2
