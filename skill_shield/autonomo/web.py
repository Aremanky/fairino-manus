"""
Web del modo autónomo (10-nov) — FastAPI + uvicorn.

Se abre desde cualquier equipo de la misma red:  http://<IP del PC del robot>:<AUTONOMO_WEB_PUERTO>/

Lectura (abierta):
  GET  /                página (skill_shield/autonomo/estatico/index.html)
  GET  /api/estado      JSON con FR5, OrcaHand, táctiles, cámara, gemelo y control
  GET  /video.mjpg      vídeo MJPEG de la cámara de cabeza ·  GET /foto.jpg  último fotograma
  GET  /sim.mjpg        gemelo de MuJoCo (vista libre) ·  GET /simcam.mjpg  gemelo visto desde la
                        cámara de cabeza calibrada (para superponerlo a la imagen real)
  POST /api/vista       girar / acercar / mover / vista predefinida del gemelo (sin contraseña)
Mandos (con contraseña AUTONOMO_WEB_CLAVE; sesión por cookie):
  POST /api/login {"clave": "..."}      ·  POST /api/logout
  POST /api/pedal {"pulsado": true}     pedal de la web: mientras se mantiene pulsado el botón,
                                        la página manda este latido cada 100 ms. Sin latido
                                        durante AUTONOMO_PEDAL_LATIDO_S → pedal SUELTO (también si
                                        se corta la red, se cierra la pestaña o muere este proceso).
  POST /api/pedal/modo {"obligatorio": bool}   pedal obligatorio (hombre muerto) o no
  POST /api/reset                       falsa colisión: borrar error → disable → enable

La web no habla con el robot: deja los mandos en un canal de memoria compartida
(skill_autonomo_mando) y el proceso del modo autónomo decide (modo.py). El paro de la web
es solo funcional: la seta física sigue siendo la parada de verdad.

Prueba aislada (enseña «sin datos» si no hay modo autónomo en marcha):
    python -m skill_shield.autonomo.web --puerto 8080
Requiere:  pip install fastapi uvicorn
"""
import argparse
import asyncio
import hmac
import json
import os
import secrets
import socket
import sys
import threading
import time
import uuid

from skill_shield.autonomo.canal import Canal

AQUI = os.path.dirname(os.path.abspath(__file__))
ESTATICO = os.path.join(AQUI, "estatico")
COOKIE = "ss_sesion"
# Vista inicial del gemelo = la del visor de MuJoCo. Delante del robot = -Y del mundo.
VISTA_INICIAL = dict(azimut=90.0, elevacion=-20.0, distancia=1.1, mirar=[0.15, 0.0, 0.25])
VISTAS = {"inicial": "inicial", "frente": dict(azimut=90.0, elevacion=-8.0),
          "lado": dict(azimut=0.0, elevacion=-8.0), "arriba": dict(azimut=90.0, elevacion=-88.0)}
SESION_S = 12 * 3600


# ══════════════════════════════════════════════════════════════════════════════
# LÓGICA (sin FastAPI: se puede probar sola)
# ══════════════════════════════════════════════════════════════════════════════

class ServicioWeb:
    """Estado de la web: sesiones, pedal, modo del pedal, reset y lectura de los canales."""

    def __init__(self, canal_img, canal_info, canal_estado, canal_mando, clave=None,
                 permitir_sin_pedal=True, pedal_obligatorio=True, latido_s=0.35,
                 canal_sim=None, canal_simcam=None, vista=None):
        self._nombres = dict(img=canal_img, info=canal_info, estado=canal_estado, mando=canal_mando,
                             sim=canal_sim, simcam=canal_simcam)
        self._vista_inicial = dict(vista or VISTA_INICIAL)
        self.vista = dict(self._vista_inicial)
        self._canales = {}
        self._lock = threading.Lock()
        self.clave = clave or None
        self.permitir_sin_pedal = bool(permitir_sin_pedal)
        self.latido_s = float(latido_s)
        self._sesiones = {}                       # token → caducidad (time.time())
        self._pulsando = {}                       # token → último latido del pedal (time.time())
        self._pedal_obligatorio = bool(pedal_obligatorio) or not self.permitir_sin_pedal
        self._reset_base = uuid.uuid4().hex[:8]   # cambia en cada arranque de la web
        self._reset_n = 0

    # ── canales ──
    def canal(self, clave):
        nombre = self._nombres.get(clave)
        if not nombre:
            return None
        with self._lock:
            if clave not in self._canales:
                try:
                    self._canales[clave] = Canal(nombre)
                except FileNotFoundError:
                    return None
            return self._canales[clave]

    def _json(self, clave):
        c = self.canal(clave)
        if c is None:
            return None, None
        datos, t = c.leer()
        if datos is None:
            return None, None
        try:
            return json.loads(datos), t
        except ValueError:
            return None, None

    def jpg(self, clave="img"):
        c = self.canal(clave)
        return c.leer()[0] if c is not None else None

    def edad(self, clave):
        """Segundos desde el último mensaje de un canal (None si no hay)."""
        c = self.canal(clave)
        t = c.tiempo() if c is not None else None
        return round(time.time() - t, 2) if t else None

    # ── vista del gemelo (compartida por todos los que miran) ──
    def mover_vista(self, d):
        accion = d.get("accion")
        with self._lock:
            v = self.vista
            if accion == "rotar":
                v["azimut"] = (float(v["azimut"]) - 0.35 * float(d.get("dx", 0))) % 360.0
                v["elevacion"] = float(min(-2.0, max(-89.0, float(v["elevacion"]) - 0.35 * float(d.get("dy", 0)))))
            elif accion == "zoom":
                f = float(d.get("factor", 1.0))
                v["distancia"] = float(min(4.0, max(0.25, float(v["distancia"]) * min(2.0, max(0.5, f)))))
            elif accion == "mover":                     # desplazar el punto al que mira
                import math
                az = math.radians(float(v["azimut"]))
                k = 0.0015 * float(v["distancia"])
                dx, dy = float(d.get("dx", 0)), float(d.get("dy", 0))
                m = list(v["mirar"])
                m[0] += k * (dx * math.sin(az))
                m[1] += k * (-dx * math.cos(az))
                m[2] += k * dy
                v["mirar"] = m
            elif accion == "preset":
                p = VISTAS.get(d.get("nombre"))
                if p is not None:
                    self.vista = dict(self._vista_inicial) if p == "inicial" else \
                        dict(self._vista_inicial, **p)
            vista = dict(self.vista)
        self._escribir_mando()
        return vista

    # ── sesiones ──
    @property
    def requiere_clave(self):
        return self.clave is not None

    def entrar(self, clave):
        """Token de sesión si la clave es buena (o si no hay clave), si no None."""
        if self.requiere_clave and not hmac.compare_digest(str(clave or ""), self.clave):
            return None
        token = secrets.token_urlsafe(24)
        with self._lock:
            ahora = time.time()
            self._sesiones = {k: v for k, v in self._sesiones.items() if v > ahora}
            self._sesiones[token] = ahora + SESION_S
        return token

    def salir(self, token):
        with self._lock:
            self._sesiones.pop(token, None)
            self._pulsando.pop(token, None)
        self._escribir_mando()

    def autorizado(self, token):
        if not self.requiere_clave:
            return True
        with self._lock:
            cad = self._sesiones.get(token or "")
            return cad is not None and cad > time.time()

    # ── mandos ──
    def pedal(self, token, pulsado):
        with self._lock:
            if pulsado:
                self._pulsando[token or "-"] = time.time()
            else:
                self._pulsando.pop(token or "-", None)
        self._escribir_mando()

    def pedal_pisado(self):
        ahora = time.time()
        with self._lock:
            return any(ahora - t < self.latido_s for t in self._pulsando.values())

    def poner_obligatorio(self, obligatorio):
        if not obligatorio and not self.permitir_sin_pedal:
            return False, "Funcionar sin pedal está bloqueado en config.py (AUTONOMO_PERMITIR_SIN_PEDAL)."
        with self._lock:
            self._pedal_obligatorio = bool(obligatorio)
        self._escribir_mando()
        return True, "Pedal obligatorio" if obligatorio else "Pedal desactivado: el robot puede moverse sin pedal"

    def reset(self):
        with self._lock:
            self._reset_n += 1
        self._escribir_mando()

    def _escribir_mando(self):
        c = self.canal("mando")
        if c is None:
            return
        with self._lock:
            ahora = time.time()      # latidos viejos (red cortada sin «soltar»): fuera
            self._pulsando = {k: v for k, v in self._pulsando.items() if ahora - v < 5.0}
            ultimo = max(self._pulsando.values(), default=0.0)
            datos = dict(pedal=bool(self._pulsando), pedal_t=ultimo,
                         pedal_obligatorio=self._pedal_obligatorio, vista=self.vista,
                         reset_id=f"{self._reset_base}-{self._reset_n}" if self._reset_n else None,
                         t=time.time())
        c.escribir(json.dumps(datos).encode())

    def latido_mando(self):
        """Reescribe el mando cada poco (el proceso principal sabe así que la web está viva)."""
        self._escribir_mando()

    # ── estado para la página ──
    def estado(self, token=None):
        ahora = time.time()
        e, t = self._json("estado")
        e = dict(e or {})
        e["edad_s"] = round(ahora - t, 2) if t else None
        info, ti = self._json("info")
        if info is not None:
            info["edad_s"] = round(ahora - ti, 2) if ti else None
        e["camara"] = info or {}
        ctrl = dict(e.get("control") or {})
        with self._lock:
            ctrl["pedal_obligatorio"] = self._pedal_obligatorio
        ctrl["pedal_pisado_web"] = self.pedal_pisado()
        ctrl["permitir_sin_pedal"] = self.permitir_sin_pedal
        e["control"] = ctrl
        e["sesion"] = dict(autorizado=self.autorizado(token), requiere_clave=self.requiere_clave)
        e["gemelo_web"] = dict(edad_s=self.edad("sim"), cam_edad_s=self.edad("simcam"))
        return e


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

def crear_app(servicio):
    from fastapi import FastAPI, Request
    from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, Response,
                                   StreamingResponse)

    app = FastAPI(title="Skill Shield · Funcionamiento autónomo", docs_url=None, redoc_url=None)

    def _token(request):
        return request.cookies.get(COOKIE)

    async def _cuerpo(request):
        try:
            return await request.json()
        except Exception:
            return {}

    def _no_autorizado():
        return JSONResponse({"ok": False, "msg": "Desbloquea los mandos con la contraseña"}, status_code=401)

    @app.get("/")
    async def pagina():
        with open(os.path.join(ESTATICO, "index.html"), encoding="utf-8") as f:
            return HTMLResponse(f.read(), headers={"Cache-Control": "no-store"})

    @app.get("/logo.png")
    async def logo():
        return FileResponse(os.path.join(ESTATICO, "logo_idi.png"), media_type="image/png")

    @app.get("/api/estado")
    async def estado(request: Request):
        return JSONResponse(servicio.estado(_token(request)), headers={"Cache-Control": "no-store"})

    @app.get("/foto.jpg")
    async def foto():
        jpg = servicio.jpg()
        if jpg is None:
            return Response("Sin imagen", status_code=503, media_type="text/plain; charset=utf-8")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def _mjpeg(request, clave):
        async def fotogramas():
            leido = None
            while not await request.is_disconnected():
                c = servicio.canal(clave)
                seq = c.secuencia() if c is not None else None
                if c is None or seq == leido or seq % 2:
                    await asyncio.sleep(0.01 if c is not None else 0.3)
                    continue
                jpg, _ = c.leer()
                leido = seq
                if jpg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        return StreamingResponse(fotogramas(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/video.mjpg")
    async def video(request: Request):
        return _mjpeg(request, "img")

    @app.get("/sim.mjpg")
    async def sim(request: Request):
        return _mjpeg(request, "sim")

    @app.get("/simcam.mjpg")
    async def simcam(request: Request):
        return _mjpeg(request, "simcam")

    @app.post("/api/vista")
    async def vista(request: Request):
        # mover la vista del gemelo no toca el robot: no pide contraseña
        return JSONResponse({"ok": True, "vista": servicio.mover_vista(await _cuerpo(request))})

    @app.post("/api/login")
    async def login(request: Request):
        token = servicio.entrar((await _cuerpo(request)).get("clave"))
        if token is None:
            return JSONResponse({"ok": False, "msg": "Contraseña incorrecta"}, status_code=401)
        r = JSONResponse({"ok": True})
        r.set_cookie(COOKIE, token, max_age=SESION_S, httponly=True, samesite="strict")
        return r

    @app.post("/api/logout")
    async def logout(request: Request):
        servicio.salir(_token(request))
        r = JSONResponse({"ok": True})
        r.delete_cookie(COOKIE)
        return r

    @app.post("/api/pedal")
    async def pedal(request: Request):
        tok = _token(request)
        pulsado = bool((await _cuerpo(request)).get("pulsado"))
        if pulsado and not servicio.autorizado(tok):
            return _no_autorizado()
        servicio.pedal(tok, pulsado)                   # soltar se acepta siempre
        return JSONResponse({"ok": True, "pisado": servicio.pedal_pisado()})

    @app.post("/api/pedal/modo")
    async def pedal_modo(request: Request):
        if not servicio.autorizado(_token(request)):
            return _no_autorizado()
        ok, msg = servicio.poner_obligatorio(bool((await _cuerpo(request)).get("obligatorio", True)))
        return JSONResponse({"ok": ok, "msg": msg}, status_code=200 if ok else 409)

    @app.post("/api/reset")
    async def reset(request: Request):
        if not servicio.autorizado(_token(request)):
            return _no_autorizado()
        servicio.reset()
        return JSONResponse({"ok": True, "msg": "Reinicio pedido: borrar error → deshabilitar → habilitar"})

    return app


def _ips_locales():
    """IPs de este PC en la red (para decir dónde abrir la web)."""
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))               # no envía nada: solo elige la interfaz
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def main():
    import config as cfg
    ap = argparse.ArgumentParser(description="Web del modo autónomo")
    ap.add_argument("--host", default=getattr(cfg, "AUTONOMO_WEB_HOST", "0.0.0.0"))
    ap.add_argument("--puerto", type=int, default=int(getattr(cfg, "AUTONOMO_WEB_PUERTO", 8080)))
    ap.add_argument("--canal-img", default="skill_autonomo_img")
    ap.add_argument("--canal-info", default="skill_autonomo_caminfo")
    ap.add_argument("--canal-estado", default="skill_autonomo_estado")
    ap.add_argument("--canal-mando", default="skill_autonomo_mando")
    ap.add_argument("--canal-sim", default="skill_autonomo_sim")
    ap.add_argument("--canal-simcam", default="skill_autonomo_simcam")
    ap.add_argument("--padre", type=int, default=0, help="PID del modo autónomo (si muere, salir)")
    args = ap.parse_args()

    try:
        import uvicorn
        import fastapi  # noqa: F401
    except ImportError as e:
        print(f"✗ [autónomo] Falta FastAPI/uvicorn ({e}). Instálalo en el entorno:\n"
              f"   ./venv_nivel6/bin/pip install fastapi uvicorn")
        return 1

    servicio = ServicioWeb(args.canal_img, args.canal_info, args.canal_estado, args.canal_mando,
                           clave=getattr(cfg, "AUTONOMO_WEB_CLAVE", None),
                           permitir_sin_pedal=getattr(cfg, "AUTONOMO_PERMITIR_SIN_PEDAL", True),
                           pedal_obligatorio=getattr(cfg, "AUTONOMO_PEDAL_OBLIGATORIO", True),
                           latido_s=getattr(cfg, "AUTONOMO_PEDAL_LATIDO_S", 0.35),
                           canal_sim=args.canal_sim, canal_simcam=args.canal_simcam,
                           vista=getattr(cfg, "AUTONOMO_SIM_VISTA", None))
    servidor = uvicorn.Server(uvicorn.Config(crear_app(servicio), host=args.host, port=args.puerto,
                                             log_level="warning", access_log=False,
                                             timeout_graceful_shutdown=1))   # el vídeo no la bloquea

    def _vigilar():
        while not servidor.should_exit:
            servicio.latido_mando()
            if args.padre and os.getppid() != args.padre:
                print("🌐 [autónomo] El proceso principal ya no existe → cierro la web.")
                servidor.should_exit = True
            time.sleep(0.1)

    threading.Thread(target=_vigilar, daemon=True).start()
    print(f"🌐 [autónomo] Web en http://localhost:{args.puerto}/")
    if args.host in ("0.0.0.0", ""):
        for ip in _ips_locales():
            print(f"   desde otro equipo de la red: http://{ip}:{args.puerto}/")
    if servicio.requiere_clave:
        print("   Mandos (pedal, reinicio) protegidos con AUTONOMO_WEB_CLAVE de config.py")
    try:
        servidor.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
