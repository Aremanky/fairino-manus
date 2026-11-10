"""
Web del modo autónomo (10-nov) — paso 2 de la hoja de ruta: web MÍNIMA de solo lectura.

Servidor HTTP de la librería estándar (sin dependencias nuevas). Se abre desde cualquier
equipo de la misma red en  http://<IP del PC del robot>:<AUTONOMO_WEB_PUERTO>/

  /             página con la imagen de la cámara de cabeza y el estado del robot
  /estado       JSON: estado del modo autónomo (FR5, cámara...)
  /video.mjpg   vídeo MJPEG de la cámara de cabeza
  /foto.jpg     último fotograma

DE MOMENTO NO TIENE NINGÚN MANDO: no hay nada que pueda mover el robot desde la web. Cuando
se añadan marcha/pausa/paro (paso 11) habrá que poner contraseña, porque cualquiera de la red
podría abrirla. El paro de la web nunca sustituye a la seta física.

Prueba aislada (enseña «sin datos» si no hay modo autónomo en marcha):
    python -m skill_shield.autonomo.web --puerto 8080
"""
import argparse
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from skill_shield.autonomo.canal import Canal

_PAGINA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Skill Shield · Autónomo</title>
<style>
:root{--fondo:#f4f5f7;--panel:#fff;--texto:#1c2430;--suave:#5d6b7c;--borde:#dde2e8;
--ok:#1f8a4c;--aviso:#b7791f;--mal:#c53030;--marca:#e2001a}
@media (prefers-color-scheme:dark){:root{--fondo:#11161d;--panel:#1a212b;--texto:#e6ebf1;
--suave:#93a1b3;--borde:#2a3441}}
*{box-sizing:border-box}body{margin:0;background:var(--fondo);color:var(--texto);
font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--borde);
background:var(--panel)}header b{font-size:17px}header .barra{width:4px;height:22px;background:var(--marca)}
#aviso{margin-left:auto;font-size:13px;color:var(--suave)}
main{display:grid;grid-template-columns:minmax(0,2fr) minmax(260px,1fr);gap:16px;padding:16px 20px;
max-width:1400px;margin:0 auto}@media (max-width:820px){main{grid-template-columns:1fr;padding:12px 16px}}
.panel{background:var(--panel);border:1px solid var(--borde);border-radius:10px;padding:14px 16px}
.panel h2{margin:0 0 10px;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--suave)}
#video{width:100%;aspect-ratio:4/3;object-fit:contain;background:#000;border-radius:6px;display:block}
.fila{display:flex;justify-content:space-between;gap:10px;padding:5px 0;border-bottom:1px solid var(--borde)}
.fila:last-child{border-bottom:0}.fila span:first-child{color:var(--suave)}
.fila span:last-child{text-align:right;font-variant-numeric:tabular-nums}
.punto{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;background:var(--suave)}
.ok{background:var(--ok)}.aviso{background:var(--aviso)}.mal{background:var(--mal)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td{padding:3px 0;border-bottom:1px solid var(--borde)}td:last-child{text-align:right}
.lateral{display:flex;flex-direction:column;gap:16px}
.nota{font-size:13px;color:var(--suave);margin:10px 0 0}
</style></head><body>
<header><span class="barra"></span><b>Skill Shield · Funcionamiento autónomo</b>
<span id="aviso">Solo lectura: desde esta página no se mueve el robot</span></header>
<main>
<section class="panel"><h2>Cámara de cabeza</h2>
<img id="video" src="/video.mjpg" alt="Imagen de la cámara de cabeza">
<p class="nota" id="cam_txt">—</p></section>
<div class="lateral">
<section class="panel"><h2>Estado</h2>
<div class="fila"><span>Modo</span><span id="modo">—</span></div>
<div class="fila"><span>FR5</span><span><i class="punto" id="fr5_p"></i><span id="fr5">—</span></span></div>
<div class="fila"><span>Movimiento</span><span id="mov">—</span></div>
<div class="fila"><span>Código controladora</span><span id="cod">—</span></div>
<div class="fila"><span>Cámara</span><span><i class="punto" id="cam_p"></i><span id="cam">—</span></span></div>
<div class="fila"><span>Gemelo (MuJoCo)</span><span id="gem">—</span></div>
</section>
<section class="panel"><h2>Articulaciones FR5 (pose real, °)</h2>
<table id="q"></table></section>
</div></main>
<script>
const $=id=>document.getElementById(id);
function punto(el,c){el.className='punto '+c}
async function refrescar(){
 try{
  const e=await (await fetch('/estado',{cache:'no-store'})).json();
  const vivo=e.edad_s!=null&&e.edad_s<2;
  $('modo').textContent=vivo?(e.modo||'—'):'sin datos del modo autónomo';
  const f=e.fr5||{};
  $('fr5').textContent=f.texto||'sin FR5';
  punto($('fr5_p'),!vivo||!f.en_linea?'mal':(f.en_error||f.colision?'aviso':'ok'));
  $('mov').textContent=f.simulacro===false?'REAL':'bloqueado (simulacro)';
  $('cod').textContent=f.codigo?f.codigo.join(' / '):'—';
  const c=e.camara||{};
  const camViva=c.ok&&c.edad_s!=null&&c.edad_s<2;
  $('cam').textContent=camViva?`${c.fps} fps · ${c.usb}`:(c.error?'error':'sin imagen');
  punto($('cam_p'),camViva?(c.usb&&c.usb.startsWith('SUPER')?'ok':'aviso'):'mal');
  $('cam_txt').textContent=camViva?`MxId ${c.mxid} · ${c.ancho}×${c.alto} px · centro: `+
    (c.prof_centro_mm?`${Math.round(c.prof_centro_mm)} mm`:'sin profundidad'):(c.error||'Esperando a la cámara…');
  $('gem').textContent=e.gemelo||'—';
  const q=f.q_real_deg;
  $('q').innerHTML=q?q.map((v,i)=>`<tr><td>J${i+1}</td><td>${v.toFixed(2)}</td></tr>`).join(''):
    '<tr><td>sin lectura</td><td></td></tr>';
 }catch(err){$('modo').textContent='sin conexión con el PC del robot'}
}
setInterval(refrescar,500);refrescar();
</script></body></html>
"""


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


class _Canales:
    """Se engancha a los canales cuando existan (la web puede arrancar antes que el resto)."""

    def __init__(self, img, info, estado):
        self._nombres = dict(img=img, info=info, estado=estado)
        self._c = {}
        self._lock = threading.Lock()

    def get(self, clave):
        nombre = self._nombres.get(clave)
        if not nombre:
            return None
        with self._lock:
            if clave not in self._c:
                try:
                    self._c[clave] = Canal(nombre)
                except FileNotFoundError:
                    return None
            return self._c[clave]

    def json(self, clave):
        c = self.get(clave)
        if c is None:
            return None, None
        datos, t = c.leer()
        if datos is None:
            return None, None
        try:
            return json.loads(datos), t
        except ValueError:
            return None, None


def crear_servidor(host, puerto, canales):
    class Manejador(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):                      # sin una línea por petición
            pass

        def _enviar(self, codigo, tipo, cuerpo):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_GET(self):
            ruta = self.path.split("?")[0]
            if ruta in ("/", "/index.html"):
                return self._enviar(200, "text/html; charset=utf-8", _PAGINA.encode())
            if ruta == "/estado":
                return self._enviar(200, "application/json", json.dumps(self._estado()).encode())
            if ruta == "/foto.jpg":
                c = canales.get("img")
                jpg = c.leer()[0] if c is not None else None
                if jpg is None:
                    return self._enviar(503, "text/plain; charset=utf-8", "Sin imagen".encode())
                return self._enviar(200, "image/jpeg", jpg)
            if ruta == "/video.mjpg":
                return self._video()
            return self._enviar(404, "text/plain; charset=utf-8", b"No encontrado")

        def _estado(self):
            ahora = time.time()
            e, t = canales.json("estado")
            e = dict(e or {})
            e["edad_s"] = round(ahora - t, 2) if t else None
            info, ti = canales.json("info")
            if info is not None:
                info["edad_s"] = round(ahora - ti, 2) if ti else None
            e["camara"] = info or {}
            return e

        def _video(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            leidos = None
            try:
                while True:
                    c = canales.get("img")
                    if c is None:
                        time.sleep(0.2)
                        continue
                    seq = c.secuencia()
                    if seq == leidos or seq % 2:
                        time.sleep(0.01)
                        continue
                    jpg, _ = c.leer()
                    leidos = seq
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass                                    # el navegador cerró la pestaña
            self.close_connection = True

    class Servidor(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    return Servidor((host, int(puerto)), Manejador)


def main():
    import config as cfg
    ap = argparse.ArgumentParser(description="Web del modo autónomo (solo lectura)")
    ap.add_argument("--host", default=getattr(cfg, "AUTONOMO_WEB_HOST", "0.0.0.0"))
    ap.add_argument("--puerto", type=int, default=int(getattr(cfg, "AUTONOMO_WEB_PUERTO", 8080)))
    ap.add_argument("--canal-img", default="skill_autonomo_img")
    ap.add_argument("--canal-info", default="skill_autonomo_caminfo")
    ap.add_argument("--canal-estado", default="skill_autonomo_estado")
    ap.add_argument("--padre", type=int, default=0, help="PID del modo autónomo (si muere, salir)")
    args = ap.parse_args()

    try:
        srv = crear_servidor(args.host, args.puerto,
                             _Canales(args.canal_img, args.canal_info, args.canal_estado))
    except OSError as e:
        print(f"✗ [autónomo] La web no puede escuchar en {args.host}:{args.puerto} ({e}). "
              f"¿Puerto ocupado? Cambia AUTONOMO_WEB_PUERTO en config.py.")
        return 1
    hilo = threading.Thread(target=srv.serve_forever, daemon=True)
    hilo.start()
    print(f"🌐 [autónomo] Web en http://localhost:{args.puerto}/")
    if args.host in ("0.0.0.0", ""):
        for ip in _ips_locales():
            print(f"   desde otro equipo de la red: http://{ip}:{args.puerto}/")
    try:
        while True:
            time.sleep(0.5)
            if args.padre and os.getppid() != args.padre:
                break
    except KeyboardInterrupt:
        pass
    srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
