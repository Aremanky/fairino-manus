"""
Proceso de RENDER del gemelo digital para la web (10-nov).

Dibuja el modelo de MuJoCo SIN ventana (mujoco.Renderer: solo la escena, sin los menús
laterales del visor) con la pose que le manda el modo autónomo, y deja las imágenes en
memoria compartida para que la web las sirva como vídeo:

  · canal «sim»:     vista libre del gemelo; se gira y se acerca desde la web (ratón/dedo).
  · canal «simcam»:  el gemelo visto DESDE la cámara de cabeza calibrada (camara_robot.json),
                     con sus mismos parámetros (focales y punto principal). La web lo superpone
                     a la imagen real: si la calibración está bien, el robot virtual tapa al real.
                     Se recarga solo cuando cambia el archivo (al recalibrar).

Entradas (canales): «qpos» (float64 × nq, lo escribe modo.py), «mando» (la vista libre y si se
ven los obstáculos, que pide la web) y «mapa» (vóxeles ocupados del mapa de obstáculos, en naranja). En un proceso aparte para no frenar el bucle del gemelo ni la web.

Backend de OpenGL sin ventana: MUJOCO_GL = egl (GPU, recomendado) · osmesa (CPU, lento) ·
glfw (necesita escritorio). Si el elegido falla, el proceso se relanza con el siguiente.

Prueba aislada (sin modo autónomo, brazo en reposo; abre una ventana de OpenCV):
    python -m skill_shield.autonomo.render_gemelo --ventana
"""
import argparse
import json
import os
import sys
import time

VISTA_DEFECTO = dict(azimut=90.0, elevacion=-20.0, distancia=1.1, mirar=[0.15, 0.0, 0.25])


def _parsear():
    import config as cfg
    ap = argparse.ArgumentParser(description="Render del gemelo para la web")
    ap.add_argument("--canal-qpos", default=None)
    ap.add_argument("--canal-sim", default=None)
    ap.add_argument("--canal-simcam", default=None)
    ap.add_argument("--canal-mando", default=None)
    ap.add_argument("--canal-mapa", default=None, help="mapa de obstáculos (vóxeles) para dibujarlo")
    ap.add_argument("--canal-herramientas", default=None, help="herramientas detectadas para dibujarlas")
    ap.add_argument("--ancho", type=int, default=int(getattr(cfg, "AUTONOMO_SIM_ANCHO", 960)))
    ap.add_argument("--alto", type=int, default=int(getattr(cfg, "AUTONOMO_SIM_ALTO", 540)))
    ap.add_argument("--fps", type=float, default=float(getattr(cfg, "AUTONOMO_SIM_FPS", 20)))
    ap.add_argument("--gl", default=getattr(cfg, "AUTONOMO_SIM_GL", "egl"))
    ap.add_argument("--calibracion", default=getattr(cfg, "CALIB_ROBOT_ARCHIVO", "camara_robot.json"))
    ap.add_argument("--calidad", type=int, default=int(getattr(cfg, "AUTONOMO_JPEG_CALIDAD", 80)))
    ap.add_argument("--padre", type=int, default=0)
    ap.add_argument("--ventana", action="store_true")
    return ap.parse_args()


BACKENDS_GL = ("egl", "osmesa", "glfw")


def _importar_mujoco(gl):
    """Importa mujoco con ese backend de OpenGL y prueba a crear un contexto. None si falla.
    (El backend solo se puede elegir UNA vez por proceso: si falla, main() se relanza con el
    siguiente.)"""
    os.environ["MUJOCO_GL"] = gl
    try:
        import mujoco
        r = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>"), 8, 8)
        _cerrar(r)
        return mujoco
    except Exception as e:                               # noqa: BLE001 — cada backend falla distinto
        print(f"⚠️  [gemelo web] MUJOCO_GL={gl} no funciona aquí ({e})")
        return None


def _cerrar(renderer):
    f = getattr(renderer, "close", None)
    if callable(f):
        f()


def _mando(mando_bytes):
    if not mando_bytes:
        return {}
    try:
        return json.loads(mando_bytes)
    except ValueError:
        return {}


def _vista(mando_bytes):
    v = _mando(mando_bytes).get("vista") or {}
    if not v:
        return dict(VISTA_DEFECTO)
    out = dict(VISTA_DEFECTO)
    out.update({k: v[k] for k in VISTA_DEFECTO if k in v})
    return out


def main():
    args = _parsear()
    import cv2
    import numpy as np
    import config as cfg
    from skill_shield.autonomo.canal import Canal
    from skill_shield.autonomo import calibracion as C

    gl = args.gl
    mujoco = _importar_mujoco(gl)
    if mujoco is None:
        siguientes = list(BACKENDS_GL[BACKENDS_GL.index(gl) + 1:]) if gl in BACKENDS_GL else []
        if siguientes:                                   # se relanza con el siguiente backend
            argv = [a for a in sys.argv[1:]]
            if "--gl" in argv:
                argv[argv.index("--gl") + 1] = siguientes[0]
            else:
                argv += ["--gl", siguientes[0]]
            os.execv(sys.executable, [sys.executable, "-m", "skill_shield.autonomo.render_gemelo"] + argv)
        print("✗ [gemelo web] Ningún backend de OpenGL sin ventana funciona: la web no tendrá gemelo "
              "(la ventana de MuJoCo sigue funcionando). Prueba  sudo apt install libegl1")
        return 1
    from skill_shield.simulacion.entorno_taller import EntornoTaller
    from skill_shield.simulacion.entorno_taladro import EntornoTaladro

    modelo = mujoco.MjModel.from_xml_path(cfg.RUTA_XML)
    for i in range(modelo.nmat):                         # sin reflejo del suelo (como el visor)
        modelo.mat_reflectance[i] = 0.0
    modelo.vis.global_.offwidth = max(int(modelo.vis.global_.offwidth), args.ancho, 1352)
    modelo.vis.global_.offheight = max(int(modelo.vis.global_.offheight), args.alto, 1014)
    datos = mujoco.MjData(modelo)
    EntornoTaller(modelo, datos)                         # escondidos, igual que en el modo autónomo
    EntornoTaladro(modelo, datos)
    mujoco.mj_forward(modelo, datos)
    render = mujoco.Renderer(modelo, args.alto, args.ancho)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    print(f"🧊 [gemelo web] Render {args.ancho}x{args.alto} a {args.fps:.0f} fps (MUJOCO_GL={gl})")

    canal_qpos = Canal.esperar(args.canal_qpos) if args.canal_qpos else None
    canal_sim = Canal.esperar(args.canal_sim) if args.canal_sim else None
    canal_simcam = Canal.esperar(args.canal_simcam) if args.canal_simcam else None
    canal_mando = Canal.esperar(args.canal_mando) if args.canal_mando else None
    canal_mapa = Canal.esperar(args.canal_mapa) if args.canal_mapa else None
    from skill_shield.autonomo.mapa import desempaquetar, dibujar_voxeles
    from skill_shield.autonomo.herramientas import dibujar_herramientas
    canal_herr = Canal.esperar(args.canal_herramientas) if args.canal_herramientas else None
    voxeles = (0.0, None)
    herr = ([], None)

    ruta_calib = args.calibracion if os.path.isabs(args.calibracion) else \
        os.path.join(os.path.dirname(os.path.abspath(cfg.__file__)), args.calibracion)
    calib, mtime_calib, render_cam = None, None, None
    periodo = 1.0 / max(args.fps, 1.0)
    t_cam = 0.0
    try:
        while True:
            t0 = time.monotonic()
            if args.padre and os.getppid() != args.padre:
                break
            if canal_qpos is not None:
                b, _ = canal_qpos.leer()
                if b and len(b) == 8 * modelo.nq:
                    datos.qpos[:] = np.frombuffer(b, dtype=np.float64)
                    mujoco.mj_forward(modelo, datos)

            # ── vista libre ──
            b_mando = canal_mando.leer()[0] if canal_mando is not None else None
            v = _vista(b_mando)
            cam.azimuth, cam.elevation, cam.distance = float(v["azimut"]), float(v["elevacion"]), float(v["distancia"])
            cam.lookat[:] = [float(x) for x in v["mirar"]]
            voxeles = (0.0, None)                        # obstáculos del mapa (si están recientes)
            if canal_mapa is not None and _mando(b_mando).get("ver_obstaculos", True):
                b_m, t_m = canal_mapa.leer()
                if b_m and t_m and time.time() - t_m < 3.0:
                    voxeles = desempaquetar(b_m)
            herr = ([], None)
            if canal_herr is not None:
                b_h, t_h = canal_herr.leer()
                if b_h and t_h and time.time() - t_h < 3.0:
                    d_h = _mando(b_h)
                    herr = (d_h.get("lista", []), d_h.get("objetivo"))
            render.update_scene(datos, camera=cam)
            if voxeles[1] is not None:
                dibujar_voxeles(render.scene, voxeles[1], voxeles[0])
            dibujar_herramientas(render.scene, *herr)
            rgb = render.render()
            if canal_sim is not None or args.ventana:
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, args.calidad])
                if ok and canal_sim is not None and len(jpg) <= canal_sim.capacidad:
                    canal_sim.escribir(jpg.tobytes())
                if args.ventana:
                    cv2.imshow("Gemelo (render web)", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

            # ── vista desde la cámara de cabeza calibrada (a 10 fps) ──
            if canal_simcam is not None and t0 - t_cam >= 0.1:
                t_cam = t0
                try:
                    m = os.path.getmtime(ruta_calib)
                except OSError:
                    m = None
                if m != mtime_calib:
                    mtime_calib = m
                    calib = C.cargar(ruta_calib) if m is not None else None
                    if render_cam is not None:
                        _cerrar(render_cam)
                        render_cam = None
                    if calib is not None:
                        # misma imagen que la web (640x480): K escalada desde el tamaño calibrado
                        i = calib["intrinsecos"]
                        w, h = int(getattr(cfg, "ANCHO_CAM", 640)), int(getattr(cfg, "ALTO_CAM", 480))
                        Kc = np.array(i["K"], float).copy()
                        Kc[0] *= w / float(i["ancho"])
                        Kc[1] *= h / float(i["alto"])
                        render_cam = mujoco.Renderer(modelo, h, w)
                        print(f"🎯 [gemelo web] Vista desde la cámara de cabeza calibrada ({calib['fecha']}, "
                              f"error medio {calib['error_medio_mm']} mm)")
                if render_cam is not None and calib is not None:
                    render_cam.update_scene(datos, camera=cam)
                    scn = render_cam.scene
                    if voxeles[1] is not None:              # así se ve si caen sobre los objetos reales
                        dibujar_voxeles(scn, voxeles[1], voxeles[0])
                    dibujar_herramientas(scn, *herr)
                    g = C.camara_gl(calib["T_mundo_cam"], Kc, w, h, float(scn.camera[0].frustum_near))
                    for c in (scn.camera[0], scn.camera[1]):
                        c.pos[:] = g["pos"]
                        c.forward[:] = g["forward"]
                        c.up[:] = g["up"]
                        c.frustum_center = g["frustum_center"]
                        c.frustum_top = g["frustum_top"]
                        c.frustum_bottom = g["frustum_bottom"]
                    rgb_c = render_cam.render()
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb_c, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, args.calidad])
                    if ok and len(jpg) <= canal_simcam.capacidad:
                        canal_simcam.escribir(jpg.tobytes())
            time.sleep(max(0.0, periodo - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        _cerrar(render)
        if render_cam is not None:
            _cerrar(render_cam)
    return 0


if __name__ == "__main__":
    sys.exit(main())
