"""
Configuración de cuRobo del FR5 + OrcaHand (17-nov). Se ejecuta UNA vez (y cada vez que cambie el
MJCF o las posturas de la mano), en el PC con la GPU y cuRobo instalado:

    python -m skill_shield.autonomo.curobo_fr5.robot              # → curobo_fr5/fr5_orca.yml
    python -m skill_shield.autonomo.curobo_fr5.robot --ver        # y enseña las esferas (http://localhost:8080)

1. genera el URDF desde el MJCF (urdf.py);
2. ajusta esferas de colisión a cada eslabón con el RobotBuilder de cuRobo (MorphIt, en la GPU);
3. calcula qué pares de eslabones no hace falta mirar (autocolisión);
4. lo guarda en YAML (lo que lee planificador.py).
"""
import argparse
import os
import time


def construir(ruta_urdf, raiz, salida, densidad=1.5, muestras=2000, ver=False):
    from curobo.robot_builder import RobotBuilder
    t0 = time.perf_counter()
    b = RobotBuilder(urdf_path=ruta_urdf, asset_path=raiz, tool_frames=["mano_mount"])
    print("🔵 Ajustando esferas de colisión (MorphIt, GPU)…")
    # el soporte del robot y el suelo ya son obstáculos del mundo: la base no lleva esferas por debajo
    b.fit_collision_spheres(sphere_density=densidad, use_collision_mesh=True, compute_metrics=True,
                            clip_links={"base_link": ("z", 0.0)})
    for nombre, m in (b.link_metrics or {}).items():
        print(f"   {nombre:<16s} {m.num_spheres:4d} esferas  cubre {m.coverage * 100:5.1f} %  "
              f"sobresale {m.protrusion_dist_mean * 1000:5.1f} mm")
    print("🔵 Autocolisión: pares que no hace falta mirar…")
    b.compute_collision_matrix(prune_collisions=True, num_samples=muestras)
    cfg = b.build()
    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    b.save(cfg, salida)
    print(f"✅ {b.num_spheres} esferas en {len(b.collision_link_names)} eslabones → {salida} "
          f"({time.perf_counter() - t0:.0f} s)")
    if ver:
        print("👀 Esferas en http://localhost:8080 (Ctrl+C para salir)")
        b.visualize(cfg, port=8080)
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
    return salida


def main():
    import config as cfg
    from skill_shield.autonomo.curobo_fr5.urdf import generar_urdf, posturas_por_defecto
    raiz = os.path.dirname(os.path.abspath(cfg.__file__))
    ap = argparse.ArgumentParser(description="Configuración de cuRobo del FR5 + OrcaHand")
    ap.add_argument("--salida", default=os.path.join(raiz, getattr(cfg, "AUTONOMO_CUROBO_ROBOT",
                                                                   "curobo_fr5/fr5_orca.yml")))
    ap.add_argument("--densidad", type=float, default=1.5, help="esferas por eslabón (1 = las de cuRobo)")
    ap.add_argument("--ver", action="store_true", help="enseñar las esferas en el navegador")
    a = ap.parse_args()
    urdf = generar_urdf(cfg.RUTA_XML, os.path.join(os.path.dirname(a.salida), "fr5_orca.urdf"), raiz=raiz,
                        posturas=posturas_por_defecto(cfg))
    print(f"📄 URDF: {urdf}")
    construir(urdf, raiz, a.salida, a.densidad, ver=a.ver)


if __name__ == "__main__":
    main()
