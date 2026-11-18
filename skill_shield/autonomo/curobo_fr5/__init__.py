"""
cuRobo (NVIDIA, GPU) para el FR5 + OrcaHand (17-nov).

  urdf.py         MJCF del proyecto → URDF (cuRobo solo lee URDF): brazo con joint1..joint6 y la
                  mano FIJA en mano_mount con la envolvente de varias posturas de los dedos.
  robot.py        URDF → configuración de cuRobo (esferas de colisión ajustadas en la GPU con su
                  RobotBuilder, matriz de autocolisión). Se ejecuta UNA vez en el PC con la GPU.
  mundo.py        el mapa del planificador (cubos ocupados, mesas, soporte) → mundo de
                  cuRobo: distancia con signo (ESDF) en una rejilla + cápsulas y cajas. Sin GPU.
  planificador.py PlanificadorCuRobo: tramos articulares con el MotionPlanner de cuRobo.

Uso en el modo autónomo (config.py): AUTONOMO_TAREA_PLANIFICADOR = "curobo". cuRobo PROPONE cada
tramo y el planificador del proyecto lo VERIFICA con todas sus comprobaciones (mallas de la mano
con su postura, lo que lleva, el J4, las mesas...); si no lo acepta, o si cuRobo no está
instalado, se usa el camino propio como siempre. Instalación: ./arrancar_nivel6.sh (si falta);
pruebas: ./arrancar_nivel6.sh --probar o probar_curobo.py en la raíz del proyecto.
"""
