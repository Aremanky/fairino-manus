"""
PlanificadorCuRobo (17-nov): tramos articulares (de una postura a otra) con el MotionPlanner de
cuRobo en la GPU, contra el mundo del planificador del proyecto (mundo.py).

Solo PROPONE: tarea.Planificador.conectar() comprueba cada postura del tramo con sus propias
comprobaciones y, si no vale, usa su camino de siempre. Necesita cuRobo V2 instalado
(instalar_curobo.sh) y la configuración del robot (python -m skill_shield.autonomo.curobo_fr5.robot).
"""
import os
import time

import numpy as np

from skill_shield.autonomo.curobo_fr5 import mundo as M


def _radio_max_esferas(robot):
    """Radio de la esfera de colisión más gruesa de la configuración de cuRobo (0,06 m si no se encuentra)."""
    rs = []

    def buscar(x):
        if isinstance(x, dict):
            if "radius" in x and isinstance(x["radius"], (int, float)):
                rs.append(float(x["radius"]))
            for v in x.values():
                buscar(v)
        elif isinstance(x, list):
            for v in x:
                buscar(v)
    buscar(robot)
    return max(rs) if rs else 0.06


class PlanificadorCuRobo:
    def __init__(self, ruta_robot, entorno, limites, res=0.02, intentos=4, cuda_graph=False, margen_entorno=0.0,
                 dispositivo="cuda:0", calentar=True, semillas=8, desconocido_extra=0.02, activacion=None):
        """semillas: trayectorias que cuRobo optimiza A LA VEZ en la GPU para cada tramo (se queda con la mejor).
        desconocido_extra (17-nov): lo no visto se le da a cuRobo como «a (esfera más gruesa + esto) de un
        obstáculo»: no es choque, pero con la distancia de activación (por defecto, esto + la esfera más
        gruesa) su optimizador lo penaliza."""
        import torch
        import yaml
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.scene import Capsule, Cuboid, Scene, VoxelGrid
        from curobo.types import JointState
        from curobo._src.types.device_cfg import DeviceCfg
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch no ve la GPU (CUDA)")
        if not os.path.exists(ruta_robot):
            raise FileNotFoundError(f"no está la configuración de cuRobo del robot ({ruta_robot}): "
                                    "python -m skill_shield.autonomo.curobo_fr5.robot")
        self.torch, self.JointState = torch, JointState
        self._Scene, self._Cuboid, self._Capsule, self._VoxelGrid = Scene, Cuboid, Capsule, VoxelGrid
        self.dev = DeviceCfg(device=torch.device(dispositivo), dtype=torch.float32)
        self.limites, self.res, self.intentos = limites, float(res), int(intentos)
        self._caps, self._cajas = M.obstaculos_entorno(entorno, margen_entorno)
        _, n = M.rejilla(limites, self.res)
        origen = np.asarray(limites, float).reshape(3, 2)[:, 0]
        vacia = np.full(tuple(n), 1.0, np.float32)
        with open(ruta_robot) as f:
            robot = yaml.safe_load(f)
        self.r_max = _radio_max_esferas(robot)
        self.desconocido_dist = self.r_max + float(desconocido_extra) if desconocido_extra is not None else None
        act = float(activacion) if activacion is not None else (
            self.desconocido_dist + 0.02 if self.desconocido_dist is not None else 0.0025)
        cfg = MotionPlannerCfg.create(
            robot=robot, scene_model=self._escena(vacia, origen + 0.5 * n * self.res, n * self.res),
            collision_cache={"cuboid": len(self._cajas) + 4, "capsule": len(self._caps) + 4, "voxel": 1},
            device_cfg=self.dev, use_cuda_graph=bool(cuda_graph), graph_planner_algorithm="rrt",
            num_trajopt_seeds=int(semillas), num_ik_seeds=max(32, 4 * int(semillas)),
            optimizer_collision_activation_distance=act)
        self.mp = MotionPlanner(cfg)
        self.nombres = list(self.mp.joint_names)
        orden = [f"joint{i}" for i in range(1, 7)]
        if sorted(self.nombres) != sorted(orden):
            raise ValueError(f"los joints de cuRobo no son joint1..joint6: {self.nombres}")
        self._a_curobo = [orden.index(nm) for nm in self.nombres]          # q (joint1..6) → orden de cuRobo
        self._de_curobo = [self.nombres.index(nm) for nm in orden]
        self._clave_mundo = None
        self.t_ultimo = 0.0
        if calentar:
            t0 = time.perf_counter()
            self.mp.warmup(enable_graph=True, num_warmup_iterations=2)
            print(f"🚀 cuRobo listo en la GPU ({torch.cuda.get_device_name(self.dev.device)}; "
                  f"calentado en {time.perf_counter() - t0:.1f} s)")

    def _escena(self, sdf, centro, dims):
        t = self.torch
        vox = self._VoxelGrid(name="mapa", pose=[float(v) for v in centro] + [1.0, 0.0, 0.0, 0.0],
                              dims=[float(v) for v in dims], voxel_size=self.res,
                              feature_tensor=t.as_tensor(sdf, dtype=t.float16, device=self.dev.device),
                              feature_dtype=t.float16)
        caps = [self._Capsule(name=c["name"], radius=c["radius"], base=c["base"], tip=c["tip"],
                              pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) for c in self._caps]
        cajas = [self._Cuboid(name=c["name"], dims=c["dims"], pose=c["pose"]) for c in self._cajas]
        return self._Scene(voxel=[vox], cuboid=cajas, capsule=caps)

    def actualizar_mundo(self, pl, clave=None):
        """Carga en cuRobo el mundo actual del planificador `pl` (si ha cambiado desde la última vez)."""
        if clave is not None and clave == self._clave_mundo:
            return
        sdf, centro, dims = M.mundo_del_planificador(pl, self.limites, self.res, self.desconocido_dist)
        self.mp.update_world(self._escena(sdf, centro, dims))
        self._clave_mundo = clave

    def plan(self, q0, q1):
        """Tramo articular de q0 a q1 (radianes, joint1..6): posturas (N x 6) o None."""
        t = self.torch
        t0 = time.perf_counter()
        a = t.tensor([np.asarray(q0, float)[self._a_curobo]], **self.dev.as_torch_dict())
        b = t.tensor([np.asarray(q1, float)[self._a_curobo]], **self.dev.as_torch_dict())
        r = self.mp.plan_cspace(self.JointState.from_position(b, joint_names=self.nombres),
                                self.JointState.from_position(a, joint_names=self.nombres),
                                max_attempts=self.intentos)
        self.t_ultimo = time.perf_counter() - t0
        if r is None or r.success is None or not bool(r.success.any()):
            return None
        pos = r.get_interpolated_plan().position.detach().float().cpu().numpy().reshape(-1, len(self.nombres))
        qs = pos[:, self._de_curobo].astype(float)
        qs[0], qs[-1] = np.asarray(q0, float), np.asarray(q1, float)          # puntas exactas
        return qs
