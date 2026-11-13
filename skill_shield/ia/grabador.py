"""
Grabación de episodios BRAZO + MANO desde la simulación (5-nov, formato 2).

Antes ([G], formato 1): solo los 17 joints de la MANO simulada, y solo en los fotogramas en
que MediaPipe veía tu mano (la grabación estaba dentro de ese `if`).

Ahora: el hilo de la física (HiloFisica) llama a `muestrear()` antes de CADA mj_step y la
grabadora guarda una fila cada 1/fps segundos de TIEMPO SIMULADO (30 Hz por defecto, como
LeRobot/ACT). Así el ritmo es fijo aunque la cámara vaya a tirones, y lo que se guarda es
exactamente lo que se le está mandando al robot en ese instante:
  · q:<joint>  posición SIMULADA de cada joint (6 del brazo FR5 + 17 de la OrcaHand)
  · a:<joint>  comando (ctrl) de ese joint: el del brazo ya suavizado por el generador
               (lo mismo que recibe fr5_servo) y el de la mano (guante MANUS / cámara)

Archivos (uno por episodio, en episodios/<tarea>/):
  episodio_0007.csv   cabecera + una fila por fotograma (t, t_reloj, q:..., a:...)
  episodio_0007.json  metadatos: formato, fps, robot, columnas, recorte...
Al guardar se recortan los ratos QUIETOS del principio y del final (dejando medio segundo):
si no, la red aprende a "esperar" sin hacer nada.
"""
import csv
import json
import os
import threading
import time

import numpy as np

FORMATO = 2


def nombres_y_indices(modelo, joints_brazo, offset_mano, n_mano):
    """(nombres de joint, direcciones qpos, índices ctrl) del brazo + la mano."""
    import mujoco
    nombres, qadr, cidx = [], [], []
    J = mujoco.mjtObj.mjOBJ_JOINT
    # brazo: el actuador cuyo joint es ese
    for nombre in joints_brazo:
        jid = mujoco.mj_name2id(modelo, J, nombre)
        act = [a for a in range(modelo.nu) if int(modelo.actuator_trnid[a, 0]) == jid]
        if jid < 0 or not act:
            raise ValueError(f"no encuentro el joint/actuador del brazo '{nombre}'")
        nombres.append(nombre)
        qadr.append(int(modelo.jnt_qposadr[jid]))
        cidx.append(act[0])
    # mano: los actuadores OFFSET_MANO … OFFSET_MANO+N_ACT (mismo orden que ctrl_mano)
    for i in range(n_mano):
        a = offset_mano + i
        jid = int(modelo.actuator_trnid[a, 0])
        nombres.append(mujoco.mj_id2name(modelo, J, jid))
        qadr.append(int(modelo.jnt_qposadr[jid]))
        cidx.append(a)
    return nombres, qadr, cidx


def recortar_quietos(acciones, fps, umbral=0.01, margen_s=0.5):
    """(inicio, fin) del tramo con movimiento: fuera los fotogramas del principio y del final
    en que ningún comando se ha movido más de 'umbral' (rad) respecto al primero/último."""
    a = np.asarray(acciones, dtype=float)
    n = len(a)
    if n < 3:
        return 0, n
    mov_ini = np.where(np.abs(a - a[0]).max(axis=1) > umbral)[0]
    mov_fin = np.where(np.abs(a - a[-1]).max(axis=1) > umbral)[0]
    if len(mov_ini) == 0:                     # no se movió nada en todo el episodio
        return 0, 0
    m = int(round(margen_s * fps))
    return max(0, int(mov_ini[0]) - m), min(n, int(mov_fin[-1]) + 1 + m)


def _siguiente_numero(carpeta):
    """Siguiente número de episodio libre en la carpeta (episodio_0001.csv, …)."""
    nums = []
    for f in os.listdir(carpeta) if os.path.isdir(carpeta) else []:
        if f.startswith("episodio_") and f.endswith(".csv"):
            try:
                nums.append(int(f[len("episodio_"):-4]))
            except ValueError:
                pass
    return max(nums) + 1 if nums else 1


class GrabadorEpisodio:
    def __init__(self, nombres, qadr, cidx, fps=30.0, robot="FR5", recortar=True):
        self.nombres, self.qadr, self.cidx = list(nombres), list(qadr), list(cidx)
        self.fps, self.robot, self.recortar = float(fps), robot, recortar
        self.activo = False
        self.tarea = None
        self._filas = []
        self._t_sig = 0.0
        self._t_ini = None
        self._hilos = []

    @classmethod
    def desde_modelo(cls, modelo, joints_brazo, offset_mano, n_mano, **kw):
        return cls(*nombres_y_indices(modelo, joints_brazo, offset_mano, n_mano), **kw)

    @property
    def columnas(self):
        return (["t", "t_reloj"] + [f"q:{n}" for n in self.nombres]
                + [f"a:{n}" for n in self.nombres])

    @property
    def n_filas(self):
        return len(self._filas)

    @property
    def segundos(self):
        return len(self._filas) / self.fps

    def iniciar(self, tarea, datos):
        """Llamar con el candado de la física (el bucle principal lo tiene casi siempre)."""
        self.tarea = tarea
        self._filas = []
        self._t_ini = float(datos.time)
        self._t_sig = float(datos.time)
        self.activo = True

    def muestrear(self, datos):
        """Desde el hilo de la física, ANTES de cada mj_step (ctrl ya escrito por el
        generador del brazo y por el guante)."""
        if not self.activo:
            return
        t = float(datos.time)
        if t + 1e-9 < self._t_sig:
            return
        q, c = datos.qpos, datos.ctrl
        self._filas.append([t - self._t_ini, time.time()]
                           + [float(q[i]) for i in self.qadr] + [float(c[i]) for i in self.cidx])
        self._t_sig += 1.0 / self.fps
        if t - self._t_sig > 2.0 / self.fps:     # la física se atrasó: no recuperar de golpe
            self._t_sig = t + 1.0 / self.fps

    def parar(self, carpeta, guardar=True):
        """Termina el episodio y lo guarda en segundo plano. Devuelve (ruta, n_fotogramas) o
        (None, 0) si no había nada que guardar."""
        self.activo = False
        filas, self._filas = self._filas, []
        if not guardar or not filas:
            return None, 0
        n = len(self.nombres)
        ini, fin = 0, len(filas)
        if self.recortar:
            ini, fin = recortar_quietos([f[2 + n:] for f in filas], self.fps)
        filas_ok = filas[ini:fin]
        if len(filas_ok) < 2:
            print("⏹ Episodio sin movimiento: no se guarda.")
            return None, 0
        os.makedirs(carpeta, exist_ok=True)
        ruta = os.path.join(carpeta, f"episodio_{_siguiente_numero(carpeta):04d}.csv")
        meta = dict(formato=FORMATO, fps=self.fps, robot=self.robot, tarea=self.tarea,
                    fecha=time.strftime("%Y-%m-%d %H:%M:%S"), n_fotogramas=len(filas_ok),
                    joints=self.nombres, columnas=self.columnas,
                    recorte=dict(inicio=ini, fin=fin, total=len(filas)))
        # se reserva el nombre ya (otro [G] seguido no debe pisarlo) y se escribe aparte
        open(ruta, "w").close()
        h = threading.Thread(target=self._escribir, args=(ruta, filas_ok, meta), daemon=False)
        h.start()
        self._hilos.append(h)
        return ruta, len(filas_ok)

    def _escribir(self, ruta, filas, meta):
        with open(ruta, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.columnas)
            w.writerows(filas)
        with open(ruta[:-4] + ".json", "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)

    def esperar(self):
        for h in self._hilos:
            h.join()
        self._hilos = []
