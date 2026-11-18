"""
Mapa de obstáculos (paso 7 de la hoja de ruta v2, 10-nov) — funciones puras (numpy + scipy).

La profundidad de la cámara de cabeza se convierte en una rejilla de cubos (vóxeles) en el
marco del MUNDO (el del MJCF), usando la calibración cámara→robot (camara_robot.json):

  imagen de profundidad → puntos 3D (cámara) → mundo → se quitan el ROBOT (sus cápsulas
  con la pose real, más un margen), el SUELO y lo que se pida excluir (la herramienta que
  se va a coger) → cada vóxel acumula evidencia:
     · OCUPADO  si cae un punto dentro (varias imágenes seguidas, no un píxel suelto);
     · LIBRE    si la cámara ve a través de él (la profundidad medida en su píxel está más
                lejos que el vóxel);
     · DESCONOCIDO si está tapado por algo (detrás de un objeto) o lleva mucho sin verse.
  Lo que no se ve durante AUTONOMO_MAPA_OLVIDO_S vuelve a «desconocido» (el robot es móvil:
  si se mueve la peana, lo antiguo deja de valer; también hay un botón «Borrar mapa»).

Para las colisiones (planificador, calibración automática…) se precalcula la DISTANCIA de
cada vóxel al obstáculo más cercano (transformada de distancia): comprobar una cápsula del
robot es mirar esa distancia en unos pocos puntos de su eje. Ver ColisionMapa.
"""
import math
import os
import time

import numpy as np

DESCONOCIDO, LIBRE, OCUPADO = 0, 1, 2


def pixeles_fiables(prof_mm, img_bgr=None, tol_rel=0.02, tol_abs=0.015, saturado=250):
    """Máscara de los píxeles de profundidad en los que fiarse para el mapa (11-nov):
      · COHERENTES con sus vecinos: la mediana 5x5 no se separa más de max(1,5 cm, 2 %). Fuera
        los «píxeles voladores» de los bordes y el ruido suelto de las zonas sin textura.
      · NO SATURADOS en color: ventanas y fluorescentes dan profundidades inventadas.
    (Sin el relleno de huecos ni el filtro temporal de la OAK, que también inventan.)"""
    p = prof_mm.astype(np.uint16) if prof_mm.dtype != np.uint16 else prof_mm
    import cv2
    med = cv2.medianBlur(p, 5).astype(np.float32)
    z = p.astype(np.float32)
    ok = (z > 0) & (med > 0) & (np.abs(z - med) <= np.maximum(tol_abs * 1000.0, tol_rel * med))
    if img_bgr is not None and img_bgr.shape[:2] == prof_mm.shape[:2]:
        ok &= img_bgr.max(axis=2) < saturado
    return ok


def nube_desde_profundidad(prof_mm, K, paso=2, z_min=0.35, z_max=2.5, fiables=None):
    """Puntos (N x 3, metros, marco de la cámara OpenCV) de una imagen de profundidad en mm.
    paso = 2 → se usa 1 de cada 2 píxeles en cada eje (sobra para vóxeles de 2 cm).
    z_min: por debajo de ~35 cm la OAK-D no mide (sin disparidad extendida): lo que salga es ruido.
    fiables: máscara de pixeles_fiables() (None = todos)."""
    p = prof_mm[::paso, ::paso].astype(np.float32) * 0.001
    h, w = p.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * paso
    v = v * paso
    ok = (p > z_min) & (p < z_max)
    if fiables is not None:
        ok &= fiables[::paso, ::paso]
    z = p[ok]
    x = (u[ok] - K[0, 2]) / K[0, 0] * z
    y = (v[ok] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1)


def distancia_a_segmentos(P, A, B):
    """Distancia (N,) de cada punto de P al más cercano de los segmentos A[k]-B[k]."""
    d = np.full(len(P), np.inf, dtype=np.float32)
    for a, b in zip(A, B):
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0) if L2 > 1e-12 else np.zeros(len(P))
        dist = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        d = np.minimum(d, dist)
    return d


def _en_caja(P, caja):
    """Puntos dentro de una caja (centro, semiejes) alineada con el mundo o (centro, semiejes, R)
    orientada (R: columnas = ejes de la caja en el mundo)."""
    c, semi = np.asarray(caja[0], float), np.asarray(caja[1], float)
    lo, hi = c - np.linalg.norm(semi), c + np.linalg.norm(semi)
    sel = np.nonzero(np.all((P >= lo) & (P <= hi), axis=1))[0]
    out = np.zeros(len(P), bool)
    if not len(sel):
        return out
    D = P[sel] - c
    if len(caja) > 2 and caja[2] is not None:
        D = D @ np.asarray(caja[2], float)
    out[sel[np.all(np.abs(D) <= semi, axis=1)]] = True
    return out


class MapaOcupacion:
    """Rejilla de vóxeles con evidencia (log-odds) de ocupado / libre."""

    L_OCUPADO, L_LIBRE = 0.85, 0.4          # cuánto suma un acierto / resta «verlo a través»
    L_MIN, L_MAX = -2.0, 3.5
    UMBRAL_OCUPADO, UMBRAL_LIBRE = 2.0, -0.6   # 11-nov: 3 aciertos seguidos (antes 2)

    W_MIN, D_OCUPADO, D_LIBRE = 3.0, 0.35, 0.6   # (TSDF) peso mínimo para fiarse; umbrales de distancia

    def __init__(self, limites, resolucion=0.02, olvido_s=30.0, z_suelo=None, margen_robot=0.03,
                 margen_suelo=0.015, min_puntos=4, min_grupo=4, z_min=0.35, metodo="impactos",
                 dist_max=None, k_trunc=0.02, w_max=20.0, motor="auto", superficies=False, banda_mesa=None):
        """metodo (17-nov): "impactos" (contar puntos por cubo, el de siempre) o "tsdf" (fusión de
        la distancia a la superficie: el ruido se promedia; ver tsdf.py). dist_max: más lejos de esto
        (m) la profundidad no pone obstáculos. superficies: buscar mesas como planos."""
        lim = np.asarray(limites, float).reshape(3, 2)
        self.res = float(resolucion)
        self.origen = lim[:, 0]
        self.n = np.maximum(np.ceil((lim[:, 1] - lim[:, 0]) / self.res).astype(int), 1)
        self.olvido_s = float(olvido_s)
        self.z_suelo = z_suelo
        self.margen_robot = float(margen_robot)
        self.margen_suelo = float(margen_suelo)
        self.min_puntos = int(min_puntos)
        self.min_grupo = int(min_grupo)         # cubos ocupados sueltos (grupos más pequeños): ruido
        self.z_min = float(z_min)
        self.L = np.zeros(tuple(self.n), np.float32)
        self.t_visto = np.zeros(tuple(self.n), np.float64)      # time.time() de la última vez que se vio
        ix, iy, iz = np.meshgrid(*[np.arange(k) for k in self.n], indexing="ij")
        self._centros = (np.stack([ix, iy, iz], -1).reshape(-1, 3) + 0.5) * self.res + self.origen
        self._dist = None                                         # transformada de distancia (caché)
        self._dist_desc = None
        self.excluir = []                                         # cajas (centro, semi) a ignorar
        self.ultima = dict(puntos=0, robot=0, suelo=0, ms=0.0)
        self.ultimos_puntos = None
        # (17-nov) TSDF, distancia máxima y mesas como planos
        self.metodo = str(metodo)
        self.dist_max = None if dist_max is None else float(dist_max)
        self.trunc_0, self.k_trunc, self.w_max = 2.0 * self.res, float(k_trunc), float(w_max)
        self.D = np.ones(self.L.size, np.float32)                 # distancia truncada (TSDF), plana
        self.W = np.zeros(self.L.size, np.float32)                # su peso
        self._motor, self.motor = str(motor), None
        self._observar = None
        self.buscar_superficies = bool(superficies)
        self.banda_mesa = banda_mesa                              # m sobre la tapa que se limpian (None: según ruido)
        from skill_shield.autonomo.tsdf import SeguidorSuperficies
        self.seguidor = SeguidorSuperficies(olvido_s=self.olvido_s)
        self._t_planos = 0.0
        # (17-nov) escena FIJA (escena_fija.py): dentro de sus cajas (+ margen) la cámara no pone obstáculos
        self.fijo, self.margen_fijo, self._mascara_fija = [], 0.03, None

    def poner_fijo(self, cajas, margen=None):
        """Cajas de la escena fija (escena_fija.py) o [] para quitarlas. Lo de dentro (+ margen) ya lo
        sabe el planificador con su forma limpia: la cámara solo añade lo que cambia."""
        from skill_shield.autonomo.escena_fija import en_cajas
        self.fijo = list(cajas or [])
        if margen is not None:
            self.margen_fijo = float(margen)
        self._mascara_fija = (en_cajas(self._centros, self.fijo, self.margen_fijo).reshape(self.L.shape)
                              if self.fijo else None)
        self._dist = self._dist_desc = None

    def _puntos_fijos(self, P):
        if not self.fijo or not len(P):
            return np.zeros(len(P), bool)
        from skill_shield.autonomo.escena_fija import en_cajas
        return en_cajas(P, self.fijo, self.margen_fijo)

    # ── utilidades ──
    def borrar(self):
        self.L[:] = 0.0
        self.t_visto[:] = 0.0
        self.D[:] = 1.0
        self.W[:] = 0.0
        self.seguidor.borrar()
        self._dist = self._dist_desc = None

    def _ocupado_bruto(self):
        """Cubos con evidencia de estar ocupados (sin mirar si son recientes ni el tamaño del grupo)."""
        if self.metodo == "tsdf":
            # cerca de la superficie: un poco por delante y medio truncado por detrás (más adentro no se
            # sabe; el planificador rellena hacia abajo lo que hay bajo lo visto)
            o = ((self.W >= self.W_MIN) & (self.D < self.D_OCUPADO) & (self.D > -0.5)).reshape(self.L.shape)
            if self.z_suelo is not None:                 # el suelo no es obstáculo (ya lo es en el modelo)
                kz = int(math.floor((self.z_suelo + self.margen_suelo - self.origen[2]) / self.res))
                if kz >= 0:
                    o[:, :, :kz + 1] = False
            return o
        return self.L >= self.UMBRAL_OCUPADO

    def _libre_bruto(self):
        if self.metodo == "tsdf":
            return ((self.W >= self.W_MIN) & (self.D >= self.D_LIBRE)).reshape(self.L.shape)
        return self.L <= self.UMBRAL_LIBRE

    def indices(self, P):
        """Índices enteros (N x 3) de los puntos y máscara de los que caen dentro."""
        idx = np.floor((P - self.origen) / self.res).astype(int)
        dentro = np.all((idx >= 0) & (idx < self.n), axis=1)
        return idx, dentro

    def estado(self, ahora=None):
        """Rejilla con DESCONOCIDO / LIBRE / OCUPADO (lo que lleva > olvido_s sin verse: desconocido)."""
        ahora = time.time() if ahora is None else ahora
        e = np.full(self.L.shape, DESCONOCIDO, np.int8)
        reciente = (ahora - self.t_visto) <= self.olvido_s
        ocupado = self._ocupado_bruto() & reciente
        if self._mascara_fija is not None:               # (17-nov) lo fijo lo pone el planificador, limpio
            ocupado &= ~self._mascara_fija
        if self.min_grupo > 1 and ocupado.any():
            from scipy.ndimage import label
            etiquetas, n = label(ocupado, structure=np.ones((3, 3, 3), bool))   # vecinos en 26 direcciones
            tam = np.bincount(etiquetas.ravel(), minlength=n + 1)
            ocupado &= tam[etiquetas] >= self.min_grupo
        e[ocupado] = OCUPADO
        e[self._libre_bruto() & reciente & ~ocupado] = LIBRE
        return e

    def centros_ocupados(self, ahora=None):
        e = self.estado(ahora)
        return self._centros[(e == OCUPADO).reshape(-1)]

    # ── actualización con una imagen de profundidad ──
    def _cerca(self, P, capsulas, margen=None):
        """Máscara de los puntos a menos de (radio + margen_robot) de alguna cápsula. Antes de
        medir se descartan los puntos fuera de la caja que envuelve TODAS las cápsulas: con las
        ~30 piezas pequeñas de la mano solo se miden los pocos puntos que hay cerca de ella."""
        fuera = np.zeros(len(P), bool)
        if capsulas is None or not len(capsulas) or not len(P):
            return fuera
        margen = self.margen_robot if margen is None else float(margen)
        if isinstance(capsulas, np.ndarray) and capsulas.ndim == 2 and capsulas.shape[1] == 7:
            A, B, r = capsulas[:, 0:3], capsulas[:, 3:6], capsulas[:, 6] + margen
        else:
            A = np.array([c[0] for c in capsulas], float)
            B = np.array([c[1] for c in capsulas], float)
            r = np.array([c[2] for c in capsulas], float) + margen
        lo = np.minimum(A, B).min(0) - r.max()
        hi = np.maximum(A, B).max(0) + r.max()
        sel = np.nonzero(np.all((P >= lo) & (P <= hi), axis=1))[0]
        if not len(sel):
            return fuera
        Q = P[sel]
        dentro = np.zeros(len(Q), bool)
        for a, b, rr in zip(A, B, r):
            dentro |= distancia_a_segmentos(Q, [a], [b]) < rr
        fuera[sel[dentro]] = True
        return fuera

    def actualizar(self, prof_mm, K, T_mundo_cam, capsulas_robot=(), ahora=None, paso=2, capsulas_mano=(),
                   img_bgr=None, nube_robot=None):
        """capsulas_robot: lista de (a, b, radio) en el mundo (la pose REAL del robot).
        capsulas_mano: lo mismo para las piezas de la OrcaHand con su postura real (PiezasMano)."""
        t0 = time.perf_counter()
        ahora = time.time() if ahora is None else ahora
        if self.metodo == "tsdf":
            return self._actualizar_tsdf(prof_mm, K, T_mundo_cam, capsulas_robot, ahora, paso, capsulas_mano,
                                         img_bgr, nube_robot, t0)
        R, t = T_mundo_cam[:3, :3], T_mundo_cam[:3, 3]
        fiables = pixeles_fiables(prof_mm, img_bgr)
        P = nube_desde_profundidad(prof_mm, K, paso, self.z_min, fiables=fiables)
        if self.dist_max:                                # (17-nov) distancia real a la cámara
            P = P[np.linalg.norm(P, axis=1) <= self.dist_max]
        P = P @ R.T + t
        self.ultimos_puntos = P                          # (para la autocorrección de la calibración)
        n_total = len(P)
        idx, dentro = self.indices(P)
        P, idx = P[dentro], idx[dentro]
        quitar = np.zeros(len(P), bool)
        n_suelo = 0
        if self.z_suelo is not None:
            s = P[:, 2] < self.z_suelo + self.margen_suelo
            n_suelo = int(s.sum())
            quitar |= s
        n_robot = 0
        for grupo in (capsulas_robot, capsulas_mano):
            q = self._cerca(P, grupo)
            n_robot += int((q & ~quitar).sum())
            quitar |= q
        if nube_robot is not None and len(nube_robot):
            # las MALLAS del brazo (12-nov): el brazo del FR5 no va por la línea entre sus joints
            # (el tramo del brazo va desplazado ~14 cm), así que con las cápsulas quedaban restos
            q = self._cerca_nube(P, nube_robot, self.margen_robot)
            n_robot += int((q & ~quitar).sum())
            quitar |= q
        for caja in self.excluir:
            quitar |= _en_caja(P, caja)
        quitar |= self._puntos_fijos(P)                  # (17-nov) la escena fija
        idx = idx[~quitar]

        # aciertos: vóxeles con al menos min_puntos puntos en esta imagen
        aciertos = np.zeros(self.L.shape, np.int32)
        np.add.at(aciertos, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
        golpe = aciertos >= self.min_puntos

        # libres: la cámara ve MÁS LEJOS que el vóxel en su píxel (pasa a través de él)
        visto, libre, tapado = self._visibilidad(prof_mm, K, T_mundo_cam)
        libre &= ~golpe
        self.L[golpe] += self.L_OCUPADO
        self.L[libre] -= self.L_LIBRE
        for caja in self.excluir:                        # lo excluido no es obstáculo (tampoco lo de antes)
            dentro = _en_caja(self._centros, caja)
            if dentro.any():
                d = dentro.reshape(self.L.shape)
                self.L[d] = np.minimum(self.L[d], 0.0)
        np.clip(self.L, self.L_MIN, self.L_MAX, out=self.L)
        self.t_visto[golpe | visto] = ahora
        # (17-nov) lo OCUPADO que ahora tapa algo (casi siempre el propio brazo, que se queda encima de la
        # mesa) no se olvida: no se puede saber si sigue ahí, así que sigue contando. Antes, a los
        # olvido_s, la mesa de debajo de la mano desaparecía del mapa y el plan la atravesaba.
        self.t_visto[tapado & (self.L >= self.UMBRAL_OCUPADO)] = ahora
        self._superficies(P[~quitar] if len(P) else P, T_mundo_cam, ahora)
        self._dist = self._dist_desc = None
        self.ultima = dict(puntos=n_total, robot=n_robot, suelo=n_suelo,
                           ms=round((time.perf_counter() - t0) * 1000, 1))

    # ── (17-nov) fusión TSDF ──
    def _actualizar_tsdf(self, prof_mm, K, T_mundo_cam, capsulas_robot, ahora, paso, capsulas_mano, img_bgr,
                         nube_robot, t0):
        from skill_shield.autonomo.tsdf import crear_observador, fundir, observar_numpy
        if self._observar is None:
            self._observar, self.motor = crear_observador(self._centros, self._motor)
            print(f"🧱 [mapa] Fusión TSDF con {self.motor}")
        R, t = T_mundo_cam[:3, :3], T_mundo_cam[:3, 3]
        fiables = pixeles_fiables(prof_mm, img_bgr)
        p = prof_mm[::paso, ::paso].astype(np.float32) * 0.001
        ok = (p > self.z_min) & fiables[::paso, ::paso]
        h, w = p.shape
        Ks = np.array(K, float).copy()
        Ks[0, 0] /= paso; Ks[0, 2] /= paso; Ks[1, 1] /= paso; Ks[1, 2] /= paso
        v, u = np.mgrid[0:h, 0:w]
        Pc = np.stack([(u - Ks[0, 2]) / Ks[0, 0] * p, (v - Ks[1, 2]) / Ks[1, 1] * p, p], -1)
        rango = np.linalg.norm(Pc, axis=-1)              # distancia REAL a la cámara (no solo el eje z)
        lejos = ok & (rango > self.dist_max) if self.dist_max else np.zeros_like(ok)
        cerca = ok & ~lejos
        ic = np.nonzero(cerca.ravel())[0]
        P = Pc.reshape(-1, 3)[ic] @ R.T + t
        self.ultimos_puntos = P
        robot = np.zeros(len(P), bool)
        for grupo in (capsulas_robot, capsulas_mano):
            robot |= self._cerca(P, grupo)
        if nube_robot is not None and len(nube_robot):
            robot |= self._cerca_nube(P, nube_robot, self.margen_robot)
        suelo = (P[:, 2] < self.z_suelo + self.margen_suelo) if self.z_suelo is not None else np.zeros(len(P), bool)
        excl = np.zeros(len(P), bool)
        for caja in self.excluir:
            excl |= _en_caja(P, caja)
        # (17-nov) la escena fija: lo que cae en sus cajas solo dice «libre hasta aquí» (como el robot)
        sup = ~(robot | suelo | excl | self._puntos_fijos(P))
        d_sup = np.zeros(h * w, np.float32)
        d_lib = np.zeros(h * w, np.float32)
        pz = p.ravel()
        d_sup[ic[sup]] = pz[ic[sup]]
        d_lib[ic] = np.where(robot, np.maximum(pz[ic] - self.margen_robot - self.res, 0.0), pz[ic])
        if self.dist_max:                                # lejos: libre hasta dist_max, sin superficies
            il = np.nonzero(lejos.ravel())[0]
            d_lib[il] = self.dist_max * pz[il] / np.maximum(rango.ravel()[il], 1e-6)   # en z de la cámara
        T_cm = np.linalg.inv(T_mundo_cam)
        args = (self._centros, T_cm, d_sup.reshape(h, w), d_lib.reshape(h, w), Ks, self.trunc_0, self.k_trunc)
        try:
            idx, obs, tap = self._observar(*args)
        except Exception as e:                           # la GPU nunca debe tirar el mapa
            if self._observar is not observar_numpy:
                print(f"⚠️  [mapa] Fallo de la TSDF en la GPU ({e}): sigo con numpy (CPU).")
                self._observar, self.motor = observar_numpy, "numpy (CPU)"
            idx, obs, tap = observar_numpy(*args)
        fundir(self.D, self.W, idx, obs, self.w_max)
        tv = self.t_visto.reshape(-1)
        tv[idx] = ahora
        ocup = self._ocupado_bruto().reshape(-1)
        tv[tap[ocup[tap]]] = ahora                       # lo ocupado que se ve tapado no se olvida
        for caja in self.excluir:                        # lo excluido no es obstáculo (tampoco lo de antes)
            dentro = _en_caja(self._centros, caja)
            self.W[dentro] = 0.0
        self._superficies(P[sup], T_mundo_cam, ahora)
        self._dist = self._dist_desc = None
        self.ultima = dict(puntos=int(len(P)), robot=int(robot.sum()), suelo=int(suelo.sum()),
                           ms=round((time.perf_counter() - t0) * 1000, 1))

    # ── (17-nov) mesas como planos ──
    def _superficies(self, P, T_mundo_cam, ahora):
        """Busca planos horizontales (como mucho 2 veces por segundo) y, en las mesas estables, deja
        la tapa como UNA capa a su altura y limpia el ruido que queda justo encima."""
        if not self.buscar_superficies:
            return
        from skill_shield.autonomo.tsdf import planos_horizontales
        if ahora - self._t_planos >= 0.5 and len(P):
            self._t_planos = ahora
            alto = P[:, 2] > ((self.z_suelo if self.z_suelo is not None else -np.inf) + 0.10)
            self.seguidor.actualizar(planos_horizontales(P[alto], self.res), ahora)
        self.aplicar_superficies(T_mundo_cam)

    def superficies(self):
        """Mesas estables: [(z, vértices xy del contorno)]."""
        return self.seguidor.estables()

    def aplicar_superficies(self, T_mundo_cam=None):
        from skill_shield.autonomo.tsdf import dentro_poligono
        sup = self.superficies()
        if not sup:
            return
        nx, ny, nz = self.n
        xy = self._centros.reshape(nx, ny, nz, 3)[:, :, 0, :2].reshape(-1, 2)
        Dg, Wg = self.D.reshape(self.L.shape), self.W.reshape(self.L.shape)
        for z, poli in sup:
            en = dentro_poligono(poli, xy).reshape(nx, ny)
            if not en.any():
                continue
            kz = int(math.floor((z - self.origen[2]) / self.res))
            if not 0 <= kz < nz:
                continue
            banda = self.banda_mesa
            if banda is None:                            # ruido del estéreo a esa distancia
                d = 1.0
                if T_mundo_cam is not None:
                    d = float(np.linalg.norm(np.r_[np.asarray(poli).mean(0), z] - T_mundo_cam[:3, 3]))
                banda = 2.0 * self.res + self.k_trunc * d * d
            k1 = min(nz, kz + 1 + int(math.ceil(banda / self.res)))
            if self.metodo == "tsdf":
                Dg[en, kz], Wg[en, kz] = 0.0, np.maximum(Wg[en, kz], self.W_MIN)
                for k in range(kz + 1, k1):
                    Dg[en, k], Wg[en, k] = 1.0, np.maximum(Wg[en, k], self.W_MIN)
            else:
                self.L[en, kz] = np.maximum(self.L[en, kz], self.UMBRAL_OCUPADO)
                self.L[en, kz + 1:k1] = np.minimum(self.L[en, kz + 1:k1], self.UMBRAL_LIBRE)

    @staticmethod
    def _cerca_nube(P, nube, margen):
        from scipy.spatial import cKDTree
        out = np.zeros(len(P), bool)
        if not len(P):
            return out
        nube = np.asarray(nube, float)
        lo, hi = nube.min(0) - margen, nube.max(0) + margen
        sel = np.nonzero(np.all((P >= lo) & (P <= hi), axis=1))[0]
        if len(sel):
            d, _ = cKDTree(nube).query(P[sel], distance_upper_bound=margen)
            out[sel[np.isfinite(d)]] = True
        return out

    def vaciar(self, capsulas_robot=(), capsulas_mano=(), nube_robot=None):
        """Los vóxeles por los que pasa (o acaba de pasar) el robot están LIBRES: el robot estaba
        ahí. Borra los «fantasmas» que deja el brazo al moverse (la imagen de profundidad llega
        con algo de retraso respecto a la pose con la que se quita el robot)."""
        dentro = np.zeros(len(self._centros), bool)
        for g in (capsulas_robot, capsulas_mano):
            dentro |= self._cerca(self._centros, g, margen=0.0)
        if nube_robot is not None and len(nube_robot):
            dentro |= self._cerca_nube(self._centros, nube_robot, 0.5 * self.res + 0.005)
        if dentro.any():
            d = dentro.reshape(self.L.shape)
            self.L[d] = np.minimum(self.L[d], self.L_MIN)
            self.D[dentro] = 1.0                         # (TSDF) el robot estaba ahí: libre
            self.W[dentro] = np.maximum(self.W[dentro], self.W_MIN)
            self.t_visto.reshape(-1)[dentro] = time.time()   # (17-nov) y eso se SABE: no es desconocido
            self._dist = self._dist_desc = None
        return int(dentro.sum())

    def _visibilidad(self, prof_mm, K, T_mundo_cam):
        """(visto, libre, tapado) por vóxel. visto = en el campo de visión con profundidad medida en su
        píxel y NO tapado; libre = además la medida está claramente más lejos que el vóxel;
        tapado = en el campo de visión pero con algo delante (la medida, claramente más cerca)."""
        h, w = prof_mm.shape
        Tcm = np.linalg.inv(T_mundo_cam)
        C = self._centros @ Tcm[:3, :3].T + Tcm[:3, 3]
        z = C[:, 2]
        delante = z > 0.15
        u = np.full(len(C), -1, int)
        v = np.full(len(C), -1, int)
        u[delante] = np.round(C[delante, 0] / z[delante] * K[0, 0] + K[0, 2]).astype(int)
        v[delante] = np.round(C[delante, 1] / z[delante] * K[1, 1] + K[1, 2]).astype(int)
        en_imagen = delante & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        medida = np.zeros(len(C), np.float32)
        medida[en_imagen] = prof_mm[v[en_imagen], u[en_imagen]].astype(np.float32) * 0.001
        valida = en_imagen & (medida > 0)
        tol = 1.5 * self.res
        libre = valida & (medida > z + tol)
        visto = valida & (medida > z - tol)                      # delante o en la superficie medida
        tapado = valida & ~visto
        return visto.reshape(self.L.shape), libre.reshape(self.L.shape), tapado.reshape(self.L.shape)

    # ── distancias para las colisiones ──
    def distancias(self, desconocido_ocupado=False, ahora=None):
        """Distancia (m) de cada vóxel al obstáculo más cercano (0 dentro de uno)."""
        cache = "_dist_desc" if desconocido_ocupado else "_dist"
        if getattr(self, cache) is None:
            from scipy.ndimage import distance_transform_edt
            e = self.estado(ahora)
            obst = e == OCUPADO
            if desconocido_ocupado:
                obst |= self._tapados(e)
            d = distance_transform_edt(~obst) * self.res if obst.any() else np.full(self.L.shape, np.inf, np.float32)
            setattr(self, cache, d.astype(np.float32))
        return getattr(self, cache)

    def _tapados(self, e):
        """Vóxeles desconocidos que están DEBAJO/DETRÁS de algo ocupado en la misma columna
        vertical (lo más probable: el interior o la sombra de un objeto). No se marcan como
        obstáculo las zonas que simplemente quedan fuera de la vista."""
        ocup = e == OCUPADO
        encima = np.flip(np.maximum.accumulate(np.flip(ocup, axis=2), axis=2), axis=2)   # hay algo ocupado más arriba
        return (e == DESCONOCIDO) & encima

    def distancia_en(self, P, desconocido_ocupado=False):
        """Distancia al obstáculo más cercano en los puntos P (fuera de la rejilla: infinito)."""
        d = self.distancias(desconocido_ocupado)
        idx, dentro = self.indices(np.asarray(P, float))
        out = np.full(len(P), np.inf, np.float32)
        out[dentro] = d[idx[dentro, 0], idx[dentro, 1], idx[dentro, 2]]
        return out


class ColisionMapa:
    """Choques de las cápsulas del robot (ModeloColision) contra el mapa de obstáculos.
    Misma interfaz que ModeloColision.colisiones: lista de (cápsula, "mapa", penetración)."""

    def __init__(self, modelo, mapa, margen=0.02, desconocido_ocupado=True, paso=0.01):
        self.m = modelo
        self.mapa = mapa
        self.margen = float(margen)
        self.desc = bool(desconocido_ocupado)
        self.paso = float(paso)

    def colisiones(self, q, marcos=None):
        out = list(self.m.colisiones(q, marcos))
        for c in self.m.capsulas(q, marcos):
            if c.nombre == "base":
                continue
            n = max(2, int(math.ceil(np.linalg.norm(c.b - c.a) / self.paso)) + 1)
            pts = c.a + np.linspace(0.0, 1.0, n)[:, None] * (c.b - c.a)
            d = float(np.min(self.mapa.distancia_en(pts, self.desc)))
            # la distancia es al CENTRO del vóxel ocupado más cercano: se resta medio vóxel
            pen = c.r + self.margen + 0.5 * self.mapa.res - d
            if pen > 0:
                out.append((c.nombre, "mapa", pen))
        return out

    def libre(self, q, marcos=None):
        return not self.colisiones(q, marcos)

    def capsulas(self, q, marcos=None):
        return self.m.capsulas(q, marcos)


def capsulas_robot(modelo, q, con_mano=True):
    """(a, b, radio) de las cápsulas del robot en la pose q, para quitar el robot del mapa.
    con_mano=False: sin la cápsula recta de la mano; en su lugar solo el tramo brida → mano_mount
    (wrist3 + adaptador), porque la mano llega aparte con su postura real (PiezasMano)."""
    q = np.asarray(q, float)
    if con_mano:
        return [(c.a, c.b, c.r) for c in modelo.capsulas(q)]
    m = modelo.c.fk_marcos(q)
    out = [(c.a, c.b, c.r) for c in modelo.capsulas(q, m) if c.nombre != "mano"]
    out.append((m[5][:3, 3], m[6][:3, 3], modelo.R_MUNECA))
    return out


class PiezasMano:
    """Cápsulas que envuelven cada pieza de la OrcaHand del gemelo (todos los geoms que cuelgan
    de mano_mount), con la muñeca de la Orca y los dedos en su postura REAL (11-nov).
    Sustituyen en el mapa a la cápsula recta de ModeloColision, que no sabía nada de la muñeca
    de la mano ni de los dedos y dejaba la mano real como obstáculo falso.

    Cada geom → su caja local (geom_aabb) → cápsula por el eje más largo de la caja, con radio
    = media diagonal de la sección (envuelve la caja entera). Sin geom_aabb (MuJoCo antiguo):
    una esfera de radio geom_rbound. Solo usa el modelo y los datos: se llama donde está MuJoCo
    (modo.py) y se mandan las cápsulas al proceso de la cámara."""

    def __init__(self, modelo, cuerpo="mano_mount"):
        import mujoco
        raiz = mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_BODY, cuerpo)
        if raiz < 0:
            raise ValueError(f"no hay cuerpo «{cuerpo}» en el modelo")
        cuerpos = set()
        for b in range(modelo.nbody):
            p = b
            while p > 0 and p != raiz:
                p = int(modelo.body_parentid[p])
            if p == raiz:
                cuerpos.add(b)
        aabb = getattr(modelo, "geom_aabb", None)
        g_, c_, e_, r_ = [], [], [], []
        for g in range(modelo.ngeom):
            if int(modelo.geom_bodyid[g]) not in cuerpos:
                continue
            if aabb is not None:
                a = np.asarray(aabb[g], float).reshape(6)
                c, h = a[:3], np.abs(a[3:])
                k = int(np.argmax(h))
                e = np.zeros(3)
                e[k] = h[k]
                r = float(np.sqrt(max(float(h @ h) - h[k] ** 2, 0.0)))
            else:
                c, e, r = np.zeros(3), np.zeros(3), float(modelo.geom_rbound[g])
            if r <= 0.0 and not e.any():
                continue
            g_.append(g)
            c_.append(c)
            e_.append(e)
            r_.append(r)
        self.geoms = np.array(g_, int)
        self._c = np.array(c_, float).reshape(-1, 3)
        self._e = np.array(e_, float).reshape(-1, 3)
        self._r = np.array(r_, float)

    def __len__(self):
        return len(self.geoms)

    def capsulas(self, datos):
        """(n, 7): a (3), b (3) y radio de cada pieza, en el mundo, con la pose de `datos`."""
        P = np.asarray(datos.geom_xpos)[self.geoms]
        M = np.asarray(datos.geom_xmat)[self.geoms].reshape(-1, 3, 3)
        c = P + np.einsum("nij,nj->ni", M, self._c)
        d = np.einsum("nij,nj->ni", M, self._e)
        return np.hstack([c - d, c + d, self._r[:, None]])


def empaquetar_pose(q, real, caps_mano=None):
    """Mensaje del canal «qbrazo»: float64 [q (6), real (1/0), n, a, b, r de cada pieza de la mano…]."""
    caps = np.zeros((0, 7)) if caps_mano is None else np.asarray(caps_mano, float).reshape(-1, 7)
    return np.r_[np.asarray(q, float)[:6], 1.0 if real else 0.0, float(len(caps)), caps.ravel()].tobytes()


def desempaquetar_pose(b):
    """(q, real, caps_mano (n x 7) o None). Acepta también el mensaje antiguo de 7 números."""
    a = np.frombuffer(b, np.float64)
    if len(a) < 7:
        return None, False, None
    q, real = a[:6].copy(), bool(a[6] > 0.5)
    caps = None
    if len(a) >= 8:
        n = int(a[7])
        if n > 0 and len(a) >= 8 + 7 * n:
            caps = a[8:8 + 7 * n].reshape(n, 7).copy()
    return q, real, caps


# ── para mandar el mapa a otros procesos (canal de memoria compartida) ──
def empaquetar(centros, res, max_voxeles=6000):
    """Bytes: float32 [res, n, x0, y0, z0, x1, …]. Si hay demasiados, se queda con una muestra."""
    c = np.asarray(centros, np.float32).reshape(-1, 3)
    if len(c) > max_voxeles:
        c = c[np.linspace(0, len(c) - 1, max_voxeles).astype(int)]
    return np.concatenate([np.array([res, len(c)], np.float32), c.ravel()]).tobytes()


def empaquetar_visto(mapa, ahora=None):
    """(17-nov) Bytes con qué cubos ha VISTO la cámara (libres u ocupados) y cuáles no (desconocidos):
    float32 [res, x0, y0, z0, nx, ny, nz] + 1 bit por cubo. Para que el planificador prefiera no
    pasar por lo que no sabe qué hay."""
    import json
    conocido = mapa.estado(ahora) != DESCONOCIDO
    if getattr(mapa, "_mascara_fija", None) is not None:
        conocido |= mapa._mascara_fija                  # (17-nov) lo fijo se sabe aunque no se vea
    cab = np.array([mapa.res, *mapa.origen, *mapa.n], np.float32)
    mesas = json.dumps([[round(float(z), 4), np.round(np.asarray(p, float), 4).tolist()]
                        for z, p in (mapa.superficies() if hasattr(mapa, "superficies") else [])]).encode()
    return cab.tobytes() + np.packbits(conocido.ravel()).tobytes() + mesas


def desempaquetar_visto(b):
    if not b or len(b) < 28:
        return None
    cab = np.frombuffer(b[:28], np.float32)
    n = tuple(int(round(v)) for v in cab[4:7])
    import json
    nb = (int(np.prod(n)) + 7) // 8
    if len(b) < 28 + nb:
        return None
    bits = np.unpackbits(np.frombuffer(b[28:28 + nb], np.uint8))
    try:
        mesas = [(float(z), np.asarray(p, float)) for z, p in json.loads(b[28 + nb:].decode() or "[]")]
    except ValueError:
        mesas = []
    return dict(res=float(cab[0]), origen=cab[1:4].astype(float), n=n,
                conocido=bits[:int(np.prod(n))].reshape(n).astype(bool), superficies=mesas)


def desempaquetar(b):
    a = np.frombuffer(b, np.float32)
    if len(a) < 2:
        return 0.0, np.zeros((0, 3), np.float32)
    n = int(a[1])
    return float(a[0]), a[2:2 + 3 * n].reshape(-1, 3)


def dibujar_voxeles(scn, centros, res, rgba=(1.0, 0.45, 0.0, 0.55), inicio=None):
    """Añade los vóxeles como cajas a una escena de MuJoCo (visor.user_scn o Renderer.scene).
    Devuelve cuántos ha dibujado."""
    import mujoco
    i = scn.ngeom if inicio is None else inicio
    tam = np.full(3, 0.5 * res * 0.92)
    mat = np.eye(3).flatten()
    rgba = np.asarray(rgba, np.float32)
    n = 0
    for c in centros:
        if i >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_BOX, tam, np.asarray(c, float), mat, rgba)
        i += 1
        n += 1
    scn.ngeom = i
    return n


def modelo_colision(cadena, cfg, T_mundo_cam=None, radio_mano=None):
    """Modelo de colisión de la teleoperación (cápsulas del FR5 + mano; soporte, suelo y
    TELEOP_FR5_OBSTACULOS de config) y, si se sabe, la cámara de cabeza como obstáculo.
    cfg: el módulo config o un dict."""
    from skill_shield.teleop_fr5.ik_teleop import Caja, Entorno, ModeloColision, entorno_soporte_hombros
    from skill_shield.teleop_fr5.obstaculos import cajas_desde_config
    g = cfg if isinstance(cfg, dict) else {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    ent = entorno_soporte_hombros(g["SOPORTE_SUELO_Z"]) if g.get("SOPORTE_HOMBROS", False) \
        else Entorno(z_suelo=0.0)
    ent.cajas = cajas_desde_config(g, ent.z_suelo, verbose=False)
    if T_mundo_cam is not None:
        ent.cajas.append(Caja("camara_cabeza", np.asarray(T_mundo_cam)[:3, 3], (0.07, 0.07, 0.07)))
    m = ModeloColision(cadena, ent)
    if radio_mano is not None:
        m.R_MANO = float(radio_mano)
    return m


class ConstructorMapa:
    """El mapa dentro del proceso de la cámara: carga la calibración (y la recarga si cambia),
    lee la pose real del brazo y los mandos de la web, actualiza el mapa a AUTONOMO_MAPA_HZ y lo
    publica. paso(prof_mm, K) se llama con cada imagen de profundidad nueva."""

    def __init__(self, cfg, raiz, canal_q=None, canal_mapa=None, canal_mando=None, canal_visto=None):
        import os
        g = cfg if isinstance(cfg, dict) else {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
        self.g = g
        ruta = g.get("CALIB_ROBOT_ARCHIVO", "camara_robot.json")
        self.ruta_calib = ruta if os.path.isabs(ruta) else os.path.join(raiz, ruta)
        z_suelo = g.get("SOPORTE_SUELO_Z") if g.get("SOPORTE_HOMBROS", False) else 0.0
        # (17-nov) antes acababa en y = −1,10 m y z = 0,60 m: la cajonera y el panel del punto de dejada
        # seguían más allá y, para el planificador, ahí no había nada
        lim = g.get("AUTONOMO_MAPA_LIMITES") or ((-1.0, 0.6), (-1.6, 0.4), (z_suelo, 0.9))
        self.mapa = MapaOcupacion(lim, float(g.get("AUTONOMO_MAPA_RES", 0.02)),
                                  olvido_s=float(g.get("AUTONOMO_MAPA_OLVIDO_S", 30.0)), z_suelo=z_suelo,
                                  margen_robot=float(g.get("AUTONOMO_MAPA_MARGEN_ROBOT", 0.03)),
                                  min_puntos=int(g.get("AUTONOMO_MAPA_MIN_PUNTOS", 4)),
                                  min_grupo=int(g.get("AUTONOMO_MAPA_MIN_GRUPO", 4)),
                                  z_min=float(g.get("AUTONOMO_MAPA_Z_MIN", 0.35)),
                                  metodo=str(g.get("AUTONOMO_MAPA_METODO", "tsdf")),
                                  dist_max=g.get("AUTONOMO_MAPA_DIST_MAX", 1.7),
                                  k_trunc=float(g.get("AUTONOMO_MAPA_TSDF_K", 0.02)),
                                  w_max=float(g.get("AUTONOMO_MAPA_TSDF_PESO_MAX", 20.0)),
                                  motor=str(g.get("AUTONOMO_MAPA_MOTOR", "auto")),
                                  superficies=bool(g.get("AUTONOMO_MAPA_MESAS", True)))
        from skill_shield.teleop_fr5.cinematica import cargar_cadena
        self.modelo = modelo_colision(cargar_cadena(g.get("RUTA_XML"), verbose=False), g)
        self.periodo = 1.0 / max(float(g.get("AUTONOMO_MAPA_HZ", 4.0)), 0.5)
        self.max_dibujo = int(g.get("AUTONOMO_MAPA_MAX_DIBUJO", 6000))
        self.canal_q, self.canal_mapa, self.canal_mando = canal_q, canal_mapa, canal_mando
        self.canal_visto = canal_visto
        self.T, self._mtime = None, None
        self._t_ult, self._t_calib = 0.0, 0.0
        self._borrado_visto = None
        self.hz, self._n, self._t_hz = 0.0, 0, time.monotonic()
        self.aviso = ""
        self.robot_quitado = False
        self.mano_real = False
        self._excluir_web = []
        self.excluir_extra = []                          # p. ej. la herramienta que se va a coger
        # poses recientes del robot (12-nov): se quita del mapa el robot en TODAS ellas (la
        # profundidad llega con retraso: si no, al moverse deja un rastro de cubos «fantasma»)
        self.barrido_s = float(g.get("AUTONOMO_MAPA_BARRIDO_S", 0.6))
        self._poses = []                                 # [(t monotónico, q, cápsulas de la mano o None)]
        # autocorrección de la calibración con el propio brazo (12-nov)
        from skill_shield.autonomo.autocalib import Correccion
        self.correccion = Correccion(self.ruta_calib, float(g.get("AUTONOMO_AUTOCALIB_GANANCIA", 0.8)),
                                     float(g.get("AUTONOMO_AUTOCALIB_MAX_M", 0.15)),
                                     float(g.get("AUTONOMO_AUTOCALIB_RESIDUO_MAX_M", 0.012)),
                                     float(g.get("AUTONOMO_AUTOCALIB_PASO_MAX_M", 0.08)))
        self.autocalib = None                            # (también da las mallas del brazo para quitarlo)
        self.autocalib_activa = bool(g.get("AUTONOMO_AUTOCALIB", True))
        try:
            from skill_shield.autonomo.autocalib import AutoCalib
            self.autocalib = AutoCalib(self.modelo.c, g.get("RUTA_XML"))
        except Exception as e:                           # sin mallas: con las cápsulas y sin autocorrección
            print(f"⚠️  [autónomo] Sin las mallas del brazo (ni autocorrección de la calibración): {e}")
        self.quieto_s = float(g.get("AUTONOMO_AUTOCALIB_QUIETO_S", 0.4))
        self._nube_barrido = None
        self._t_autocalib = 0.0
        self._t_mapa = np.zeros(3)                       # corrección con la que se hizo el mapa actual
        # (17-nov) escena FIJA (escena_fija.py): se escanea una vez desde la web y, desde entonces, el mapa
        # solo pone obstáculos fuera de sus cajas
        from skill_shield.autonomo.escena_fija import EscenaFija
        self.escena = EscenaFija(g)
        self.margen_fijo = float(g.get("AUTONOMO_ESCENA_FIJA_MARGEN", 0.03))
        self.escaneo_s = float(g.get("AUTONOMO_ESCENA_FIJA_ESCANEO_S", 10.0))
        self.fijo_grupo_min = int(g.get("AUTONOMO_ESCENA_FIJA_GRUPO_MIN", 20))
        self.fijo_llenado_min = float(g.get("AUTONOMO_ESCENA_FIJA_LLENADO_MIN", 0.45))
        self._escaneo_hasta = None                       # time.monotonic() en que acaba el escaneo en marcha
        self._fijo_vistos = {}                           # último id visto de cada mando de la escena fija
        self.aviso_fijo = ""
        self._aplicar_fijo()

    # ── (17-nov) escena fija ──
    def _aplicar_fijo(self):
        """Mientras se escanea, nada fijo (hay que ver TODO); si no, las cajas de la escena fija."""
        self.mapa.poner_fijo([] if self._escaneo_hasta is not None else self.escena.cajas, self.margen_fijo)

    def escanear_fijo(self, s=None):
        """Empieza a escanear la escena fija: mapa en blanco y, a los s segundos, se ajustan las cajas."""
        self._escaneo_hasta = time.monotonic() + float(s if s is not None else self.escaneo_s)
        self.aviso_fijo = ""
        self._aplicar_fijo()
        self.mapa.borrar()
        print(f"🧱 Escena fija: escaneando {self.escaneo_s:.0f} s (que no haya nada que se mueva ni el brazo delante)…")

    def terminar_escaneo(self):
        from skill_shield.autonomo import escena_fija as EF
        self._escaneo_hasta = None
        C = self.mapa.centros_ocupados()
        caps = [(c.a, c.b, c.r) for c in getattr(self.modelo.entorno, "capsulas", []) or []]
        quitar = self.mapa._cerca(C, caps, margen=0.05) if caps else None      # el soporte ya está en el modelo
        esc = EF.ajustar(C, self.mapa.res, self.mapa.superficies(), self.mapa.z_suelo, quitar=quitar,
                         grupo_min=self.fijo_grupo_min, llenado_min=self.fijo_llenado_min)
        if not esc["cajas"] and not esc["voxeles"]:
            self.aviso_fijo = "el escaneo no ha encontrado nada: no se guarda"
            print(f"⚠️  Escena fija: {self.aviso_fijo}.")
        else:
            EF.guardar(self.escena.ruta, esc, calibracion=os.path.basename(self.ruta_calib),
                       correccion_calibracion=[round(float(v), 4) for v in self.correccion.t])
            print(f"💾 Escena fija guardada en {self.escena.ruta}: {len(esc['cajas'])} cajas"
                  + (f" y {len(esc['voxeles'])} cubos sueltos" if esc["voxeles"] else ""))
            for c in esc["cajas"]:
                print(f"     · {c['nombre']}: centro {np.round(c['centro'], 2).tolist()}, "
                      f"{2 * c['semi'][0]:.2f} × {2 * c['semi'][1]:.2f} × {2 * c['semi'][2]:.2f} m, giro {c['yaw_deg']:+.0f}°")
            self.escena.actualizar()
        self._aplicar_fijo()
        return esc

    def olvidar_fijo(self):
        from skill_shield.autonomo.escena_fija import olvidar
        olvidar(self.escena.ruta)
        self._escaneo_hasta = None
        self.escena.actualizar()
        self._aplicar_fijo()
        print("🧱 Escena fija olvidada (el archivo queda como .bak).")

    def _mando_nuevo(self, m, clave):
        """¿Hay una orden nueva con este id? (la primera vez solo se apunta: no se repite una orden vieja)"""
        i = m.get(clave)
        nuevo = i is not None and clave in self._fijo_vistos and i != self._fijo_vistos[clave]
        self._fijo_vistos[clave] = i
        return nuevo

    def _recargar_calibracion(self):
        import os
        try:
            m = os.path.getmtime(self.ruta_calib)
        except OSError:
            m = None
        if m != self._mtime:
            self._mtime = m
            if m is None:
                self.T = None
            else:
                from skill_shield.autonomo.calibracion import cargar
                self.T = cargar(self.ruta_calib)["T_mundo_cam"]
                self.mapa.borrar()                       # calibración nueva: lo de antes no vale
            self.correccion.nueva_calibracion(m)
            self._t_mapa = self.correccion.t.copy()

    @property
    def T_corr(self):
        """La corrección de la calibración (4x4, en el mundo) para los detectores."""
        return self.correccion.T()

    def T_efectiva(self):
        return None if self.T is None else self.correccion.T() @ self.T

    def _recordar_pose(self, q, caps_mano, ahora):
        if q is None:
            return
        q = np.asarray(q, float)
        self._poses = [p for p in self._poses if ahora - p[0] <= self.barrido_s]
        self._poses.append((ahora, q.copy(), caps_mano))

    def _quieto(self, ahora):
        """El brazo lleva quieto al menos quieto_s (para medir sin el retraso de la imagen)."""
        if len(self._poses) < 2 or ahora - self._poses[0][0] < min(self.quieto_s, self.barrido_s) * 0.9:
            return False
        q = self._poses[-1][1]
        return all(float(np.max(np.abs(p[1] - q))) < math.radians(0.3) for p in self._poses)

    def _capsulas_barrido(self):
        """Cápsulas (y mallas del brazo) del robot en las poses recientes (sin repetir las casi iguales)."""
        robot, mano, nube, ult = [], [], [], None
        for _, q, caps in self._poses:
            if ult is not None and float(np.max(np.abs(q - ult))) < math.radians(0.5):
                continue
            ult = q
            robot += capsulas_robot(self.modelo, q, con_mano=caps is None)
            if caps is not None and len(caps):
                mano.append(np.asarray(caps, float))
            if self.autocalib is not None:
                nube.append(self.autocalib.modelo(q)[0])
        self._nube_barrido = np.vstack(nube) if nube else None
        return robot, (np.vstack(mano) if mano else ())

    def _mandos(self):
        if self.canal_mando is None:
            return
        import json
        b, _ = self.canal_mando.leer()
        if not b:
            return
        try:
            m = json.loads(b)
        except ValueError:
            return
        bid = m.get("mapa_borrar_id")
        if bid is not None and bid != self._borrado_visto:
            if self._borrado_visto is not None or m.get("mapa_borrar_nuevo"):
                self.mapa.borrar()
                self.correccion.borrar()                 # y vuelta a la calibración del archivo
                self._t_mapa = self.correccion.t.copy()
            self._borrado_visto = bid
        if self._mando_nuevo(m, "fijo_escanear_id"):
            self.escanear_fijo()
        if self._mando_nuevo(m, "fijo_olvidar_id"):
            self.olvidar_fijo()
        self._excluir_web = [(np.asarray(c, float), np.asarray(s, float)) for c, s in (m.get("mapa_excluir") or [])]

    def _q_real(self):
        """(pose REAL del brazo o None, cápsulas de la mano con su postura real o None)."""
        if self.canal_q is None:
            return None, None
        b, t = self.canal_q.leer()
        if not b or t is None or time.time() - t > 0.5:
            return None, None
        q, real, caps = desempaquetar_pose(b)
        return (q, caps) if real else (None, None)

    @staticmethod
    def quitar_pixeles(prof_mm, halos, K=None, T_mundo_cam=None):
        """Copia de la profundidad sin los píxeles de los halos [(u, v, radio px, z mín, z máx[, fila, z mundo mín])].
        Con K y T_mundo_cam, lo que esté por debajo de «z mundo mín» (la mesa) no se quita (17-nov)."""
        if not halos:
            return prof_mm
        prof = prof_mm.copy()
        h, w = prof.shape[:2]
        for halo in halos:
            u, v, r, z0, z1 = halo[:5]
            v_max = halo[5] if len(halo) > 5 else v + r           # filas de más abajo: la mesa, se queda
            x0, x1 = max(int(u - r), 0), min(int(u + r) + 1, w)
            y0, y1 = max(int(v - r), 0), min(int(min(v + r, v_max)) + 1, h)
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            trozo = prof[y0:y1, x0:x1]
            quita = ((xx - u) ** 2 + (yy - v) ** 2 <= r * r) & (trozo >= z0 * 1000.0) & (trozo <= z1 * 1000.0)
            if len(halo) > 6 and halo[6] is not None and K is not None and T_mundo_cam is not None and quita.any():
                z = trozo.astype(np.float32) * 0.001
                P = np.stack([(xx - K[0, 2]) / K[0, 0] * z, (yy - K[1, 2]) / K[1, 1] * z, z], -1)
                zw = P @ np.asarray(T_mundo_cam, float)[2, :3] + float(T_mundo_cam[2, 3])
                quita &= zw > float(halo[6])                 # la mesa (por debajo de la pelota) se queda
            trozo[quita] = 0
        return prof

    def paso(self, prof_mm, K, img_bgr=None, halos=None):
        ahora = time.monotonic()
        if ahora - self._t_ult < self.periodo:
            return False
        self._t_ult = ahora
        if ahora - self._t_calib > 1.0:
            self._t_calib = ahora
            self._recargar_calibracion()
            if self.escena.actualizar():                 # (17-nov) escena fija nueva o cambiada a mano
                self._aplicar_fijo()
        self._mandos()
        if self.T is None:
            self.aviso = "sin calibrar (camara_robot.json)"
            return False
        self.mapa.excluir = list(self._excluir_web) + [tuple(np.asarray(x, float) for x in caja)
                                                       for caja in self.excluir_extra]
        q, caps_mano = self._q_real()
        self.robot_quitado = q is not None
        self.mano_real = caps_mano is not None
        if q is None:
            self.aviso = "sin pose real del robot: puede salir como obstáculo"
        elif caps_mano is None:
            self.aviso = "sin la postura real de la mano: se quita con una cápsula recta (puede dejar restos)"
        else:
            self.aviso = ""
        self._recordar_pose(q, caps_mano, ahora)
        if q is not None:
            caps_r, caps_m = self._capsulas_barrido()
        else:
            caps_r, caps_m = (), ()
        nube = self._nube_barrido if q is not None else None
        prof_mm = self.quitar_pixeles(prof_mm, halos, K, self.T_efectiva())   # sin el halo de las pelotas (12-nov)
        self.mapa.actualizar(prof_mm, K, self.T_efectiva(), caps_r, capsulas_mano=caps_m, img_bgr=img_bgr,
                             nube_robot=nube)
        if q is not None:
            self.mapa.vaciar(caps_r, caps_m, nube)
            # autocorrección: con el brazo quieto y a la vista, como mucho 2 veces por segundo
            if (self.autocalib is not None and self.autocalib_activa and ahora - self._t_autocalib >= 0.5
                    and self._quieto(ahora)):
                self._t_autocalib = ahora
                try:
                    antes = self.correccion.t.copy()
                    medida = self.autocalib.estimar(self.mapa.ultimos_puntos, q, prof_mm=prof_mm, K=K,
                                                    T_mundo_cam=self.T_efectiva())
                    if self.correccion.aplicar(medida):
                        # lo que se marcó con la calibración de antes alrededor del brazo era el
                        # propio brazo mal colocado: fuera
                        if np.linalg.norm(self.correccion.t - self._t_mapa) > 0.01:
                            self.mapa.borrar()            # lo de antes se marcó con otra calibración
                            self._t_mapa = self.correccion.t.copy()
                        h = float(np.linalg.norm(self.correccion.t - antes)) + self.mapa.margen_robot + self.mapa.res
                        if h > self.mapa.margen_robot + self.mapa.res + 0.003:
                            gr = [(a, b, r + h) for a, b, r in caps_r]
                            gm = caps_m.copy() if len(caps_m) else ()
                            if len(gm):
                                gm[:, 6] += h
                            self.mapa.vaciar(gr, gm)
                except Exception as e:
                    print(f"⚠️  [autónomo] Autocorrección de la calibración: {e}")
        if self._escaneo_hasta is not None and ahora >= self._escaneo_hasta:
            try:
                self.terminar_escaneo()
            except Exception as e:                       # que un fallo del ajuste no tire la cámara
                import traceback
                traceback.print_exc()
                self._escaneo_hasta = None
                self.aviso_fijo = f"fallo al ajustar la escena fija: {e}"
                self._aplicar_fijo()
        if self.canal_mapa is not None:
            b = empaquetar(self.mapa.centros_ocupados(), self.mapa.res, self.max_dibujo)
            if len(b) <= self.canal_mapa.capacidad:
                self.canal_mapa.escribir(b)
        if self.canal_visto is not None:
            b = empaquetar_visto(self.mapa)
            if len(b) <= self.canal_visto.capacidad:
                self.canal_visto.escribir(b)
        self._n += 1
        if ahora - self._t_hz >= 2.0:
            self.hz, self._n, self._t_hz = self._n / (ahora - self._t_hz), 0, ahora
        return True

    def estado(self):
        if self.T is None:
            return dict(estado="sin", texto="sin calibrar")
        e = self.mapa.estado()
        n_ocu = int((e == OCUPADO).sum())
        return dict(estado="ok" if self.robot_quitado else "aviso",
                    texto=f"{n_ocu} vóxeles" + ("" if self.robot_quitado else " (robot sin quitar)")
                    + (" (mano aproximada)" if self.robot_quitado and not self.mano_real else "")
                    + (f" · calibración: {self.correccion.texto()}" if np.linalg.norm(self.correccion.t) > 0.003
                       else ""),
                    mano_real=self.mano_real,
                    ocupados=n_ocu, libres=int((e == LIBRE).sum()), res=self.mapa.res,
                    hz=round(self.hz, 1), ms=self.mapa.ultima.get("ms"), aviso=self.aviso,
                    correccion=[round(float(v), 4) for v in self.correccion.t],
                    correccion_texto=self.correccion.texto() if self.correccion.n_medidas or
                    np.linalg.norm(self.correccion.t) > 0 else "",
                    fijo=self.estado_fijo())

    def estado_fijo(self):
        if self._escaneo_hasta is not None:
            quedan = max(0.0, self._escaneo_hasta - time.monotonic())
            return dict(escaneando=True, texto=f"escaneando la escena fija… {quedan:.0f} s", cajas=0)
        return dict(escaneando=False, texto=self.escena.texto() + (f" · {self.aviso_fijo}" if self.aviso_fijo else ""),
                    cajas=len(self.escena.cajas), fecha=self.escena.fecha)
