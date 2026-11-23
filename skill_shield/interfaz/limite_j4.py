"""
Límite del J4 para el modo «Gemelo con sliders» (23-nov).

Si la muñeca se dobla mucho, la OrcaHand vuelve hacia atrás y se acerca al giro del J4 del FR5, y eso
rompe una pieza de la mano. Aquí se comprueba SOLO eso: el hueco entre la mano (sus mallas reales del
gemelo, skill_shield/autonomo/choque_mano.py) y el giro del J4 (wrist1_Link). Nada de soporte, mesa
ni pedal: en este modo no hace falta.

Ese hueco solo depende de J5 y J6 (el giro del J4 va justo antes de ellos), así que se calcula una
tabla J5 × J6 una vez al arrancar (unos segundos, en un hilo) y luego cada comprobación es inmediata.
El generador suave mueve cada joint por su cuenta, de forma monótona, así que el camino de (J5, J6)
entre la postura actual y la pedida queda dentro del rectángulo que forman: se exige el hueco en todo
ese rectángulo.

Regla: el hueco mínimo del camino tiene que ser ≥ SLIDERS_J4_HOLGURA. Si el brazo ya está más cerca
que eso (p. ej. al armar), se le deja moverse siempre que no se acerque más.
"""
import math
import threading

import numpy as np


class LimiteJ4:
    def __init__(self, cadena, ruta_xml, holgura=0.05, paso_j5_deg=2.0, paso_j6_deg=30.0):
        self.cadena = cadena
        self.ruta_xml = ruta_xml
        self.holgura = float(holgura)
        lim = np.asarray(cadena.limites, float)
        self._j5 = self._rejilla(lim[4], math.radians(paso_j5_deg))
        self._j6 = self._rejilla(lim[5], math.radians(paso_j6_deg))
        self.tabla = None                     # hueco (m) en cada (J5, J6) de la rejilla
        self.error = None
        self._hueco_1 = None

    @classmethod
    def desde_config(cls, cfg):
        from skill_shield.teleop_fr5.cinematica import cargar_cadena
        return cls(cargar_cadena(cfg["RUTA_XML"], verbose=False), cfg["RUTA_XML"],
                   holgura=float(cfg.get("SLIDERS_J4_HOLGURA", 0.05)))

    @staticmethod
    def _rejilla(rango, paso):
        a, b = float(rango[0]), float(rango[1])
        n = max(2, int(math.ceil((b - a) / paso)) + 1)
        return np.linspace(a, b, n)

    # ── preparación ──
    def preparar(self):
        """Calcula la tabla (bloquea unos segundos). True si ha ido bien."""
        try:
            from skill_shield.autonomo.choque_mano import ChoqueManoBrazo, _dist_arbol
            ch = ChoqueManoBrazo(self.cadena, self.ruta_xml, self.holgura)
            i, arbol = ch.arboles["wrist1_Link"]
            P, G = ch.mano, ch._grueso["mano"]
            cota = self.holgura + 0.05

            def hueco(q):
                m = self.cadena.fk_marcos(np.asarray(q, float))
                return _dist_arbol(arbol, G, P, np.linalg.inv(m[i]) @ m[5], cota)
            self._hueco_1 = hueco
            q = np.zeros(6)
            tabla = np.empty((len(self._j5), len(self._j6)))
            for a, q5 in enumerate(self._j5):
                for b, q6 in enumerate(self._j6):
                    q[4], q[5] = q5, q6
                    d = hueco(q)
                    tabla[a, b] = d if math.isfinite(d) else cota
            self.tabla = tabla
            return True
        except Exception as e:
            self.error = f"no se pudo preparar el límite del J4: {e}"
            print(f"⚠️  {self.error}")
            return False

    def iniciar(self):
        threading.Thread(target=self.preparar, daemon=True, name="LimiteJ4").start()
        return self

    @property
    def listo(self):
        return self.tabla is not None

    # ── consultas ──
    def _indices(self, eje, a, b):
        lo, hi = min(a, b), max(a, b)
        i0 = max(int(np.searchsorted(eje, lo, side="right")) - 1, 0)
        i1 = min(int(np.searchsorted(eje, hi, side="left")), len(eje) - 1)
        return i0, max(i1, i0)

    def hueco_rect(self, q_a, q_b):
        """Hueco mínimo (m) con (J5, J6) en el rectángulo entre q_a y q_b (celdas de la rejilla que lo tocan)."""
        a0, a1 = self._indices(self._j5, q_a[4], q_b[4])
        b0, b1 = self._indices(self._j6, q_a[5], q_b[5])
        return float(self.tabla[a0:a1 + 1, b0:b1 + 1].min())

    def permitido(self, q_desde, q_hasta):
        """(ok, hueco mínimo del camino en m). ok = False si la tabla no está lista."""
        if not self.listo:
            return False, float("nan")
        h_camino = self.hueco_rect(q_desde, q_hasta)
        h_inicio = self.hueco_rect(q_desde, q_desde)
        # (el rectángulo incluye la celda de partida: si ya estaba cerca, vale mientras no se acerque más)
        return h_camino >= min(self.holgura, h_inicio), h_camino
