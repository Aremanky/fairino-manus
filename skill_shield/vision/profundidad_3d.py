"""
Utilidades de percepción 3D (píxel + profundidad estéreo → punto 3D).

(Extraído de Skill_Shield_Nivel_6.py — código sin cambios, solo reubicado.)
"""

from config import *


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE PERCEPCIÓN 3D
# ══════════════════════════════════════════════════════════════════════════════

def pixel_a_3d(px: int, py: int, z_m: float,
               ancho: int = ANCHO_CAM, alto: int = ALTO_CAM, f_px: float = F_PX):
    """Proyección inversa: pixel + profundidad → coordenadas 3D métricas.
    El centro óptico se toma en el centro del fotograma (ancho/2, alto/2)."""
    cx = ancho / 2.0
    cy = alto  / 2.0
    return (px - cx) * z_m / f_px, (py - cy) * z_m / f_px, z_m
