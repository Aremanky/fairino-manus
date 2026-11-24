"""
Mano física: construcción del array de motores desde la simulación + límite de seguridad.

El puente con el SDK de la OrcaHand está en orca_hardware.py (en esta misma carpeta).

(Extraído de Skill_Shield_Nivel_6.py — código sin cambios, solo reubicado.)
"""

from config import *


# ══════════════════════════════════════════════════════════════════════════════
# HARDWARE REAL — construcción del array de motores + límite de seguridad
# ══════════════════════════════════════════════════════════════════════════════

def construir_hardware_target(datos) -> list:
    """Convierte datos.ctrl[] (orden de actuadores MuJoCo) al orden ESTRICTO
    que espera OrcaHand (Motor 1 a 17). Misma lógica que ya usaba la
    teleoperación, factorizada aquí para reutilizarla también en Modo IA Real."""
    # ctrl_mano: vista sobre la porción de mano de datos.ctrl[], desplazada
    # OFFSET_MANO posiciones porque ahora el brazo ocupa los primeros índices.
    ctrl_mano = datos.ctrl[OFFSET_MANO: OFFSET_MANO + N_ACT]
    mu_pos = {
        "wrist":      float(ctrl_mano[0]),
        "pinky_abd":  float(ctrl_mano[1]),
        "pinky_mcp":  float(ctrl_mano[2]),
        "pinky_pip":  float(ctrl_mano[3]),
        "ring_abd":   float(ctrl_mano[4]),
        "ring_mcp":   float(ctrl_mano[5]),
        "ring_pip":   float(ctrl_mano[6]),
        "middle_abd": float(ctrl_mano[7]),
        "middle_mcp": float(ctrl_mano[8]),
        "middle_pip": float(ctrl_mano[9]),
        "index_abd":  float(ctrl_mano[10]),
        "index_mcp":  float(ctrl_mano[11]),
        "index_pip":  float(ctrl_mano[12]),
        "thumb_cmc":  float(ctrl_mano[13]),
        "thumb_abd":  float(ctrl_mano[14]),
        "thumb_mcp":  float(ctrl_mano[15]),
        "thumb_dip":  float(ctrl_mano[16]),
    }
    return [
        mu_pos["wrist"], mu_pos["index_pip"], mu_pos["index_mcp"], mu_pos["index_abd"],
        mu_pos["middle_abd"], mu_pos["ring_abd"], mu_pos["ring_mcp"], mu_pos["ring_pip"],
        mu_pos["middle_pip"], mu_pos["middle_mcp"], mu_pos["pinky_pip"], mu_pos["pinky_mcp"],
        mu_pos["pinky_abd"], mu_pos["thumb_abd"], mu_pos["thumb_mcp"], mu_pos["thumb_dip"],
        mu_pos["thumb_cmc"],
    ]


# Máximo cambio permitido por motor entre un envío y el siguiente al hardware
# REAL, en radianes. Pensado para el Modo IA Real: como la IA puede predecir
# saltos más bruscos que una teleoperación humana (que ya viene naturalmente
# suavizada por tu propio movimiento), esto evita tirones mecánicos si la red
# predice algo poco fiable. NO se aplica a la teleoperación manual (ya es segura
# por venir de un humano moviéndose a ritmo natural).
MAX_DELTA_HARDWARE_POR_PASO = 0.12  # rad (~7°) por paso de envío


def limitar_salto_hardware(objetivo: list, anterior: list,
                            max_delta: float = MAX_DELTA_HARDWARE_POR_PASO) -> list:
    """Recorta cada componente de 'objetivo' para que no se aleje de 'anterior'
    más de 'max_delta' radianes. Devuelve una nueva lista, ya limitada."""
    limitado = []
    for obj, ant in zip(objetivo, anterior):
        delta = obj - ant
        if delta > max_delta:
            obj = ant + max_delta
        elif delta < -max_delta:
            obj = ant - max_delta
        limitado.append(obj)
    return limitado
