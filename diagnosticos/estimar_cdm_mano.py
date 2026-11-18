"""
Centro de masas de la mano (+ adaptador) en el marco de la BRIDA, estimado con el MJCF (18-nov).

FR5_CARGA_CDM_MM = None hace que la controladora no sepa dónde está el peso que cuelga de la brida: su
modelo dinámico se equivoca en los pares esperados y la detección de colisiones salta sin choque (el botón
«Reiniciar tras colisión» de la web). Lo mejor es la IDENTIFICACIÓN DE CARGA de la WebApp del FR5 (mueve
el robot y lo mide). Si no se puede todavía, esto da una primera estimación con la geometría del modelo:

    python diagnosticos/estimar_cdm_mano.py

Lo que sale:
  · el centro de masas del subárbol de mano_mount, en el MISMO marco que se usó para FR5_TCP_MM (los ejes de
    wrist3_Link del MJCF, con el origen en la brida: d6 más allá del joint6);
  · la masa que da el modelo, solo como referencia: las mallas de la OrcaHand no llevan masas reales, así que
    la masa buena es la PESADA (FR5_CARGA_KG). El centro sale bien si la masa está repartida como el volumen.
Necesita mujoco (el entorno del proyecto).
"""
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)


def main():
    import mujoco
    import config as cfg
    from skill_shield.teleop_fr5.cinematica import DH_OFICIAL_FR5
    m = mujoco.MjModel.from_xml_path(cfg.RUTA_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    b_mano = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mano_mount")
    b_w3 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist3_Link")
    if b_mano < 0 or b_w3 < 0:
        sys.exit("El MJCF no tiene los cuerpos mano_mount y wrist3_Link.")
    R = d.xmat[b_w3].reshape(3, 3)
    c_w3 = R.T @ (d.subtree_com[b_mano] - d.xpos[b_w3])
    d6 = float(DH_OFICIAL_FR5["d6"])
    c_brida = (c_w3 - np.array([0.0, 0.0, d6])) * 1000.0
    masa_modelo = float(m.body_subtreemass[b_mano])
    print(f"Centro de masas de la mano en la brida: x={c_brida[0]:.1f}  y={c_brida[1]:.1f}  z={c_brida[2]:.1f} mm")
    print(f"(masa según el modelo: {masa_modelo:.2f} kg; la pesada, FR5_CARGA_KG = {getattr(cfg, 'FR5_CARGA_KG', None)})")
    print(f"\n→ config.py:  FR5_CARGA_CDM_MM = ({c_brida[0]:.1f}, {c_brida[1]:.1f}, {c_brida[2]:.1f})")
    print("   Es una estimación: compruébala con la identificación de carga de la WebApp en cuanto se pueda.")


if __name__ == "__main__":
    main()
