"""Pruebas sin hardware (no hace falta robot, mano, guante ni cámaras).

Desde la raíz del repo:
    python -m pytest tests -q                      # todas (~2 min)
    python -m pytest tests/test_tarea.py -q        # una
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:                      # config.py, orca_hardware.py, skill_shield/…
    sys.path.insert(0, RAIZ)

# (17-nov) las pruebas no dependen de la escena fija escaneada en el laboratorio (escena_fija.json):
# las de la escena fija la activan ellas mismas con su propio archivo
os.environ.setdefault("SKILL_SHIELD_ESCENA_FIJA", "0")
