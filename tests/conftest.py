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
