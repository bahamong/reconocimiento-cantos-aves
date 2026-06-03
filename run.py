"""
Abre la interfaz grafica del reconocedor de cantos de aves.

Uso:
  python run.py
"""

import subprocess
import sys
import os

RAIZ = os.path.dirname(os.path.abspath(__file__))

subprocess.run([sys.executable, os.path.join(RAIZ, "interfaz", "interfaz.py")])
