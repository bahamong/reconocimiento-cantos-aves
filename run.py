"""
Abre la interfaz Streamlit del reconocedor de cantos de aves.

Uso:
  python run.py
"""

import subprocess
import sys
import os

RAIZ = os.path.dirname(os.path.abspath(__file__))
INTERFAZ = os.path.join(RAIZ, "interfaz", "interfaz.py")

subprocess.run([sys.executable, "-m", "streamlit", "run", INTERFAZ])
