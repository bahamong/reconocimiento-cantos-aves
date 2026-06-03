"""
EJECUCION AUTOMATICA — Proyecto de Reconocimiento de Cantos de Aves
====================================================================
Corre en orden todas las fases del proyecto:

  1. Entrenamiento  → calcula vectores E_C y los escribe en umbrales_energia.py
  2. Demo           → genera audios de prueba en audios_demo/
  3. Prueba precision  → matriz de confusion sobre el banco de datos
  4. Prueba robustez   → precision con ruido sintetico
  5. Interfaz grafica  → abre la GUI (opcional, al final)

Uso:
  python run.py                  # corre fases 1-4 y luego la GUI
  python run.py --sin-gui        # corre fases 1-4 sin abrir la GUI
  python run.py --solo-gui       # abre solo la GUI (asume que ya se entrenó)
  python run.py --tiempo-real    # abre el reconocedor de microfono en lugar de la GUI
"""

import argparse
import subprocess
import sys
import os

RAIZ = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable  # mismo interprete que lanzó este script


def correr(descripcion, *args, **kwargs):
    print()
    print("=" * 64)
    print(f"  {descripcion}")
    print("=" * 64)
    resultado = subprocess.run([PY, *args], cwd=RAIZ, **kwargs)
    if resultado.returncode != 0:
        print(f"\n[ERROR] '{descripcion}' terminó con código {resultado.returncode}.")
        sys.exit(resultado.returncode)


def main():
    ap = argparse.ArgumentParser(description="Ejecución automática del proyecto")
    grupo = ap.add_mutually_exclusive_group()
    grupo.add_argument("--sin-gui",     action="store_true", help="no abrir la interfaz al final")
    grupo.add_argument("--solo-gui",    action="store_true", help="abrir solo la interfaz gráfica")
    grupo.add_argument("--tiempo-real", action="store_true", help="abrir el reconocedor de micrófono")
    args = ap.parse_args()

    if args.solo_gui:
        correr("Interfaz gráfica", os.path.join("interfaz", "interfaz.py"))
        return

    if args.tiempo_real:
        correr("Reconocimiento en tiempo real", os.path.join("tiempo_real", "reconocimiento_tiempo_real.py"))
        return

    # ── FASE 1: Entrenamiento ────────────────────────────────────────────────
    correr(
        "FASE 1 · Entrenamiento — calcula vectores E_C",
        os.path.join("entrenamiento", "entrenar.py"),
    )

    # ── FASE 2: Generar audios de demo ───────────────────────────────────────
    correr(
        "FASE 2 · Generar audios de demo (audios_demo/)",
        os.path.join("pruebas", "generar_audios_demo.py"),
    )

    # ── FASE 3: Prueba de precisión ──────────────────────────────────────────
    correr(
        "FASE 3 · Prueba de precisión — matriz de confusión",
        os.path.join("pruebas", "prueba_precision.py"),
    )

    # ── FASE 4: Prueba de robustez ───────────────────────────────────────────
    correr(
        "FASE 4 · Prueba de robustez — ruido sintético",
        os.path.join("pruebas", "prueba_robustez.py"),
    )

    # ── FASE 5: Interfaz gráfica (opcional) ─────────────────────────────────
    if not args.sin_gui:
        correr(
            "FASE 5 · Interfaz gráfica",
            os.path.join("interfaz", "interfaz.py"),
        )

    print()
    print("=" * 64)
    print("  Todas las fases completadas correctamente.")
    print("=" * 64)


if __name__ == "__main__":
    main()
