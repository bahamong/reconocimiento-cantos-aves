"""
RECONOCIMIENTO EN TIEMPO REAL POR MICROFONO
============================================
Captura audio del microfono del PC en ventanas de unos segundos, calcula su
vector de energia E_X y lo compara contra los VECTORES DE UMBRALES DE ENERGIA
(constantes en umbrales_energia.py) mediante la DIFERENCIA TOTAL ABSOLUTA
E_XC = suma_i |E_Xi - E_Ci|.  Decide el tipo con argmin y aplica el umbral tau.

Uso (demostracion):
  1. Copia los audios de audios_demo a una memoria USB/SD.
  2. Reproducelos en un parlante externo.
  3. Ejecuta:   python tiempo_real/reconocimiento_tiempo_real.py
  4. Acerca el parlante al microfono del PC: el programa reconoce el canto.

Opciones:
  python reconocimiento_tiempo_real.py --duracion 8 --tau 1.0 --device 1 --listar
"""
import os
import sys
import argparse
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from reconocer import reconocer_arreglo, TAU_RECHAZO
from umbrales_energia import NOMBRES_ESPECIES, LIMITES_SUBBANDAS_HZ
from caracteristicas import FS

try:
    import sounddevice as sd
except ImportError:
    print("Falta sounddevice. Instala con:  pip install sounddevice")
    sys.exit(1)

import librosa


def listar_dispositivos():
    print("\nDispositivos de ENTRADA disponibles:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}  ({int(d['default_samplerate'])} Hz)")
    print()


def capturar(duracion, dispositivo):
    """Graba 'duracion' s del microfono y devuelve la senal a 22050 Hz mono."""
    if dispositivo is not None:
        info = sd.query_devices(dispositivo)
    else:
        info = sd.query_devices(kind="input")
    fs_nativo = int(info["default_samplerate"])
    n = int(duracion * fs_nativo)
    kw = dict(samplerate=fs_nativo, channels=1, dtype="float32")
    if dispositivo is not None:
        kw["device"] = dispositivo
    grabacion = sd.rec(n, **kw)
    sd.wait()
    y = grabacion.flatten().astype(np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y - np.mean(y)                       # quitar offset DC
    pico_crudo = float(np.max(np.abs(y)))
    if fs_nativo != FS:
        y = librosa.resample(y.astype(np.float32), orig_sr=fs_nativo, target_sr=FS)
    return y.astype(np.float32), pico_crudo


def barra(valor, vmax=2.0, ancho=24):
    n = int(np.clip(valor / vmax, 0, 1) * ancho)
    return "#" * n + "-" * (ancho - n)


def main():
    ap = argparse.ArgumentParser(description="Reconocimiento de cantos de aves en tiempo real")
    ap.add_argument("--duracion", type=float, default=8.0, help="segundos por ventana")
    ap.add_argument("--tau", type=float, default=TAU_RECHAZO, help="umbral de rechazo")
    ap.add_argument("--device", type=int, default=None, help="indice del microfono")
    ap.add_argument("--listar", action="store_true", help="listar dispositivos y salir")
    args = ap.parse_args()

    if args.listar:
        listar_dispositivos()
        return

    print("=" * 64)
    print(" RECONOCIMIENTO DE CANTOS DE AVES EN TIEMPO REAL")
    print(" Comparacion con vectores de umbrales de energia CONSTANTES")
    print(f" Subbandas: {len(LIMITES_SUBBANDAS_HZ)-1} | tau = {args.tau} | ventana = {args.duracion}s")
    print("=" * 64)
    listar_dispositivos()
    print("Reproduce un canto en el parlante junto al microfono.")
    print("Pulsa Ctrl+C para terminar.\n")

    try:
        while True:
            print("Escuchando..." + " " * 30, end="\r")
            y, pico_crudo = capturar(args.duracion, args.device)

            if pico_crudo < 0.005:
                print("[sin senal] el microfono no capto audio (revisa el dispositivo)        ")
                continue

            r = reconocer_arreglo(y, tau_rechazo=args.tau)

            print("-" * 64)
            if r["rechazado"]:
                print(f" >> NO CLASIFICABLE   (E_XC min = {r['minimo']:.3f} > tau = {args.tau})")
            else:
                print(f" >> {r['nombre_comun'].upper()}  ({r['cientifico']})")
                print(f"    E_XC = {r['minimo']:.3f}   (tau = {args.tau})")
            print("    Diferencia total absoluta por especie  E_XC = suma|E_Xi - E_Ci|:")
            for sp, d in sorted(r["diferencias"].items(), key=lambda x: x[1]):
                marca = " <=" if sp == r["mejor_especie"] else "   "
                print(f"      {NOMBRES_ESPECIES[sp][0]:18s} {d:5.3f}  |{barra(d)}|{marca}")
            print("-" * 64 + "\n")
    except KeyboardInterrupt:
        print("\nFin del reconocimiento.")


if __name__ == "__main__":
    main()
