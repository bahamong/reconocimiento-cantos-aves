"""
Prueba de robustez de los audios demo simulando la cadena altavoz->microfono:
ruido rosa a varios SNR y coloracion del microfono. Verifica que se mantenga la
especie correcta.
"""
import os, sys, glob
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from reconocer import reconocer_arreglo
from umbrales_energia import NOMBRES_ESPECIES
from caracteristicas import FS

DIR = os.path.join(RAIZ, "audios_demo")
NOMBRE_A_CLAVE = {v[0].replace(" ", "_"): k for k, v in NOMBRES_ESPECIES.items()}
rng = np.random.default_rng(0)


def ruido_rosa(n):
    blanco = rng.standard_normal(n)
    sos = butter(1, 0.02, btype="low", output="sos")
    rosa = sosfilt(sos, blanco) + 0.3 * blanco
    return rosa / (np.std(rosa) + 1e-9)


def agregar_ruido(y, snr_db):
    p_senal = np.mean(y ** 2) + 1e-12
    ruido = ruido_rosa(len(y))
    p_ruido = np.mean(ruido ** 2) + 1e-12
    k = np.sqrt(p_senal / (p_ruido * (10 ** (snr_db / 10))))
    return y + k * ruido


def coloracion_microfono(y):
    sos = butter(2, [800/(FS/2), 6000/(FS/2)], btype="band", output="sos")
    return np.tanh(1.2 * (0.6 * y + 0.4 * sosfilt(sos, y)))


def clave_desde_nombre(fname):
    base = os.path.splitext(fname)[0]
    base = base.split("_", 1)[1] if base[0].isdigit() else base
    return NOMBRE_A_CLAVE.get(base)


def main():
    archivos = sorted(glob.glob(os.path.join(DIR, "*.wav")))
    condiciones = ["limpio", "SNR20", "SNR15", "SNR10", "color+SNR12"]
    print(f"{'archivo':28s} " + " ".join(f"{c:>11s}" for c in condiciones))
    total = ok_total = 0
    fallos = []
    for f in archivos:
        fname = os.path.basename(f)
        clave = clave_desde_nombre(fname)
        y, _ = sf.read(f)
        if y.ndim > 1:
            y = y[:, 0]
        y = y.astype(np.float32)
        versiones = {
            "limpio": y, "SNR20": agregar_ruido(y, 20), "SNR15": agregar_ruido(y, 15),
            "SNR10": agregar_ruido(y, 10), "color+SNR12": coloracion_microfono(agregar_ruido(y, 12)),
        }
        celdas = []
        for c in condiciones:
            r = reconocer_arreglo(versiones[c], tau_rechazo=1e9)
            bien = r["mejor_especie"] == clave
            total += 1; ok_total += bien
            if not bien:
                fallos.append((fname, c, NOMBRES_ESPECIES[r["mejor_especie"]][0]))
            celdas.append(f"{'OK' if bien else 'XX'} {r['minimo']:.2f}")
        print(f"{fname:28s} " + " ".join(f"{c:>11s}" for c in celdas))
    print(f"\nRobustez global: {ok_total}/{total} = {100*ok_total/total:.1f}%")
    if fallos:
        print("Fallos:")
        for fn, c, pred in fallos:
            print(f"   {fn} [{c}] -> {pred}")


if __name__ == "__main__":
    main()
