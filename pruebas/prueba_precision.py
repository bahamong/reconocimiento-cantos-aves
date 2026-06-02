"""
Prueba de precision: recorre todas las grabaciones de datos/ y arma la matriz de
confusion con el reconocedor (constantes + diferencia total absoluta).
"""
import os, sys, glob
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import librosa
from reconocer import reconocer_arreglo
from caracteristicas import FS
from umbrales_energia import E_C, NOMBRES_ESPECIES

DIR_DATOS = os.path.join(RAIZ, "datos")
ESPECIES = list(E_C.keys())

def corto(sp): return NOMBRES_ESPECIES[sp][0]

def main(tau=None):
    conf = {s: {p: 0 for p in ESPECIES} for s in ESPECIES}
    d_ok, d_mal = [], []
    total = aciertos = 0
    for sp in ESPECIES:
        for f in sorted(glob.glob(os.path.join(DIR_DATOS, sp, "*.mp3"))):
            try:
                y, _ = librosa.load(f, sr=FS, mono=True)
            except Exception:
                continue
            r = reconocer_arreglo(y, tau_rechazo=(tau if tau is not None else 1e9))
            pred = r["mejor_especie"]
            conf[sp][pred] += 1
            total += 1
            if pred == sp:
                aciertos += 1; d_ok.append(r["minimo"])
            else:
                d_mal.append(r["minimo"])

    print("\n=== MATRIZ DE CONFUSION (fila=real, col=predicho) ===")
    print("real \\ pred  " + " ".join(f"{corto(p)[:8]:>9s}" for p in ESPECIES))
    for s in ESPECIES:
        fila = " ".join(f"{conf[s][p]:>9d}" for p in ESPECIES)
        t = sum(conf[s].values())
        acc = 100.0 * conf[s][s] / t if t else 0
        print(f"{corto(s)[:11]:11s} {fila}  acc={acc:5.1f}% (n={t})")
    print(f"\nPRECISION GLOBAL: {100.0*aciertos/total:.2f}%  ({aciertos}/{total})")
    do, dm = np.array(d_ok), np.array(d_mal)
    if len(do):
        print(f"E_XC aciertos: med={np.median(do):.3f} p90={np.percentile(do,90):.3f} "
              f"p95={np.percentile(do,95):.3f} max={do.max():.3f}")
    if len(dm):
        print(f"E_XC errores : med={np.median(dm):.3f} min={dm.min():.3f}")

if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else None)
