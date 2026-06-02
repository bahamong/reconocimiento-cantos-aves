"""
Genera AUDIOS DE PRUEBA de 10 segundos (varios por especie) para la demostracion:
se reproducen en un parlante y se capturan con el microfono para el
reconocimiento en tiempo real.

Para cada especie:
  1. Evalua sus grabaciones con el reconocedor (constantes + diferencia absoluta).
  2. Se queda con las que se clasifican CORRECTAMENTE, con menor distancia y
     mayor margen de confianza.
  3. Extrae la mejor ventana de 10 s (la de mayor actividad de canto).
  4. Normaliza, aplica desvanecimiento y guarda como WAV.
  5. Verifica el audio final reconociendolo de nuevo.
"""
import os, sys, glob
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from reconocer import reconocer_arreglo
from umbrales_energia import E_C, NOMBRES_ESPECIES
from caracteristicas import FS

DIR_DATOS = os.path.join(RAIZ, "datos")
DIR_SALIDA = os.path.join(RAIZ, "audios_demo")
SEGUNDOS = 10.0
N_CLIP = int(FS * SEGUNDOS)
POR_ESPECIE = 3
ESPECIES = list(E_C.keys())


def mejor_ventana(y, ventana=N_CLIP, salto=FS):
    """Ventana de 'ventana' muestras con mayor energia en 300-8000 Hz."""
    if len(y) <= ventana:
        repeticiones = int(np.ceil(ventana / max(len(y), 1)))
        return np.tile(y, repeticiones)[:ventana]
    sos = butter(4, [300/(FS/2), 8000/(FS/2)], btype="band", output="sos")
    yf = sosfilt(sos, y)
    inicio, mejor = 0, -1.0
    for s in range(0, len(y) - ventana + 1, salto):
        e = float(np.sum(yf[s:s+ventana] ** 2))
        if e > mejor:
            mejor, inicio = e, s
    return y[inicio:inicio+ventana].copy()


def hacer_clip(y):
    """Mejor ventana de 10 s + desvanecimiento + normalizacion a -3 dBFS."""
    clip = mejor_ventana(y).astype(np.float32)
    fundido = int(0.05 * FS)
    if len(clip) > 2 * fundido:
        clip[:fundido] *= np.linspace(0, 1, fundido)
        clip[-fundido:] *= np.linspace(1, 0, fundido)
    m = np.max(np.abs(clip))
    if m > 0:
        clip = clip / m * 0.707
    return clip


def candidatos(sp):
    lista = []
    for f in sorted(glob.glob(os.path.join(DIR_DATOS, sp, "*.mp3"))):
        try:
            y, _ = librosa.load(f, sr=FS, mono=True)
        except Exception:
            continue
        if np.max(np.abs(y)) < 1e-4:
            continue
        r = reconocer_arreglo(y, tau_rechazo=1e9)
        if r["mejor_especie"] != sp:
            continue
        d = sorted(r["diferencias"].values())
        margen = (d[1] - d[0]) if len(d) > 1 else 0.0
        lista.append((margen, r["minimo"], f, y))
    lista = [c for c in lista if c[0] > 0.12]      # margen minimo
    lista.sort(key=lambda t: t[1])                 # menor distancia primero
    return lista


def main():
    os.makedirs(DIR_SALIDA, exist_ok=True)
    print("Generando audios demo de 10 s por especie...\n")
    resumen = []
    for sp in ESPECIES:
        comun, cient = NOMBRES_ESPECIES[sp]
        cands = candidatos(sp)
        print(f"== {comun} ({cient}) : {len(cands)} grabaciones correctas ==")
        hechos = 0
        for margen, dmin, ruta, y in cands:
            if hechos >= POR_ESPECIE:
                break
            clip = hacer_clip(y)
            rv = reconocer_arreglo(clip, tau_rechazo=1e9)
            ok = rv["mejor_especie"] == sp
            dv = sorted(rv["diferencias"].values())
            margen_clip = dv[1] - dv[0]
            if not ok or rv["minimo"] > 0.85 or margen_clip < 0.12:
                continue
            hechos += 1
            nombre = f"{hechos}_{comun.replace(' ', '_')}.wav"
            sf.write(os.path.join(DIR_SALIDA, nombre), clip, FS, subtype="PCM_16")
            print(f"   OK {nombre:26s} E_XC={rv['minimo']:.3f} margen={margen_clip:.3f}  <- {os.path.basename(ruta)}")
            resumen.append((nombre, ok, rv["minimo"], margen_clip))
        if hechos == 0:
            print("   [AVISO] No se genero ningun clip robusto.")
        print()
    print("=" * 60)
    print(" RESUMEN DE AUDIOS DEMO")
    print("=" * 60)
    n_ok = sum(1 for r in resumen if r[1])
    for nombre, ok, dmin, margen in resumen:
        print(f"  [{'OK' if ok else 'XX'}] {nombre:28s} E_XC={dmin:.3f} margen={margen:.3f}")
    print(f"\n  {n_ok}/{len(resumen)} audios reconocidos correctamente.")
    print(f"  Carpeta: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
