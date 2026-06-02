"""
RECONOCIMIENTO DE CANTOS DE AVES.

El reconocimiento NO procesa el banco de grabaciones. Se compara el vector de
energia de la senal (E_X) contra los VECTORES DE UMBRALES DE ENERGIA que son
CONSTANTES en el programa (archivo umbrales_energia.py), mediante la DIFERENCIA
TOTAL ABSOLUTA.

Fase de reconocimiento:
  1. X -> senal ;  ||X(k)|| ;  E_Xi = (1/N_i) suma ||X_i(k)||^2  ->  E_X.
  2. E_XC = suma_i |E_Xi - E_Ci|   (diferencia total absoluta) por cada tipo C.
  3. tipo_X = argmin { E_XC1, ..., E_XC5 }.
  4. Si E_XC minimo > tau  ->  "No clasificable".
"""

import os
import sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# ── PASO 6 ──────────────────────────────────────────────────────────────────
# Cargamos los 5 vectores de referencia que ya estan escritos como numeros
# fijos en umbrales_energia.py.  El programa NO abre ningun archivo de audio
# del banco en este momento: solo lee estas constantes numericas.
# (Si abres umbrales_energia.py veras 5 lineas np.array([0.13, 0.06, ...]))
from umbrales_energia import E_C, SIGMA_C, LIMITES_SUBBANDAS_HZ, NOMBRES_ESPECIES
from caracteristicas import energia_de_senal, FS

# Si la diferencia minima supera este valor, el sonido no se parece a ninguna
# de las 5 aves y se devuelve "No clasificable".
# Los cantos reales de las aves dan valores entre 0.3 y 0.7; el ruido da ~1.0.
TAU_RECHAZO = 1.0


# ── PASO 7 ──────────────────────────────────────────────────────────────────
def diferencia_absoluta_total(E_X, E_Ci):
    # Compara el "perfil de energia" del audio nuevo (E_X) con el perfil de
    # referencia de una especie (E_Ci) subbanda por subbanda.
    # Ejemplo: si E_X[0]=0.20 y E_Ci[0]=0.13  ->  |0.20 - 0.13| = 0.07
    # Se suman esas diferencias en las 20 subbandas: E_XC = suma|E_Xi - E_Ci|
    # Cuanto menor sea E_XC, mas se parece el audio a esa especie.
    n = min(len(E_X), len(E_Ci))
    return float(np.sum(np.abs(E_X[:n] - E_Ci[:n])))


def reconocer_arreglo(y, tau_rechazo=None):
    if tau_rechazo is None:
        tau_rechazo = TAU_RECHAZO

    # ── PASOS 2-5: preprocesar la señal y calcular su vector de energia E_X
    # (Butterworth + STFT + magnitudes + energia por subbanda, todo en una linea)
    E_X = energia_de_senal(np.asarray(y, dtype=np.float32))

    # ── PASO 7: calcular E_XC para cada una de las 5 especies
    # Pregunta: "¿cuanto se diferencia este audio del Canario? ¿del Sinsonte?..."
    # E_C.items() son los 5 vectores constantes; NUNCA se leen archivos de audio.
    diferencias = {sp: diferencia_absoluta_total(E_X, E_Ci) for sp, E_Ci in E_C.items()}

    # ── PASO 8: argmin → la especie con menor diferencia es la ganadora
    # Ejemplo: Canario=0.33, Sinsonte=0.91, Siriri=1.05 → gana Canario (0.33)
    mejor = min(diferencias, key=diferencias.get)
    minimo = diferencias[mejor]

    # ── PASO 9: umbral de rechazo
    # Si hasta la especie mas cercana da E_XC > 1.0, el sonido es ruido/ajeno.
    rechazado = minimo > tau_rechazo

    base = {"E_X": E_X, "diferencias": diferencias, "minimo": minimo,
            "tau_rechazo": tau_rechazo, "mejor_especie": mejor}
    if rechazado:
        return {**base, "clave_especie": None, "nombre_comun": "No clasificable",
                "cientifico": "", "rechazado": True}
    comun, cientifico = NOMBRES_ESPECIES.get(mejor, (mejor, ""))
    return {**base, "clave_especie": mejor, "nombre_comun": comun,
            "cientifico": cientifico, "E_C_mejor": E_C[mejor],
            "sigma_mejor": SIGMA_C.get(mejor), "rechazado": False}


def reconocer_archivo(ruta, tau_rechazo=None):
    """Reconoce un archivo de audio (.wav/.mp3)."""
    import librosa
    y, _ = librosa.load(ruta, sr=FS, mono=True)
    return reconocer_arreglo(y, tau_rechazo=tau_rechazo)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("Uso: python reconocer.py <ruta_audio> [tau]")
        sys.exit(1)
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else None
    r = reconocer_archivo(sys.argv[1], tau_rechazo=tau)
    print(f"\nResultado : {r['nombre_comun']}")
    if not r["rechazado"]:
        print(f"Especie   : {r['cientifico']}")
    print(f"E_XC min  : {r['minimo']:.5f}   (tau = {r['tau_rechazo']:.3f})")
    print("Diferencia total absoluta por especie  E_XC = suma|E_Xi-E_Ci| :")
    for sp, d in sorted(r["diferencias"].items(), key=lambda x: x[1]):
        print(f"   {NOMBRES_ESPECIES.get(sp, (sp,))[0]:18s} {d:.5f}")
