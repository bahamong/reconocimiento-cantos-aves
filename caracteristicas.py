"""
Extraccion de caracteristicas: espectro de magnitud y vector de energia por
subbanda.

Este modulo concentra el calculo del espectro y del vector de energia para que
el entrenamiento y el reconocimiento usen exactamente el mismo procedimiento y,
asi, los vectores de referencia E_C (constantes) y el vector E_X de la senal en
tiempo real sean comparables.

Procedimiento:
  1. Preprocesamiento (segmento activo + filtro Butterworth + normalizacion).
  2. STFT y magnitud:  ||X(k)|| = | FFT |  por ventana, promediada en el tiempo.
  3. Energia por subbanda:  E_i = (1/N_i) * suma de ||X_i(k)||^2.
  4. Normalizacion del vector por su suma (independiente del volumen).
"""

import numpy as np
from scipy.signal import stft as stft_scipy

from preprocesamiento import FS, preparar_senal

# Parametros de la STFT.
N_FFT = 1024
SALTO = 256

# ── PASO 5a ─────────────────────────────────────────────────────────────────
# Los 20 "cajones de frecuencia" (subbandas) que se usan para describir el canto.
# Ejemplo: el Currucutu canta fuerte en 300-1500 Hz (cajones graves);
#          el Canario coronado canta fuerte en 5000-8000 Hz (cajones agudos).
# Esta diferencia es la que permite distinguir una especie de otra.
LIMITES_SUBBANDAS_HZ = [300, 500, 700, 900, 1100, 1300, 1500, 1800, 2100, 2500,
                        3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]


def hz_a_bin(frecuencia_hz):
    # Convierte Hz al indice del arreglo de la STFT.
    # Ejemplo: 1000 Hz → columna 46 del espectro.
    return int(round(frecuencia_hz * N_FFT / FS))


def _bins_subbandas():
    bins = [hz_a_bin(f) for f in LIMITES_SUBBANDAS_HZ]
    return [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]


BINS_SUBBANDAS = _bins_subbandas()


# ── PASO 3 ───────────────────────────────────────────────────────────────────
def espectro_de_senal(y):
    # STFT: divide el audio en ventanitas de ~46 ms y le aplica la FFT a cada una.
    # Ejemplo: un audio de 3 s queda dividido en ~250 ventanas pequeñas.
    _, _, Zxx = stft_scipy(y, fs=FS, nperseg=N_FFT, noverlap=N_FFT - SALTO)

    # ── PASO 4 ────────────────────────────────────────────────────────────────
    # np.abs(Zxx): saca la magnitud de cada componente  → ||X(k)||
    # np.mean(..., axis=1): promedia en el tiempo todas las ventanitas.
    # Resultado: un solo vector que resume "cuanta energia hay en cada frecuencia".
    return np.mean(np.abs(Zxx), axis=1)


# ── PASO 5b ──────────────────────────────────────────────────────────────────
def vector_energia(espectro):
    # Recorre los 20 cajones de frecuencia y calcula cuanta energia hay en cada uno.
    E = np.zeros(len(BINS_SUBBANDAS))
    for i, (a, b) in enumerate(BINS_SUBBANDAS):
        segmento = espectro[a:b]         # frecuencias que caen en el cajon i
        n = len(segmento)                # cuantas frecuencias hay en ese cajon
        if n > 0:
            # Promedio de ||X_i(k)||² en el cajon i  (formula del profesor)
            # Ejemplo: cajon 300-500 Hz tiene 9 frecuencias; se suma su energia y se divide entre 9.
            E[i] = (1.0 / n) * np.sum(segmento ** 2)

    # Normaliza dividiendo entre la suma total.
    # Esto hace que no importe si el audio es fuerte o suave: solo importa
    # en qué cajones está concentrada la energia (la "forma" del canto).
    suma = E.sum()
    return E / suma if suma > 0 else E


def energia_de_senal(y, ya_preparada=False):
    # Funcion principal: recibe audio crudo y devuelve el vector E_X de 20 numeros.
    # Internamente aplica los pasos 2, 3, 4 y 5 en secuencia.
    if not ya_preparada:
        y = preparar_senal(np.asarray(y, dtype=np.float32), especie=None)
    return vector_energia(espectro_de_senal(y))
