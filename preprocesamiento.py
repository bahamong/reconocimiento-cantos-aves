"""
Preprocesamiento de la senal de audio.

Pasos:
  1. Seleccionar el segmento de 3 s con mayor actividad de canto.
  2. Aplicar el filtro pasabanda Butterworth (unico mecanismo de reduccion de
     ruido del sistema): atenua todo lo que esta fuera del rango de las aves.
  3. Normalizar la amplitud (pico = 1) para que el resultado sea independiente
     del volumen de la grabacion.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

FS = 22050                       # frecuencia de muestreo (Hz)
DURACION = 3.0                   # duracion estandar de analisis (s)
N_MUESTRAS = int(FS * DURACION)

# Rango de frecuencia del canto de cada especie (solo para la descarga del banco).
RANGOS_ESPECIE = {
    "Sicalis_flaveola":       (2000, 8000),
    "Mimus_gilvus":           (1500, 8000),
    "Tyrannus_melancholicus": (1000, 6000),
    "Pitangus_sulphuratus":   (800,  5000),
    "Megascops_choliba":      (300,  1500),
}

# Banda comun usada en el reconocimiento (cubre a las cinco especies).
PASABANDA = (300, 8000)


# ── PASO 2a ──────────────────────────────────────────────────────────────────
def filtro_pasabanda(corte_bajo, corte_alto, fs, orden=4):
    # Diseña matematicamente el filtro IIR Butterworth.
    # "Pasabanda" significa: deja pasar solo las frecuencias entre corte_bajo y corte_alto.
    # Ejemplo: corte_bajo=300 Hz, corte_alto=8000 Hz → elimina el ruido de motores
    # (muy grave, <300 Hz) y el sibido electrico (muy agudo, >8000 Hz).
    nyquist = fs / 2.0                              # frecuencia maxima representable
    bajo = max(corte_bajo / nyquist, 1e-4)          # normaliza entre 0 y 1
    alto = min(corte_alto / nyquist, 1.0 - 1e-4)
    return butter(orden, [bajo, alto], btype="band", output="sos")  # coeficientes del filtro


def aplicar_pasabanda(senal, corte_bajo, corte_alto, fs):
    # ── PASO 2b ───────────────────────────────────────────────────────────────
    # Aplica el filtro a la senal muestra por muestra.
    # Antes: la senal contiene voces humanas, viento, motores, etc.
    # Despues: solo quedan las frecuencias donde cantan las aves (300-8000 Hz).
    sos = filtro_pasabanda(corte_bajo, corte_alto, fs)
    return sosfilt(sos, senal)                      # sosfilt = aplica el filtro IIR


def segmento_activo(y, fs, duracion=DURACION, salto=0.5):
    # Recorre el audio en ventanas de 0.5 s buscando donde el ave canta mas fuerte.
    # Ejemplo: grabacion de 30 s donde el ave canta solo en el segundo 12-15.
    # Esta funcion detecta esos 3 segundos y descarta los silencios del resto.
    n_ventana = int(fs * duracion)                  # 3 s = 66150 muestras a 22050 Hz
    n_salto = int(fs * salto)                       # avanza de 0.5 en 0.5 segundos
    if len(y) <= n_ventana:
        return np.pad(y, (0, n_ventana - len(y)))   # si el audio es corto, rellena con ceros
    sos = filtro_pasabanda(300, 8000, fs)
    y_filtrada = sosfilt(sos, y)                    # filtra para medir solo el canto del ave
    mejor_inicio, mejor_energia = 0, -1.0
    for inicio in range(0, len(y) - n_ventana + 1, n_salto):
        energia = float(np.sum(y_filtrada[inicio:inicio + n_ventana] ** 2))  # energia de esa ventana
        if energia > mejor_energia:
            mejor_energia = energia
            mejor_inicio = inicio                   # guarda donde encontro mas canto
    return y[mejor_inicio:mejor_inicio + n_ventana].copy()  # devuelve esos 3 segundos


def preparar_senal(y, especie=None):
    # Aplica los tres pasos de preprocesamiento en orden.
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)  # limpia valores invalidos

    # ── PASO 1b: recortar a los 3 segundos con mas actividad de canto ────────
    y = segmento_activo(y, FS, duracion=DURACION)

    # ── PASO 2: aplicar el filtro Butterworth 300-8000 Hz ────────────────────
    if especie and especie in RANGOS_ESPECIE:
        bajo, alto = RANGOS_ESPECIE[especie]        # banda especifica de la especie
    else:
        bajo, alto = PASABANDA                      # banda comun para las 5 especies
    y = aplicar_pasabanda(y, bajo, alto, FS)

    # Normalizacion de amplitud: hace que el volumen no afecte el resultado.
    # Ejemplo: el mismo canto grabado fuerte o suave dara el mismo vector E_X.
    maximo = np.max(np.abs(y))
    if maximo > 0:
        y = y / maximo                              # escala para que el pico sea 1.0
    return y.astype(np.float32)


# ── PASO 1a ──────────────────────────────────────────────────────────────────
def cargar_y_preparar(ruta, especie=None):
    # Lee el archivo de audio desde el disco y lo convierte a:
    # - mono (un solo canal, no estereo)
    # - 22050 muestras por segundo (FS)
    # Luego aplica preparar_senal() con los pasos 1b, 2 y normalizacion.
    y, _ = librosa.load(ruta, sr=FS, mono=True)    # ← aqui se lee el archivo
    return preparar_senal(y, especie=especie)


def preparar_directorio(dir_entrada, dir_salida, especie):
    """Preprocesa todos los audios de un directorio y los guarda como WAV."""
    os.makedirs(dir_salida, exist_ok=True)
    archivos = [f for f in os.listdir(dir_entrada)
                if f.lower().endswith((".mp3", ".wav", ".ogg", ".flac"))]
    procesados = 0
    for nombre in archivos:
        origen = os.path.join(dir_entrada, nombre)
        destino = os.path.join(dir_salida, os.path.splitext(nombre)[0] + ".wav")
        try:
            y = cargar_y_preparar(origen, especie)
            sf.write(destino, y, FS)
            procesados += 1
        except Exception as e:
            print(f"  Error procesando {nombre}: {e}")
    return procesados
