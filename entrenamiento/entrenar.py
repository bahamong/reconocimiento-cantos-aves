"""
ENTRENAMIENTO: calcula los vectores de umbrales de energia y los escribe como
CONSTANTES numericas en el archivo umbrales_energia.py (en la raiz del proyecto).

Fase de entrenamiento:
  1. A cada grabacion:  FFT -> magnitud -> ||X(k)||  (espectro).
  2. Se promedian los espectros de cada carpeta:  ||C(k)||_m (espectro promedio).
  3. A cada subbanda del espectro promedio se le calcula la energia:
         E_Ci = (1/N_i) * suma ||C_i(k)||^2
     obteniendo el vector de umbrales de energia  E_C = [E_C1, ..., E_CS].
  4. Se anade la desviacion estandar por subbanda (SIGMA_C) como complemento
     estadistico de la variabilidad de la energia.

El calculo del espectro y de la energia vive en caracteristicas.py, de modo que
entrenamiento y reconocimiento sean identicos.
"""

import os
import sys
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Raiz del proyecto (carpeta superior a esta) para importar el nucleo.
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

DIR_DATOS = os.path.join(RAIZ, "datos")
DIR_MODELOS = os.path.join(RAIZ, "modelos")

import librosa
from preprocesamiento import preparar_senal
from caracteristicas import (FS, N_FFT, SALTO, LIMITES_SUBBANDAS_HZ,
                             espectro_de_senal, vector_energia)

POR_ESPECIE = 60   # misma cantidad de grabaciones por cada tipo de sonido

ESPECIES = ["Sicalis_flaveola", "Mimus_gilvus", "Tyrannus_melancholicus",
            "Pitangus_sulphuratus", "Megascops_choliba"]

NOMBRES = {
    "Sicalis_flaveola":       ("Canario coronado", "Sicalis flaveola"),
    "Mimus_gilvus":           ("Sinsonte tropical", "Mimus gilvus"),
    "Tyrannus_melancholicus": ("Siriri comun", "Tyrannus melancholicus"),
    "Pitangus_sulphuratus":   ("Bichofue", "Pitangus sulphuratus"),
    "Megascops_choliba":      ("Currucutu", "Megascops choliba"),
}


def entrenar():
    os.makedirs(DIR_MODELOS, exist_ok=True)
    print("=" * 70)
    print(" ENTRENAMIENTO: vectores de umbrales de energia")
    print("=" * 70)
    print(f" FS={FS} Hz | ventana=3 s | STFT N_FFT={N_FFT} salto={SALTO}")
    print(f" Subbandas S={len(LIMITES_SUBBANDAS_HZ)-1} -> cortes {LIMITES_SUBBANDAS_HZ} Hz")
    print(f" Grabaciones por especie: {POR_ESPECIE}\n")

    espectros_promedio, E_C, SIGMA_C, n_usadas = {}, {}, {}, {}

    for especie in ESPECIES:
        carpeta = os.path.join(DIR_DATOS, especie)
        if not os.path.isdir(carpeta):
            print(f" [AVISO] No existe {carpeta}")
            continue
        archivos = sorted(f for f in os.listdir(carpeta)
                          if f.lower().endswith((".mp3", ".wav", ".ogg", ".flac")))[:POR_ESPECIE]

        espectros, energias = [], []
        for nombre in archivos:
            try:
                y, _ = librosa.load(os.path.join(carpeta, nombre), sr=FS, mono=True)
                y = preparar_senal(y, especie=None)
                esp = espectro_de_senal(y)              # ||X(k)||
                espectros.append(esp)
                energias.append(vector_energia(esp))
            except Exception as e:
                print(f"   error leyendo {nombre}: {e}")

        if not espectros:
            print(f" [AVISO] sin datos para {especie}")
            continue

        n_usadas[especie] = len(espectros)
        espectros_promedio[especie] = np.mean(espectros, axis=0)        # ||C(k)||_m
        E_C[especie] = vector_energia(espectros_promedio[especie])      # E_C
        SIGMA_C[especie] = np.std(np.array(energias), axis=0) + 1e-8    # SIGMA_C

        print(f" {NOMBRES[especie][0]:18s} ({len(espectros)} grab.)")
        print(f"   E_C = {np.round(E_C[especie], 5).tolist()}")

    if not E_C:
        print("\n[ERROR] No hay grabaciones en datos/.")
        return

    np.save(os.path.join(DIR_MODELOS, "vectores_energia.npy"), E_C)
    np.save(os.path.join(DIR_MODELOS, "vectores_sigma.npy"), SIGMA_C)
    np.save(os.path.join(DIR_MODELOS, "limites_subbandas.npy"), np.array(LIMITES_SUBBANDAS_HZ))
    np.save(os.path.join(DIR_MODELOS, "espectros_promedio.npy"), espectros_promedio)
    print(f"\n Modelos .npy guardados en {DIR_MODELOS}/")

    _escribir_constantes(E_C, SIGMA_C, n_usadas)
    _graficar_espectros(espectros_promedio)


def _formato_vector(v):
    return "[" + ", ".join(f"{x:.8f}" for x in v) + "]"


def _escribir_constantes(E_C, SIGMA_C, n_usadas):
    salida = os.path.join(RAIZ, "umbrales_energia.py")
    L = []
    A = L.append
    A('"""')
    A("VECTORES DE UMBRALES DE ENERGIA  (CONSTANTES del programa)")
    A("===========================================================")
    A("Generado automaticamente por entrenar.py el " + datetime.date.today().isoformat() + ".")
    A("")
    A("Estos vectores estan escritos de forma EXPLICITA y NUMERICA. El")
    A("reconocimiento NO procesa el banco de grabaciones: solo compara el vector")
    A("de energia de la senal contra estas constantes con la diferencia total")
    A("absoluta  E_XC = suma_i |E_Xi - E_Ci|.")
    A("")
    A("Cada vector E_C tiene " + str(len(LIMITES_SUBBANDAS_HZ) - 1) + " componentes (una por subbanda),")
    A("normalizado (suma = 1) para ser independiente del volumen.")
    A('"""')
    A("")
    A("import numpy as np")
    A("")
    A("LIMITES_SUBBANDAS_HZ = " + repr(list(LIMITES_SUBBANDAS_HZ)))
    A("FS    = %d" % FS)
    A("N_FFT = %d" % N_FFT)
    A("SALTO = %d" % SALTO)
    A("")
    A("# E_C -> VECTOR DE UMBRALES DE ENERGIA de cada tipo de canto (CONSTANTE)")
    A("#        E_C[especie] = [E_C1, E_C2, ..., E_CS]")
    A("E_C = {")
    for sp in ESPECIES:
        if sp in E_C:
            A(f"    # {NOMBRES[sp][0]}  ({n_usadas.get(sp,0)} grabaciones)")
            A(f'    "{sp}": np.array({_formato_vector(E_C[sp])}),')
    A("}")
    A("")
    A("# SIGMA_C -> desviacion estandar por subbanda (variabilidad de la energia).")
    A("SIGMA_C = {")
    for sp in ESPECIES:
        if sp in SIGMA_C:
            A(f'    "{sp}": np.array({_formato_vector(SIGMA_C[sp])}),')
    A("}")
    A("")
    A("NOMBRES_ESPECIES = {")
    for sp in ESPECIES:
        c, s = NOMBRES[sp]
        A(f'    "{sp}": ("{c}", "{s}"),')
    A("}")
    A("")
    with open(salida, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f" Constantes escritas en {salida}")


def _graficar_espectros(espectros_promedio):
    colores = ["#FFD700", "#00D4FF", "#7FFF00", "#FF6B35", "#FF1493"]
    eje_frec = np.fft.rfftfreq(N_FFT, d=1.0 / FS)
    mascara = eje_frec <= 8500
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    for i, sp in enumerate(ESPECIES):
        if sp in espectros_promedio:
            ax.plot(eje_frec[mascara], espectros_promedio[sp][mascara],
                    color=colores[i % len(colores)], label=NOMBRES[sp][0],
                    linewidth=1.4, alpha=0.9)
    ymax = ax.get_ylim()[1]
    for hz in LIMITES_SUBBANDAS_HZ[1:-1]:
        ax.axvline(x=hz, color="#9Fb8d8", linestyle="--", linewidth=1.0, alpha=0.95)
        ax.text(hz, ymax * 0.99, f"{hz}", color="#AFC4DF", fontsize=6.5,
                rotation=90, va="top", ha="right")
    ax.set_xlabel("Frecuencia (Hz)", color="#CCCCCC")
    ax.set_ylabel("||C(k)||_m  (magnitud STFT promedio)", color="#CCCCCC")
    ax.set_title("Espectros promedio por especie y lineas verticales de subbandas",
                 color="#FFFFFF", fontsize=12)
    ax.legend(facecolor="#161B22", labelcolor="#FFFFFF", fontsize=9, edgecolor="#30363D")
    ax.tick_params(colors="#CCCCCC")
    for s in ax.spines.values():
        s.set_edgecolor("#30363D")
    plt.tight_layout()
    for destino in (os.path.join(DIR_MODELOS, "espectros_promedio.png"),
                    os.path.join(RAIZ, "espectros_promedio.png")):
        plt.savefig(destino, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f" Grafico de espectros guardado: {os.path.join(RAIZ, 'espectros_promedio.png')}")


if __name__ == "__main__":
    entrenar()
