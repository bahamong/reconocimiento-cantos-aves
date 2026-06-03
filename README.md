<div align="center">

# 🦜 BioAcústica — Reconocimiento de Cantos de Aves

### Procesamiento Digital de Señales · Llanos Orientales, Colombia

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![librosa](https://img.shields.io/badge/librosa-0.10+-darkgreen?style=flat-square)](https://librosa.org)
[![License](https://img.shields.io/badge/Licencia-Académica-blue?style=flat-square)](#)

Sistema de reconocimiento automático de cantos de aves mediante **bancos de filtros de frecuencia** y **vectores de energía por subbanda**. Identifica 5 especies nativas del Meta (Colombia) a partir de un archivo de audio o grabación en tiempo real.

</div>

---

## Tabla de contenidos

- [¿Qué hace este proyecto?](#qué-hace-este-proyecto)
- [Especies reconocidas](#especies-reconocidas)
- [Fundamento matemático del algoritmo](#fundamento-matemático-del-algoritmo)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Interfaz gráfica (Streamlit)](#1-interfaz-gráfica-streamlit--recomendado)
  - [Reconocimiento por línea de comandos](#2-reconocimiento-por-línea-de-comandos)
  - [Re-entrenar el modelo](#3-re-entrenar-el-modelo)
  - [Descargar nuevos datos](#4-descargar-nuevos-datos)
- [Descripción de cada módulo](#descripción-de-cada-módulo)
- [Flujo completo paso a paso](#flujo-completo-paso-a-paso)
- [Parámetros del sistema](#parámetros-del-sistema)
- [Requerimientos](#requerimientos)
- [Preguntas frecuentes](#preguntas-frecuentes)

---

## ¿Qué hace este proyecto?

Dado un archivo de audio (`.wav` o `.mp3`) o una grabación en tiempo real por micrófono, el sistema:

1. **Preprocesa** la señal: extrae el segmento de 3 segundos con mayor actividad de canto y aplica un filtro Butterworth pasa-banda 300–8000 Hz para eliminar ruido ambiental.
2. **Extrae características**: calcula la STFT (transformada de Fourier de tiempo corto), obtiene el espectro de magnitud y lo comprime en un **vector de 20 energías por subbanda** — el "perfil espectral" del canto.
3. **Clasifica**: compara ese vector contra los 5 vectores de referencia entrenados (uno por especie) usando la **diferencia total absoluta** `E_XC = Σ|E_Xi − E_Ci|`. La especie con menor diferencia gana.
4. **Rechaza** sonidos que no son cantos de estas aves si la diferencia mínima supera el umbral τ = 1.0.

> El reconocimiento es **instantáneo**: no accede al banco de grabaciones en tiempo de ejecución. Solo usa 5 vectores de 20 números ya escritos como constantes en el código.

---

## Especies reconocidas

| Especie | Nombre científico | Rango de canto | Hábitat |
|---|---|---|---|
| 🐦 **Canario coronado** | *Sicalis flaveola* | 2000–8000 Hz | Sabanas abiertas |
| 🎵 **Sinsonte tropical** | *Mimus gilvus* | 1500–8000 Hz | Bordes de bosque, zonas urbanas |
| 🦜 **Sirirí común** | *Tyrannus melancholicus* | 1000–6000 Hz | Áreas abiertas con árboles dispersos |
| 🦅 **Bichofué / Cristofué** | *Pitangus sulphuratus* | 800–5000 Hz | Bordes de cuerpos de agua, jardines |
| 🦉 **Currucutú** | *Megascops choliba* | 300–1500 Hz | Bosques de galería, sabanas arboladas |

Todas las grabaciones de entrenamiento provienen de **Colombia** (API [xeno-canto](https://xeno-canto.org)), con **60 grabaciones de calidad A/B por especie**.

---

## Fundamento matemático del algoritmo

El sistema implementa un clasificador de **banco de filtros de energía** en dos fases:

### Fase A — Entrenamiento (se ejecuta una sola vez)

```
Para cada especie C:
  Para cada grabación de entrenamiento x_n:
    1. Preprocesar: segmento activo 3s + Butterworth 300-8000 Hz + normalización
    2. STFT:   X(k, t) = STFT{x_n}  →  N_FFT=1024, salto=256
    3. Magnitud promediada en el tiempo:  ||C(k)||_m = (1/T) Σ_t |X(k,t)|

  4. Espectro promedio de la especie:  ||C(k)||_m ← promedio sobre las 60 grabaciones
  5. Vector de energía de referencia (20 subbandas):
         E_Ci = (1/N_i) · Σ_{k ∈ subbanda_i} ||C(k)||_m²    (i = 1...20)
         E_C  = E_C / Σ E_Ci                                   (normalizado, suma = 1)
```

El vector resultante **E_C** se escribe como constante numérica en `umbrales_energia.py`.

### Fase B — Reconocimiento (tiempo real)

```
Dado un audio nuevo X:
  1. Preprocesar X  (igual que entrenamiento)
  2. STFT + magnitud + vector de energía  →  E_X  (20 números)
  3. Para cada especie C:
         E_XC = Σ_{i=1}^{20} |E_Xi − E_Ci|        (diferencia total absoluta)
  4. especie* = argmin { E_XC1, ..., E_XC5 }
  5. Si min(E_XC) > τ  →  "No clasificable"
     Si no              →  nombre de especie*
```

### Subbandas de frecuencia (20 cajones)

| # | Rango (Hz) | | # | Rango (Hz) | | # | Rango (Hz) | | # | Rango (Hz) |
|---|---|-|---|---|-|---|---|-|---|---|
| S1 | 300–500 | | S6 | 1300–1500 | | S11 | 3000–3500 | | S16 | 5500–6000 |
| S2 | 500–700 | | S7 | 1500–1800 | | S12 | 3500–4000 | | S17 | 6000–6500 |
| S3 | 700–900 | | S8 | 1800–2100 | | S13 | 4000–4500 | | S18 | 6500–7000 |
| S4 | 900–1100 | | S9 | 2100–2500 | | S14 | 4500–5000 | | S19 | 7000–7500 |
| S5 | 1100–1300 | | S10 | 2500–3000 | | S15 | 5000–5500 | | S20 | 7500–8000 |

Cada especie tiene un "perfil" de energía característico: el Currucutú concentra su energía en S1–S3 (graves), el Canario coronado en S15–S19 (agudos). Esta diferencia espectral es lo que permite la clasificación.

---

## Estructura del proyecto

```
bird_recognition/
│
├── 📄 README.md                      ← Este archivo
├── 📄 GUIA_PROFESOR.md               ← Guía técnica del algoritmo paso a paso
├── 📄 requirements.txt               ← Dependencias Python
├── 📄 umbrales_energia.py            ← Vectores E_C constantes (generado por entrenar.py)
├── 📄 caracteristicas.py             ← STFT, magnitudes, vector de energía por subbanda
├── 📄 preprocesamiento.py            ← Filtro Butterworth, segmento activo, normalización
├── 📄 reconocer.py                   ← Diferencia total absoluta + clasificación + CLI
│
├── 📁 interfaz/
│   └── interfaz.py                   ← App Streamlit (interfaz gráfica web)
│
├── 📁 entrenamiento/
│   └── entrenar.py                   ← Calcula E_C a partir de datos/ y actualiza umbrales_energia.py
│
├── 📁 adquisicion/
│   └── descargar_datos.py            ← Descarga grabaciones desde xeno-canto API v3
│
├── 📁 pruebas/
│   ├── prueba_precision.py           ← Evalúa precisión sobre audios_demo/
│   ├── prueba_robustez.py            ← Prueba con ruido añadido (SNR variable)
│   └── generar_audios_demo.py        ← Genera audios de demostración desde datos/
│
├── 📁 tiempo_real/
│   └── reconocimiento_tiempo_real.py ← Grabación desde micrófono + clasificación
│
├── 📁 modelos/                       ← Artefactos del entrenamiento (generados)
│   ├── vectores_energia.npy          ← Vectores E_C serializados
│   ├── vectores_sigma.npy            ← Desviaciones estándar SIGMA_C
│   ├── limites_subbandas.npy         ← Límites de las 20 subbandas
│   └── espectros_promedio.npy/.png   ← Espectros ||C(k)||_m y su gráfica
│
├── 📁 audios_demo/                   ← 15 audios de muestra (3 por especie)
│   ├── 1_Bichofue.wav
│   ├── 1_Canario_coronado.wav
│   └── ...  (formato: N_NombreEspecie.wav)
│
├── 📁 datos/                         ← Banco de grabaciones MP3 (descargado)
│   ├── Sicalis_flaveola/
│   ├── Mimus_gilvus/
│   ├── Tyrannus_melancholicus/
│   ├── Pitangus_sulphuratus/
│   └── Megascops_choliba/
│
└── 📁 datos_procesados/              ← Versiones WAV preprocesadas del banco
    └── (misma estructura que datos/)
```

---

## Instalación

### Prerrequisitos

- Python **3.11** o superior
- `pip` actualizado (`python -m pip install --upgrade pip`)
- Micrófono (opcional, solo para modo tiempo real)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/bird_recognition.git
cd bird_recognition

# 2. Crear entorno virtual (recomendado)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **Nota para Windows:** si `sounddevice` falla al instalar, ejecuta primero:
> `pip install pipwin && pipwin install pyaudio`

### Verificar instalación

```bash
python reconocer.py audios_demo/1_Canario_coronado.wav
```

Deberías ver algo como:

```
Resultado : Canario coronado
Especie   : Sicalis flaveola
E_XC min  : 0.33412   (tau = 1.000)
Diferencia total absoluta por especie  E_XC = suma|E_Xi-E_Ci| :
   Canario coronado   0.33412
   Bichofue           0.55891
   Siriri comun       0.71034
   Sinsonte tropical  0.89203
   Currucutu          1.01847
```

---

## Uso

### 1. Interfaz gráfica (Streamlit) — recomendado

```bash
streamlit run interfaz/interfaz.py
```

Se abre automáticamente en `http://localhost:8501`. La interfaz tiene tres pestañas:

| Pestaña | Descripción |
|---|---|
| 🎙 **Reconocimiento** | Sube un archivo `.wav` o `.mp3` y obtén la clasificación con gráficas |
| 🎤 **Tiempo real** | Graba directamente desde el micrófono con cuenta regresiva |
| 📊 **Espectros de entrenamiento** | Visualiza los espectros promedio y el mapa de calor de vectores E_C |

**Controles disponibles en la barra lateral:**
- **Umbral τ** (slider 0.4–1.6): regula la sensibilidad. Valor recomendado: `1.0`. Cantos reales dan E_XC < 0.7; ruido da ~1.0.
- **Ficha de cada especie**: rango de frecuencia, hábitat y curiosidad biológica.

---

### 2. Reconocimiento por línea de comandos

```bash
# Reconocer un archivo de audio
python reconocer.py <ruta_audio>

# Con umbral personalizado
python reconocer.py audios_demo/1_Currucutu.wav 0.8

# Ejemplos con audios de demostración
python reconocer.py audios_demo/2_Sinsonte_tropical.wav
python reconocer.py audios_demo/3_Bichofue.wav
```

**Salida de ejemplo:**

```
Resultado : Currucutu
Especie   : Megascops choliba
E_XC min  : 0.28901   (tau = 1.000)
Diferencia total absoluta por especie  E_XC = suma|E_Xi-E_Ci| :
   Currucutu          0.28901
   Sinsonte tropical  0.76432
   Bichofue           0.91034
   Canario coronado   1.02145
   Siriri comun       1.15678
```

---

### 3. Re-entrenar el modelo

Solo es necesario si cambias el banco de grabaciones en `datos/`.

```bash
python entrenamiento/entrenar.py
```

El script:
1. Lee todas las grabaciones MP3 de `datos/<especie>/` (máximo 60 por especie)
2. Calcula los espectros STFT y los vectores de energía E_C y SIGMA_C
3. Sobreescribe `umbrales_energia.py` con las nuevas constantes
4. Guarda los artefactos `.npy` en `modelos/`
5. Genera la gráfica de espectros promedio `modelos/espectros_promedio.png`

---

### 4. Descargar nuevos datos

Requiere una clave gratuita de [xeno-canto.org](https://xeno-canto.org/account).

```bash
# Opción A: variable de entorno (recomendado)
set XC_API_KEY=tu_clave_aqui          # Windows
export XC_API_KEY=tu_clave_aqui       # macOS/Linux

python adquisicion/descargar_datos.py

# Opción B: editar directamente la variable XC_CLAVE_API en descargar_datos.py
```

El script descarga hasta **60 grabaciones de calidad A/B por especie** desde Colombia y las guarda en `datos/<especie>/`.

---

## Descripción de cada módulo

### `preprocesamiento.py`

Prepara cualquier señal de audio antes del análisis:

| Función | Descripción |
|---|---|
| `segmento_activo(y, fs)` | Desliza una ventana de 3 s buscando el fragmento de mayor energía filtrada (donde canta el ave) |
| `filtro_pasabanda(bajo, alto, fs)` | Diseña el filtro IIR Butterworth de orden 4 |
| `aplicar_pasabanda(senal, bajo, alto, fs)` | Aplica el filtro con `sosfilt`, dejando solo 300–8000 Hz |
| `preparar_senal(y, especie)` | Pipeline completo: segmento activo → filtro → normalización pico = 1 |
| `cargar_y_preparar(ruta, especie)` | Lee un archivo de disco y llama a `preparar_senal` |

### `caracteristicas.py`

Extrae la huella espectral de la señal:

| Función | Descripción |
|---|---|
| `espectro_de_senal(y)` | STFT con N_FFT=1024 y salto=256 → magnitud promediada en el tiempo |
| `vector_energia(espectro)` | Calcula E_i = (1/N_i)·Σ\|\|X_i(k)\|\|² en cada subbanda y normaliza |
| `energia_de_senal(y)` | Función todo-en-uno: preprocesa + STFT + vector E_X |

### `umbrales_energia.py`

Archivo **generado automáticamente** por `entrenar.py`. Contiene:
- `E_C` — diccionario con los 5 vectores de 20 energías de referencia
- `SIGMA_C` — desviación estándar por subbanda (variabilidad de las grabaciones)
- `NOMBRES_ESPECIES` — mapeo clave → (nombre común, nombre científico)

### `reconocer.py`

Motor de clasificación. No lee archivos del banco en tiempo de ejecución:

| Función | Descripción |
|---|---|
| `diferencia_absoluta_total(E_X, E_Ci)` | `Σ\|E_Xi − E_Ci\|` para una especie |
| `reconocer_arreglo(y, tau_rechazo)` | Recibe un array numpy; devuelve diccionario con especie, E_XC, ranking |
| `reconocer_archivo(ruta, tau_rechazo)` | Carga el audio con librosa y llama a `reconocer_arreglo` |

**Diccionario de retorno:**
```python
{
  "nombre_comun":  "Canario coronado",
  "cientifico":    "Sicalis flaveola",
  "clave_especie": "Sicalis_flaveola",
  "minimo":        0.33412,           # E_XC de la especie ganadora
  "diferencias":   {                  # E_XC de las 5 especies
      "Sicalis_flaveola": 0.334,
      "Mimus_gilvus":     0.892, ...
  },
  "E_X":           np.array([...]),   # vector de 20 energías del audio
  "rechazado":     False,
}
```

### `interfaz/interfaz.py`

App Streamlit con diseño oscuro estilo dark-mode científico. Genera cuatro tipos de gráficas:
- **Forma de onda** del segmento analizado
- **Espectrograma STFT** en escala dB (colormap inferno)
- **Comparación de vectores** E_X vs E_C de la especie ganadora (barras dobles)
- **Ranking de diferencias** E_XC de las 5 especies (barras horizontales con τ marcado)

---

## Flujo completo paso a paso

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━ FASE A: ENTRENAMIENTO ━━━━━━━━━━━━━━━━━━━━━━━━━━━

 datos/Sicalis_flaveola/*.mp3  ──┐
 datos/Mimus_gilvus/*.mp3      ──┤
 datos/Tyrannus_*.mp3          ──┼─► entrenar.py
 datos/Pitangus_*.mp3          ──┤
 datos/Megascops_*.mp3         ──┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │  Por cada grabación:                     │
                    │  1. cargar_y_preparar() → y (3 s)        │
                    │  2. Butterworth 300-8000 Hz              │
                    │  3. STFT (N_FFT=1024, salto=256)         │
                    │  4. ||X(k)|| = np.abs(Zxx).mean(axis=1) │
                    │  5. E_i = (1/N_i)·Σ||X_i(k)||²          │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │  Por especie:                            │
                    │  6. ||C(k)||_m = promedio de espectros  │
                    │  7. E_C = vector_energia(||C(k)||_m)    │
                    │  8. SIGMA_C = std de los 60 vectores    │
                    └────────────────────┬────────────────────┘
                                         │
                              umbrales_energia.py
                              modelos/*.npy


━━━━━━━━━━━━━━━━━━━━━━━━━━ FASE B: RECONOCIMIENTO ━━━━━━━━━━━━━━━━━━━━━━━━━━

 audio_nuevo.wav / micrófono
          │
          ▼  preprocesamiento.py
 1. segmento_activo() → 3 s de mayor energía
          │
          ▼  preprocesamiento.py
 2. Butterworth 300-8000 Hz + normalización pico=1
          │
          ▼  caracteristicas.py
 3. STFT → ||X(k)|| promedio temporal
          │
          ▼  caracteristicas.py
 4. E_X = vector_energia()  →  [E_X1, ..., E_X20]
          │
          ▼  reconocer.py  ←── lee umbrales_energia.py (5 constantes)
 5. E_XC_i = Σ|E_Xj − E_Cij|   para i = 1..5
          │
          ▼  reconocer.py
 6. especie* = argmin(E_XC)
          │
          ▼  reconocer.py
 7. ¿E_XC_min > τ = 1.0?
       SÍ  →  "No clasificable"
       NO  →  nombre de la especie reconocida
```

---

## Parámetros del sistema

| Parámetro | Valor | Descripción |
|---|---|---|
| `FS` | 22 050 Hz | Frecuencia de muestreo |
| `DURACION` | 3.0 s | Longitud del segmento analizado |
| `N_FFT` | 1 024 | Tamaño de ventana STFT |
| `SALTO` | 256 | Hop length de la STFT |
| `PASABANDA` | 300–8 000 Hz | Rango del filtro Butterworth |
| `N_SUBBANDAS` | 20 | Número de cajones de frecuencia |
| `POR_ESPECIE` | 60 | Grabaciones de entrenamiento por especie |
| `TAU_RECHAZO` | 1.0 | Umbral de rechazo τ (ajustable en la UI) |

---

## Requerimientos

```
librosa>=0.10.2        # Carga y análisis de audio
numpy>=1.26.4          # Álgebra vectorial
scipy>=1.15.0          # STFT, filtro Butterworth (sosfilt)
matplotlib>=3.9.0      # Gráficas de espectros
streamlit>=1.38.0      # Interfaz web
requests>=2.32.3       # Descarga desde xeno-canto API
soundfile>=0.12.1      # Lectura/escritura de archivos WAV
tqdm>=4.66.5           # Barra de progreso en descargas
Pillow>=10.4.0         # Procesamiento de imágenes para Streamlit
sounddevice>=0.4.6     # Grabación desde micrófono
```

Instalar todo con:
```bash
pip install -r requirements.txt
```

---

## Preguntas frecuentes

**¿Por qué el resultado es "No clasificable"?**
La diferencia mínima E_XC supera el umbral τ. Intenta bajar τ en la barra lateral de la interfaz o verifica que el audio sea de una de las 5 especies soportadas y tenga buena calidad.

**¿Por qué usa vectores de energía constantes y no el banco de grabaciones en tiempo real?**
La decisión de diseño es deliberada: los vectores E_C se calculan una sola vez durante el entrenamiento y se fijan como constantes. El reconocimiento es instantáneo (sin I/O de disco) y completamente determinista.

**¿Puedo agregar más especies?**
Sí. Agrega una carpeta con grabaciones en `datos/<NombreCientifico>/`, actualiza el diccionario `ESPECIES` en `entrenamiento/entrenar.py` y vuelve a entrenar con `python entrenamiento/entrenar.py`.

**¿El sistema funciona con cantos mezclados o ruido de fondo?**
El filtro Butterworth atenúa significativamente el ruido fuera de 300–8000 Hz. Para ruido intenso dentro de ese rango (voces humanas, tráfico) el sistema puede fallar o devolver "No clasificable", que es el comportamiento esperado con el umbral τ.

**¿Cómo funciona el modo tiempo real?**
La app graba N segundos desde el micrófono usando `sounddevice`, aplica el mismo pipeline de preprocesamiento y clasifica. Incluye una cuenta regresiva de preparación de 3 segundos.

---

<div align="center">

Proyecto académico — Ingeniería de Sistemas · Meta, Colombia  
Datos de entrenamiento: [xeno-canto.org](https://xeno-canto.org) · 300 grabaciones · 5 especies

</div>
