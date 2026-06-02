# Reconocimiento de Cantos de Aves — Llanos Orientales, Colombia

Sistema de reconocimiento automático de cantos de aves en Python, basado en un
banco de filtros con **vectores de umbrales de energía por subbanda** y
clasificación por **diferencia total absoluta**. Identifica cinco especies del
departamento del Meta (Colombia).

## Especies reconocidas

| Especie | Nombre científico | Rango de frecuencia |
|---|---|---|
| Canario coronado | *Sicalis flaveola* | 2000–8000 Hz |
| Sinsonte tropical | *Mimus gilvus* | 1500–8000 Hz |
| Sirirí común | *Tyrannus melancholicus* | 1000–6000 Hz |
| Bichofué | *Pitangus sulphuratus* | 800–5000 Hz |
| Currucutú | *Megascops choliba* | 300–1500 Hz |

## Algoritmo

**Entrenamiento:**
`grabaciones → FFT → ‖X(k)‖ → promedio ‖C(k)‖_m → E_Ci=(1/Nᵢ)Σ‖Cᵢ(k)‖² → E_C (constante)`

**Reconocimiento:**
`señal → Butterworth 300-8000 Hz → STFT → E_X → E_XC=Σ|E_Xi−E_Ci| → argmin → (E_XC>τ → "No clasificable")`

Los vectores `E_C` de cada especie están escritos como **constantes numéricas** en
`umbrales_energia.py`. El reconocimiento **no** procesa el banco de grabaciones;
solo compara contra esas constantes.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Interfaz gráfica (Streamlit)
```bash
streamlit run app.py
```
Incluye tres pestañas: reconocimiento por archivo, reconocimiento en tiempo real
y visualización de espectros de entrenamiento.

### Reconocimiento por línea de comandos
```bash
python reconocer.py audios_demo/1_Currucutu.wav
```

### Reconocimiento en tiempo real (demo con parlante)
```bash
python reconocimiento_tiempo_real.py --listar          # ver micrófonos
python reconocimiento_tiempo_real.py --device 1 --duracion 8
```
1. Copia los audios de `audios_demo/` a una memoria USB/SD.
2. Reprodúcelos en un parlante externo.
3. Ejecuta el script y acerca el parlante al micrófono del PC.

### Re-entrenar (regenerar vectores constantes)
```bash
python entrenar.py
```
Requiere tener las grabaciones en `data/<especie>/`.
Los vectores se escriben automáticamente en `umbrales_energia.py`.

## Resultados

- **Precisión global:** 70 % sobre 344 grabaciones de calidad A/B (xeno-canto)
- **Audios demo:** 15/15 reconocidos correctamente (3 por especie, 10 s c/u)
- **Robustez:** 97.3 % bajo simulación del canal altavoz→micrófono (SNR 10-20 dB)

## Espectros promedio por especie

![Espectros promedio](mean_spectra_plot.png)

Los picos diferenciados permiten clasificar por subbanda:
Currucutú (~700 Hz), Sinsonte (~2 kHz), Bichofué (~3.3 kHz), Canario/Sirirí (~5.5 kHz).

## Estructura del proyecto

```
bird_recognition/
├── umbrales_energia.py          # Vectores de umbrales E_C (CONSTANTES numéricas)
├── reconocer.py                 # Reconocimiento: E_X + diferencia total absoluta
├── caracteristicas.py           # Espectro STFT y vector de energía por subbanda
├── preprocesamiento.py          # Segmento activo + Butterworth + normalización
├── entrenar.py                  # Entrenamiento → escribe umbrales_energia.py
├── app.py                       # Interfaz Streamlit
├── reconocimiento_tiempo_real.py# Reconocimiento en vivo por micrófono
├── prueba_precision.py          # Matriz de confusión sobre el banco completo
├── prueba_robustez.py           # Simulación canal altavoz→micrófono
├── generar_audios_demo.py       # Genera los audios de demostración de 10 s
├── audios_demo/                 # 15 audios WAV de 10 s listos para la demo
├── informe_proyecto.tex         # Informe académico (compilable en Overleaf)
├── mean_spectra_plot.png        # Espectros promedio con líneas de subbanda
└── requirements.txt
```

## Dependencias

```
librosa, numpy, scipy, matplotlib, streamlit, soundfile, sounddevice, tqdm, Pillow
```

## Fuente de datos

Grabaciones de [xeno-canto](https://xeno-canto.org) (API v3, calidad A/B).
Por su tamaño (~772 MB), no se incluyen en el repositorio.

---

Proyecto académico — Procesamiento Digital de Señales  
Universidad de los Llanos · Ingeniería de Sistemas
