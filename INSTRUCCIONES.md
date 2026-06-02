# BioAcustica — Reconocimiento de Cantos de Aves

Reconocimiento de cantos de 5 aves del Meta (Llanos Orientales) con
**vectores de umbrales de energia constantes** y
**diferencia total absoluta** `E_XC = Σ|E_Xi − E_Ci|`.

## Instalar dependencias
```bash
pip install -r requirements.txt
```

## Reconocer un audio (linea de comandos)
```bash
python reconocer.py audios_demo/1_Currucutu.wav
```

## Reconocimiento EN TIEMPO REAL por microfono  (la demo)
```bash
python tiempo_real/reconocimiento_tiempo_real.py
python tiempo_real/reconocimiento_tiempo_real.py --listar
python tiempo_real/reconocimiento_tiempo_real.py --device 1 --duracion 8 --tau 1.0
```
1. Copia los WAV de `audios_demo/` a una memoria **USB/SD**.
2. Reproducelos en un **parlante externo**.
3. Ejecuta el comando y acerca el parlante al **microfono del PC**.
4. El programa imprime el ave reconocida y la diferencia `E_XC` de cada especie.

## Interfaz grafica (Streamlit)
```bash
streamlit run interfaz/interfaz.py
```
Pestanas: **Reconocimiento** (subir audio), **Tiempo real** (graba del microfono)
y **Espectros de entrenamiento** (espectros promedio + subbandas + vectores E_C).

## Re-entrenar (regenerar los vectores constantes)
```bash
python entrenamiento/entrenar.py
```
Lee `datos/`, calcula `E_C` y `sigma_C` y los **escribe como constantes numericas**
en `umbrales_energia.py`. Tambien genera `espectros_promedio.png`.

## Comprobaciones
```bash
python pruebas/prueba_precision.py       # matriz de confusion sobre todo el banco
python pruebas/generar_audios_demo.py    # regenera y verifica los audios de 10 s
python pruebas/prueba_robustez.py        # simula altavoz -> microfono (ruido + coloracion)
```

---

## Estructura del proyecto
```
bird_recognition/
├── umbrales_energia.py        <- VECTORES DE ENERGIA (constantes numericas)
├── reconocer.py               <- Reconocimiento: E_X + diferencia total absoluta
├── caracteristicas.py         <- FFT/STFT, espectro y vector de energia
├── preprocesamiento.py        <- Segmento activo + Butterworth + normalizacion
│
├── entrenamiento/
│   └── entrenar.py            <- Genera umbrales_energia.py con los vectores E_C
│
├── tiempo_real/
│   └── reconocimiento_tiempo_real.py   <- Reconocimiento en vivo por microfono
│
├── interfaz/
│   └── interfaz.py            <- Interfaz Streamlit
│
├── adquisicion/
│   └── descargar_datos.py     <- Descarga del banco (NO va en el informe)
│
├── pruebas/
│   ├── prueba_precision.py    <- Exactitud sobre el banco completo
│   ├── prueba_robustez.py     <- Simulacion altavoz -> microfono
│   └── generar_audios_demo.py <- Crea los audios de demostracion
│
├── audios_demo/               <- 15 audios de 10 s verificados (3 por especie)
├── datos/                     <- Grabaciones crudas (.mp3)
├── modelos/                   <- vectores_energia.npy + espectros_promedio.npy + .png
└── requirements.txt
```

## Algoritmo

**Entrenamiento:**
`FFT → ‖X(k)‖ → promedio ‖C(k)‖_m → E_Ci=(1/N_i)Σ‖C_i(k)‖² → E_C (constante)`

**Reconocimiento:**
`E_X → E_XC=Σ|E_Xi−E_Ci| → tipo_X=argmin → (E_XC>τ ⇒ "No clasificable")`

- Vectores `E_C` **constantes** en `umbrales_energia.py`.
- El reconocimiento **no** procesa el banco, solo compara contra esas constantes.
