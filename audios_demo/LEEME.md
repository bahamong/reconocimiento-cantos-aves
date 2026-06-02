# Audios de prueba (10 segundos) para la demo

Estos audios son cantos reales de las 5 especies, recortados a **10 segundos**,
listos para copiar a una **memoria USB/SD**, reproducir en un **parlante externo**
y reconocer en tiempo real con el **microfono del PC**.

Hay **3 audios por especie** (un principal y dos alternativos). **Todos** fueron
verificados: el sistema los reconoce correctamente con el umbral por defecto.

## Como usar en la demo
1. Copia estos `.wav` a la USB/SD.
2. Reproducelos en el parlante.
3. En el PC ejecuta:
   ```
   python tiempo_real/reconocimiento_tiempo_real.py
   ```
   (o usa la pestana **Tiempo real** de `streamlit run interfaz/interfaz.py`)
4. Acerca el parlante al microfono: aparecera el ave reconocida.

## Audios y su verificacion  (E_XC = diferencia total absoluta; menor = mas seguro)

| Archivo | Especie | E_XC | Margen |
|---|---|---|---|
| `1_Canario_coronado.wav` | Canario coronado (*Sicalis flaveola*) | 0.33 | 0.16 |
| `2_Canario_coronado.wav` | Canario coronado | 0.37 | 0.38 |
| `3_Canario_coronado.wav` | Canario coronado | 0.37 | 0.38 |
| `1_Sinsonte_tropical.wav` | Sinsonte tropical (*Mimus gilvus*) | 0.41 | 0.69 |
| `2_Sinsonte_tropical.wav` | Sinsonte tropical | 0.50 | 0.44 |
| `3_Sinsonte_tropical.wav` | Sinsonte tropical | 0.51 | 0.41 |
| `1_Siriri_comun.wav` | Siriri comun (*Tyrannus melancholicus*) | 0.49 | 0.37 |
| `2_Siriri_comun.wav` | Siriri comun | 0.59 | 0.22 |
| `3_Siriri_comun.wav` | Siriri comun | 0.60 | 0.27 |
| `1_Bichofue.wav` | Bichofue (*Pitangus sulphuratus*) | 0.37 | 0.30 |
| `2_Bichofue.wav` | Bichofue | 0.33 | 0.65 |
| `3_Bichofue.wav` | Bichofue | 0.46 | 0.60 |
| `1_Currucutu.wav` | Currucutu (*Megascops choliba*) | 0.45 | 0.94 |
| `2_Currucutu.wav` | Currucutu | 0.42 | 0.89 |
| `3_Currucutu.wav` | Currucutu | 0.45 | 0.95 |

Todos: 10.00 s · 22050 Hz · mono · WAV PCM-16.

> Recomendacion para la demo en vivo: usa el archivo `1_...` de cada especie.
> Sube el volumen del parlante y acercalo al microfono.
