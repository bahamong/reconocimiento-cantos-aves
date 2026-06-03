

## FASE A — ENTRENAMIENTO
> Se ejecuta una sola vez con `python entrenamiento/entrenar.py`.  
> Produce los vectores E_C constantes que quedan escritos en `umbrales_energia.py`.

### Paso 1 · Leer audio de entrenamiento
 [entrenamiento/entrenar.py:55](entrenamiento/entrenar.py)  
 abre cada grabación `.mp3` del banco, la convierte a mono y la lleva a 22050 Hz.

```python
y = cargar_y_preparar(ruta, especie=nombre_especie)
```

La función `cargar_y_preparar` está definida en:  
→ [preprocesamiento.py:97](preprocesamiento.py) — aquí se llama `librosa.load()` que abre el archivo físicamente.

---

### Paso 2 · Quitar ruido — Filtro Butterworth pasabanda 300–8000 Hz
[preprocesamiento.py:35](preprocesamiento.py)  
**Qué hace:** diseña matemáticamente el filtro que deja pasar solo las frecuencias de 300 a 8000 Hz (donde cantan las aves) y elimina el resto (motores, viento, corriente eléctrica).

```python
def filtro_pasabanda(corte_bajo, corte_alto, fs, orden=4):
    return butter(orden, [bajo, alto], btype="band", output="sos")
```

**Dónde se aplica a la señal:**  
→ [preprocesamiento.py:44](preprocesamiento.py) — `sosfilt(sos, senal)` pasa muestra por muestra por el filtro.  
→ [preprocesamiento.py:83](preprocesamiento.py) — se llama dentro de `preparar_senal()`, paso 2 explícito.

---

Paso 3 · STFT — Espectro de tiempo corto de Fourier
[caracteristicas.py:50](caracteristicas.py)  
**Qué hace:** divide el audio de 3 s en ~250 ventanitas de 46 ms y le aplica la FFT a cada una. Convierte la señal de "tiempo" a "frecuencias".

```python
_, _, Zxx = stft_scipy(y, fs=FS, nperseg=N_FFT, noverlap=N_FFT - SALTO)
```

---

### Paso 4 · Magnitudes ||X(k)||
**Archivo:** [caracteristicas.py:51](caracteristicas.py)  
**Qué hace:** saca el valor absoluto de cada componente de la STFT (descarta la fase, queda solo la energía) y promedia las ~250 ventanitas en un solo vector.

```python
return np.mean(np.abs(Zxx), axis=1)
```

---

### Paso 5 · Energías por subbanda  E_i = (1/N_i) · Σ||X_i(k)||²
**Archivo:** [caracteristicas.py:55](caracteristicas.py)  
**Qué hace:** recorre los 20 cajones de frecuencia y calcula cuánta energía hay en cada uno. Normaliza dividiendo entre la suma total para que el volumen no afecte el resultado.

```python
E[i] = (1.0 / n) * np.sum(segmento ** 2)   # energía del cajón i
...
return E / suma   # normalizado → suma = 1.0
```

Los 20 límites de los cajones están definidos en:  
→ [caracteristicas.py:28](caracteristicas.py) — `LIMITES_SUBBANDAS_HZ = [300, 500, 700, ...]`

---

### Paso 6 · Vectores de umbrales de energía E_C (constantes numéricas escritas en el código)
**Archivo:** [umbrales_energia.py:24](umbrales_energia.py)  
**Qué hace:** almacena los 5 vectores de referencia con los 20 números de cada especie. Estos no se calculan en tiempo de reconocimiento: ya están escritos como constantes.

```python
E_C = {
    "Sicalis_flaveola":       np.array([0.13353, 0.06733, ...]),   # Canario coronado
    "Mimus_gilvus":           np.array([0.07649, 0.05177, ...]),   # Sinsonte tropical
    "Tyrannus_melancholicus": np.array([0.14834, 0.07699, ...]),   # Sirirí común
    "Pitangus_sulphuratus":   np.array([0.08326, 0.05092, ...]),   # Bichofué
    "Megascops_choliba":      np.array([0.05397, 0.22867, ...]),   # Currucutú
}
```

**Desviación estándar por subbanda (σ_C):**  
→ [umbrales_energia.py:38](umbrales_energia.py) — `SIGMA_C = { ... }` — mide qué tan variables son las grabaciones de cada especie en cada cajón.

---

---

## FASE B — RECONOCIMIENTO EN TIEMPO REAL
> Se ejecuta con `python tiempo_real/reconocimiento_tiempo_real.py` o subiendo un audio a la app.  
> **No procesa el banco.** Solo usa los vectores constantes de `umbrales_energia.py`.

---

### Paso 1 · Leer audio nuevo (micrófono o archivo)
**Desde archivo:**  
→ [reconocer.py:67](reconocer.py) — `reconocer_archivo()` llama a `librosa.load()`.

**Desde micrófono:**  
→ [tiempo_real/reconocimiento_tiempo_real.py:51](tiempo_real/reconocimiento_tiempo_real.py) — `sd.rec()` graba del micrófono.

---

### Paso 2 · Quitar ruido + preparar señal
**Archivo:** [preprocesamiento.py:70](preprocesamiento.py)  
**Qué hace:** igual que en entrenamiento — segmento activo de 3 s, filtro Butterworth 300-8000 Hz y normalización. Se aplica automáticamente dentro de `energia_de_senal()`.

→ [caracteristicas.py:73](caracteristicas.py) — `y = preparar_senal(...)` es la llamada.

---

### Pasos 3, 4 y 5 · STFT → magnitudes → vector E_X
**Archivo:** [caracteristicas.py:70](caracteristicas.py)  
**Qué hace:** igual que en entrenamiento, pero ahora para el audio nuevo. Produce E_X: 20 números que describen el canto recién grabado.

```python
def energia_de_senal(y, ya_preparada=False):
    y = preparar_senal(...)         # pasos 1b y 2
    return vector_energia(espectro_de_senal(y))  # pasos 3, 4 y 5
```

---

### Paso 7 · Diferencia total absoluta  E_XC = Σ|E_Xi − E_Ci|
**Archivo:** [reconocer.py:32](reconocer.py)  
**Qué hace:** compara los 20 números del audio nuevo (E_X) con los 20 números de cada especie de referencia (E_C). Suma las diferencias. Resultado: un número por especie que indica qué tan diferente es el audio de esa especie.

```python
def diferencia_absoluta_total(E_X, E_Ci):
    return float(np.sum(np.abs(E_X[:n] - E_Ci[:n])))
```

**Dónde se calcula para las 5 especies:**  
→ [reconocer.py:49](reconocer.py) — `diferencias = {sp: diferencia_absoluta_total(E_X, E_Ci) for sp, E_Ci in E_C.items()}`  
> Nota: `E_C.items()` son los **5 vectores constantes del código**, no archivos de audio del banco.

---

### Paso 8 · argmin → tipo_X (especie ganadora)
**Archivo:** [reconocer.py:51](reconocer.py)  
**Qué hace:** de los 5 valores E_XC, toma el más pequeño. Esa es la especie más parecida al audio.

```python
mejor = min(diferencias, key=diferencias.get)
```

Ejemplo de resultado:

| Especie | E_XC |
|---|---|
| **Canario coronado** ← ganadora | **0.33** |
| Bichofué | 0.49 |
| Currucutú | 0.91 |
| Sirirí común | 1.05 |
| Sinsonte tropical | 1.12 |

---

### Paso 9 · Umbral de rechazo τ
**Archivo:** [reconocer.py:53](reconocer.py)  
**Qué hace:** si incluso la especie ganadora tiene E_XC > 1.0 (umbral τ), el sonido no se parece a ninguna de las 5 aves y se responde "No clasificable". Así el sistema no inventa una especie cuando le hablas o le metes ruido.

```python
rechazado = minimo > tau_rechazo   # tau_rechazo = 1.0 por defecto
```

→ [reconocer.py:29](reconocer.py) — `TAU_RECHAZO = 1.0` valor por defecto definido aquí.

---

## Resumen visual del flujo

```
AUDIO NUEVO
    │
    ▼ [preprocesamiento.py:97]
Paso 1 · Leer audio  (librosa.load o sd.rec)
    │
    ▼ [preprocesamiento.py:35]
Paso 2 · Filtro Butterworth 300-8000 Hz  (quitar ruido)
    │
    ▼ [caracteristicas.py:50]
Paso 3 · STFT  (dividir en ventanitas y aplicar FFT)
    │
    ▼ [caracteristicas.py:51]
Paso 4 · Magnitudes ||X(k)||  (np.abs → promedio en el tiempo)
    │
    ▼ [caracteristicas.py:60]
Paso 5 · Vector E_X  (energia por cada uno de los 20 cajones)
    │
    ▼ [reconocer.py:49]
Paso 7 · E_XC = Σ|E_Xi − E_Ci|  ← compara con los 5 vectores CONSTANTES
    │                               del archivo umbrales_energia.py [línea 24]
    │                               NO se abre ningún archivo del banco
    ▼ [reconocer.py:51]
Paso 8 · tipo_X = argmin{E_XC}   (especie con menor diferencia)
    │
    ▼ [reconocer.py:53]
Paso 9 · ¿E_XC_min > τ?  →  "No clasificable"
         ¿E_XC_min ≤ τ?  →  nombre de la especie reconocida
```
