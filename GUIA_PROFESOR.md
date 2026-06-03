

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

---

## EJEMPLO NUMÉRICO — Ejercicio de presustentación

Se tienen dos conjuntos de señales, tipo A y tipo B. Cada conjunto tiene dos vectores de 4 muestras.

**Conjunto A** — señal suave, cambia poco entre muestra y muestra:
```
A₁ = [2, 3, 3, 2]
A₂ = [1, 2, 2, 1]
```

**Conjunto B** — señal que oscila entre valores altos y bajos:
```
B₁ = [3, 1, 3, 1]
B₂ = [2, 0, 2, 0]
```

---

### 1) Espectro de cada vector — DFT

La DFT convierte cada vector de muestras en un vector de frecuencias: dice "cuánta energía tiene esta señal en cada frecuencia".

En el proyecto esto lo hace `espectro_de_senal()` → [caracteristicas.py:50](caracteristicas.py)
La línea clave es la STFT (`stft_scipy`) seguida de `np.abs(Zxx)` para quedarse solo con las magnitudes.

Para N = 4 muestras la fórmula es:

```
X(k) = Σ x(n) · e^(−j·2π·kn/4),   k = 0, 1, 2, 3
```

Desarrollando (usando e^(−jπ/2) = −j, e^(−jπ) = −1, e^(−j3π/2) = j):

**A₁ = [2, 3, 3, 2]:**
```
X(0) = 2+3+3+2 = 10                              → |X(0)| = 10
X(1) = 2 + 3(−j) + 3(−1) + 2(j) = −1 − j        → |X(1)| = √2 ≈ 1.41
X(2) = 2 + 3(−1) + 3(1) + 2(−1) = 0             → |X(2)| = 0
X(3) = 2 + 3(j) + 3(−1) + 2(−j) = −1 + j        → |X(3)| = 1.41
```

**A₂ = [1, 2, 2, 1]:**
```
X(0) = 1+2+2+1 = 6                               → |X(0)| = 6
X(1) = 1 + 2(−j) + 2(−1) + 1(j) = −1 − j        → |X(1)| = 1.41
X(2) = 1 + 2(−1) + 2(1) + 1(−1) = 0             → |X(2)| = 0
X(3) = 1 + 2(j) + 2(−1) + 1(−j) = −1 + j        → |X(3)| = 1.41
```

**B₁ = [3, 1, 3, 1]:**
```
X(0) = 3+1+3+1 = 8                               → |X(0)| = 8
X(1) = 3 + 1(−j) + 3(−1) + 1(j) = 0             → |X(1)| = 0
X(2) = 3 + 1(−1) + 3(1) + 1(−1) = 4             → |X(2)| = 4
X(3) = 3 + 1(j) + 3(−1) + 1(−j) = 0             → |X(3)| = 0
```

**B₂ = [2, 0, 2, 0]:**
```
X(0) = 2+0+2+0 = 4                               → |X(0)| = 4
X(1) = 2 + 0 + 2(−1) + 0 = 0                    → |X(1)| = 0
X(2) = 2 + 0 + 2(1) + 0 = 4                     → |X(2)| = 4
X(3) = 2 + 0 + 2(−1) + 0 = 0                    → |X(3)| = 0
```

---

### 2) Espectro promedio de cada conjunto

Se promedian las magnitudes de los dos vectores del mismo conjunto. Así se obtiene un solo espectro representativo por conjunto.

En el código esto lo hace `np.mean(np.abs(Zxx), axis=1)` → [caracteristicas.py:59](caracteristicas.py)

**Espectro promedio del Conjunto A:**

| k | A₁ | A₂ | Promedio |
|---|----|----|----------|
| 0 | 10 | 6 | **(10+6)/2 = 8** |
| 1 | 1.41 | 1.41 | **(1.41+1.41)/2 = 1.41** |
| 2 | 0 | 0 | **0** |
| 3 | 1.41 | 1.41 | **1.41** |

**Espectro promedio del Conjunto B:**

| k | B₁ | B₂ | Promedio |
|---|----|----|----------|
| 0 | 8 | 4 | **(8+4)/2 = 6** |
| 1 | 0 | 0 | **0** |
| 2 | 4 | 4 | **(4+4)/2 = 4** |
| 3 | 0 | 0 | **0** |

---

### 3) Energías por subanda

Las 4 frecuencias se dividen en 2 subandas:
- **Subanda 1** (frecuencias bajas): k = 0 y k = 1
- **Subanda 2** (frecuencias altas): k = 2 y k = 3

La energía de cada subanda se calcula con la fórmula:
```
E_i = (1/N_i) · Σ |X_i(k)|²
```
donde N_i es cuántas frecuencias hay en esa subanda.

En el código esto lo hace `vector_energia()` → [caracteristicas.py:63](caracteristicas.py)
Al final se divide entre la suma total para que no importe si la señal es fuerte o suave.

**Conjunto A:**
```
E_A1 = (1/2) × (8² + 1.41²) = (1/2) × (64 + 2) = 33
E_A2 = (1/2) × (0² + 1.41²) = (1/2) × (0  + 2) = 1

Suma total = 33 + 1 = 34

E_A normalizado = [33/34, 1/34] = [0.97, 0.03]
```

**Conjunto B:**
```
E_B1 = (1/2) × (6² + 0²) = (1/2) × 36 = 18
E_B2 = (1/2) × (4² + 0²) = (1/2) × 16 = 8

Suma total = 18 + 8 = 26

E_B normalizado = [18/26, 8/26] = [0.69, 0.31]
```

**Comparación final:**

| Subanda | E_A | E_B |
|---------|-----|-----|
| Subanda 1 (bajas, k=0,1) | **0.97** | **0.69** |
| Subanda 2 (altas, k=2,3) | **0.03** | **0.31** |

Las energías de subandas equivalentes son distintas entre A y B (0.97 ≠ 0.69 en la subanda baja, y 0.03 ≠ 0.31 en la alta). Esto confirma que el método puede distinguir los dos tipos de señal.

En el proyecto real, los vectores [0.97, 0.03] y [0.69, 0.31] serían los vectores de referencia E_C guardados como constantes numéricas en [umbrales_energia.py:24](umbrales_energia.py). Para reconocer una señal nueva, se calcula su vector E_X y se compara contra cada E_C usando la diferencia absoluta total de [reconocer.py:38](reconocer.py).
