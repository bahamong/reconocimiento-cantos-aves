# Mapeo del algoritmo → codigo

Este documento muestra **donde** esta implementada cada formula y como el codigo
cumple los tres requisitos del sistema:

> 1. Vectores de umbrales de energia escritos de forma explicita (constantes numericas).
> 2. Reconocimiento mediante la **diferencia total absoluta** con los vectores constantes.
> 3. El reconocimiento NO procesa el banco de grabaciones.

---

## Archivos del nucleo (van en el informe)

| Archivo | Rol | En el informe |
|---|---|---|
| `umbrales_energia.py` | **Vectores E_C CONSTANTES** (numericos) | Si |
| `reconocer.py` | **Reconocimiento**: E_X + diferencia total absoluta | Si |
| `caracteristicas.py` | Calculo de `‖X(k)‖` y del vector de energia `E` | Si |
| `preprocesamiento.py` | Segmento activo + Butterworth + normalizacion | Si |

## Archivos auxiliares (por carpeta)

| Carpeta / Archivo | Rol | En el informe |
|---|---|---|
| `tiempo_real/reconocimiento_tiempo_real.py` | Captura microfono + reconocimiento en vivo | Si (lineas de reconocimiento) |
| `entrenamiento/entrenar.py` | Genera los vectores constantes | Opcional (fase de entrenamiento) |
| `adquisicion/descargar_datos.py` | Descarga del banco | No |
| `interfaz/interfaz.py` | Interfaz Streamlit | No |
| `pruebas/*.py` | Validacion y pruebas | No |

---

## Notacion → variable del codigo

| Notacion | Variable | Significado |
|---|---|---|
| `X(k)` | `Zxx` (STFT) | Transformada de la senal |
| `‖X(k)‖` | `np.abs(Zxx)` → `espectro_de_senal` | Magnitud del espectro |
| `‖C(k)‖_m` | `espectros_promedio[especie]` | Espectro promedio de la especie |
| `E_Ci` | `E_C[especie][i]` | Energia de la subbanda i (CONSTANTE) |
| `E_C` | `E_C[especie]` | Vector de umbrales de energia (CONSTANTE) |
| `E_Xi` | `E_X[i]` | Energia de la subbanda i de la senal |
| `E_X` | `E_X` | Vector de energia de la senal en tiempo real |
| `E_XC` | `diferencias[especie]` | **Diferencia total absoluta** `Σ|E_Xi−E_Ci|` |
| `tipo_X` | `mejor_especie` | `argmin` de `E_XC` |
| `τ` | `tau_rechazo` | Umbral de rechazo |
| `σ_C` | `SIGMA_C[especie]` | Desviacion estandar por subbanda |

---

## FASE 1 — ENTRENAMIENTO  ·  `entrenamiento/entrenar.py`

1. **`y = FFT(x)` ; `z = |y|` → `‖X(k)‖`** — `caracteristicas.espectro_de_senal()`
   (STFT y magnitud; se promedia en el tiempo).
2. **Espectro promedio `‖C(k)‖_m = promedio{‖C(k)‖}`** — `entrenar.py`:
   `espectros_promedio[especie] = np.mean(espectros, axis=0)`.
3. **Energia de cada subbanda del espectro promedio:**
   `E_Ci = (1/N_i)·Σ‖C_i(k)‖²` — `caracteristicas.vector_energia()`.
4. **Vector de umbrales `E_C = [E_C1,…,E_CS]`** — `entrenar.py`:
   `E_C[especie] = vector_energia(espectros_promedio[especie])`.
5. **Desviacion estandar por subbanda `σ_C`** →
   `SIGMA_C[especie] = np.std(energias, axis=0)`.
6. `entrenar.py` **escribe los vectores como constantes numericas** en
   `umbrales_energia.py`.

---

## FASE 2 — RECONOCIMIENTO  ·  `reconocer.py`

1. **`X → senal` ; `‖X(k)‖` ; `E_Xi = (1/N_i)·Σ‖X_i(k)‖²`** —
   `caracteristicas.energia_de_senal()` produce `E_X`.
2. **DIFERENCIA TOTAL ABSOLUTA**:

   ```python
   # reconocer.py
   def diferencia_absoluta_total(E_X, E_Ci):
       return float(np.sum(np.abs(E_X - E_Ci)))   # E_XC = Σ |E_Xi − E_Ci|
   ```

3. **Clasificacion `tipo_X = argmin{E_XC1,…,E_XC5}`** —
   `mejor = min(diferencias, key=diferencias.get)`.
4. **Umbral de rechazo `τ`** — si `E_XC_min > τ` → `"No clasificable"`.

```python
# reconocer.py — nucleo del reconocimiento
from umbrales_energia import E_C          # VECTORES CONSTANTES
E_X = energia_de_senal(y)                 # vector de energia de la senal
diferencias = {esp: np.sum(np.abs(E_X - E_Ci))   # E_XC = Σ|E_Xi − E_Ci|
               for esp, E_Ci in E_C.items()}
mejor = min(diferencias, key=diferencias.get)     # tipo_X = argmin
```

---

## Flujo completo

```
ENTRENAMIENTO (entrenamiento/entrenar.py)  →  umbrales_energia.py (CONSTANTES)
  audio → Butterworth 300-8000 Hz → STFT → ‖C(k)‖ → promedio ‖C(k)‖_m
        → E_Ci=(1/N_i)Σ‖C_i(k)‖² → E_C  (+ σ_C)

RECONOCIMIENTO (reconocer.py / tiempo_real/reconocimiento_tiempo_real.py)
  senal → Butterworth 300-8000 Hz → STFT → ‖X(k)‖ → E_Xi=(1/N_i)Σ‖X_i(k)‖²
        → E_X → E_XC=Σ|E_Xi−E_Ci| (vs CONSTANTES) → argmin → (E_XC>τ? → rechazo)
```
