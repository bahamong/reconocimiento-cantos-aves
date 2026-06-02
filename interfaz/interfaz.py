"""
Interfaz Streamlit - BioAcustica: Reconocimiento de Cantos de Aves
Tematica: bioacustica cientifica, modo oscuro, Llanos Orientales de Colombia
"""

import os
import sys
import io
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft as stft_scipy
import soundfile as sf
import librosa

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
DIR_MODELOS = os.path.join(RAIZ, "modelos")

from umbrales_energia import LIMITES_SUBBANDAS_HZ   # subbandas (constantes del programa)

FS = 22050
N_MUESTRAS = int(FS * 3.0)

INFO_ESPECIES = {
    "Sicalis_flaveola": {
        "comun":      "Canario coronado",
        "cientifico": "Sicalis flaveola",
        "rango":      "2000-8000 Hz",
        "region":     "Sabanas abiertas, Llanos Orientales",
        "curiosidad": "Su canto es una serie de trinos rapidos y melodiosos. El macho canta desde perchas elevadas para defender su territorio.",
        "color":      "#FFD700",
        "emoji":      "🐦",
    },
    "Mimus_gilvus": {
        "comun":      "Sinsonte tropical",
        "cientifico": "Mimus gilvus",
        "rango":      "1500-8000 Hz",
        "region":     "Bordes de bosque, areas urbanas, Llanos",
        "curiosidad": "Imitador extraordinario: puede incorporar cantos de docenas de otras especies en su repertorio vocal.",
        "color":      "#00D4FF",
        "emoji":      "🎵",
    },
    "Tyrannus_melancholicus": {
        "comun":      "Siriri comun",
        "cientifico": "Tyrannus melancholicus",
        "rango":      "1000-6000 Hz",
        "region":     "Areas abiertas con arboles dispersos, Llanos",
        "curiosidad": "Su nombre onomatopeyico 'siriri' imita su llamado. Es uno de los tiranidos mas comunes de Suramerica.",
        "color":      "#7FFF00",
        "emoji":      "🦜",
    },
    "Pitangus_sulphuratus": {
        "comun":      "Bichofue / Cristofue",
        "cientifico": "Pitangus sulphuratus",
        "rango":      "800-5000 Hz",
        "region":     "Bordes de cuerpos de agua, jardines, Llanos",
        "curiosidad": "Su grito Bichofue! es uno de los mas reconocibles de los Llanos. Agresivo defensor de su territorio.",
        "color":      "#FF6B35",
        "emoji":      "🦅",
    },
    "Megascops_choliba": {
        "comun":      "Currucutu",
        "cientifico": "Megascops choliba",
        "rango":      "300-1500 Hz",
        "region":     "Bosques de galeria, sabanas arboladas, Llanos",
        "curiosidad": "Buho nocturno de voz grave y pulsante. Su canto 'cu-cu-cu-cu' resuena en las noches llaneras.",
        "color":      "#FF1493",
        "emoji":      "🦉",
    },
}

ETIQUETAS_SUBBANDAS = [f"{LIMITES_SUBBANDAS_HZ[i]}-{LIMITES_SUBBANDAS_HZ[i+1]} Hz"
                       for i in range(len(LIMITES_SUBBANDAS_HZ) - 1)]

PALETA = {
    "fondo":     "#0D1117",
    "panel":     "#161B22",
    "acento":    "#00D4FF",
    "acento2":   "#7FFF00",
    "texto":     "#E6EDF3",
    "subtexto":  "#8B949E",
    "borde":     "#30363D",
    "exito":     "#3FB950",
    "aviso":     "#F0883E",
    "peligro":   "#FF4444",
}

# ─── CSS personalizado ────────────────────────────────────────────────────────

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: {PALETA['fondo']} !important;
    color: {PALETA['texto']} !important;
}}

.stApp {{ background-color: {PALETA['fondo']}; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1117 0%, #0A0F1A 100%);
    border-right: 1px solid {PALETA['borde']};
}}

.hero-header {{
    background: linear-gradient(135deg, #0D1117 0%, #111827 50%, #0D1117 100%);
    border: 1px solid {PALETA['borde']};
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}}
.hero-header::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(0,212,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}}
.hero-title {{
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, {PALETA['acento']}, {PALETA['acento2']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}}
.hero-sub {{
    font-size: 0.95rem;
    color: {PALETA['subtexto']};
    font-weight: 300;
    letter-spacing: 0.5px;
}}

.bio-card {{
    background: {PALETA['panel']};
    border: 1px solid {PALETA['borde']};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
}}
.result-card-success {{
    background: linear-gradient(135deg, #0D1117, #0F1E0F);
    border: 1px solid {PALETA['exito']};
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 0 20px rgba(63,185,80,0.15);
}}
.result-card-reject {{
    background: linear-gradient(135deg, #0D1117, #1A0D0D);
    border: 1px solid {PALETA['peligro']};
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 0 20px rgba(255,68,68,0.15);
}}
.species-name {{
    font-size: 1.7rem;
    font-weight: 700;
    color: {PALETA['exito']};
    margin: 0 0 4px 0;
}}
.scientific-name {{
    font-size: 1rem;
    color: {PALETA['subtexto']};
    font-style: italic;
    font-family: 'JetBrains Mono', monospace;
}}
.distance-badge {{
    display: inline-block;
    background: rgba(0,212,255,0.12);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    color: {PALETA['acento']};
    font-family: 'JetBrains Mono', monospace;
    margin-top: 8px;
}}

.metric-row {{
    display: flex;
    gap: 12px;
    margin: 12px 0;
    flex-wrap: wrap;
}}
.metric-pill {{
    background: rgba(255,255,255,0.04);
    border: 1px solid {PALETA['borde']};
    border-radius: 8px;
    padding: 8px 14px;
    text-align: center;
    flex: 1;
    min-width: 100px;
}}
.metric-pill .label {{
    font-size: 0.72rem;
    color: {PALETA['subtexto']};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}}
.metric-pill .value {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {PALETA['acento']};
    font-family: 'JetBrains Mono', monospace;
}}

.sp-card {{
    background: rgba(255,255,255,0.03);
    border-left: 3px solid;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.85rem;
}}
.sp-name {{ font-weight: 600; margin-bottom: 2px; }}
.sp-detail {{ color: {PALETA['subtexto']}; font-size: 0.78rem; line-height: 1.5; }}

.stTabs [data-baseweb="tab-list"] {{
    background: {PALETA['panel']};
    border-radius: 8px;
    gap: 4px;
    padding: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px;
    color: {PALETA['subtexto']};
}}
.stTabs [aria-selected="true"] {{
    background: rgba(0,212,255,0.15) !important;
    color: {PALETA['acento']} !important;
}}

.section-title {{
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: {PALETA['subtexto']};
    border-bottom: 1px solid {PALETA['borde']};
    padding-bottom: 6px;
    margin-bottom: 14px;
}}

[data-testid="stFileUploader"] {{
    border: 1px dashed {PALETA['borde']};
    border-radius: 10px;
    padding: 10px;
    background: rgba(255,255,255,0.02);
}}

.stButton > button {{
    background: linear-gradient(90deg, {PALETA['acento']}, #0099BB);
    color: #000;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.3px;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,212,255,0.35);
}}

.stSlider [data-testid="stThumbValue"] {{ color: {PALETA['acento']}; }}

.plot-container {{
    background: {PALETA['panel']};
    border: 1px solid {PALETA['borde']};
    border-radius: 10px;
    padding: 4px;
    margin: 8px 0;
}}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {PALETA['fondo']}; }}
::-webkit-scrollbar-thumb {{ background: {PALETA['borde']}; border-radius: 3px; }}

.footer {{
    text-align: center;
    font-size: 0.75rem;
    color: {PALETA['subtexto']};
    padding: 20px 0 8px;
    border-top: 1px solid {PALETA['borde']};
    margin-top: 32px;
}}
</style>
"""

# ─── Utilidades de graficado ──────────────────────────────────────────────────

def fig_a_imagen(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


def grafica_forma_onda(y: np.ndarray, fs: int = FS) -> io.BytesIO:
    t = np.linspace(0, len(y) / fs, len(y))
    fig, ax = plt.subplots(figsize=(10, 2.2), facecolor=PALETA["panel"])
    ax.set_facecolor(PALETA["panel"])
    ax.fill_between(t, y, alpha=0.35, color=PALETA["acento"])
    ax.plot(t, y, color=PALETA["acento"], linewidth=0.6, alpha=0.9)
    ax.axhline(0, color=PALETA["borde"], linewidth=0.5)
    ax.set_xlabel("Tiempo (s)", color=PALETA["subtexto"], fontsize=9)
    ax.set_ylabel("Amplitud", color=PALETA["subtexto"], fontsize=9)
    ax.set_title("Forma de onda", color=PALETA["texto"], fontsize=10, pad=8)
    ax.tick_params(colors=PALETA["subtexto"], labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(PALETA["borde"])
    plt.tight_layout(pad=0.5)
    return fig_a_imagen(fig)


def grafica_espectrograma(y: np.ndarray, fs: int = FS) -> io.BytesIO:
    nperseg = 512
    noverlap = 384
    f, t, Zxx = stft_scipy(y, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = 20 * np.log10(np.abs(Zxx) + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=PALETA["panel"])
    ax.set_facecolor(PALETA["panel"])
    im = ax.pcolormesh(t, f / 1000, Sxx, shading="gouraud", cmap="inferno",
                       vmin=np.percentile(Sxx, 5), vmax=np.percentile(Sxx, 99))
    ax.set_ylim(0, 11)
    ax.set_xlabel("Tiempo (s)", color=PALETA["subtexto"], fontsize=9)
    ax.set_ylabel("Frecuencia (kHz)", color=PALETA["subtexto"], fontsize=9)
    ax.set_title("Espectrograma STFT", color=PALETA["texto"], fontsize=10, pad=8)
    ax.tick_params(colors=PALETA["subtexto"], labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(PALETA["borde"])
    cbar = plt.colorbar(im, ax=ax, pad=0.01, shrink=0.9)
    cbar.ax.tick_params(colors=PALETA["subtexto"], labelsize=7)
    cbar.set_label("dB", color=PALETA["subtexto"], fontsize=8)
    plt.tight_layout(pad=0.5)
    return fig_a_imagen(fig)


def grafica_comparacion_energia(E_X: np.ndarray, E_C_mejor: np.ndarray,
                                nombre_especie: str, color: str) -> io.BytesIO:
    S = min(len(E_X), len(E_C_mejor), len(ETIQUETAS_SUBBANDAS))
    x = np.arange(S)
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=PALETA["panel"])
    ax.set_facecolor(PALETA["panel"])
    ax.bar(x - ancho/2, E_X[:S], ancho, label="Audio subido", color=PALETA["acento"],
           alpha=0.85, edgecolor=PALETA["fondo"], linewidth=0.5)
    ax.bar(x + ancho/2, E_C_mejor[:S], ancho, label=f"Referencia: {nombre_especie}",
           color=color, alpha=0.85, edgecolor=PALETA["fondo"], linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ETIQUETAS_SUBBANDAS[:S], rotation=30, ha="right",
                       fontsize=8, color=PALETA["subtexto"])
    ax.set_ylabel("Energia por subbanda  E_Ci", color=PALETA["subtexto"], fontsize=9)
    ax.set_title("Vector de energias: audio vs. referencia de especie", color=PALETA["texto"], fontsize=10)
    ax.legend(facecolor=PALETA["panel"], labelcolor=PALETA["texto"], fontsize=9,
              edgecolor=PALETA["borde"])
    ax.tick_params(colors=PALETA["subtexto"], labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(PALETA["borde"])
    plt.tight_layout(pad=0.5)
    return fig_a_imagen(fig)


def grafica_distancias(diferencias: dict, mejor: str, tau_rechazo: float) -> io.BytesIO:
    items = sorted(diferencias.items(), key=lambda x: x[1])
    etiquetas = [INFO_ESPECIES[k]["comun"] for k, _ in items]
    valores = [v for _, v in items]
    colores = [INFO_ESPECIES[k]["color"] for k, _ in items]
    alfa = [1.0 if k == mejor else 0.5 for k, _ in items]

    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=PALETA["panel"])
    ax.set_facecolor(PALETA["panel"])
    barras = []
    for i, (lbl, val, col, alp) in enumerate(zip(etiquetas, valores, colores, alfa)):
        b = ax.barh(lbl, val, color=col, alpha=alp,
                    edgecolor=PALETA["fondo"], linewidth=0.5, height=0.55)
        barras.append(b[0])
    ax.axvline(tau_rechazo, color=PALETA["peligro"], linestyle="--",
               linewidth=1.2, label=f"tau rechazo = {tau_rechazo:.2f}")
    val_max = max(valores) if valores else 1
    for barra, val in zip(barras, valores):
        ax.text(val + val_max * 0.01, barra.get_y() + barra.get_height()/2,
                f"{val:.3f}", va="center", color=PALETA["texto"], fontsize=8,
                fontfamily="monospace")
    ax.set_xlabel("E_XC = suma|E_Xi - E_Ci|", color=PALETA["subtexto"], fontsize=9)
    ax.set_title("Ranking de diferencias por especie", color=PALETA["texto"], fontsize=10)
    ax.legend(facecolor=PALETA["panel"], labelcolor=PALETA["texto"], fontsize=9,
              edgecolor=PALETA["borde"])
    ax.tick_params(colors=PALETA["subtexto"], labelsize=9)
    for s in ax.spines.values():
        s.set_edgecolor(PALETA["borde"])
    plt.tight_layout(pad=0.5)
    return fig_a_imagen(fig)


def grafica_espectros_promedio(espectros_promedio: dict, limites_subbandas_hz: list) -> io.BytesIO:
    colores = [m["color"] for m in INFO_ESPECIES.values()]
    claves = list(INFO_ESPECIES.keys())

    N_FFT = 1024
    eje_frec = np.fft.rfftfreq(N_FFT, d=1.0 / FS)
    mascara = eje_frec <= 8500

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETA["panel"])
    ax.set_facecolor(PALETA["panel"])

    for i, clave in enumerate(claves):
        if clave in espectros_promedio:
            esp = espectros_promedio[clave]
            ax.plot(eje_frec[mascara], esp[mascara], color=colores[i],
                    label=INFO_ESPECIES[clave]["comun"], linewidth=1.2, alpha=0.88)

    for hz in limites_subbandas_hz[1:-1]:
        ax.axvline(x=hz, color="#334455", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.text(hz + 40, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1,
                f"{hz} Hz", color="#556677", fontsize=7.5, rotation=90, va="top")

    ax.set_xlabel("Frecuencia (Hz)", color=PALETA["subtexto"], fontsize=10)
    ax.set_ylabel("||C(k)||_m  - Magnitud promedio", color=PALETA["subtexto"], fontsize=10)
    ax.set_title("Espectros de magnitud promedio por especie\n(lineas = puntos de corte de subbandas)",
                 color=PALETA["texto"], fontsize=11, pad=10)
    ax.legend(facecolor=PALETA["panel"], labelcolor=PALETA["texto"], fontsize=9,
              edgecolor=PALETA["borde"])
    ax.tick_params(colors=PALETA["subtexto"])
    for s in ax.spines.values():
        s.set_edgecolor(PALETA["borde"])
    plt.tight_layout(pad=0.6)
    return fig_a_imagen(fig)


def grafica_mapa_calor_energia(E_C: dict) -> io.BytesIO:
    claves = list(INFO_ESPECIES.keys())
    matriz = np.array([E_C[k] for k in claves if k in E_C])
    etiquetas_y = [INFO_ESPECIES[k]["comun"] for k in claves if k in E_C]
    S = min(matriz.shape[1], len(ETIQUETAS_SUBBANDAS))

    fig, ax = plt.subplots(figsize=(12, 4), facecolor=PALETA["panel"])
    ax.set_facecolor(PALETA["panel"])
    im = ax.imshow(matriz[:, :S], aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_xticks(range(S))
    ax.set_xticklabels(ETIQUETAS_SUBBANDAS[:S], rotation=35, ha="right",
                       fontsize=9, color=PALETA["subtexto"])
    ax.set_yticks(range(len(etiquetas_y)))
    ax.set_yticklabels(etiquetas_y, fontsize=9, color=PALETA["texto"])
    ax.set_title("Mapa de calor - Vectores de energia de referencia E_C por especie",
                 color=PALETA["texto"], fontsize=10, pad=8)
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.tick_params(colors=PALETA["subtexto"], labelsize=8)
    cbar.set_label("Energia", color=PALETA["subtexto"], fontsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(PALETA["borde"])
    plt.tight_layout(pad=0.5)
    return fig_a_imagen(fig)


# ─── Barra lateral ────────────────────────────────────────────────────────────

def construir_barra_lateral():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 10px 0 20px;">
            <div style="font-size:2.2rem;">🦜</div>
            <div style="font-size:1.1rem; font-weight:700; color:#00D4FF;">BioAcustica</div>
            <div style="font-size:0.75rem; color:#8B949E;">Meta - Colombia - Llanos Orientales</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Umbral de rechazo</div>', unsafe_allow_html=True)
        tau = st.slider(
            "tau (umbral de rechazo)",
            min_value=0.4,
            max_value=1.6,
            value=1.0,
            step=0.05,
            help="Umbral de la diferencia total absoluta E_XC = suma|E_Xi - E_Ci|. "
                 "Los cantos de las 5 especies dan E_XC < ~0.7; el ruido da ~1.0. "
                 "Sube tau para aceptar mas audios; bajalo para ser mas estricto.",
        )
        st.markdown(
            f'<div style="font-family:monospace;font-size:0.8rem;color:#8B949E;">tau = {tau:.2f}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title" style="margin-top:20px;">Especies objetivo</div>', unsafe_allow_html=True)
        for clave, info in INFO_ESPECIES.items():
            st.markdown(f"""
            <div class="sp-card" style="border-color:{info['color']}">
                <div class="sp-name" style="color:{info['color']}">
                    {info['emoji']} {info['comun']}
                </div>
                <div class="sp-detail">
                    <i>{info['cientifico']}</i><br>
                    🎵 {info['rango']}<br>
                    📍 {info['region']}<br>
                    💡 {info['curiosidad']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="footer">Algoritmo: Banco de filtros<br>vectores de energia por subbanda<br>Procesamiento Digital de Senales</div>', unsafe_allow_html=True)

    return tau


# ─── Pagina principal ─────────────────────────────────────────────────────────

def pagina_reconocimiento(tau_rechazo: float):
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🎙 Reconocimiento de Cantos de Aves</div>
        <div class="hero-sub">
            Banco de filtros - Vectores de energia por subbanda - 5 especies - Llanos Orientales, Colombia
        </div>
    </div>
    """, unsafe_allow_html=True)

    archivo = st.file_uploader(
        "Sube un archivo de audio (WAV o MP3)",
        type=["wav", "mp3"],
        help="El audio se normalizara a 22050 Hz, mono, 3 segundos.",
    )

    if archivo is None:
        st.markdown("""
        <div class="bio-card" style="text-align:center; padding:40px; color:#8B949E;">
            <div style="font-size:3rem;">🎵</div>
            <div style="margin-top:10px; font-size:1rem;">
                Sube un audio de canto de ave para comenzar el analisis
            </div>
            <div style="font-size:0.82rem; margin-top:6px;">
                Formatos soportados: WAV - MP3
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Leer audio
    bytes_audio = archivo.read()
    with io.BytesIO(bytes_audio) as buf:
        try:
            y_crudo, fs_crudo = sf.read(buf)
        except Exception:
            buf.seek(0)
            y_crudo, fs_crudo = librosa.load(buf, sr=None, mono=True)

    if y_crudo.ndim > 1:
        y_crudo = y_crudo[:, 0]
    if fs_crudo != FS:
        y_crudo = librosa.resample(y_crudo.astype(np.float32), orig_sr=fs_crudo, target_sr=FS)

    # Seleccionar segmento activo (mas energia de canto)
    from preprocesamiento import segmento_activo
    y_vista = segmento_activo(y_crudo.astype(np.float32), FS).astype(np.float32)
    # Normalizar para visualizacion (el reconocedor aplica los filtros internamente)
    max_a = np.max(np.abs(y_vista))
    if max_a > 0:
        y_vista = y_vista / max_a

    # Reproducir audio
    st.audio(bytes_audio, format=f"audio/{archivo.name.split('.')[-1]}")

    # Visualizaciones
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Forma de onda</div>', unsafe_allow_html=True)
        st.image(grafica_forma_onda(y_vista), use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">Espectrograma STFT</div>', unsafe_allow_html=True)
        st.image(grafica_espectrograma(y_vista), use_container_width=True)

    st.markdown("---")

    # Boton de analisis
    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        analizar = st.button("🔍  Analizar canto", use_container_width=True)

    if not analizar:
        return

    # Verificar modelo entrenado
    ruta_ev = os.path.join(DIR_MODELOS, "vectores_energia.npy")
    if not os.path.exists(ruta_ev):
        st.error("⚠️ Modelos no encontrados. Ejecuta `python entrenamiento/entrenar.py` primero.")
        return

    with st.spinner("Analizando espectro de frecuencias..."):
        try:
            from reconocer import reconocer_arreglo
            # Se pasa la senal COMPLETA: el reconocedor aplica internamente el
            # preprocesamiento (segmento activo + Butterworth 300-8000 Hz) y calcula E_X.
            resultado = reconocer_arreglo(y_crudo.astype(np.float32).copy(),
                                          tau_rechazo=tau_rechazo)
        except Exception as e:
            st.error(f"Error en el clasificador: {e}")
            return

    # ── Resultado ──
    if resultado["rechazado"]:
        st.markdown(f"""
        <div class="result-card-reject">
            <div style="font-size:1.8rem; color:#FF4444; font-weight:700;">
                ❌ No clasificable
            </div>
            <div style="color:#8B949E; margin-top:6px; font-size:0.9rem;">
                La diferencia minima <span style="font-family:monospace; color:#FF4444;">
                E_XC = {resultado['minimo']:.4f}</span> supera el umbral tau = {tau_rechazo:.2f}
            </div>
            <div style="color:#8B949E; font-size:0.82rem; margin-top:4px;">
                Especie mas cercana: {INFO_ESPECIES.get(resultado['mejor_especie'], {}).get('comun', resultado['mejor_especie'])}
                - prueba bajar el umbral tau en la barra lateral.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        info = INFO_ESPECIES[resultado["clave_especie"]]
        st.markdown(f"""
        <div class="result-card-success">
            <div style="display:flex; align-items:center; gap:16px;">
                <div style="font-size:3rem;">{info['emoji']}</div>
                <div>
                    <div class="species-name" style="color:{info['color']}">
                        {info['comun']}
                    </div>
                    <div class="scientific-name">{info['cientifico']}</div>
                    <div class="distance-badge">E_XC min = {resultado['minimo']:.5f}</div>
                </div>
            </div>
            <div style="margin-top:14px; padding-top:12px; border-top:1px solid #1E3A1E;
                        font-size:0.87rem; color:#8B949E; line-height:1.6;">
                🎵 Rango: {info['rango']} &nbsp;|&nbsp;
                📍 {info['region']}<br>
                💡 {info['curiosidad']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:20px;">Comparacion de vectores de energia</div>', unsafe_allow_html=True)
        E_C_mejor = resultado.get("E_C_mejor", np.zeros_like(resultado["E_X"]))
        st.image(grafica_comparacion_energia(resultado["E_X"], E_C_mejor,
                                             info["comun"], info["color"]),
                 use_container_width=True)

    # Ranking de diferencias (siempre)
    st.markdown('<div class="section-title" style="margin-top:20px;">Ranking de diferencias a todas las especies</div>', unsafe_allow_html=True)
    st.image(grafica_distancias(resultado["diferencias"], resultado["mejor_especie"],
                                resultado["tau_rechazo"]),
             use_container_width=True)

    # Tabla numerica
    with st.expander("Ver tabla de diferencias y vector E_X"):
        col_d, col_e = st.columns(2)
        with col_d:
            st.markdown("**E_XC por especie**")
            datos_dif = {
                "Especie": [INFO_ESPECIES[k]["comun"] for k in resultado["diferencias"]],
                "E_XC = suma|E_Xi - E_Ci|": [f"{v:.6f}" for v in resultado["diferencias"].values()],
            }
            st.dataframe(datos_dif, hide_index=True, use_container_width=True)
        with col_e:
            st.markdown("**E_X - Vector de energia del audio**")
            datos_ex = {
                "Subbanda": ETIQUETAS_SUBBANDAS[:len(resultado["E_X"])],
                "E_Xi": [f"{v:.6f}" for v in resultado["E_X"]],
            }
            st.dataframe(datos_ex, hide_index=True, use_container_width=True)


# ─── Pagina espectros de entrenamiento ───────────────────────────────────────

def pagina_espectros_entrenamiento():
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📊 Espectros de Entrenamiento</div>
        <div class="hero-sub">
            Espectros promedio ||C(k)||_m - Vectores E_C - Puntos de corte de subbandas
        </div>
    </div>
    """, unsafe_allow_html=True)

    ruta_ev = os.path.join(DIR_MODELOS, "vectores_energia.npy")
    ruta_ms = os.path.join(DIR_MODELOS, "espectros_promedio.npy")
    ruta_sb = os.path.join(DIR_MODELOS, "limites_subbandas.npy")

    if not os.path.exists(ruta_ev):
        st.warning("⚠️ No se encontraron modelos entrenados. Ejecuta `python entrenamiento/entrenar.py` primero.")
        return

    E_C = np.load(ruta_ev, allow_pickle=True).item()
    limites_subbandas_hz = np.load(ruta_sb).tolist()

    # Espectros promedio superpuestos
    st.markdown('<div class="section-title">Espectros promedio superpuestos - ||C(k)||_m</div>', unsafe_allow_html=True)
    if os.path.exists(ruta_ms):
        espectros_promedio = np.load(ruta_ms, allow_pickle=True).item()
        st.image(grafica_espectros_promedio(espectros_promedio, limites_subbandas_hz),
                 use_container_width=True)
    else:
        st.info("Espectros promedio no disponibles (ejecuta entrenar.py con datos).")

    # Info de subbandas
    st.markdown('<div class="section-title" style="margin-top:20px;">Definicion de subbandas</div>', unsafe_allow_html=True)
    cols = st.columns(len(ETIQUETAS_SUBBANDAS))
    for i, (col, etiqueta) in enumerate(zip(cols, ETIQUETAS_SUBBANDAS)):
        with col:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="label">S{i+1}</div>
                <div class="value" style="font-size:0.75rem;">{etiqueta}</div>
            </div>
            """, unsafe_allow_html=True)

    # Mapa de calor de vectores de energia
    st.markdown('<div class="section-title" style="margin-top:20px;">Mapa de calor de vectores E_C por especie</div>', unsafe_allow_html=True)
    st.image(grafica_mapa_calor_energia(E_C), use_container_width=True)

    # Tabla de vectores E_C
    st.markdown('<div class="section-title" style="margin-top:20px;">Vectores de energia de referencia E_C</div>', unsafe_allow_html=True)
    filas = []
    for clave in INFO_ESPECIES:
        if clave in E_C:
            fila = {"Especie": INFO_ESPECIES[clave]["comun"]}
            for i, etiqueta in enumerate(ETIQUETAS_SUBBANDAS[:len(E_C[clave])]):
                fila[etiqueta] = f"{E_C[clave][i]:.5f}"
            filas.append(fila)
    if filas:
        import pandas as pd
        df = pd.DataFrame(filas)
        st.dataframe(df, hide_index=True, use_container_width=True)


# ─── Pagina grabacion en tiempo real ─────────────────────────────────────────

def pagina_tiempo_real(tau_rechazo: float):
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🎤 Reconocimiento en Tiempo Real</div>
        <div class="hero-sub">
            Graba el canto directamente desde tu microfono - Analisis automatico al terminar
        </div>
    </div>
    """, unsafe_allow_html=True)

    ruta_ev = os.path.join(DIR_MODELOS, "vectores_energia.npy")
    if not os.path.exists(ruta_ev):
        st.error("⚠️ Modelos no encontrados. Ejecuta `python entrenamiento/entrenar.py` primero.")
        return

    try:
        import sounddevice as sd
    except ImportError:
        st.error("⚠️ Instala sounddevice: `pip install sounddevice`")
        return

    # Listar dispositivos de entrada disponibles
    try:
        todos_dispositivos = sd.query_devices()
        dispositivos_entrada = [(i, d["name"]) for i, d in enumerate(todos_dispositivos)
                                if d["max_input_channels"] > 0]
    except Exception:
        dispositivos_entrada = []

    # Configuracion de grabacion
    col1, col2 = st.columns([2, 1])
    with col1:
        if dispositivos_entrada:
            idx_defecto = 0
            try:
                dev_defecto = sd.default.device[0]
                for k, (i, _) in enumerate(dispositivos_entrada):
                    if i == dev_defecto:
                        idx_defecto = k
                        break
            except Exception:
                pass
            eleccion_dev = st.selectbox(
                "🎙 Microfono",
                options=dispositivos_entrada,
                index=idx_defecto,
                format_func=lambda x: f"[{x[0]}] {x[1]}",
                help="Selecciona el microfono que captara el audio.",
            )
            dispositivo_sel = eleccion_dev[0]
        else:
            dispositivo_sel = None
            st.info("No se detectaron dispositivos de entrada.")
    with col2:
        duracion = st.selectbox("Duracion (s)", [5, 8, 10, 12],
                                index=1, help="Segundos a grabar")

    grabar = st.button("🔴  Iniciar grabacion", use_container_width=True)

    st.markdown(f"""
    <div class="bio-card" style="padding:12px 18px; margin-top:4px;">
        <span style="color:#8B949E; font-size:0.82rem;">
        💡 <b>Como usar:</b> presiona Iniciar grabacion -> durante la cuenta Preparate (3,2,1)
        inicia el audio del ave en tu telefono y pegalo al microfono -> manten el sonido durante
        toda la grabacion. El analisis es automatico al terminar.
        </span>
    </div>
    """, unsafe_allow_html=True)

    if "audio_grabado" not in st.session_state:
        st.session_state.audio_grabado = None
        st.session_state.resultado_grabado = None

    if grabar:
        st.session_state.audio_grabado = None
        st.session_state.resultado_grabado = None

        import time as _time
        import librosa as _librosa

        # Obtener FS nativo del dispositivo seleccionado para evitar distorsion
        try:
            if dispositivo_sel is not None:
                info_dev = sd.query_devices(dispositivo_sel)
            else:
                info_dev = sd.query_devices(kind='input')
            fs_nativo = int(info_dev['default_samplerate'])
        except Exception:
            fs_nativo = 44100

        marcador = st.empty()

        # ── Cuenta regresiva de PREPARACION (antes de grabar) ──
        for prep in range(3, 0, -1):
            marcador.markdown(f"""
            <div class="bio-card" style="text-align:center; padding:30px;">
                <div style="font-size:3.5rem;">📱</div>
                <div style="font-size:1.6rem; color:#F0883E; font-weight:700; margin-top:8px;">
                    Preparate... {prep}
                </div>
                <div style="color:#8B949E; font-size:0.9rem; margin-top:6px;">
                    Inicia el audio del ave en tu telefono AHORA y acercalo al microfono
                </div>
            </div>
            """, unsafe_allow_html=True)
            _time.sleep(1)

        # ── Iniciar grabacion NO bloqueante al FS nativo del dispositivo ──
        n_muestras = int(duracion * fs_nativo)
        kwargs_grab = dict(samplerate=fs_nativo, channels=1, dtype='float32')
        if dispositivo_sel is not None:
            kwargs_grab["device"] = dispositivo_sel
        grabacion = sd.rec(n_muestras, **kwargs_grab)

        for restante in range(duracion, 0, -1):
            marcador.markdown(f"""
            <div class="bio-card" style="text-align:center; padding:30px;">
                <div style="font-size:3.5rem;">🎤</div>
                <div style="font-size:1.6rem; color:#FF4444; font-weight:700; margin-top:8px;">
                    ● Grabando... {restante}s
                </div>
                <div style="color:#8B949E; font-size:0.85rem; margin-top:6px;">
                    Manten el audio sonando cerca del microfono
                </div>
            </div>
            """, unsafe_allow_html=True)
            _time.sleep(1)

        sd.wait()  # esperar a que termine la grabacion
        marcador.empty()

        y_crudo = grabacion.flatten().astype(np.float64)

        # ── LIMPIEZA ROBUSTA de la grabacion cruda ──
        y_crudo = np.nan_to_num(y_crudo, nan=0.0, posinf=0.0, neginf=0.0)
        y_crudo = y_crudo - np.mean(y_crudo)              # quitar offset DC
        p999 = np.percentile(np.abs(y_crudo), 99.9)       # recortar picos extremos
        if p999 > 0:
            y_crudo = np.clip(y_crudo, -p999, p999)

        # Medir nivel CRUDO antes de normalizar (para detectar mic en silencio)
        pico_crudo = float(np.max(np.abs(y_crudo)))
        rms_crudo = float(np.sqrt(np.mean(y_crudo ** 2)))
        st.session_state.pico_crudo_grabado = pico_crudo
        st.session_state.rms_crudo_grabado = rms_crudo

        # Resamplear al FS del modelo (22050 Hz)
        if fs_nativo != FS:
            y_crudo = _librosa.resample(y_crudo.astype(np.float32),
                                        orig_sr=fs_nativo, target_sr=FS)
        y_crudo = y_crudo.astype(np.float32)

        # Normalizar amplitud
        max_a = np.max(np.abs(y_crudo))
        if max_a > 0:
            y_crudo = y_crudo / max_a

        st.session_state.audio_grabado = y_crudo.copy()

        # Mismo flujo que el reconocimiento de archivos: se pasa la senal COMPLETA
        # grabada y el reconocedor calcula E_X y la diferencia absoluta.
        from reconocer import reconocer_arreglo
        st.session_state.resultado_grabado = reconocer_arreglo(y_crudo, tau_rechazo=tau_rechazo)
        st.rerun()

    # Mostrar resultado de la ultima grabacion
    if st.session_state.audio_grabado is not None and st.session_state.resultado_grabado is not None:
        y = st.session_state.audio_grabado
        resultado = st.session_state.resultado_grabado

        # Aviso de calidad de senal capturada (nivel CRUDO antes de normalizar)
        pico_crudo = st.session_state.get("pico_crudo_grabado", 1.0)
        if pico_crudo < 0.01:
            st.error(
                f"🔇 El microfono no capto senal (nivel pico = {pico_crudo:.5f}). "
                "Verifica que: 1) el microfono correcto este seleccionado en Windows, "
                "2) el volumen del telefono este alto, 3) el telefono este pegado al microfono. "
                "**El resultado de abajo NO es confiable** - es ruido amplificado."
            )
        elif pico_crudo < 0.05:
            st.warning(
                f"⚠️ Senal debil (pico = {pico_crudo:.4f}). Acerca mas el telefono al microfono "
                "y sube el volumen para un resultado confiable."
            )

        st.markdown('<div class="section-title">Audio grabado</div>', unsafe_allow_html=True)
        col_w, col_s = st.columns(2)
        with col_w:
            st.image(grafica_forma_onda(y[:N_MUESTRAS]), use_container_width=True)
        with col_s:
            st.image(grafica_espectrograma(y[:N_MUESTRAS]), use_container_width=True)

        # Reproduccion
        import soundfile as sf_mod
        buf = io.BytesIO()
        sf_mod.write(buf, y[:N_MUESTRAS], FS, format="WAV", subtype="FLOAT")
        buf.seek(0)
        st.audio(buf.read(), format="audio/wav")

        st.markdown("---")

        # Resultado
        if resultado["rechazado"]:
            st.markdown(f"""
            <div class="result-card-reject">
                <div style="font-size:1.8rem; color:#FF4444; font-weight:700;">❌ No clasificable</div>
                <div style="color:#8B949E; margin-top:6px; font-size:0.9rem;">
                    E_XC min = <span style="font-family:monospace; color:#FF4444;">{resultado['minimo']:.4f}</span>
                    supera tau = {tau_rechazo:.2f} - Especie mas cercana:
                    {INFO_ESPECIES.get(resultado['mejor_especie'], {}).get('comun', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            info = INFO_ESPECIES[resultado["clave_especie"]]
            st.markdown(f"""
            <div class="result-card-success">
                <div style="display:flex; align-items:center; gap:16px;">
                    <div style="font-size:3rem;">{info['emoji']}</div>
                    <div>
                        <div class="species-name" style="color:{info['color']}">
                            {info['comun']}
                        </div>
                        <div class="scientific-name">{info['cientifico']}</div>
                        <div class="distance-badge">E_XC min = {resultado['minimo']:.5f}</div>
                    </div>
                </div>
                <div style="margin-top:14px; padding-top:12px; border-top:1px solid #1E3A1E;
                            font-size:0.87rem; color:#8B949E; line-height:1.6;">
                    🎵 {info['rango']} &nbsp;|&nbsp; 📍 {info['region']}<br>
                    💡 {info['curiosidad']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            E_C_mejor = resultado.get("E_C_mejor", np.zeros_like(resultado["E_X"]))
            st.markdown('<div class="section-title" style="margin-top:16px;">Comparacion de energias</div>', unsafe_allow_html=True)
            st.image(grafica_comparacion_energia(resultado["E_X"], E_C_mejor,
                                                 info["comun"], info["color"]),
                     use_container_width=True)

        st.image(grafica_distancias(resultado["diferencias"], resultado["mejor_especie"],
                                    resultado["tau_rechazo"]), use_container_width=True)

    elif not grabar:
        st.markdown("""
        <div class="bio-card" style="text-align:center; padding:50px; color:#8B949E;">
            <div style="font-size:3.5rem;">🎤</div>
            <div style="font-size:1rem; margin-top:12px;">
                Presiona <b style="color:#00D4FF;">Iniciar grabacion</b> para capturar el canto
            </div>
            <div style="font-size:0.82rem; margin-top:6px;">
                Funciona con el microfono del sistema o cualquier dispositivo de audio
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Principal ────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="BioAcustica - Cantos de Aves",
        page_icon="🦜",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    tau_rechazo = construir_barra_lateral()

    tab_rec, tab_rt, tab_ent = st.tabs([
        "🎙 Reconocimiento",
        "🎤 Tiempo real",
        "📊 Espectros de entrenamiento",
    ])

    with tab_rec:
        pagina_reconocimiento(tau_rechazo)

    with tab_rt:
        pagina_tiempo_real(tau_rechazo)

    with tab_ent:
        pagina_espectros_entrenamiento()


if __name__ == "__main__":
    main()
