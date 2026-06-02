"""
Descarga automatica de grabaciones desde la API xeno-canto v3.
Filtra calidad A/B, descarga MP3, aplica preprocesamiento y guarda en
datos_procesados/.
"""

import os
import sys
import time
import requests
from tqdm import tqdm

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from preprocesamiento import preparar_directorio

DIR_DATOS = os.path.join(RAIZ, "datos")
DIR_DATOS_PROC = os.path.join(RAIZ, "datos_procesados")

# Clave de la API de xeno-canto (gratis en https://xeno-canto.org/account).
# Se puede definir aqui o como variable de entorno: set XC_API_KEY=tu_clave
XC_CLAVE_API = os.environ.get("XC_API_KEY", "17eda5b840d41a9e16ded85a6ff5c5fba6d83fd9")

# genero y especie de cada ave (formato de la API v3).
ESPECIES = {
    "Sicalis_flaveola":       ('Sicalis',   'flaveola'),
    "Mimus_gilvus":           ('Mimus',     'gilvus'),
    "Tyrannus_melancholicus": ('Tyrannus',  'melancholicus'),
    "Pitangus_sulphuratus":   ('Pitangus',  'sulphuratus'),
    "Megascops_choliba":      ('Megascops', 'choliba'),
}

MAX_POR_ESPECIE = 60
FILTRO_CALIDAD = {"A", "B"}
API_XENO = "https://xeno-canto.org/api/3/recordings"
DESCARGA_XENO = "https://xeno-canto.org/{id}/download"

SESION = requests.Session()
SESION.headers.update({"User-Agent": "ReconocedorCantosAves/1.0 (proyecto academico)"})


def consultar_grabaciones(genero: str, especie: str, max_paginas: int = 5):
    consulta = f'sp:"{genero} {especie}" cnt:Colombia'
    grabaciones = []
    for pagina in range(1, max_paginas + 1):
        try:
            resp = SESION.get(API_XENO, params={"query": consulta, "page": pagina,
                                                "key": XC_CLAVE_API}, timeout=30)
            resp.raise_for_status()
            datos = resp.json()
        except Exception as e:
            print(f"  Error consultando API (pag {pagina}): {e}")
            break
        recs = datos.get("recordings", [])
        grabaciones.extend(recs)
        if pagina >= int(datos.get("numPages", 1)):
            break
        time.sleep(0.5)
    return grabaciones


def descargar_archivo(url: str, ruta_destino: str, reintentos: int = 3) -> bool:
    for intento in range(1, reintentos + 1):
        try:
            with SESION.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(ruta_destino, "wb") as f:
                    for bloque in r.iter_content(chunk_size=8192):
                        f.write(bloque)
            return True
        except Exception as e:
            print(f"    Intento {intento}/{reintentos} fallido: {e}")
            time.sleep(2 * intento)
    return False


def descargar_especie(nombre_carpeta: str, genero_especie: tuple):
    genero, especie = genero_especie
    nombre_cientifico = f"{genero} {especie}"
    dir_crudo = os.path.join(DIR_DATOS, nombre_carpeta)
    dir_proc = os.path.join(DIR_DATOS_PROC, nombre_carpeta)
    os.makedirs(dir_crudo, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Especie: {nombre_cientifico} -> {nombre_carpeta}")
    print(f"{'='*60}")

    grabaciones = consultar_grabaciones(genero, especie)
    grabaciones_ab = [r for r in grabaciones if r.get("q") in FILTRO_CALIDAD]
    grabaciones_ab = grabaciones_ab[:MAX_POR_ESPECIE]

    print(f"  Grabaciones disponibles (A/B): {len(grabaciones_ab)}")

    descargadas = 0
    for rec in tqdm(grabaciones_ab, desc=f"  Descargando {nombre_carpeta}", unit="audio"):
        id_rec = rec.get("id", "")
        nombre = f"{id_rec}.mp3"
        destino = os.path.join(dir_crudo, nombre)
        if os.path.exists(destino):
            descargadas += 1
            continue
        url = DESCARGA_XENO.format(id=id_rec)
        if descargar_archivo(url, destino):
            descargadas += 1
        time.sleep(0.3)

    print(f"  Descargadas: {descargadas}")

    print(f"  Preprocesando...")
    n_proc = preparar_directorio(dir_crudo, dir_proc, nombre_carpeta)
    print(f"  Procesadas: {n_proc}")


def main():
    os.makedirs(DIR_DATOS, exist_ok=True)
    os.makedirs(DIR_DATOS_PROC, exist_ok=True)

    print("=" * 60)
    print("Descarga de banco de grabaciones - Cantos de aves del Meta, Colombia")
    print("=" * 60)

    for nombre_carpeta, genero_especie in ESPECIES.items():
        descargar_especie(nombre_carpeta, genero_especie)

    print("\nBanco de grabaciones descargado y preprocesado.")
    print(f"Audios crudos:      {DIR_DATOS}")
    print(f"Audios procesados:  {DIR_DATOS_PROC}")


if __name__ == "__main__":
    main()
