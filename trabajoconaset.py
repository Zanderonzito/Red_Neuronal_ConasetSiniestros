"""
Evaluacion 1 - Fundamentos de Data Science
Tema: Seguridad Vial - CONASET
Fuente principal: Excel oficial del observatorio de CONASET
https://www.conaset.cl/programa/observatorio-datos-estadistica/biblioteca-observatorio/estadisticas-generales/
----
Fuente secundaria (validacion): API publica ArcGIS de CONASET
https://mapas-conaset.opendata.arcgis.com
-----
Integrantes: Rigo Vega, Martín Caamaño, Favi Muñoz, Nikolas Maldonado.
"""

import pandas as pd
import numpy as np
import os
import requests
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------
# configuracion general
# -----------------------------------------------------------------------

# nombre del excel que bajamos de CONASET, tiene que estar en la misma carpeta
ARCHIVOS_EXCEL = [
    "Regionesdeocurrencia2000-2024.xlsx"
]

# cache para no tener que procesar el excel cada vez que corremos el codigo
CSV_CACHE = "datos_conaset.csv"

# features para cada modelo predictivo
# simple: solo el año, para ver tendencia en el tiempo
# multiple: mas variables porque claramente influyen en el resultado po :v
FEATURES_SIMPLE = ["anio"]
FEATURES_MULTI  = ["anio", "region_num", "fallecidos", "lesionados_graves"]

COLORES = {
    "azul":    "#2563EB",
    "rojo":    "#DC2626",
    "verde":   "#16A34A",
    "naranja": "#EA580C",
    "gris":    "#6B7280",
}

# -----------------------------------------------------------------------
# validacion de la API (mas que to pa justificar la fuente de datos)
# -----------------------------------------------------------------------

def verificar_api_conaset():
    # confirmamos que la API de CONASET está activa y el catalogo es publico
    # no pudimos extraer tablas directo porque el servidor lo bloquea,
    # pero esto sirve mas que todo para mostrar que la fuente existe y es oficial
    print("\n[ Validando fuente de datos — API publica CONASET / ArcGIS ]\n")

    url = "https://mapas-conaset.opendata.arcgis.com/data.json"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            print("  Conexion exitosa a la API de ArcGIS Open Data (status 200)")
            print("  Catalogo de datos CONASET verificado y disponible publicamente")
            print("  Proveedor: CONASET — Ministerio de Transportes y Telecomunicaciones")
        else:
            print(f"  La API respondio pero con codigo {r.status_code}")
            print("  Igual usamos los Excel oficiales que publica el mismo organismo")
    except Exception:
        # si falla la conexion no es el fin del mundo, igual tenemos los datos kbros, pero igual siempre me funca
        print("  Sin conexion al servidor de CONASET en este momento")
        print("  Continuamos con los Excel descargados directamente del observatorio")


# -----------------------------------------------------------------------
# lectura del excel
# -----------------------------------------------------------------------

def mapear_columnas(df):
    """
    El Excel de CONASET tiene un formato medio raro con celdas combinadas
    y los subtitulos a veces quedan en la fila de abajo en vez del header.
    Esta funcion mas que todo detecta las columnas por palabras clave en vez de confiar
    en el nombre exacto, que puede variar entre versiones del Excel, profe aqui tuvimos hartos problemitas pero con las 
    palabras quedó bien
    """
    nuevas = {}
    for i, col in enumerate(df.columns):
        # juntamos el nombre de la columna con lo que hay en la primera fila
        # para no perder los subtitulos que quedaron desplazados
        encabezado  = str(col).lower().strip()
        primera_fila = str(df.iloc[0, i]).lower().strip() if len(df) > 0 else ""
        texto = encabezado + " " + primera_fila

        # el orden importa: primero los casos mas especificos
        if "regi" in texto:
            nuevas[col] = "region"
        elif "siniestro" in texto:
            nuevas[col] = "siniestros"
        elif "fallecido" in texto or "muerto" in texto:
            nuevas[col] = "fallecidos"
        elif "menos graves" in texto:
            nuevas[col] = "lesionados_menos_graves"
        elif "graves" in texto:
            nuevas[col] = "lesionados_graves"
        elif "leves" in texto:
            nuevas[col] = "lesionados_leves"
        elif "ileso" in texto:
            nuevas[col] = "ilesos"

    df = df.rename(columns=nuevas)
    # si quedaron columnas duplicadas nos quedamos con la primera
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def cargar_datos():
    # si ya corrimos antes, cargamos el csv que guardamos
    # para no tener que procesar el Excel cada vez
    if os.path.exists(CSV_CACHE):
        df = pd.read_csv(CSV_CACHE)
        print(f"\n[ Datos cargados desde cache — {len(df):,} registros ]\n")
        return df

    print("\n[ Leyendo Excel de CONASET por primera vez... ]\n")
    partes = []
    ok = 0

    for archivo in ARCHIVOS_EXCEL:
        if not os.path.exists(archivo):
            print(f"  no encontre '{archivo}', ponlo en la misma carpeta que este .py")
            continue
        try:
            # el Excel tiene una hoja por año, entonces las leemos todas
            # skiprows=3 porque las primeras filas son titulo y encabezados decorativos
            hojas = pd.read_excel(archivo, sheet_name=None, skiprows=3)
            for nombre_hoja, df_hoja in hojas.items():
                df_hoja["anio"] = nombre_hoja
                df_hoja = mapear_columnas(df_hoja)
                partes.append(df_hoja)
            ok += 1
            print(f"  listo: {archivo} ({len(hojas)} hojas/años procesadas)")
        except Exception as e:
            print(f"  error leyendo {archivo}: {e}")

    if ok == 0:
        print("\n  no se cargo ningun Excel, revisa los nombres en ARCHIVOS_EXCEL")
        raise SystemExit(1)

    df = pd.concat(partes, ignore_index=True)
    # limpiar_datos() aqui lo dejo mas que todo pa que continuen con lo que habiamos hecho por discord kbros:V 
    df.to_csv(CSV_CACHE, index=False)
    print(f"\n  datos guardados en '{CSV_CACHE}' para las proximas ejecuciones\n")
    return df

if __name__ == "__main__":
    verificar_api_conaset()
    df = cargar_datos()
    print(df.tail(10).to_string())