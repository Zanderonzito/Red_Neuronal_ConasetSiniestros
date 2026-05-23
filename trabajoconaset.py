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

# -----------------------------------------------------------------------
# limpieza de datos
# -----------------------------------------------------------------------

def limpiar_datos(df):
    print("\n" + "-"*50)
    print("  LIMPIEZA DE DATOS")
    print("-"*50)
    n_inicio = len(df)

    # convertir a numerico las columnas que deberían serlo
    # errors='coerce' convierte lo que no sea numero a NaN en vez de explotar
    cols_numericas = ["anio", "siniestros", "fallecidos",
                      "lesionados_graves", "lesionados_leves"]
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # chequear que tenemos las columnas minimas para trabajar
    if "region" not in df.columns or "siniestros" not in df.columns:
        print("  PROBLEMA: no se encontraron las columnas region o siniestros")
        print("  columnas disponibles:", df.columns.tolist())
        print("  ajusta el diccionario de mapeo en mapear_columnas()")
        raise SystemExit(1)

    # sacar filas donde falten los datos principales
    antes = len(df)
    df = df.dropna(subset=["anio", "region", "siniestros"])
    if antes - len(df) > 0:
        print(f"  filas con nulos eliminadas: {antes - len(df)}")

    # las filas de "Total" o "TOTAL" no son regiones, son sumas parciales
    # si las dejamos van a inflar los modelos
    df = df[~df["region"].astype(str).str.contains("Total|TOTAL", na=False)]

    # crear columna numerica de region para los modelos de ML
    # LabelEncoder le asigna un numero a cada region (0, 1, 2...)
    df["region_num"] = LabelEncoder().fit_transform(df["region"].astype(str))

    # si alguna columna de victimas no existe la creamos con 0
    # para no tener problemas mas adelante
    for col in ["fallecidos", "lesionados_graves", "lesionados_leves", "siniestros"]:
        if col not in df.columns:
            df[col] = 0

    # tasa de mortalidad: de cada 100 siniestros cuantos terminaron con muertos
    # np.where para evitar division por cero
    df["tasa_mortalidad"] = np.where(
        df["siniestros"] > 0,
        (df["fallecidos"] / df["siniestros"] * 100).round(2),
        0
    )

    # outliers en siniestros usando IQR × 3 (bastante permisivo para no perder casos reales)
    Q1 = df["siniestros"].quantile(0.25)
    Q3 = df["siniestros"].quantile(0.75)
    limite = Q3 + 3 * (Q3 - Q1)
    antes = len(df)
    df = df[df["siniestros"] <= limite]
    if antes - len(df) > 0:
        print(f"  outliers removidos en siniestros: {antes - len(df)} filas")

    df = df.reset_index(drop=True)
    print(f"  resultado: {n_inicio:,} filas al inicio -> {len(df):,} filas limpias")
    print("-"*50 + "\n")
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
    df = limpiar_datos(df)  
    df.to_csv(CSV_CACHE, index=False)
    print(f"\n  datos guardados en '{CSV_CACHE}' para las proximas ejecuciones\n")
    return df

# -----------------------------------------------------------------------
# aqui faltaria la estadistica descriptiva
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# graficos exploratorios
# -----------------------------------------------------------------------

def hacer_graficos_exploratorios(df):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Análisis Exploratorio — Siniestros Viales Chile (2000-2024)",
                 fontsize=14, fontweight="bold")
    
    # 1. tendencia anual
    por_anio = df.groupby("anio")["siniestros"].sum().reset_index()
    axes[0,0].plot(por_anio["anio"], por_anio["siniestros"],
                   marker="o", color=COLORES["azul"], linewidth=2.5, markersize=6)
    axes[0,0].fill_between(por_anio["anio"], por_anio["siniestros"],
                            alpha=0.1, color=COLORES["azul"])
    axes[0,0].set_title("Evolucion anual de siniestros")
    axes[0,0].set_xlabel("Año")
    axes[0,0].set_ylabel("Total siniestros")

    # 2. promedio por region (top 8)
    if "region" in df.columns:
        top_reg = (df.groupby("region")["siniestros"]
                     .mean()
                     .sort_values(ascending=True)
                     .tail(8))
        colores_barra = [COLORES["rojo"] if r == top_reg.idxmax()
                         else COLORES["azul"] for r in top_reg.index]
        axes[0,1].barh(top_reg.index, top_reg.values,
                        color=colores_barra, edgecolor="white")
        axes[0,1].set_title("Promedio de siniestros por region (top 8)")
        axes[0,1].set_xlabel("Promedio anual")

    # 3. mapa de calor correlaciones
    cols_corr = [c for c in ["siniestros", "fallecidos",
                               "lesionados_graves", "lesionados_leves",
                               "tasa_mortalidad"] if c in df.columns]
    sns.heatmap(df[cols_corr].corr(), annot=True, fmt=".2f",
                cmap="Blues", ax=axes[1,0], linewidths=0.5, annot_kws={"size": 9})
    axes[1,0].set_title("Mapa de calor — correlaciones")

    # 4. scatter siniestros vs fallecidos con linea de tendencia
    muestra = df.sample(min(600, len(df)), random_state=42)
    axes[1,1].scatter(muestra["siniestros"], muestra["fallecidos"],
                      alpha=0.4, color=COLORES["verde"], edgecolors="none", s=20)
    m, b = np.polyfit(df["siniestros"], df["fallecidos"], 1)
    xs = np.linspace(df["siniestros"].min(), df["siniestros"].max(), 100)
    corr_val = df[["siniestros","fallecidos"]].corr().iloc[0,1]
    axes[1,1].plot(xs, m*xs + b, color=COLORES["rojo"],
                   linewidth=2, label=f"tendencia (r={corr_val:.2f})")
    axes[1,1].set_title("Siniestros vs Fallecidos")
    axes[1,1].set_xlabel("Siniestros")
    axes[1,1].set_ylabel("Fallecidos")
    axes[1,1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------
# modelos
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# menu
# -----------------------------------------------------------------------

if __name__ == "__main__":
    verificar_api_conaset()
    df = cargar_datos()
    print(df.tail(10).to_string())