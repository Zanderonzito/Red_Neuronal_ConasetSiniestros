"""
Evaluacion 2 - Fundamentos de Data Science
Tema: Seguridad Vial - CONASET
Fuente principal: Excel oficial del observatorio de CONASET
https://www.conaset.cl/programa/observatorio-datos-estadistica/biblioteca-observatorio/estadisticas-generales/
----
Fuente secundaria (validacion): API publica ArcGIS de CONASET
https://mapas-conaset.opendata.arcgis.com
-----
Integrantes: Rigo Vega, Martín Caamaño, Favio Muñoz, Nikolas Maldonado.
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
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
FEATURES_SIMPLE = ["año"]
FEATURES_MULTI  = ["año", "region_num", "fallecidos", "lesionados_graves"]
TARGET = "siniestros"

COLORES = {
    "azul":    "#2563EB",
    "rojo":    "#DC2626",
    "verde":   "#16A34A",
    "naranja": "#EA580C",
    "gris":    "#6B7280",
    "morado":  "#7C3AED",
}
# mas que todo para las regiones
REGIONES_NUM = {
    "Tarapacá": 1, "Antofagasta": 2, "Atacama": 3, "Coquimbo": 4,
    "Valparaíso": 5, "L.B.O´Higgins": 6, "Maule": 7, "Biobio": 8,
    "Araucanía": 9, "Los Lagos": 10, "Aysén": 11, "Magallanes": 12,
    "Metropolitana": 13, "Los Ríos": 14, "Arica y Parinacota": 15, "Ñuble": 16,
}
# estado global de los modelos entrenados
RESULTADOS_CLASICOS = {"simple": None, "multiple": None, "rf": None}
RESULTADOS_RED = None
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

def leer_hoja(df_hoja, año):
    # aqui mas que to sacamos la fila 0 que tiene los subheaders y resetear indice
    df = df_hoja.iloc[1:].copy() 
    df = df.reset_index(drop=True)

    cols = df_hoja.columns.tolist()
    rename = {cols[0]: "region"} 
    
    for i, c in enumerate(cols):
        cl = str(c).lower()
        if "siniestro" in cl: rename[c] = "siniestros"
        elif "fallecido" in cl: rename[c] = "fallecidos"

    subheader = df_hoja.iloc[0]
    for i, c in enumerate(cols):
        if c in rename: continue
        sub = str(subheader.iloc[i]).lower().strip()
        if sub == "graves": rename[c] = "lesionados_graves"
        elif "menos" in sub: rename[c] = "lesionados_menos_graves"
        elif sub == "leves": rename[c] = "lesionados_leves"

    df = df.rename(columns=rename)
    df["año"] = int(año)
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
    cols_numericas = ["año", "siniestros", "fallecidos",
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
    df = df.dropna(subset=["año", "region", "siniestros"])
    if antes - len(df) > 0:
        print(f"  filas con nulos eliminadas: {antes - len(df)}")

    # las filas de "Total" o "TOTAL" no son regiones, son sumas parciales
    # si las dejamos van a inflar los modelos
    df = df[~df["region"].astype(str).str.contains("Total|TOTAL", na=False)]
    df["region_num"] = df["region"].map(REGIONES_NUM)
    df["region_num"] = df["region_num"].fillna(-1).astype(int) # Por si hay datos raros

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
                df_hoja = leer_hoja(df_hoja, nombre_hoja)
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
# estadistica descriptiva
# -----------------------------------------------------------------------

def mostrar_estadistica_descriptiva(df):
    print("\n" + "="*60)
    print("     ESTADÍSTICA DESCRIPTIVA — ESTADO DEL ARTE")
    print("="*60)

    cols = ["siniestros", "fallecidos", "lesionados_graves", "tasa_mortalidad"]
    
    print("\n[ Métricas detalladas ]\n")
    print(f"  {'Variable':<20} {'Promedio':>10} {'Varianza':>12} {'Desv.Std':>10}")
    print(f"  {'-'*56}")
    for col in cols:
        if col in df.columns:
            # Convertir a numérico y eliminar errores/texto
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            # Evitar errores si la columna queda vacía
            if len(s) > 0:
                print(f"  {col:<20} "f"{s.mean():>10.2f} "f"{s.var(ddof=1):>12.2f} "f"{s.std(ddof=1):>10.2f}")

    print("\n[ Análisis de Varianza y Comportamiento ]")
    print(f"  → La alta varianza en 'siniestros' ({df['siniestros'].var():.1f}) refleja la disparidad")
    print("    geográfica: regiones como la Metropolitana concentran el volumen,")
    print("    mientras zonas extremas muestran valores atípicamente bajos.")

    print("\n[ Covarianza y Correlación ]\n")
    pares = [("siniestros", "fallecidos"), ("siniestros", "lesionados_graves")]
    for v1, v2 in pares:

        if v1 in df.columns and v2 in df.columns:

            # Crear copia temporal
            temp = df[[v1, v2]].copy()

            # Convertir ambas columnas a numéricas
            temp[v1] = pd.to_numeric(temp[v1], errors='coerce')
            temp[v2] = pd.to_numeric(temp[v2], errors='coerce')

            # Eliminar filas inválidas
            temp = temp.dropna()

            # Verificar que existan suficientes datos
            if len(temp) > 1:
                cov = temp.cov().iloc[0, 1]
                corr = temp.corr().iloc[0, 1]
                print(f"  {v1} ↔ {v2}: "f"Correlación = {corr:.4f} "f"(Covarianza = {cov:.2f})")
# -----------------------------------------------------------------------
# graficos exploratorios
# -----------------------------------------------------------------------

def hacer_graficos_exploratorios(df):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Análisis Exploratorio — Siniestros Viales Chile (2000-2024)",
                 fontsize=14, fontweight="bold")
    
    # 1. tendencia anual
    por_año = df.groupby("año")["siniestros"].sum().reset_index()
    axes[0,0].plot(por_año["año"], por_año["siniestros"],
                   marker="o", color=COLORES["azul"], linewidth=2.5, markersize=6)
    axes[0,0].fill_between(por_año["año"], por_año["siniestros"],
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
# modelos predictivos
# -----------------------------------------------------------------------

def preparar_split(df, features):
    # Profe, aquí separamos 80% fit y 20% test y después
    # y después los ordenamos por año mas que todo para que el test quede con datos mas recientes.
    df_limpio = df.dropna(subset=features + ["siniestros"]).copy()
    df_limpio = df_limpio.sort_values(["año", "region"]).reset_index(drop=True)
    corte = int(len(df_limpio) * 0.80)
    train = df_limpio.iloc[:corte]
    test = df_limpio.iloc[corte:]
    X_train = train[features]
    X_test = test[features]
    y_train = train["siniestros"]
    y_test = test["siniestros"]
    print(f"\n  Split de datos: {len(train)} registros fit y {len(test)} registros test")
    print(f"  Fit:  {train['año'].min()}-{train['año'].max()}")
    print(f"  Test: {test['año'].min()}-{test['año'].max()}")
    return X_train, X_test, y_train, y_test

def imprimir_metricas(nombre, y_train, y_pred_train, y_test, y_pred_test):
    r2_tr = r2_score(y_train, y_pred_train)
    r2_te = r2_score(y_test,  y_pred_test)
    gap   = r2_tr - r2_te

    print(f"\n  Metricas — {nombre}")
    print(f"  {'':10} {'train':>10} {'test':>10}")
    print(f"  {'R2':10} {r2_tr:>10.4f} {r2_te:>10.4f}")
    print(f"  {'RMSE':10} {np.sqrt(mean_squared_error(y_train,y_pred_train)):>10.2f} "
          f"{np.sqrt(mean_squared_error(y_test,y_pred_test)):>10.2f}")
    print(f"  {'MAE':10} {mean_absolute_error(y_train,y_pred_train):>10.2f} "
          f"{mean_absolute_error(y_test,y_pred_test):>10.2f}")

    if abs(gap) > 0.15:
        print(f"  aviso: gap R2 = {gap:+.4f} — puede haber sobreajuste")
    else:
        print(f"  gap R2 = {gap:+.4f} — ok, generaliza bien")
    return r2_te


def graficar_diagnostico(y_test, y_pred, nombre):
    residuos = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Diagnostico — {nombre}", fontweight="bold")

    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].scatter(y_test, y_pred, alpha=0.4, color=COLORES["azul"],
                    edgecolors="none", s=25)
    axes[0].plot([mn,mx],[mn,mx], "r--", linewidth=1.5, label="prediccion perfecta")
    axes[0].set_xlabel("valor real")
    axes[0].set_ylabel("valor predicho")
    axes[0].set_title("Real vs Predicho")
    axes[0].legend(fontsize=9)

    axes[1].hist(residuos, bins=25, color=COLORES["azul"], edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color=COLORES["rojo"], linestyle="--", linewidth=1.5,
                    label="residuo = 0")
    axes[1].axvline(residuos.mean(), color=COLORES["naranja"], linewidth=1.5,
                    label=f"media = {residuos.mean():.2f}")
    axes[1].set_xlabel("residuo")
    axes[1].set_ylabel("frecuencia")
    axes[1].set_title("Histograma de residuos")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()

def matriz_confusion(y_test, y_pred, nombre):
    bins = [0, 2000, 4500, float("inf")]
    labels = ["bajo", "medio", "alto"]

    y_test_s = pd.Series(np.array(y_test).flatten()).reset_index(drop=True)
    y_pred_s = pd.Series(np.array(y_pred).flatten()).reset_index(drop=True)

    y_test_cat = pd.cut(y_test_s, bins=bins, labels=labels)
    y_pred_cat = pd.cut(y_pred_s, bins=bins, labels=labels)

    mask = y_test_cat.notna() & y_pred_cat.notna()
    y_test_cat = y_test_cat[mask].astype(str).values
    y_pred_cat = y_pred_cat[mask].astype(str).values

    cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de Confusion — {nombre}")
    plt.tight_layout()
    plt.show()

def comparar_modelos(resultados):
    print("\n" + "="*50)
    print("  COMPARACION FINAL DE MODELOS")
    print("="*50)

    nombres = ["Lineal Simple", "Regresion Multiple", "Random Forest"]
    r2s     = [resultados["simple"][1], resultados["multiple"][1], resultados["rf"][1]]
    colores = [COLORES["naranja"], COLORES["azul"], COLORES["verde"]]

    ranking = sorted(zip(r2s, nombres), reverse=True)
    print(f"\n  {'Modelo':<22} {'R2 Test':>10}")
    print(f"  {'-'*34}")
    for i, (r2, nombre) in enumerate(ranking, 1):
        marca = "  <- mejor" if i == 1 else ""
        print(f"  {nombre:<22} {r2:>10.4f}{marca}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Comparacion de Modelos — Seguridad Vial Chile", fontweight="bold")

    nombres_graf = ["Lineal\nSimple", "Regresion\nMultiple", "Random\nForest"]
    bars = axes[0].bar(nombres_graf, r2s, color=colores, edgecolor="white",
                       width=0.5, alpha=0.9)
    for bar, val in zip(bars, r2s):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")
    axes[0].set_ylabel("R2 en conjunto de prueba")
    axes[0].set_title("R2 por modelo")
    axes[0].set_ylim(0, max(r2s) * 1.3)

    # importancia de variables del Random Forest
    mod_rf   = resultados["rf"][0]
    feats_ok = resultados["rf"][2]
    imp = pd.Series(mod_rf.feature_importances_, index=feats_ok).sort_values()
    axes[1].barh(imp.index, imp.values, color=COLORES["verde"],
                 edgecolor="white", alpha=0.85)
    axes[1].set_title("Variables mas importantes (Random Forest)")
    axes[1].set_xlabel("Importancia relativa")

    plt.tight_layout()
    plt.show()

    mejor = ranking[0][1]
    print(f"\n  El mejor modelo es '{mejor}'.")
    print("  Igual la Regresion Multiple sirve porque los coeficientes se pueden")
    print("  explicar en terminos reales, algo que con Random Forest es mas dificil.")

# -----------------------------------------------------------------------
# entrenamiento de modelos clasicos
# -----------------------------------------------------------------------

def _entrenar_simple(df):
    global RESULTADOS_CLASICOS
    print("\n" + "="*50)
    print("  MODELO 1 — Regresion Lineal Simple")
    print("="*50)

    X_train, X_test, y_train, y_test = preparar_split(df, FEATURES_SIMPLE)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    r2 = imprimir_metricas("Lineal Simple", y_train, y_pred_train, y_test, y_pred_test)
    graficar_diagnostico(y_test, y_pred_test, "Lineal Simple")
    matriz_confusion(y_test, y_pred_test, "Lineal Simple")

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    RESULTADOS_CLASICOS["simple"] = (
        modelo,
        {"r2_test": r2, "rmse_test": rmse, "mae_test": mae},
        FEATURES_SIMPLE,
    )
    return modelo, r2, FEATURES_SIMPLE


def _entrenar_multiple(df):
    global RESULTADOS_CLASICOS
    print("\n" + "="*50)
    print("  MODELO 2 — Regresion Multiple")
    print("="*50)

    X_train, X_test, y_train, y_test = preparar_split(df, FEATURES_MULTI)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    r2 = imprimir_metricas("Regresion Multiple", y_train, y_pred_train, y_test, y_pred_test)
    graficar_diagnostico(y_test, y_pred_test, "Regresion Multiple")
    matriz_confusion(y_test, y_pred_test, "Regresion Multiple")

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    RESULTADOS_CLASICOS["multiple"] = (
        modelo,
        {"r2_test": r2, "rmse_test": rmse, "mae_test": mae},
        FEATURES_MULTI,
    )
    return modelo, r2, FEATURES_MULTI


def _entrenar_rf(df):
    global RESULTADOS_CLASICOS
    print("\n" + "="*50)
    print("  MODELO 3 — Random Forest")
    print("="*50)

    X_train, X_test, y_train, y_test = preparar_split(df, FEATURES_MULTI)
    modelo = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    r2 = imprimir_metricas("Random Forest", y_train, y_pred_train, y_test, y_pred_test)
    graficar_diagnostico(y_test, y_pred_test, "Random Forest")
    matriz_confusion(y_test, y_pred_test, "Random Forest")

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    RESULTADOS_CLASICOS["rf"] = (
        modelo,
        {"r2_test": r2, "rmse_test": rmse, "mae_test": mae},
        FEATURES_MULTI,
    )
    return modelo, r2, FEATURES_MULTI


# -----------------------------------------------------------------------
# Redes neuronales
# -----------------------------------------------------------------------


FEATURES_NN_NUM = [
    "año",
    "region_num",
    "fallecidos",
    "lesionados_graves",
    "lesionados_leves",
    "tasa_mortalidad",
]

FEATURES_NN_CAT = ["region"]

def preparar_datos_red_neuronal(df):
    # Mismo criterio del profesor: 80% fit y 20% test.
    # El scaler y encoder se ajustan solo con fit para evitar fuga de datos.
    features = FEATURES_NN_NUM + FEATURES_NN_CAT
    df_limpio = df.dropna(subset=features + [TARGET]).copy()
    df_limpio = df_limpio.sort_values(["año", "region"]).reset_index(drop=True)

    corte = int(len(df_limpio) * 0.80)
    train = df_limpio.iloc[:corte]
    test = df_limpio.iloc[corte:]

    X_train = train[features]
    X_test = test[features]
    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                FEATURES_NN_NUM,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", encoder),
                ]),
                FEATURES_NN_CAT,
            ),
        ]
    )

    X_train_prep = preprocesador.fit_transform(X_train)
    X_test_prep = preprocesador.transform(X_test)

    print(f"\n  Split red neuronal: {len(train)} registros fit y {len(test)} registros test")
    print(f"  Fit:  {train['año'].min()}-{train['año'].max()}")
    print(f"  Test: {test['año'].min()}-{test['año'].max()}")

    return X_train_prep, X_test_prep, y_train, y_test, preprocesador


def _entrenar_red_neuronal(df):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ModuleNotFoundError:
        print("\n  Falta TensorFlow. Instala dependencias antes de entrenar la red neuronal.")
        return None, 0, FEATURES_NN_NUM + FEATURES_NN_CAT, None, None, None

    print("\n" + "="*50)
    print("  MODELO 4 — Red Neuronal MLP")
    print("="*50)

    np.random.seed(42)
    tf.random.set_seed(42)

    X_tr, X_te, y_tr, y_te, preprocesador = preparar_datos_red_neuronal(df)

    modelo = keras.Sequential([
        keras.layers.Input(shape=(X_tr.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.30),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.20),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="relu"),
    ])

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )

    print("\n  Arquitectura: 64 -> 32 -> 16 -> 1")
    print("  Activacion: ReLU")
    print("  Entrenando red neuronal...")

    historial = modelo.fit(
        X_tr,
        y_tr,
        validation_split=0.15,
        epochs=150,
        batch_size=16,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
            )
        ],
    )

    y_pred_tr = modelo.predict(X_tr).reshape(-1)
    y_pred_te = modelo.predict(X_te).reshape(-1)

    r2 = imprimir_metricas("Red Neuronal MLP", y_tr, y_pred_tr, y_te, y_pred_te)
    graficar_diagnostico(y_te, y_pred_te, "Red Neuronal MLP")
    matriz_confusion(y_te, y_pred_te, "Red Neuronal MLP")

    return modelo, r2, FEATURES_NN_NUM + FEATURES_NN_CAT, historial, y_te, y_pred_te

def entrenar_red_neuronal(df):
    global RESULTADOS_RED
    modelo, r2, features, historial, y_te, y_pred_te = _entrenar_red_neuronal(df)

    if modelo is None:
        return

    # construir matriz de confusion como DataFrame para los graficos
    bins = [0, 2000, 4500, float("inf")]
    labels = ["bajo", "medio", "alto"]
    y_test_s = pd.Series(np.array(y_te).flatten()).reset_index(drop=True)
    y_pred_s = pd.Series(np.array(y_pred_te).flatten()).reset_index(drop=True)
    y_test_cat = pd.cut(y_test_s, bins=bins, labels=labels)
    y_pred_cat = pd.cut(y_pred_s, bins=bins, labels=labels)
    mask = y_test_cat.notna() & y_pred_cat.notna()
    cm = confusion_matrix(
        y_test_cat[mask].astype(str).values,
        y_pred_cat[mask].astype(str).values,
        labels=labels,
    )
    matriz_df = pd.DataFrame(cm, index=labels, columns=labels)

    rmse = np.sqrt(mean_squared_error(y_te, y_pred_te))
    mae = mean_absolute_error(y_te, y_pred_te)

    RESULTADOS_RED = {
        "historial": historial.history,
        "predicciones": pd.DataFrame({
            "y_real": np.array(y_te).flatten(),
            "y_pred": np.array(y_pred_te).flatten(),
        }),
        "matriz": matriz_df,
        "metricas": {"r2_test": r2, "rmse_test": rmse, "mae_test": mae},
    }
    print("\n  Resultados guardados. Usa opcion 9 para ver graficos.")


def mostrar_metricas_red():
    if RESULTADOS_RED is None:
        print("\n  Todavia no hay resultados. Primero entrena la red con la opcion 7.")
        return

    m = RESULTADOS_RED["metricas"]
    print("\n" + "="*50)
    print("  METRICAS — Red Neuronal MLP")
    print("="*50)
    print(f"  R2 Test:   {m['r2_test']:.4f}")
    print(f"  RMSE Test: {m['rmse_test']:.2f}")
    print(f"  MAE Test:  {m['mae_test']:.2f}")


def graficar_resultados_red():
    """
    Grafica 3 paneles del entrenamiento de la red neuronal:
    1. Evolucion de la perdida (train vs validacion)
    2. Real vs predicho
    3. Matriz de confusion
    """
    if RESULTADOS_RED is None:
        print("\n  Todavia no hay resultados. Primero entrena la red con la opcion 7.")
        return

    historial = RESULTADOS_RED["historial"]
    predicciones = RESULTADOS_RED["predicciones"]
    matriz = RESULTADOS_RED["matriz"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Resultados Red Neuronal — CONASET", fontweight="bold")

    # 1. evolucion de la perdida por epoca
    axes[0].plot(historial["loss"], label="train", color=COLORES["azul"])
    axes[0].plot(historial["val_loss"], label="validacion", color=COLORES["naranja"])
    axes[0].set_title("Evolucion de perdida")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    # 2. scatter real vs predicho
    axes[1].scatter(
        predicciones["y_real"], predicciones["y_pred"],
        alpha=0.65, edgecolors="none", color=COLORES["azul"],
    )
    minimo = min(predicciones["y_real"].min(), predicciones["y_pred"].min())
    maximo = max(predicciones["y_real"].max(), predicciones["y_pred"].max())
    axes[1].plot([minimo, maximo], [minimo, maximo], "r--", label="prediccion perfecta")
    axes[1].set_title("Real vs predicho")
    axes[1].set_xlabel("Siniestros reales")
    axes[1].set_ylabel("Siniestros predichos")
    axes[1].legend()

    # 3. matriz de confusion por categorias
    im = axes[2].imshow(matriz.values, cmap="Blues")
    axes[2].set_title("Matriz de confusion")
    axes[2].set_xticks(range(len(matriz.columns)))
    axes[2].set_yticks(range(len(matriz.index)))
    axes[2].set_xticklabels(matriz.columns)
    axes[2].set_yticklabels(matriz.index)
    axes[2].set_xlabel("Predicho")
    axes[2].set_ylabel("Real")
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            axes[2].text(j, i, matriz.iloc[i, j], ha="center", va="center")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("resultados_red_neuronal.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Grafico guardado como 'resultados_red_neuronal.png'")


# =====================================================================
# COMPARACION DE TODOS LOS MODELOS
# =====================================================================

def comparar_todos_los_modelos():
    """
    Compara los 4 modelos (3 clasicos + red neuronal) en una tabla con:
    - R2 test, RMSE test, MAE test
    - Interpretabilidad, tiempo de entrenamiento, riesgo de sobreajuste
    - Grafico de barras comparativo de R2, RMSE y MAE
    - Reporte de texto con recomendaciones

    Prerequisito: los 4 modelos deben estar entrenados.
    """
    # verificar que los modelos clasicos esten entrenados
    faltantes_clasicos = [k for k, v in RESULTADOS_CLASICOS.items() if v is None]
    if faltantes_clasicos:
        nombres_menu = {"simple": "4", "multiple": "5", "rf": "6"}
        print("\n  Faltan modelos clasicos por entrenar:")
        for f in faltantes_clasicos:
            print(f"    -> opcion {nombres_menu[f]} ({f})")
        return

    # verificar que la red neuronal este entrenada
    if RESULTADOS_RED is None:
        print("\n  Falta entrenar la red neuronal.")
        print("    -> opcion 8")
        return

    print("\n" + "=" * 70)
    print("  COMPARATIVA DE TODOS LOS MODELOS")
    print("=" * 70)

    # recopilar metricas de cada modelo
    modelos = {
        "Lineal Simple": RESULTADOS_CLASICOS["simple"][1],
        "Reg. Multiple": RESULTADOS_CLASICOS["multiple"][1],
        "Random Forest": RESULTADOS_CLASICOS["rf"][1],
        "Red Neuronal":  {
            "r2_test": RESULTADOS_RED["metricas"]["r2_test"],
            "rmse_test": RESULTADOS_RED["metricas"]["rmse_test"],
            "mae_test": RESULTADOS_RED["metricas"]["mae_test"],
        },
    }

    # caracteristicas cualitativas de cada modelo
    interpretabilidad = {
        "Lineal Simple": "Muy alta",
        "Reg. Multiple": "Alta",
        "Random Forest": "Media",
        "Red Neuronal":  "Baja",
    }
    tiempo_entreno = {
        "Lineal Simple": "<1s",
        "Reg. Multiple": "<1s",
        "Random Forest": "~5s",
        "Red Neuronal":  "~30s",
    }
    riesgo_overfit = {
        "Lineal Simple": "Bajo",
        "Reg. Multiple": "Bajo",
        "Random Forest": "Medio",
        "Red Neuronal":  "Posible",
    }

    # imprimir tabla comparativa en consola
    header = (
        f"  {'Modelo':<18} {'R2 Test':>10} {'RMSE Test':>12} {'MAE Test':>10} "
        f"{'Interpret.':>14} {'Tiempo':>8} {'Overfit':>10}"
    )
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")

    for nombre, m in modelos.items():
        print(
            f"  {nombre:<18} {m['r2_test']:>10.4f} {m['rmse_test']:>12.2f} "
            f"{m['mae_test']:>10.2f} {interpretabilidad[nombre]:>14} "
            f"{tiempo_entreno[nombre]:>8} {riesgo_overfit[nombre]:>10}"
        )

    # ranking por R2 (de mayor a menor)
    ranking = sorted(modelos.items(), key=lambda x: x[1]["r2_test"], reverse=True)
    print(f"\n  Mejor modelo por R2: {ranking[0][0]} ({ranking[0][1]['r2_test']:.4f})")

    # grafico comparativo con 3 paneles: R2, RMSE, MAE
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Comparativa de Modelos — Siniestralidad Vial Chile",
        fontsize=13, fontweight="bold",
    )

    nombres_graf = list(modelos.keys())
    colores = [COLORES["naranja"], COLORES["azul"], COLORES["verde"], COLORES["morado"]]

    # panel 1: R2 por modelo (mas alto = mejor)
    r2_vals = [m["r2_test"] for m in modelos.values()]
    bars = axes[0].bar(nombres_graf, r2_vals, color=colores, edgecolor="white", width=0.5)
    for bar, val in zip(bars, r2_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
    axes[0].set_ylabel("R2 Test")
    axes[0].set_title("R2 por modelo")
    axes[0].set_ylim(0, max(r2_vals) * 1.3 if max(r2_vals) > 0 else 1)
    axes[0].tick_params(axis="x", rotation=15)

    # panel 2: RMSE por modelo (mas bajo = mejor)
    rmse_vals = [m["rmse_test"] for m in modelos.values()]
    bars2 = axes[1].bar(nombres_graf, rmse_vals, color=colores, edgecolor="white", width=0.5)
    for bar, val in zip(bars2, rmse_vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rmse_vals) * 0.01,
            f"{val:.2f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
    axes[1].set_ylabel("RMSE Test")
    axes[1].set_title("RMSE por modelo (menor es mejor)")
    axes[1].tick_params(axis="x", rotation=15)

    # panel 3: MAE por modelo (mas bajo = mejor)
    mae_vals = [m["mae_test"] for m in modelos.values()]
    bars3 = axes[2].bar(nombres_graf, mae_vals, color=colores, edgecolor="white", width=0.5)
    for bar, val in zip(bars3, mae_vals):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mae_vals) * 0.01,
            f"{val:.2f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
    axes[2].set_ylabel("MAE Test")
    axes[2].set_title("MAE por modelo (menor es mejor)")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig("comparativa_modelos.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Grafico guardado como 'comparativa_modelos.png'")

    # generar reporte de texto con analisis y recomendaciones
    reporte = []
    reporte.append("=" * 70)
    reporte.append("REPORTE — COMPARATIVA DE TODOS LOS MODELOS")
    reporte.append("Trabajo de Redes Neuronales y Modelos Predictivos Data Science")
    reporte.append("Datos: CONASET — Siniestralidad Vial Chile (2000-2024)")
    reporte.append("=" * 70)
    reporte.append("")
    reporte.append("TABLA DE RESULTADOS")
    reporte.append(f"{'Modelo':<18} {'R2 Test':>10} {'RMSE Test':>12} {'MAE Test':>10}")
    reporte.append("-" * 52)
    for nombre, m in modelos.items():
        reporte.append(
            f"{nombre:<18} {m['r2_test']:>10.4f} {m['rmse_test']:>12.2f} {m['mae_test']:>10.2f}"
        )
    reporte.append("")
    reporte.append(f"Mejor modelo por R2: {ranking[0][0]} ({ranking[0][1]['r2_test']:.4f})")
    reporte.append("")
    reporte.append("ANALISIS POR MODELO")
    reporte.append("")
    reporte.append("Regresion Lineal Simple:")
    reporte.append("  Ideal para explicar tendencias globales en el tiempo.")
    reporte.append("  Requiere pocos datos y es facil de interpretar.")
    reporte.append("  Limitado: no captura relaciones no lineales entre variables.")
    reporte.append("")
    reporte.append("Regresion Multiple:")
    reporte.append("  Permite cuantificar el efecto de cada variable con coeficientes.")
    reporte.append("  Los coeficientes tienen interpretacion directa y real.")
    reporte.append("  Recomendado para reportes publicos de CONASET.")
    reporte.append("")
    reporte.append("Random Forest:")
    reporte.append("  Captura interacciones y relaciones no lineales entre variables.")
    reporte.append("  Maneja datos heterogeneos con robustez.")
    reporte.append("  La importancia de variables ayuda a identificar los factores clave.")
    reporte.append("")
    reporte.append("Red Neuronal:")
    reporte.append("  Maxima capacidad predictiva con suficientes datos y features.")
    reporte.append("  Dificil de explicar (caja negra).")
    reporte.append("  Recomendada para alertas internas de mayor precision.")
    reporte.append("")
    reporte.append("RECOMENDACION PARA CONASET:")
    reporte.append("  - Regresion Multiple para reportes publicos (coeficientes explicables)")
    reporte.append("  - Red Neuronal para alertas internas (mayor precision)")
    reporte.append("  - Random Forest como baseline robusto intermedio")

    with open("reporte.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(reporte))
    print("  Reporte guardado como 'reporte.txt'")


# =====================================================================
# MENU PRINCIPAL
# =====================================================================

def main():
    print("\n" + "=" * 60)
    print("  TRABAJO DE REDES NEURONALES Y MODELOS PREDICTIVOS")
    print("  DATA SCIENCE")
    print("  Fuente: Observatorio CONASET (2000-2024)")
    print("=" * 60)

    verificar_api_conaset()
    df = cargar_datos()

    if "region" not in df.columns or "siniestros" not in df.columns:
        print("\n  Error: no se mapearon bien las columnas del Excel.")
        print("  Revisa las funciones de lectura y ajusta segun tu archivo.")
        return

    while True:
        print("\n" + "=" * 50)
        print("                  MENU PRINCIPAL")
        print("=" * 50)
        print("  [ Exploracion ]")
        print("   1. Ver datos (ultimas 10 filas)")
        print("   2. Estadistica descriptiva")
        print("   3. Graficos exploratorios")
        print("")
        print("  [ Modelos Clasicos ]")
        print("   4. Entrenar Regresion Lineal Simple")
        print("   5. Entrenar Regresion Multiple")
        print("   6. Entrenar Random Forest")
        print("")
        print("  [ Red Neuronal ]")
        print("   7. Entrenar red neuronal")
        print("   8. Ver metricas red neuronal")
        print("   9. Ver graficos red neuronal")
        print("")
        print("  [ Comparativa ]")
        print("  10. Comparar todos los modelos")
        print("")
        print("  11. Salir")

        op = input("\n>> ").strip()

        if op == "1":
            print(df.tail(10).to_string())

        elif op == "2":
            mostrar_estadistica_descriptiva(df)

        elif op == "3":
            hacer_graficos_exploratorios(df)

        elif op == "4":
            _entrenar_simple(df)

        elif op == "5":
            _entrenar_multiple(df)

        elif op == "6":
            _entrenar_rf(df)

        elif op == "7":
            entrenar_red_neuronal(df)

        elif op == "8":
            mostrar_metricas_red()

        elif op == "9":
            graficar_resultados_red()

        elif op == "10":
            comparar_todos_los_modelos()

        elif op == "11":
            print("\n  Fin del Programa.\n")
            break

        else:
            print("  Eso no es una opcion valida")


if __name__ == "__main__":
    main()


