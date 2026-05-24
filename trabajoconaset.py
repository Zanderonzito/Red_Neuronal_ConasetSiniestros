"""
Evaluacion 1 - Fundamentos de Data Science
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
# modelos predictivos
# -----------------------------------------------------------------------

def preparar_split(df, features):
    # nos aseguramos de no incluir filas con nulos en las features que vamos a usar
    df_limpio = df.dropna(subset=features + ["siniestros"]).copy()
    X = df_limpio[features]
    y = df_limpio["siniestros"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


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
# menu principal
# -----------------------------------------------------------------------

def main():
    print("\n" + "█"*60)
    print("  SEGURIDAD VIAL EN CHILE — ANALISIS DE SINIESTRALIDAD")
    print("  Fuente: Observatorio CONASET / ArcGIS Open Data")
    print("█"*60)

    verificar_api_conaset()
    df = cargar_datos()

    if "region" not in df.columns or "siniestros" not in df.columns:
        print("\n  Error: no se mapearon bien las columnas del Excel.")
        print("  Revisa la funcion mapear_columnas() y ajusta segun tu archivo.")
        return

    # guardamos los modelos en memoria a medida que se van entrenando
    # al principio todos en None, se van llenando con las opciones del menu
    resultados = {
        "simple":   None,
        "multiple": None,
        "rf":       None,
    }

    while True:
        print("\n" + "="*45)
        print("             MENU")
        print("="*45)
        print("  [ Exploracion ]")
        print("  1. Ver datos (ultimas 10 filas)")
        print("  2. Estadistica descriptiva")
        print("  3. Graficos exploratorios")
        print("")
        print("  [ Modelos predictivos ]")
        print("  4. Entrenar Regresion Lineal Simple")
        print("  5. Entrenar Regresion Multiple")
        print("  6. Entrenar Random Forest")
        print("")
        print("  [ Resultados ]")
        print("  7. Comparar modelos")
        print("")
        print("  8. Salir")

        op = input("\n>> ").strip()

        if op == "1":
            print(df.tail(10).to_string())

        elif op == "2":
            mostrar_estadistica_descriptiva(df)

        elif op == "3":
            hacer_graficos_exploratorios(df)

        elif op == "4":
            mod, r2, feats = _entrenar_simple(df)
            resultados["simple"] = (mod, r2, feats)

        elif op == "5":
            mod, r2, feats = _entrenar_multiple(df)
            resultados["multiple"] = (mod, r2, feats)

        elif op == "6":
            mod, r2, feats = _entrenar_rf(df)
            resultados["rf"] = (mod, r2, feats)

        elif op == "7":
            # verificar que esten entrenados los 3 antes de comparar
            faltantes = [k for k, v in resultados.items() if v is None]
            if faltantes:
                nombres = {"simple": "4", "multiple": "5", "rf": "6"}
                print("\n  faltan modelos por entrenar:")
                for f in faltantes:
                    print(f"    -> opcion {nombres[f]} ({f})")
            else:
                comparar_modelos(resultados)

        elif op == "8":
            print("\n  suerte con la presentacion!!\n")
            break

        else:
            print("  eso no es una opcion valida")


# -----------------------------------------------------------------------
# funciones internas de entrenamiento (separadas para el menu)
# -----------------------------------------------------------------------

def _entrenar_simple(df):
    print("\n" + "="*50)
    print("  MODELO 1 — Regresion Lineal Simple (año -> siniestros)")
    print("="*50)
    X_tr, X_te, y_tr, y_te = preparar_split(df, FEATURES_SIMPLE)
    mod = LinearRegression().fit(X_tr, y_tr)
    coef = mod.coef_[0]
    print(f"\n  ecuacion: siniestros = {coef:.2f} * año + ({mod.intercept_:.2f})")
    print(f"  por cada año que pasa, los siniestros cambian en {coef:.2f} unidades")
    r2 = imprimir_metricas("Lineal Simple",
                            y_tr, mod.predict(X_tr),
                            y_te, mod.predict(X_te))
    graficar_diagnostico(y_te, mod.predict(X_te), "Regresion Lineal Simple")
    return mod, r2, FEATURES_SIMPLE


def _entrenar_multiple(df):
    feats_ok = [f for f in FEATURES_MULTI if f in df.columns]
    print("\n" + "="*50)
    print(f"  MODELO 2 — Regresion Multiple")
    print(f"  Features: {feats_ok}")
    print("="*50)
    X_tr, X_te, y_tr, y_te = preparar_split(df, feats_ok)
    mod = LinearRegression().fit(X_tr, y_tr)
    print("\n  coeficientes (cuanto aporta cada variable):")
    for feat, c in zip(feats_ok, mod.coef_):
        print(f"    {feat:<22}: {c:+.4f}")
    r2 = imprimir_metricas("Regresion Multiple",
                            y_tr, mod.predict(X_tr),
                            y_te, mod.predict(X_te))
    graficar_diagnostico(y_te, mod.predict(X_te), "Regresion Multiple")
    return mod, r2, feats_ok


def _entrenar_rf(df):
    feats_ok = [f for f in FEATURES_MULTI if f in df.columns]
    print("\n" + "="*50)
    print("  MODELO 3 — Random Forest (100 arboles, profundidad max 6)")
    print("="*50)
    X_tr, X_te, y_tr, y_te = preparar_split(df, feats_ok)
    print("  entrenando... puede tardar unos segundos")
    mod = RandomForestRegressor(
        n_estimators=100, max_depth=6,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    mod.fit(X_tr, y_tr)
    print("\n  importancia de variables:")
    importancias = pd.Series(mod.feature_importances_, index=feats_ok)
    for feat, imp in importancias.sort_values(ascending=False).items():
        barra = "█" * int(imp * 35)
        print(f"    {feat:<22}: {barra} {imp:.4f}")
    r2 = imprimir_metricas("Random Forest",
                            y_tr, mod.predict(X_tr),
                            y_te, mod.predict(X_te))
    graficar_diagnostico(y_te, mod.predict(X_te), "Random Forest")
    return mod, r2, feats_ok


if _name_ == "_main_":
    main()
