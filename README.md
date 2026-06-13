## Fuente de datos
- **Datos históricos:** Excel oficial descargado desde el Observatorio CONASET  
  Archivo: `Regionesdeocurrencia2000-2024.xlsx`  
  Descarga directa: https://www.conaset.cl/programa/observatorio-datos-estadistica/biblioteca-observatorio/estadisticas-generales/

- **Validación de fuente:** API pública ArcGIS Open Data de CONASET  
  https://mapas-conaset.opendata.arcgis.com

Profesor aqui para correr el código, descarga el Excel desde el link de arriba
y lo pone en la carpeta "trabajoconaset"

El proyecto compara modelos clasicos de Machine Learning con una red neuronal MLP:

- Regresion lineal simple
- Regresion multiple
- Random Forest
- Red neuronal MLP

Todos los modelos usan una separacion 80% fit y 20% test, siguiendo la indicacion del profesor. El objetivo es comparar el rendimiento predictivo sobre la variable siniestros.

## Metricas usadas

Para regresion:

- MAE
- MSE
- RMSE
- R2

Ademas, las predicciones se agrupan en categorias bajo, medio y alto para calcular:

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusion

Integrantes:
Martin Caamaño
Nikolas Maldonado
Favio Muños
Rigo Vega
