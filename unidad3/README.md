# Unidad 3 - Analisis Supervisado

## Caso de estudio

Prediccion de ventas y clasificacion de demanda
mediante el dataset Online Retail.

## Objetivos

- Predecir el importe diario de ventas.
- Clasificar los dias de demanda alta y baja.
- Comparar y optimizar modelos supervisados.

## Dataset

Se reutilizaron los 524,878 registros procesados
durante la Unidad 2.

- Dias disponibles: 360.
- Entrenamiento: 288 dias.
- Prueba: 72 dias.
- Umbral de demanda alta: 23926.16 GBP.

## Algoritmos utilizados

Regresion:
- Regresion lineal.
- Random Forest Regressor.

Clasificacion:
- Regresion logistica.
- Random Forest Classifier.

## Optimizacion

GridSearchCV con TimeSeriesSplit de 5 divisiones.

## Instalacion

python -m pip install pandas scikit-learn matplotlib seaborn joblib

## Ejecucion

python unidad3/scripts/preparar_datos.py

python unidad3/scripts/modelo_regresion.py

python unidad3/scripts/modelo_random_forest.py

python unidad3/scripts/modelo_clasificacion.py

python unidad3/scripts/comparar_modelos.py

## Resultados

- Mejor R2: 0.4170.
- Mejor Accuracy: 93.06%.
- Mejor F1-score: 95.41%.

## Herramientas

Python, Pandas, Scikit-learn, Matplotlib,
Seaborn, Joblib, VS Code y GitHub.