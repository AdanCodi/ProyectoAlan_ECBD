from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Rutas del proyecto
unidad3 = Path(__file__).resolve().parents[1]
resultados = unidad3 / "resultados"
modelos = unidad3 / "modelos"

# Leer conjuntos de datos
train = pd.read_csv(resultados / "entrenamiento.csv")
test = pd.read_csv(resultados / "prueba.csv")

variables = [
    "Ventas_ayer",
    "Ventas_hace_7_dias",
    "Promedio_7_dias",
    "Promedio_14_dias",
    "Dia_semana",
    "Mes",
    "Fin_semana"
]

X_train = train[variables]
y_train = train["Ventas"]

X_test = test[variables]
y_test = test["Ventas"]

# Modelo inicial
print("Entrenando Random Forest inicial...")

modelo_inicial = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

modelo_inicial.fit(X_train, y_train)

pred_inicial = modelo_inicial.predict(X_test)

# Optimizar hiperparametros
print("Optimizando hiperparametros...")

parametros = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 3, 5]
}

validacion = TimeSeriesSplit(n_splits=5)

busqueda = GridSearchCV(
    RandomForestRegressor(random_state=42),
    parametros,
    cv=validacion,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

busqueda.fit(X_train, y_train)

modelo_optimizado = busqueda.best_estimator_

pred_optimizada = modelo_optimizado.predict(X_test)

# Funcion para evaluar los modelos
def evaluar(nombre, predicciones):
    return {
        "Modelo": nombre,
        "MAE": mean_absolute_error(
            y_test, predicciones
        ),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predicciones)
        ),
        "R2": r2_score(y_test, predicciones)
    }

# Recuperar predicciones de regresion lineal
lineal = pd.read_csv(
    resultados / "predicciones_regresion.csv"
)

# Comparar todos los modelos
metricas = pd.DataFrame([
    evaluar(
        "Referencia semanal",
        test["Ventas_hace_7_dias"]
    ),
    evaluar(
        "Regresion lineal",
        lineal["Ventas_predichas"]
    ),
    evaluar(
        "Random Forest inicial",
        pred_inicial
    ),
    evaluar(
        "Random Forest optimizado",
        pred_optimizada
    )
])

print("\n===== MEJORES HIPERPARAMETROS =====")
print(busqueda.best_params_)

print("\n===== COMPARACION DE MODELOS =====")
print(metricas.round(4).to_string(index=False))

# Guardar resultados
metricas.to_csv(
    resultados / "metricas_regresion.csv",
    index=False
)

joblib.dump(
    modelo_optimizado,
    modelos / "random_forest_regresion.joblib"
)

predicciones = pd.DataFrame({
    "Fecha": test["Fecha"],
    "Ventas_reales": y_test,
    "Random_Forest_inicial": pred_inicial,
    "Random_Forest_optimizado": pred_optimizada
})

predicciones.to_csv(
    resultados / "predicciones_random_forest.csv",
    index=False
)

# Grafica de predicciones
fechas = pd.to_datetime(test["Fecha"])

plt.figure(figsize=(14, 6))

plt.plot(
    fechas,
    y_test,
    label="Ventas reales",
    color="royalblue"
)

plt.plot(
    fechas,
    pred_optimizada,
    label="Random Forest optimizado",
    color="darkorange",
    linestyle="--"
)

plt.title("Random Forest Regressor - Online Retail")
plt.xlabel("Fecha")
plt.ylabel("Ventas (GBP)")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    resultados / "grafica_random_forest.png",
    dpi=300
)

plt.show()

print("\nProceso completado correctamente.")