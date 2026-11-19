from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Rutas del proyecto
unidad3 = Path(__file__).resolve().parents[1]
resultados = unidad3 / "resultados"
modelos = unidad3 / "modelos"

# Cargar conjuntos de datos
train = pd.read_csv(resultados / "entrenamiento.csv")
test = pd.read_csv(resultados / "prueba.csv")

# Variables predictoras
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
y_train = train["Demanda_alta"]

X_test = test[variables]
y_test = test["Demanda_alta"]

# MODELO 1: Regresion logistica
print("Entrenando Regresion Logistica...")

logistica = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        random_state=42
    )
)

logistica.fit(X_train, y_train)
pred_logistica = logistica.predict(X_test)

# MODELO 2: Random Forest inicial
print("Entrenando Random Forest Classifier...")

forest_inicial = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_inicial.fit(X_train, y_train)
pred_inicial = forest_inicial.predict(X_test)

# MODELO 3: Random Forest optimizado
print("Optimizando hiperparametros...")

parametros = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 3, 5]
}

busqueda = GridSearchCV(
    RandomForestClassifier(random_state=42),
    parametros,
    cv=TimeSeriesSplit(n_splits=5),
    scoring="f1",
    n_jobs=-1
)

busqueda.fit(X_train, y_train)

forest_optimizado = busqueda.best_estimator_
pred_optimizada = forest_optimizado.predict(X_test)

# Funcion para evaluar cada modelo
def evaluar(nombre, predicciones):
    return {
        "Modelo": nombre,
        "Accuracy": accuracy_score(
            y_test, predicciones
        ),
        "Precision": precision_score(
            y_test, predicciones, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predicciones, zero_division=0
        ),
        "F1": f1_score(
            y_test, predicciones, zero_division=0
        )
    }

# Comparacion de resultados
metricas = pd.DataFrame([
    evaluar(
        "Regresion Logistica",
        pred_logistica
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

print("\n===== METRICAS DE CLASIFICACION =====")
print(metricas.round(4).to_string(index=False))

print("\n===== REPORTE DE CLASIFICACION =====")

print(classification_report(
    y_test,
    pred_optimizada,
    labels=[0, 1],
    target_names=["Demanda baja", "Demanda alta"],
    zero_division=0
))

# Guardar metricas
metricas.to_csv(
    resultados / "metricas_clasificacion.csv",
    index=False
)

# Guardar modelos entrenados
joblib.dump(
    logistica,
    modelos / "regresion_logistica.joblib"
)

joblib.dump(
    forest_optimizado,
    modelos / "random_forest_clasificacion.joblib"
)

# Guardar predicciones
pd.DataFrame({
    "Fecha": test["Fecha"],
    "Demanda_real": y_test,
    "Regresion_logistica": pred_logistica,
    "Random_Forest_inicial": pred_inicial,
    "Random_Forest_optimizado": pred_optimizada
}).to_csv(
    resultados / "predicciones_clasificacion.csv",
    index=False
)

# Graficar matrices de confusion
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

predicciones = [
    pred_logistica,
    pred_inicial,
    pred_optimizada
]

nombres = [
    "Regresion Logistica",
    "Random Forest inicial",
    "Random Forest optimizado"
]

for ax, pred, nombre in zip(
    axes, predicciones, nombres
):
    matriz = confusion_matrix(
        y_test, pred, labels=[0, 1]
    )

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Baja", "Alta"],
        yticklabels=["Baja", "Alta"],
        ax=ax
    )

    ax.set_title(nombre)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor real")

plt.tight_layout()

plt.savefig(
    resultados / "matrices_confusion.png",
    dpi=300
)

plt.show()

print("\nClasificacion completada correctamente.")