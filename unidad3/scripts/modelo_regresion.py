from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Rutas
unidad3 = Path(__file__).resolve().parents[1]
resultados = unidad3 / "resultados"
modelos = unidad3 / "modelos"

modelos.mkdir(exist_ok=True)

# Cargar los datos
train = pd.read_csv(resultados / "entrenamiento.csv")
test = pd.read_csv(resultados / "prueba.csv")

# Variables de entrada
variables = [
    "Ventas_ayer",
    "Ventas_hace_7_dias",
    "Promedio_7_dias",
    "Promedio_14_dias",
    "Dia_semana",
    "Mes",
    "Fin_semana"
]

# Datos de entrenamiento
X_train = train[variables]
y_train = train["Ventas"]

# Datos de prueba
X_test = test[variables]
y_test = test["Ventas"]

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Realizar predicciones
predicciones = modelo.predict(X_test)

# Evaluar modelo
mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
r2 = r2_score(y_test, predicciones)

print("\n===== REGRESION LINEAL =====")
print("Dias de entrenamiento:", len(X_train))
print("Dias de prueba:", len(X_test))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")

# Modelo de referencia
referencia = test["Ventas_hace_7_dias"]

print("\n===== MODELO DE REFERENCIA =====")
print(f"MAE: {mean_absolute_error(y_test, referencia):.2f}")
print(
    f"RMSE: "
    f"{np.sqrt(mean_squared_error(y_test, referencia)):.2f}"
)
print(f"R2: {r2_score(y_test, referencia):.4f}")

# Guardar las predicciones
comparacion = pd.DataFrame({
    "Fecha": test["Fecha"],
    "Ventas_reales": y_test,
    "Ventas_predichas": predicciones
})

comparacion.to_csv(
    resultados / "predicciones_regresion.csv",
    index=False
)

# Guardar el modelo
joblib.dump(
    modelo,
    modelos / "regresion_lineal.joblib"
)

# Generar grafica
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
    predicciones,
    label="Ventas predichas",
    color="darkorange",
    linestyle="--"
)

plt.title("Regresion lineal - Online Retail")
plt.xlabel("Fecha")
plt.ylabel("Ventas (GBP)")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    resultados / "grafica_regresion_lineal.png",
    dpi=300
)

plt.show()

print("\nModelo guardado correctamente.")