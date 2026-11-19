from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Rutas del proyecto
unidad3 = Path(__file__).resolve().parents[1]
resultados = unidad3 / "resultados"

# Cargar las metricas obtenidas
regresion = pd.read_csv(
    resultados / "metricas_regresion.csv"
)

clasificacion = pd.read_csv(
    resultados / "metricas_clasificacion.csv"
)

sns.set_theme(style="whitegrid")

# 1. Comparacion MAE y RMSE
fig, ax = plt.subplots(figsize=(12, 6))

regresion.set_index("Modelo")[["MAE", "RMSE"]].plot(
    kind="bar",
    ax=ax,
    color=["#2563eb", "#f59e0b"],
    rot=15
)

ax.set_title("Comparacion de errores de regresion")
ax.set_ylabel("Error en GBP")
ax.set_xlabel("Modelo")

plt.tight_layout()
plt.savefig(
    resultados / "comparacion_errores_regresion.png",
    dpi=300
)
plt.close()

# 2. Comparacion R2
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=regresion,
    x="Modelo",
    y="R2",
    color="#2563eb",
    ax=ax
)

for contenedor in ax.containers:
    ax.bar_label(contenedor, fmt="%.4f")

ax.set_title("Comparacion del coeficiente R2")
ax.set_ylim(0, 1)
ax.set_xlabel("Modelo")
ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig(
    resultados / "comparacion_r2.png",
    dpi=300
)
plt.close()

# 3. Comparacion de clasificacion
fig, ax = plt.subplots(figsize=(12, 6))

clasificacion.set_index("Modelo")[
    ["Accuracy", "Precision", "Recall", "F1"]
].plot(
    kind="bar",
    ax=ax,
    color=["#2563eb", "#16a34a", "#f59e0b", "#7c3aed"],
    rot=10
)

ax.set_title("Comparacion de modelos de clasificacion")
ax.set_ylabel("Valor de la metrica")
ax.set_ylim(0, 1.1)
ax.set_xlabel("Modelo")

for contenedor in ax.containers:
    ax.bar_label(
        contenedor,
        fmt="%.2f",
        fontsize=8
    )

plt.tight_layout()
plt.savefig(
    resultados / "comparacion_clasificacion.png",
    dpi=300
)
plt.close()

# Mostrar resultados finales
print("\n===== RESULTADOS FINALES DE REGRESION =====")
print(regresion.round(4).to_string(index=False))

print("\n===== RESULTADOS FINALES DE CLASIFICACION =====")
print(clasificacion.round(4).to_string(index=False))

print("\nGraficas generadas correctamente:")
print("- comparacion_errores_regresion.png")
print("- comparacion_r2.png")
print("- comparacion_clasificacion.png")

print("\nAnalisis comparativo completado.")