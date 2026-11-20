from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Rutas del proyecto
proyecto = Path(__file__).resolve().parents[2]
resultados = proyecto / "unidad4" / "resultados"
modelos = proyecto / "unidad4" / "modelos"

# Cargar datos
rfm = pd.read_csv(resultados / "datos_rfm.csv")

escalados = pd.read_csv(
    resultados / "datos_rfm_escalados.csv"
)

variables = ["Recencia", "Frecuencia", "Monetario"]

X = escalados[variables]

# Evaluar diferentes hiperparametros
evaluaciones = []

print("Evaluando configuraciones de DBSCAN...")

for eps in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:

    for min_samples in [5, 10, 15, 20]:

        modelo = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        etiquetas = modelo.fit_predict(X)

        clusters = len(set(etiquetas) - {-1})
        ruido = (etiquetas == -1).sum()

        mascara = etiquetas != -1

        if clusters >= 2 and mascara.sum() > clusters:

            evaluaciones.append({
                "Eps": eps,
                "Min_samples": min_samples,
                "Clusters": clusters,
                "Ruido": ruido,
                "Porcentaje_ruido": ruido / len(X) * 100,
                "Silhouette": silhouette_score(
                    X[mascara],
                    etiquetas[mascara]
                ),
                "Davies_Bouldin": davies_bouldin_score(
                    X[mascara],
                    etiquetas[mascara]
                ),
                "Calinski_Harabasz": calinski_harabasz_score(
                    X[mascara],
                    etiquetas[mascara]
                )
            })

metricas = pd.DataFrame(evaluaciones)

if metricas.empty:
    raise RuntimeError(
        "No se encontraron configuraciones validas."
    )

metricas.to_csv(
    resultados / "metricas_dbscan.csv",
    index=False
)

print("\n===== EVALUACION DE DBSCAN =====")
print(metricas.round(4).to_string(index=False))

# Priorizar configuraciones con maximo 20% de ruido
candidatos = metricas[
    metricas["Porcentaje_ruido"] <= 20
]

if candidatos.empty:
    candidatos = metricas

mejor = candidatos.sort_values(
    ["Silhouette", "Porcentaje_ruido"],
    ascending=[False, True]
).iloc[0]

print("\n===== MEJOR CONFIGURACION =====")
print(mejor.round(4).to_string())

# Entrenar modelo final
modelo_final = DBSCAN(
    eps=float(mejor["Eps"]),
    min_samples=int(mejor["Min_samples"])
)

rfm["Cluster"] = modelo_final.fit_predict(X)

# Perfil de los grupos
perfiles = rfm.groupby("Cluster").agg(
    Clientes=("CustomerID", "count"),
    Recencia_Promedio=("Recencia", "mean"),
    Frecuencia_Promedio=("Frecuencia", "mean"),
    Monetario_Promedio=("Monetario", "mean"),
    Monetario_Total=("Monetario", "sum")
)

print("\n===== PERFIL DE LOS CLUSTERS =====")
print(perfiles.round(2).to_string())

# Guardar archivos
rfm.to_csv(
    resultados / "clientes_dbscan.csv",
    index=False
)

perfiles.to_csv(
    resultados / "perfiles_dbscan.csv"
)

joblib.dump(
    modelo_final,
    modelos / "dbscan.joblib"
)

# Grafica de distribucion
plt.figure(figsize=(10, 6))

perfiles["Clientes"].plot(
    kind="bar",
    color="darkorange"
)

plt.title("Distribucion de clientes - DBSCAN")
plt.xlabel("Cluster (-1 representa ruido)")
plt.ylabel("Cantidad de clientes")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    resultados / "distribucion_dbscan.png",
    dpi=300
)

plt.close()

print("\nModelo DBSCAN generado correctamente.")