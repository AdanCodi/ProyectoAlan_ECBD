
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

BASE = Path(__file__).resolve().parents[1]
RESULTADOS = BASE / "resultados"

sns.set_theme(style="whitegrid")

print("Cargando resultados de los modelos...")

# Cargar resultados anteriores
kmeans = pd.read_csv(RESULTADOS / "metricas_kmeans.csv")
dbscan = pd.read_csv(RESULTADOS / "metricas_dbscan.csv")
pca = pd.read_csv(RESULTADOS / "metricas_pca.csv")

# Seleccionar mejores configuraciones
mejor_kmeans = kmeans.loc[kmeans["Silhouette"].idxmax()]

dbscan_validos = dbscan[
    dbscan["Porcentaje_ruido"] <= 20
]

if dbscan_validos.empty:
    dbscan_validos = dbscan

mejor_dbscan = dbscan_validos.loc[
    dbscan_validos["Silhouette"].idxmax()
]

comparacion = pd.DataFrame([
    {
        "Modelo": "K-Means",
        "Clusters": int(mejor_kmeans["K"]),
        "Silhouette": mejor_kmeans["Silhouette"],
        "Davies_Bouldin": mejor_kmeans["Davies_Bouldin"],
        "Calinski_Harabasz": mejor_kmeans["Calinski_Harabasz"],
        "Porcentaje_ruido": 0
    },
    {
        "Modelo": "DBSCAN",
        "Clusters": int(mejor_dbscan["Clusters"]),
        "Silhouette": mejor_dbscan["Silhouette"],
        "Davies_Bouldin": mejor_dbscan["Davies_Bouldin"],
        "Calinski_Harabasz": mejor_dbscan["Calinski_Harabasz"],
        "Porcentaje_ruido": mejor_dbscan["Porcentaje_ruido"]
    }
])

print("\n===== RESULTADOS ORIGINALES =====")
print(comparacion.round(4).to_string(index=False))

comparacion.to_csv(
    RESULTADOS / "comparacion_clustering.csv",
    index=False
)

# Comparacion sobre los mismos clientes
rfm = pd.read_csv(RESULTADOS / "datos_rfm_escalados.csv")

clientes_kmeans = pd.read_csv(
    RESULTADOS / "clientes_kmeans.csv"
)

clientes_dbscan = pd.read_csv(
    RESULTADOS / "clientes_dbscan.csv"
)

datos = rfm.merge(
    clientes_kmeans[["CustomerID", "Cluster"]].rename(
        columns={"Cluster": "Cluster_KMeans"}
    ),
    on="CustomerID"
)

datos = datos.merge(
    clientes_dbscan[["CustomerID", "Cluster"]].rename(
        columns={"Cluster": "Cluster_DBSCAN"}
    ),
    on="CustomerID"
)

# Excluir ruido para evaluar ambos con la misma muestra
comunes = datos[datos["Cluster_DBSCAN"] != -1]

X = comunes[["Recencia", "Frecuencia", "Monetario"]]

metricas_comunes = []

for modelo, columna in [
    ("K-Means", "Cluster_KMeans"),
    ("DBSCAN", "Cluster_DBSCAN")
]:
    etiquetas = comunes[columna]

    metricas_comunes.append({
        "Modelo": modelo,
        "Clientes_evaluados": len(comunes),
        "Silhouette": silhouette_score(X, etiquetas),
        "Davies_Bouldin": davies_bouldin_score(X, etiquetas),
        "Calinski_Harabasz": calinski_harabasz_score(
            X, etiquetas
        )
    })

metricas_comunes = pd.DataFrame(metricas_comunes)

print("\n===== COMPARACION SOBRE LA MISMA MUESTRA =====")
print(metricas_comunes.round(4).to_string(index=False))

metricas_comunes.to_csv(
    RESULTADOS / "comparacion_muestra_comun.csv",
    index=False
)

# Evaluacion de la optimizacion
kmeans_inicial = kmeans[kmeans["K"] == 3].iloc[0]

dbscan_inicial = dbscan[
    (dbscan["Eps"] == 0.5) &
    (dbscan["Min_samples"] == 5)
].iloc[0]

optimizacion = pd.DataFrame([
    {
        "Modelo": "K-Means",
        "Configuracion": "Inicial: K=3",
        "Silhouette": kmeans_inicial["Silhouette"]
    },
    {
        "Modelo": "K-Means",
        "Configuracion": "Optimizado",
        "Silhouette": mejor_kmeans["Silhouette"]
    },
    {
        "Modelo": "DBSCAN",
        "Configuracion": "Inicial: eps=0.5, min=5",
        "Silhouette": dbscan_inicial["Silhouette"]
    },
    {
        "Modelo": "DBSCAN",
        "Configuracion": "Optimizado",
        "Silhouette": mejor_dbscan["Silhouette"]
    }
])

print("\n===== OPTIMIZACION DE MODELOS =====")
print(optimizacion.round(4).to_string(index=False))

optimizacion.to_csv(
    RESULTADOS / "comparacion_optimizacion.csv",
    index=False
)

print("\n===== RESULTADOS DE PCA =====")
print(pca.round(6).to_string(index=False))

# Grafica comparativa de metricas
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metricas = [
    "Silhouette",
    "Davies_Bouldin",
    "Calinski_Harabasz"
]

for ax, metrica in zip(axes, metricas):
    ax.bar(
        metricas_comunes["Modelo"],
        metricas_comunes[metrica],
        color=["steelblue", "darkorange"]
    )
    ax.set_title(metrica)

fig.suptitle("Comparacion de algoritmos de clustering")
plt.tight_layout()

plt.savefig(
    RESULTADOS / "comparacion_clustering.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Grafica antes y despues de optimizar
fig, ax = plt.subplots(figsize=(10, 5))

iniciales = [
    kmeans_inicial["Silhouette"],
    dbscan_inicial["Silhouette"]
]

optimizados = [
    mejor_kmeans["Silhouette"],
    mejor_dbscan["Silhouette"]
]

posiciones = [0, 1]

ax.bar(
    [x - 0.18 for x in posiciones],
    iniciales,
    width=0.36,
    label="Inicial",
    color="gray"
)

ax.bar(
    [x + 0.18 for x in posiciones],
    optimizados,
    width=0.36,
    label="Optimizado",
    color="seagreen"
)

ax.set_xticks(posiciones)
ax.set_xticklabels(["K-Means", "DBSCAN"])
ax.set_ylabel("Coeficiente Silhouette")
ax.set_title("Evaluacion de la optimizacion")
ax.legend()

plt.tight_layout()

plt.savefig(
    RESULTADOS / "comparacion_optimizacion.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nArchivos generados:")
print("- comparacion_clustering.csv")
print("- comparacion_muestra_comun.csv")
print("- comparacion_optimizacion.csv")
print("- comparacion_clustering.png")
print("- comparacion_optimizacion.png")

print("\nAnalisis comparativo completado correctamente.")
