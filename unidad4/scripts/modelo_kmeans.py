from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import KMeans
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

# Evaluar diferentes cantidades de clusters
evaluaciones = []

print("Evaluando diferentes valores de K...")

for k in range(2, 11):

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    etiquetas = modelo.fit_predict(X)

    evaluaciones.append({
        "K": k,
        "Inercia": modelo.inertia_,
        "Silhouette": silhouette_score(X, etiquetas),
        "Davies_Bouldin": davies_bouldin_score(
            X, etiquetas
        ),
        "Calinski_Harabasz": calinski_harabasz_score(
            X, etiquetas
        )
    })

metricas = pd.DataFrame(evaluaciones)

print("\n===== EVALUACION DE K-MEANS =====")
print(metricas.round(4).to_string(index=False))

# Seleccionar K por coeficiente de silueta
mejor_k = int(
    metricas.loc[metricas["Silhouette"].idxmax(), "K"]
)

print("\nMejor cantidad de clusters:", mejor_k)

# Entrenar modelo seleccionado
modelo_final = KMeans(
    n_clusters=mejor_k,
    random_state=42,
    n_init=20
)

rfm["Cluster"] = modelo_final.fit_predict(X)

# Perfil de cada cluster
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
metricas.to_csv(
    resultados / "metricas_kmeans.csv",
    index=False
)

rfm.to_csv(
    resultados / "clientes_kmeans.csv",
    index=False
)

perfiles.to_csv(
    resultados / "perfiles_kmeans.csv"
)

joblib.dump(
    modelo_final,
    modelos / "kmeans.joblib"
)

# Grafica del metodo del codo
plt.figure(figsize=(10, 6))

plt.plot(
    metricas["K"],
    metricas["Inercia"],
    marker="o",
    color="royalblue"
)

plt.title("Metodo del codo - K-Means")
plt.xlabel("Numero de clusters")
plt.ylabel("Inercia")
plt.xticks(range(2, 11))
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    resultados / "metodo_codo.png",
    dpi=300
)

plt.close()

# Grafica de coeficiente de silueta
plt.figure(figsize=(10, 6))

plt.plot(
    metricas["K"],
    metricas["Silhouette"],
    marker="o",
    color="seagreen"
)

plt.axvline(
    mejor_k,
    color="red",
    linestyle="--",
    label=f"Mejor K = {mejor_k}"
)

plt.title("Coeficiente de silueta - K-Means")
plt.xlabel("Numero de clusters")
plt.ylabel("Silhouette Score")
plt.xticks(range(2, 11))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    resultados / "silueta_kmeans.png",
    dpi=300
)

plt.close()

# Distribucion de clientes
plt.figure(figsize=(10, 6))

perfiles["Clientes"].plot(
    kind="bar",
    color="steelblue"
)

plt.title("Distribucion de clientes por cluster")
plt.xlabel("Cluster")
plt.ylabel("Cantidad de clientes")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    resultados / "distribucion_kmeans.png",
    dpi=300
)

plt.close()

print("\nModelo y graficas generados correctamente.")