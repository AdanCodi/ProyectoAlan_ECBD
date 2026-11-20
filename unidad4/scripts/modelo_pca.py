from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# Rutas del proyecto
proyecto = Path(__file__).resolve().parents[2]
resultados = proyecto / "unidad4" / "resultados"
modelos = proyecto / "unidad4" / "modelos"

# Cargar datos
escalados = pd.read_csv(
    resultados / "datos_rfm_escalados.csv"
)

kmeans = pd.read_csv(
    resultados / "clientes_kmeans.csv"
)

dbscan = pd.read_csv(
    resultados / "clientes_dbscan.csv"
)

variables = ["Recencia", "Frecuencia", "Monetario"]
X = escalados[variables]

# Evaluar diferentes cantidades de componentes
evaluaciones = []

print("Evaluando componentes principales...")

for n in range(1, 4):

    modelo = PCA(n_components=n)

    transformados = modelo.fit_transform(X)

    reconstruidos = modelo.inverse_transform(
        transformados
    )

    evaluaciones.append({
        "Componentes": n,
        "Varianza_Explicada":
            modelo.explained_variance_ratio_.sum(),

        "Porcentaje_Varianza":
            modelo.explained_variance_ratio_.sum() * 100,

        "Error_Reconstruccion":
            mean_squared_error(X, reconstruidos)
    })

metricas = pd.DataFrame(evaluaciones)

print("\n===== EVALUACION DE PCA =====")
print(metricas.round(6).to_string(index=False))

# Seleccionar el menor numero de componentes
# que conserve al menos 90% de la varianza
mejor_n = int(
    metricas.loc[
        metricas["Porcentaje_Varianza"] >= 90,
        "Componentes"
    ].min()
)

print("\nComponentes recomendados:", mejor_n)

# Entrenar modelo optimizado
modelo_optimo = PCA(n_components=mejor_n)
modelo_optimo.fit(X)

joblib.dump(
    modelo_optimo,
    modelos / "pca_optimizado.joblib"
)

# Modelo PCA para visualizacion en dos dimensiones
modelo_2d = PCA(n_components=2)
componentes = modelo_2d.fit_transform(X)

pca_df = pd.DataFrame(
    componentes,
    columns=["PC1", "PC2"]
)

pca_df["CustomerID"] = escalados["CustomerID"]

# Incorporar etiquetas de ambos algoritmos
pca_df = pca_df.merge(
    kmeans[["CustomerID", "Cluster"]].rename(
        columns={"Cluster": "Cluster_KMeans"}
    ),
    on="CustomerID"
)

pca_df = pca_df.merge(
    dbscan[["CustomerID", "Cluster"]].rename(
        columns={"Cluster": "Cluster_DBSCAN"}
    ),
    on="CustomerID"
)

print("\n===== VARIANZA EXPLICADA =====")
print("PC1:", modelo_2d.explained_variance_ratio_[0])
print("PC2:", modelo_2d.explained_variance_ratio_[1])

print(
    "Varianza total conservada:",
    round(
        modelo_2d.explained_variance_ratio_.sum() * 100,
        2
    ),
    "%"
)

# Cargas de los componentes principales
cargas = pd.DataFrame(
    modelo_2d.components_,
    columns=variables,
    index=["PC1", "PC2"]
)

print("\n===== COMPONENTES PRINCIPALES =====")
print(cargas.round(4).to_string())

# Guardar resultados
metricas.to_csv(
    resultados / "metricas_pca.csv",
    index=False
)

pca_df.to_csv(
    resultados / "clientes_pca.csv",
    index=False
)

cargas.to_csv(
    resultados / "componentes_pca.csv"
)

joblib.dump(
    modelo_2d,
    modelos / "pca_2d.joblib"
)

# Grafica de varianza explicada
plt.figure(figsize=(9, 5))

plt.plot(
    metricas["Componentes"],
    metricas["Porcentaje_Varianza"],
    marker="o",
    color="royalblue"
)

plt.axhline(
    90,
    color="red",
    linestyle="--",
    label="Umbral del 90%"
)

plt.title("Varianza acumulada - PCA")
plt.xlabel("Numero de componentes")
plt.ylabel("Varianza explicada (%)")
plt.xticks([1, 2, 3])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    resultados / "varianza_pca.png",
    dpi=300
)

plt.close()

# Grafica de K-Means con PCA
plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster_KMeans",
    palette="Set1",
    alpha=0.6,
    s=20
)

plt.title("Segmentacion K-Means visualizada con PCA")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.tight_layout()

plt.savefig(
    resultados / "pca_kmeans.png",
    dpi=300
)

plt.close()

# Grafica de DBSCAN con PCA
plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster_DBSCAN",
    palette="Dark2",
    alpha=0.6,
    s=20
)

plt.title("Segmentacion DBSCAN visualizada con PCA")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.tight_layout()

plt.savefig(
    resultados / "pca_dbscan.png",
    dpi=300
)

plt.close()

print("\nModelos PCA y graficas generados correctamente.")