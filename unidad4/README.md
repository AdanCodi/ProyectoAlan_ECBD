
# Unidad 4 - Analisis no supervisado

## Proyecto Online Retail

Universidad Tecnologica de Coahuila.

## Objetivo

Aplicar algoritmos de aprendizaje no supervisado
para identificar segmentos de clientes mediante
sus comportamientos de compra.

## Dataset

Online Retail, obtenido del repositorio UCI
Machine Learning Repository y procesado
previamente en la Unidad 2.

## Preparacion de datos

- Registros procesados: 524878.
- Registros utilizados: 392692.
- Clientes identificados: 4338.
- Transformacion logaritmica.
- Estandarizacion mediante StandardScaler.

## Variables RFM

- Recencia: dias desde la ultima compra.
- Frecuencia: cantidad de compras realizadas.
- Monetario: importe acumulado de ventas validas.

## Algoritmos implementados

- K-Means.
- DBSCAN.
- PCA.

## Resultados

### K-Means

- Numero optimo de clusters: 2.
- Silhouette: 0.4325.
- Davies-Bouldin: 0.8913.
- Calinski-Harabasz: 4376.8324.

### DBSCAN

- Eps: 0.6.
- Min_samples: 5.
- Clusters: 2.
- Clientes identificados como ruido: 36.
- Silhouette: 0.5724, sin considerar el ruido.

### PCA

- Componentes originales: 3.
- Componentes seleccionados: 2.
- Varianza conservada: 93.86%.
- Error de reconstruccion: 0.061402.

## Instalacion de dependencias

```bash
python -m pip install -r unidad4/requirements.txt
```

## Ejecucion

Ejecutar desde la raiz del repositorio.

```bash
python unidad4/scripts/preparar_rfm.py
python unidad4/scripts/modelo_kmeans.py
python unidad4/scripts/modelo_dbscan.py
python unidad4/scripts/modelo_pca.py
python unidad4/scripts/comparar_modelos.py
```

## Conclusiones

K-Means permite generar segmentos comerciales
utiles, DBSCAN facilita la identificacion de
clientes atipicos y PCA permite reducir la
dimensionalidad de los datos conservando
la mayor parte de su varianza.

Los resultados corresponden a un analisis
exploratorio sobre los clientes disponibles.
