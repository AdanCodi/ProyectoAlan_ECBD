from pathlib import Path
import pandas as pd

# Rutas del proyecto
proyecto = Path(__file__).resolve().parents[2]
archivo = proyecto / "datos_limpios" / "dataset_limpio.csv"
resultados = proyecto / "unidad3" / "resultados"

resultados.mkdir(exist_ok=True)

# Leer el dataset de la Unidad 2
print("Cargando dataset limpio...")

df = pd.read_csv(
    archivo,
    parse_dates=["InvoiceDate"]
)

print("Registros originales:", len(df))

# Agrupar las ventas por fecha
df["Fecha"] = df["InvoiceDate"].dt.normalize()

ventas = (
    df.groupby("Fecha")["TotalVenta"]
    .sum()
    .to_frame("Ventas")
    .sort_index()
)

# Crear un calendario continuo
ventas = ventas.asfreq("D", fill_value=0)

# Variables predictoras históricas
ventas["Ventas_ayer"] = ventas["Ventas"].shift(1)
ventas["Ventas_hace_7_dias"] = ventas["Ventas"].shift(7)

ventas["Promedio_7_dias"] = (
    ventas["Ventas"].shift(1).rolling(7).mean()
)

ventas["Promedio_14_dias"] = (
    ventas["Ventas"].shift(1).rolling(14).mean()
)

# Variables de calendario
ventas["Dia_semana"] = ventas.index.dayofweek
ventas["Mes"] = ventas.index.month
ventas["Fin_semana"] = (
    ventas.index.dayofweek >= 5
).astype(int)

# Eliminar filas sin suficiente historial
ventas = ventas.dropna().copy()

# Separar cronológicamente los datos
limite = int(len(ventas) * 0.8)

entrenamiento = ventas.iloc[:limite].copy()
prueba = ventas.iloc[limite:].copy()

# Definir el umbral utilizando solo el entrenamiento
umbral = entrenamiento["Ventas"].median()

# Variable objetivo de clasificación
ventas["Demanda_alta"] = (
    ventas["Ventas"] >= umbral
).astype(int)

entrenamiento = ventas.iloc[:limite].copy()
prueba = ventas.iloc[limite:].copy()

# Guardar los conjuntos de datos
ventas.to_csv(resultados / "datos_supervisados.csv")
entrenamiento.to_csv(resultados / "entrenamiento.csv")
prueba.to_csv(resultados / "prueba.csv")

# Mostrar resultados
print("\n===== PREPARACION DE DATOS =====")
print("Dias disponibles:", len(ventas))
print("Datos de entrenamiento:", len(entrenamiento))
print("Datos de prueba:", len(prueba))
print(f"Umbral de demanda alta: GBP {umbral:.2f}")

print("\nPeriodo de entrenamiento:")
print(entrenamiento.index.min(), "-", entrenamiento.index.max())

print("\nPeriodo de prueba:")
print(prueba.index.min(), "-", prueba.index.max())

print("\nDistribucion de demanda:")
print(ventas["Demanda_alta"].value_counts())

print("\nPrimeros registros:")
print(ventas.head().to_string())

print("\nPreparacion completada correctamente.")