
from pathlib import Path
import pandas as pd

# 1. Rutas del proyecto
proyecto = Path(__file__).resolve().parent.parent
origen = proyecto / "datos_originales" / "Online Retail.xlsx"
destino = proyecto / "datos_limpios"

destino.mkdir(exist_ok=True)

# 2. EXTRAER: cargar datos originales
print("Cargando dataset original...")
df = pd.read_excel(origen, engine="openpyxl")

originales = len(df)
print(f"Registros originales: {originales}")

# 3. TRANSFORMAR: eliminar duplicados exactos
duplicados = df.duplicated().sum()
df = df.drop_duplicates().copy()

print(f"Duplicados eliminados: {duplicados}")

# 4. Estandarizar campos de texto
df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
df["StockCode"] = df["StockCode"].astype("string").str.strip()

df["Description"] = (
    df["Description"]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
    .fillna("SIN DESCRIPCION")
)

df["Country"] = (
    df["Country"].astype("string").str.strip()
)

# 5. Identificadores de clientes
# No inventamos identificadores faltantes
df["CustomerID"] = (
    df["CustomerID"]
    .astype("Int64")
    .astype("string")
)

# 6. Validar fechas y números
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df["UnitPrice"] = pd.to_numeric(
    df["UnitPrice"], errors="coerce"
)

# 7. Separar operaciones de venta y otras operaciones
cancelaciones = df["InvoiceNo"].str.upper().str.startswith(
    "C", na=False
)

ventas_validas = (
    ~cancelaciones
    & df["Quantity"].gt(0)
    & df["UnitPrice"].gt(0)
    & df["InvoiceDate"].notna()
    & df["InvoiceNo"].notna()
    & df["StockCode"].notna()
)

ventas = df.loc[ventas_validas].copy()
excluidos = df.loc[~ventas_validas].copy()

# 8. Crear nuevas columnas para análisis
ventas["TotalVenta"] = (
    ventas["Quantity"] * ventas["UnitPrice"]
).round(2)

ventas["Anio"] = ventas["InvoiceDate"].dt.year
ventas["Mes"] = ventas["InvoiceDate"].dt.month

# 9. CARGAR: guardar resultados en CSV
ventas.to_csv(
    destino / "dataset_limpio.csv",
    index=False,
    encoding="utf-8-sig"
)

excluidos.to_csv(
    destino / "registros_excluidos.csv",
    index=False,
    encoding="utf-8-sig"
)

# 10. Mostrar reporte del procesamiento
print("\n========== REPORTE DE LIMPIEZA ==========")
print(f"Registros originales: {originales}")
print(f"Duplicados eliminados: {duplicados}")
print(f"Operaciones excluidas: {len(excluidos)}")
print(f"Ventas válidas: {len(ventas)}")
print(f"Registros finales: {len(ventas)}")

print("\n========== VALIDACIÓN FINAL ==========")
print("Duplicados:", ventas.duplicated().sum())
print("Nulos por columna:")
print(ventas.isnull().sum())

print("\nArchivos generados:")
print("datos_limpios/dataset_limpio.csv")
print("datos_limpios/registros_excluidos.csv")

print("\nProceso de limpieza terminado.")
