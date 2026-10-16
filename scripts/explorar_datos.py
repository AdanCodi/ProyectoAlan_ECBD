
from pathlib import Path
import pandas as pd

# PASO 1: Localizar el dataset original
carpeta_proyecto = Path(__file__).resolve().parent.parent

ruta_excel = carpeta_proyecto / "datos_originales" / "Online Retail.xlsx"

# PASO 2: Leer el archivo Excel
print("Cargando dataset original...")
df = pd.read_excel(ruta_excel, engine="openpyxl")

# PASO 3: Mostrar información general
print("\n========== INFORMACION GENERAL ==========")
print("Cantidad de registros:", len(df))
print("Cantidad de columnas:", len(df.columns))

# PASO 4: Mostrar nombres de columnas
print("\n========== COLUMNAS ==========")
print(df.columns.tolist())

# PASO 5: Mostrar primeros registros
print("\n========== PRIMEROS 5 REGISTROS ==========")
print(df.head().to_string())

# PASO 6: Detectar valores nulos
print("\n========== VALORES NULOS ==========")
print(df.isnull().sum())

# PASO 7: Detectar registros duplicados
print("\n========== REGISTROS DUPLICADOS ==========")
print(df.duplicated().sum())

# PASO 8: Mostrar tipos de datos
print("\n========== TIPOS DE DATOS ==========")
print(df.dtypes)

print("\nExploracion completada correctamente.")
