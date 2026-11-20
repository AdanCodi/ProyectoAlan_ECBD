from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Rutas del proyecto
proyecto = Path(__file__).resolve().parents[2]
archivo = proyecto / "datos_limpios" / "dataset_limpio.csv"
resultados = proyecto / "unidad4" / "resultados"
modelos = proyecto / "unidad4" / "modelos"

resultados.mkdir(exist_ok=True)
modelos.mkdir(exist_ok=True)

# Cargar dataset limpio de la Unidad 2
print("Cargando dataset limpio...")

df = pd.read_csv(
    archivo,
    dtype={"InvoiceNo": str},
    parse_dates=["InvoiceDate"]
)

print("Registros originales:", len(df))

# Excluir registros sin identificador de cliente
sin_cliente = df["CustomerID"].isna().sum()

df = df.dropna(subset=["CustomerID"]).copy()
df["CustomerID"] = df["CustomerID"].astype(int)

# Fecha de referencia
fecha_referencia = df["InvoiceDate"].max().normalize()
fecha_referencia += pd.Timedelta(days=1)

# Construir variables RFM
rfm = df.groupby("CustomerID").agg(
    Recencia=(
        "InvoiceDate",
        lambda x: (
            fecha_referencia - x.max().normalize()
        ).days
    ),
    Frecuencia=("InvoiceNo", "nunique"),
    Monetario=("TotalVenta", "sum")
).reset_index()

# Transformacion logaritmica y escalamiento
variables = ["Recencia", "Frecuencia", "Monetario"]

rfm_log = np.log1p(rfm[variables])

escalador = StandardScaler()
datos_escalados = escalador.fit_transform(rfm_log)

rfm_escalado = pd.DataFrame(
    datos_escalados,
    columns=variables
)

rfm_escalado.insert(0, "CustomerID", rfm["CustomerID"])

# Guardar resultados
rfm.to_csv(
    resultados / "datos_rfm.csv",
    index=False
)

rfm_escalado.to_csv(
    resultados / "datos_rfm_escalados.csv",
    index=False
)

joblib.dump(
    escalador,
    modelos / "escalador_rfm.joblib"
)

# Mostrar resultados
print("\n===== PREPARACION DE DATOS RFM =====")
print("Registros sin cliente excluidos:", sin_cliente)
print("Registros utilizados:", len(df))
print("Clientes identificados:", len(rfm))
print("Fecha de referencia:", fecha_referencia.date())

print("\n===== PRIMEROS 10 CLIENTES =====")
print(rfm.head(10).to_string(index=False))

print("\n===== ESTADISTICAS DESCRIPTIVAS =====")
print(rfm[variables].describe().round(2))

print("\n===== VALIDACION =====")
print("Valores nulos:", rfm.isna().sum().sum())
print("Clientes duplicados:", rfm["CustomerID"].duplicated().sum())

print("\nArchivos generados:")
print("- datos_rfm.csv")
print("- datos_rfm_escalados.csv")
print("- escalador_rfm.joblib")

print("\nPreparacion RFM completada correctamente.")