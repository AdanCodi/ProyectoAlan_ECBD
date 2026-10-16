
from pathlib import Path
import pandas as pd

proyecto = Path(__file__).resolve().parent.parent
archivo = proyecto / "datos_limpios" / "dataset_limpio.csv"

df = pd.read_csv(
    archivo,
    dtype={"StockCode": "string"}
)

coincidencias = df[
    df["StockCode"].str.contains(
        "15056", case=False, na=False
    )
]

print("REGISTROS CON CODIGO 15056:")
print(coincidencias["StockCode"].value_counts())

print("\nDESCRIPCIONES:")
print(
    coincidencias[
        ["StockCode", "Description"]
    ].drop_duplicates().to_string(index=False)
)
