
from pathlib import Path
from decimal import Decimal
from dotenv import load_dotenv
import mysql.connector
import pandas as pd
import os

# Rutas del proyecto
proyecto = Path(__file__).resolve().parent.parent
load_dotenv(proyecto / ".env")

archivo = proyecto / "datos_limpios" / "dataset_limpio.csv"

print("Leyendo dataset limpio...")
df = pd.read_csv(
    archivo,
    dtype={
        "InvoiceNo": "string",
        "StockCode": "string",
        "Description": "string",
        "CustomerID": "string",
        "Country": "string"
    },
    parse_dates=["InvoiceDate"]
)

df["CustomerID"] = df["CustomerID"].fillna("SIN_IDENTIFICAR")


# Estandarizar los codigos de productos
# para evitar diferencias entre mayusculas y minusculas
df["StockCode"] = (
    df["StockCode"]
    .astype("string")
    .str.strip()
    .str.upper()
)

print("Codigos de productos normalizados.")


print("Registros encontrados:", len(df))

conexion = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", "3306")),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)

cursor = conexion.cursor()

try:
    # Comprobar que no haya datos cargados anteriormente
    cursor.execute("SELECT COUNT(*) FROM fact_ventas")
    existentes = cursor.fetchone()[0]

    if existentes > 0:
        raise RuntimeError(
            "La tabla fact_ventas ya contiene datos. "
            "No se realizará una carga duplicada."
        )

    # DIMENSION PRODUCTO
    print("Cargando productos...")

    productos = (
        df[["StockCode", "Description"]]
        .drop_duplicates(subset=["StockCode"])
    )

    cursor.executemany(
        """
        INSERT INTO dim_producto
        (codigo_producto, descripcion)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE
        descripcion = VALUES(descripcion)
        """,
        list(productos.itertuples(index=False, name=None))
    )

    # DIMENSION CLIENTE
    print("Cargando clientes...")

    clientes = df["CustomerID"].drop_duplicates().tolist()

    cursor.executemany(
        """
        INSERT IGNORE INTO dim_cliente (codigo_cliente)
        VALUES (%s)
        """,
        [(cliente,) for cliente in clientes]
    )

    # DIMENSION PAIS
    print("Cargando países...")

    paises = df["Country"].drop_duplicates().tolist()

    cursor.executemany(
        """
        INSERT IGNORE INTO dim_pais (nombre_pais)
        VALUES (%s)
        """,
        [(pais,) for pais in paises]
    )

    # DIMENSION TIEMPO
    print("Cargando fechas...")

    fechas = sorted(df["InvoiceDate"].dt.date.unique())

    cursor.executemany(
        """
        INSERT IGNORE INTO dim_tiempo
        (fecha, anio, mes, dia)
        VALUES (%s, %s, %s, %s)
        """,
        [
            (fecha, fecha.year, fecha.month, fecha.day)
            for fecha in fechas
        ]
    )

    # Obtener los identificadores generados
    cursor.execute(
        "SELECT codigo_producto, id_producto FROM dim_producto"
    )
    mapa_productos = dict(cursor.fetchall())

    cursor.execute(
        "SELECT codigo_cliente, id_cliente FROM dim_cliente"
    )
    mapa_clientes = dict(cursor.fetchall())

    cursor.execute(
        "SELECT nombre_pais, id_pais FROM dim_pais"
    )
    mapa_paises = dict(cursor.fetchall())

    cursor.execute(
        "SELECT fecha, id_tiempo FROM dim_tiempo"
    )
    mapa_fechas = dict(cursor.fetchall())

    # TABLA DE HECHOS
    print("Cargando ventas por lotes...")

    sql_ventas = """
        INSERT INTO fact_ventas
        (numero_factura, fecha_hora, id_producto,
         id_cliente, id_pais, id_tiempo,
         cantidad, precio_unitario, total_venta)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    lote = []
    tamano_lote = 2000
    insertados = 0

    for fila in df.itertuples(index=False):
        fecha = fila.InvoiceDate.to_pydatetime()

        registro = (
            str(fila.InvoiceNo),
            fecha,
            mapa_productos[str(fila.StockCode)],
            mapa_clientes[str(fila.CustomerID)],
            mapa_paises[str(fila.Country)],
            mapa_fechas[fecha.date()],
            int(fila.Quantity),
            Decimal(str(fila.UnitPrice)),
            Decimal(str(fila.TotalVenta))
        )

        lote.append(registro)

        if len(lote) >= tamano_lote:
            cursor.executemany(sql_ventas, lote)
            conexion.commit()
            insertados += len(lote)
            lote.clear()
            print(f"Registros cargados: {insertados}")

    if lote:
        cursor.executemany(sql_ventas, lote)
        conexion.commit()
        insertados += len(lote)

    print("\n========== CARGA FINALIZADA ==========")
    print("Registros cargados:", insertados)

    cursor.execute("SELECT COUNT(*) FROM fact_ventas")
    total_mysql = cursor.fetchone()[0]

    print("Registros en MySQL:", total_mysql)

    if total_mysql != len(df):
        print("ADVERTENCIA: Los totales no coinciden.")
    else:
        print("VALIDACION EXITOSA: todos los registros coinciden.")


except Exception as error:
    import traceback

    conexion.rollback()

    print("\nERROR DURANTE LA CARGA:")
    print("Tipo de error:", type(error).__name__)
    print("Detalle:", error)

    traceback.print_exc()

    print("Si la carga de ventas quedó incompleta,")
    print("no ejecutes otra vez el programa sin revisar MySQL.")

finally:
    cursor.close()
    conexion.close()
