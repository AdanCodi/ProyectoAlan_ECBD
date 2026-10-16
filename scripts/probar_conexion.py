
from pathlib import Path
from dotenv import load_dotenv
import mysql.connector
import os

# Localizar la carpeta principal del proyecto
proyecto = Path(__file__).resolve().parent.parent

# Cargar las variables de configuración
load_dotenv(proyecto / ".env")

try:
    # Establecer conexión con MySQL
    conexion = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

    if conexion.is_connected():
        print("================================")
        print("CONEXION EXITOSA CON MYSQL")
        print("================================")

        cursor = conexion.cursor()

        cursor.execute("SELECT DATABASE()")
        resultado = cursor.fetchone()

        print("Base de datos:", resultado[0])

        cursor.execute("SHOW TABLES")

        print("\nTablas disponibles:")

        for tabla in cursor.fetchall():
            print("-", tabla[0])

        cursor.close()

    conexion.close()

except mysql.connector.Error as error:
    print("ERROR AL CONECTAR CON MYSQL:")
    print(error)
