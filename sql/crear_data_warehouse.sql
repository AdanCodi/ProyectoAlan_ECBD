
USE dw_online_retail;

-- Dimensión de productos
CREATE TABLE IF NOT EXISTS dim_producto (
    id_producto INT AUTO_INCREMENT PRIMARY KEY,
    codigo_producto VARCHAR(30) NOT NULL UNIQUE,
    descripcion VARCHAR(255)
);

-- Dimensión de clientes
CREATE TABLE IF NOT EXISTS dim_cliente (
    id_cliente INT AUTO_INCREMENT PRIMARY KEY,
    codigo_cliente VARCHAR(30) NOT NULL UNIQUE
);

-- Dimensión de países
CREATE TABLE IF NOT EXISTS dim_pais (
    id_pais INT AUTO_INCREMENT PRIMARY KEY,
    nombre_pais VARCHAR(100) NOT NULL UNIQUE
);

-- Dimensión de tiempo
CREATE TABLE IF NOT EXISTS dim_tiempo (
    id_tiempo INT AUTO_INCREMENT PRIMARY KEY,
    fecha DATE NOT NULL UNIQUE,
    anio SMALLINT,
    mes TINYINT,
    dia TINYINT
);

-- Tabla de hechos de ventas
CREATE TABLE IF NOT EXISTS fact_ventas (
    id_venta BIGINT AUTO_INCREMENT PRIMARY KEY,
    numero_factura VARCHAR(30) NOT NULL,
    fecha_hora DATETIME NOT NULL,
    id_producto INT NOT NULL,
    id_cliente INT NOT NULL,
    id_pais INT NOT NULL,
    id_tiempo INT NOT NULL,
    cantidad INT NOT NULL,
    precio_unitario DECIMAL(12,2) NOT NULL,
    total_venta DECIMAL(15,2) NOT NULL,

    FOREIGN KEY (id_producto)
        REFERENCES dim_producto(id_producto),

    FOREIGN KEY (id_cliente)
        REFERENCES dim_cliente(id_cliente),

    FOREIGN KEY (id_pais)
        REFERENCES dim_pais(id_pais),

    FOREIGN KEY (id_tiempo)
        REFERENCES dim_tiempo(id_tiempo)
);

-- Verificar tablas creadas
SHOW TABLES;
