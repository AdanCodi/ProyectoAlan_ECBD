
USE dw_online_retail;

-- 1. Cantidad de registros de cada tabla
SELECT 'Ventas' AS Tabla, COUNT(*) AS Registros
FROM fact_ventas
UNION ALL
SELECT 'Productos', COUNT(*) FROM dim_producto
UNION ALL
SELECT 'Clientes', COUNT(*) FROM dim_cliente
UNION ALL
SELECT 'Paises', COUNT(*) FROM dim_pais
UNION ALL
SELECT 'Fechas', COUNT(*) FROM dim_tiempo;

-- 2. Importe total de las ventas
SELECT
    COUNT(*) AS total_registros,
    ROUND(SUM(total_venta), 2) AS importe_total
FROM fact_ventas;

-- 3. Comprobar registros de ejemplo
SELECT
    v.numero_factura,
    p.descripcion,
    v.cantidad,
    v.precio_unitario,
    v.total_venta
FROM fact_ventas v
JOIN dim_producto p
    ON v.id_producto = p.id_producto
LIMIT 10;
