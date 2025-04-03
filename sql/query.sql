-- 慢查询示例1：不合理的JOIN顺序（从大表开始）
EXPLAIN ANALYZE
SELECT
    u.name,
    COUNT(*) as order_count,
    SUM(od.quantity) as total_quantity
FROM order_details od   -- 从最大的表(1M行)开始
JOIN orders o ON od.order_id = o.id  -- 然后连接orders(100K行)
JOIN users u ON o.user_id = u.id     -- 最后连接users(1K行)
WHERE od.quantity > 5
GROUP BY u.name
ORDER BY total_quantity DESC;

-- 慢查询示例2：没有使用索引的条件
EXPLAIN ANALYZE
SELECT
    o.id,
    o.order_date,
    u.name
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE YEAR(o.order_date) = 2020   -- 使用函数导致无法使用索引
AND u.name LIKE '%User1%';        -- 前缀模糊匹配导致无法使用索引

-- 慢查询示例3：不必要的笛卡尔积
EXPLAIN ANALYZE
SELECT
    u1.name as user1,
    u2.name as user2,
    o.order_date
FROM users u1
CROSS JOIN users u2   -- 产生 1K x 1K 的笛卡尔积
JOIN orders o ON o.user_id = u1.id
WHERE u1.city = u2.city
AND o.order_date >= '2020-01-01';

-- 慢查询示例4：低效的子查询
EXPLAIN ANALYZE
SELECT
    u.name,
    u.city,
    (SELECT COUNT(*)
     FROM orders o
     WHERE o.user_id = u.id) as order_count,
    (SELECT SUM(od.quantity)
     FROM order_details od
     JOIN orders o ON od.order_id = o.id
     WHERE o.user_id = u.id) as total_quantity
FROM users u
WHERE u.city = 'New York';

-- 慢查询示例5：GROUP BY 后的排序和分页
EXPLAIN ANALYZE
SELECT
    u.city,
    DATE(o.order_date) as order_day,
    COUNT(DISTINCT u.id) as user_count,
    COUNT(DISTINCT o.id) as order_count,
    SUM(od.quantity) as total_quantity
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
LEFT JOIN order_details od ON o.id = od.order_id
GROUP BY u.city, DATE(o.order_date)
HAVING total_quantity > 100
ORDER BY total_quantity DESC
LIMIT 10000, 10;  -- 大偏移量的分页

-- 慢查询示例6：IN子查询
EXPLAIN ANALYZE
SELECT *
FROM orders o
WHERE user_id IN (
    SELECT id
    FROM users
    WHERE city IN (
        SELECT DISTINCT city
        FROM users
        WHERE id IN (
            SELECT user_id
            FROM orders
            GROUP BY user_id
            HAVING COUNT(*) > 5
        )
    )
);
