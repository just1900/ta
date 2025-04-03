
DROP TABLE IF EXISTS order_details;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS users;

-- 用户表（1K 条数据）
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50),
    city VARCHAR(50)
);

-- 订单表（100K 条数据）
CREATE TABLE orders (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    order_date DATE,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 订单详情表（1M 条数据）
CREATE TABLE order_details (
    id INT PRIMARY KEY AUTO_INCREMENT,
    order_id INT,
    product_id INT,
    quantity INT,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);



INSERT INTO users (name, city)
SELECT
    CONCAT('User', FLOOR(RAND() * 100000)),
    CASE WHEN RAND() < 0.1 THEN 'New York' ELSE 'Other' END
FROM
    (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4) a, -- 4
    (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4) b, -- 4x4=16
    (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4) c, -- 16x4=64
    (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4) d; -- 64x4=256 rows (adjust to reach 1000)


INSERT INTO orders (user_id, order_date)
SELECT
    u.id,
    DATE('2020-01-01') + INTERVAL FLOOR(RAND() * 1095) DAY
FROM
    users u
CROSS JOIN
    (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5) a, -- 5
    (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5) b, -- 5x5=25 rows per user
    (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5) c; -- 5*5*5=125 rows per user


INSERT INTO order_details (order_id, product_id, quantity)
SELECT
    o.id,
    FLOOR(RAND() * 1000) + 1,
    FLOOR(RAND() * 10) + 1
FROM
    orders o
CROSS JOIN
    (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5) a,
    (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5) b,
    (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5) c;

-- CREATE INDEX idx_users_city ON users(city);
-- CREATE INDEX idx_orders_user_id ON orders(user_id);
-- CREATE INDEX idx_order_details_order_id ON order_details(order_id);

