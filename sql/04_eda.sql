-- Exploratory queries. Each block starts with "-- name: <query_name>" so
-- src/fraud/eda.py can run them one at a time and save the results.

-- name: overview
SELECT
    COUNT(*)                                    AS transactions,
    SUM(is_fraud)                               AS frauds,
    ROUND(AVG(is_fraud) * 100, 2)               AS fraud_rate_pct,
    ROUND(SUM(amount), 0)                       AS total_amount,
    ROUND(SUM(amount) FILTER (WHERE is_fraud = 1), 0) AS fraud_amount,
    MIN(event_ts)                               AS first_ts,
    MAX(event_ts)                               AS last_ts,
    COUNT(DISTINCT card_key)                    AS distinct_cards
FROM transactions;

-- name: fraud_by_week
-- Fraud rate drifts over time, which is why the train/test split must be time-based.
SELECT
    CAST(date_trunc('week', event_ts) AS DATE)  AS week,
    COUNT(*)                                    AS transactions,
    ROUND(AVG(is_fraud) * 100, 2)               AS fraud_rate_pct
FROM transactions
GROUP BY ALL
ORDER BY week;

-- name: fraud_by_hour
SELECT
    hour_of_day,
    COUNT(*)                                    AS transactions,
    ROUND(AVG(is_fraud) * 100, 2)               AS fraud_rate_pct
FROM features
GROUP BY ALL
ORDER BY hour_of_day;

-- name: fraud_by_product
SELECT
    product_cd,
    COUNT(*)                                    AS transactions,
    SUM(is_fraud)                               AS frauds,
    ROUND(AVG(is_fraud) * 100, 2)               AS fraud_rate_pct
FROM transactions
GROUP BY ALL
ORDER BY fraud_rate_pct DESC;

-- name: fraud_by_amount_decile
WITH banded AS (
    SELECT *, NTILE(10) OVER (ORDER BY amount) AS amount_decile
    FROM transactions
)
SELECT
    amount_decile,
    ROUND(MIN(amount), 2)                       AS min_amount,
    ROUND(MAX(amount), 2)                       AS max_amount,
    ROUND(AVG(is_fraud) * 100, 2)               AS fraud_rate_pct
FROM banded
GROUP BY ALL
ORDER BY amount_decile;

-- name: fraud_by_card_type
SELECT
    coalesce(card_network, 'unknown')           AS card_network,
    coalesce(card_type, 'unknown')              AS card_type,
    COUNT(*)                                    AS transactions,
    ROUND(AVG(is_fraud) * 100, 2)               AS fraud_rate_pct
FROM transactions
GROUP BY ALL
HAVING COUNT(*) >= 100
ORDER BY transactions DESC;

-- name: fraud_by_identity
SELECT
    has_identity,
    coalesce(device_type, 'none')               AS device_type,
    COUNT(*)                                    AS transactions,
    ROUND(AVG(is_fraud) * 100, 2)               AS fraud_rate_pct
FROM transactions
GROUP BY ALL
ORDER BY has_identity, transactions DESC;

-- name: fraud_by_email_domain
SELECT
    coalesce(p_email_domain, 'missing')         AS p_email_domain,
    COUNT(*)                                    AS transactions,
    ROUND(AVG(is_fraud) * 100, 2)               AS fraud_rate_pct
FROM transactions
GROUP BY ALL
HAVING COUNT(*) >= 1000
ORDER BY fraud_rate_pct DESC
LIMIT 15;

-- name: fraud_by_card_velocity
-- Does fraud cluster on cards that were busy in the previous 24 hours?
SELECT
    CASE
        WHEN card_txn_count_24h = 0 THEN '0'
        WHEN card_txn_count_24h = 1 THEN '1'
        WHEN card_txn_count_24h <= 4 THEN '2-4'
        WHEN card_txn_count_24h <= 9 THEN '5-9'
        ELSE '10+'
    END                                         AS prior_txns_24h,
    COUNT(*)                                    AS transactions,
    ROUND(AVG(is_fraud) * 100, 2)               AS fraud_rate_pct
FROM features
GROUP BY ALL
ORDER BY MIN(card_txn_count_24h);
