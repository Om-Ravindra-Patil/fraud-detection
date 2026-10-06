-- Exploratory queries. Each block starts with "-- name: <query_name>" so
-- src/fraud/eda.py can run them one at a time and save the results.

-- name: overview
SELECT
    COUNT(*)                                            AS transactions,
    SUM(is_fraud)                                       AS frauds,
    ROUND(AVG(is_fraud) * 100, 3)                       AS fraud_rate_pct,
    ROUND(SUM(amount), 0)                               AS total_amount,
    ROUND(SUM(amount) FILTER (WHERE is_fraud = 1), 0)   AS fraud_amount,
    MIN(step)                                           AS first_step,
    MAX(step)                                           AS last_step,
    COUNT(DISTINCT name_orig)                           AS distinct_origin_accounts,
    COUNT(DISTINCT name_dest)                           AS distinct_destination_accounts
FROM transactions;

-- name: fraud_by_type
-- Fraud only appears in TRANSFER and CASH_OUT, which is why the model is scoped to them.
SELECT
    type,
    COUNT(*)                                            AS transactions,
    SUM(is_fraud)                                       AS frauds,
    ROUND(AVG(is_fraud) * 100, 3)                       AS fraud_rate_pct,
    ROUND(AVG(amount), 0)                               AS avg_amount
FROM transactions
GROUP BY ALL
ORDER BY frauds DESC, transactions DESC;

-- name: fraud_by_day
-- Volume and fraud rate over time. A shift here is why the split must be time-based.
SELECT
    day,
    COUNT(*)                                            AS transactions,
    SUM(is_fraud)                                       AS frauds,
    ROUND(AVG(is_fraud) * 100, 3)                       AS fraud_rate_pct
FROM features
GROUP BY ALL
ORDER BY day;

-- name: fraud_by_hour
SELECT
    hour_of_day,
    COUNT(*)                                            AS transactions,
    ROUND(AVG(is_fraud) * 100, 3)                       AS fraud_rate_pct
FROM features
GROUP BY ALL
ORDER BY hour_of_day;

-- name: fraud_by_amount_decile
WITH banded AS (
    SELECT *, NTILE(10) OVER (ORDER BY amount) AS amount_decile
    FROM features
)
SELECT
    amount_decile,
    ROUND(MIN(amount), 0)                               AS min_amount,
    ROUND(MAX(amount), 0)                               AS max_amount,
    ROUND(AVG(is_fraud) * 100, 3)                       AS fraud_rate_pct
FROM banded
GROUP BY ALL
ORDER BY amount_decile;

-- name: fraud_by_balance_pattern
-- Does fraud tend to empty the sender's account?
SELECT
    empties_orig_account,
    orig_balance_zero,
    COUNT(*)                                            AS transactions,
    SUM(is_fraud)                                       AS frauds,
    ROUND(AVG(is_fraud) * 100, 3)                       AS fraud_rate_pct
FROM features
GROUP BY ALL
ORDER BY fraud_rate_pct DESC;

-- name: fraud_by_dest_velocity
-- Does fraud cluster on destination accounts that received money in the previous 24 hours?
SELECT
    CASE
        WHEN dest_txn_count_24h = 0 THEN '0'
        WHEN dest_txn_count_24h = 1 THEN '1'
        WHEN dest_txn_count_24h <= 4 THEN '2-4'
        WHEN dest_txn_count_24h <= 9 THEN '5-9'
        ELSE '10+'
    END                                                 AS dest_txns_prior_24h,
    COUNT(*)                                            AS transactions,
    ROUND(AVG(is_fraud) * 100, 3)                       AS fraud_rate_pct
FROM features
GROUP BY ALL
ORDER BY MIN(dest_txn_count_24h);

-- name: existing_rule_performance
-- How well does PaySim's built-in rule (isFlaggedFraud) do? This is the bar the model must beat.
SELECT
    SUM(is_flagged_fraud)                               AS flagged,
    SUM(is_flagged_fraud * is_fraud)                    AS flagged_and_fraud,
    SUM(is_fraud)                                       AS total_fraud,
    ROUND(SUM(is_flagged_fraud * is_fraud) / nullif(SUM(is_flagged_fraud), 0), 3) AS precision,
    ROUND(SUM(is_flagged_fraud * is_fraud) / nullif(SUM(is_fraud), 0), 4)          AS recall
FROM features;
