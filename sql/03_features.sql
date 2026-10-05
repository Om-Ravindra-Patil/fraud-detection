-- Interpretable features for the model and for SHAP / analyst notes.
--
-- Leakage rule: every card-history feature uses only transactions STRICTLY EARLIER
-- in time than the current one ("... AND 1 PRECEDING" on dt_seconds). Transactions
-- in the same second are excluded, because in production their order is not reliable.
-- No feature uses is_fraud.

CREATE OR REPLACE TABLE features AS
WITH windowed AS (
    SELECT
        transaction_id,
        is_fraud,
        dt_seconds,
        event_ts,
        card_key,

        -- The transaction itself
        amount,
        ln(1 + amount)                                          AS log_amount,
        CAST(floor(dt_seconds / 3600) % 24 AS INTEGER)          AS hour_of_day,
        CAST(floor(dt_seconds / 86400) % 7 AS INTEGER)          AS day_of_week,
        product_cd,
        card_network,
        card_type,
        device_type,
        has_identity,
        dist1,
        p_email_domain,
        r_email_domain,
        CASE
            WHEN p_email_domain IS NULL OR r_email_domain IS NULL THEN NULL
            ELSE (p_email_domain = r_email_domain)::INTEGER
        END                                                     AS email_domains_match,

        -- Card velocity: how busy has this card been recently?
        COUNT(*)           OVER card_24h                        AS card_txn_count_24h,
        coalesce(SUM(amount) OVER card_24h, 0)                  AS card_amount_sum_24h,
        COUNT(*)           OVER card_7d                         AS card_txn_count_7d,

        -- Card history: what is normal for this card?
        COUNT(*)           OVER card_history                    AS card_prior_txn_count,
        AVG(amount)        OVER card_history                    AS card_prior_avg_amount,
        dt_seconds - MAX(dt_seconds) OVER card_history          AS secs_since_prev_card_txn
    FROM transactions
    WINDOW
        card_24h     AS (PARTITION BY card_key ORDER BY dt_seconds
                         RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING),
        card_7d      AS (PARTITION BY card_key ORDER BY dt_seconds
                         RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING),
        card_history AS (PARTITION BY card_key ORDER BY dt_seconds
                         RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT
    *,
    amount / nullif(card_prior_avg_amount, 0)                   AS amount_vs_card_avg
FROM windowed
ORDER BY dt_seconds, transaction_id;
