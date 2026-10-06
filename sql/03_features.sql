-- Interpretable features for the model and for SHAP / analyst notes.
--
-- Scope: PaySim only contains fraud in TRANSFER and CASH_OUT transactions, so the model
-- table keeps those two types. History features are still computed over ALL transactions,
-- so a destination account's earlier payments of any type count towards its history.
--
-- Leakage rule: every history feature uses only transactions in STRICTLY EARLIER hours
-- ("... AND 1 PRECEDING" on step). Transactions in the same hour are excluded, because
-- PaySim does not record their order within the hour. No feature uses is_fraud.

CREATE OR REPLACE TABLE features AS
WITH history AS (
    SELECT
        *,
        -- Destination velocity: is money suddenly flowing into this account?
        COUNT(*)            OVER dest_24h                       AS dest_txn_count_24h,
        coalesce(SUM(amount) OVER dest_24h, 0)                  AS dest_amount_sum_24h,
        COUNT(*)            OVER dest_history                   AS dest_prior_txn_count,
        step - MAX(step)    OVER dest_history                   AS hours_since_prev_dest_txn,
        -- Origin history: has this customer transacted before?
        COUNT(*)            OVER orig_history                   AS orig_prior_txn_count
    FROM transactions
    WINDOW
        dest_24h     AS (PARTITION BY name_dest ORDER BY step
                         RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING),
        dest_history AS (PARTITION BY name_dest ORDER BY step
                         RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        orig_history AS (PARTITION BY name_orig ORDER BY step
                         RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT
    transaction_id,
    is_fraud,
    is_flagged_fraud,
    step,
    step % 24                                                   AS hour_of_day,
    step // 24                                                  AS day,

    -- The transaction itself
    type,
    amount,
    ln(1 + amount)                                              AS log_amount,

    -- Balances known before the payment
    old_balance_orig,
    old_balance_dest,
    amount / nullif(old_balance_orig, 0)                        AS amount_to_orig_balance,
    (old_balance_orig > 0 AND amount >= old_balance_orig)::INTEGER AS empties_orig_account,
    (old_balance_orig = 0)::INTEGER                             AS orig_balance_zero,
    (old_balance_dest = 0)::INTEGER                             AS dest_balance_zero,

    -- History
    dest_txn_count_24h,
    dest_amount_sum_24h,
    dest_prior_txn_count,
    hours_since_prev_dest_txn,
    orig_prior_txn_count
FROM history
WHERE type IN ('TRANSFER', 'CASH_OUT')
ORDER BY step, transaction_id;
