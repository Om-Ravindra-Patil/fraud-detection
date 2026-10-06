-- Clean, consistently named transactions table.
--
-- Leakage decision: the post-transaction balance columns (newbalanceOrig, newbalanceDest)
-- are dropped here. They describe the account AFTER the payment, and in PaySim they
-- contain simulator artifacts that almost perfectly reveal fraud. A real bank scoring a
-- payment before approving it would only know the balances before the payment.
-- isFlaggedFraud is kept for comparison only: it is the simulator's existing rule-based flag.

CREATE OR REPLACE TABLE transactions AS
SELECT
    transaction_id,
    CAST(step AS INTEGER)               AS step,          -- 1 step = 1 hour, 744 steps = 30 days
    type,
    CAST(amount AS DOUBLE)              AS amount,
    nameOrig                            AS name_orig,
    nameDest                            AS name_dest,
    CAST(oldbalanceOrg AS DOUBLE)       AS old_balance_orig,
    CAST(oldbalanceDest AS DOUBLE)      AS old_balance_dest,
    CAST(isFraud AS INTEGER)            AS is_fraud,
    CAST(isFlaggedFraud AS INTEGER)     AS is_flagged_fraud
FROM raw_paysim;
