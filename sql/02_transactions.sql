-- One clean row per labelled transaction, with identity data joined on.
-- Only about a quarter of transactions have an identity record, so this is a LEFT JOIN.
--
-- card_key approximates "the same card" by combining the anonymised card fields and
-- billing region. IEEE-CIS has no true customer or card ID, so this is a proxy.

CREATE OR REPLACE TABLE transactions AS
SELECT
    t.TransactionID                                         AS transaction_id,
    CAST(t.isFraud AS INTEGER)                              AS is_fraud,
    CAST(t.TransactionDT AS BIGINT)                         AS dt_seconds,
    TIMESTAMP '{{ reference_date }}' + to_seconds(CAST(t.TransactionDT AS BIGINT)) AS event_ts,
    CAST(t.TransactionAmt AS DOUBLE)                        AS amount,
    t.ProductCD                                             AS product_cd,
    t.card4                                                 AS card_network,
    t.card6                                                 AS card_type,
    t.P_emaildomain                                         AS p_email_domain,
    t.R_emaildomain                                         AS r_email_domain,
    CAST(t.dist1 AS DOUBLE)                                 AS dist1,
    concat_ws('_',
        coalesce(CAST(t.card1 AS VARCHAR), 'na'),
        coalesce(CAST(t.card2 AS VARCHAR), 'na'),
        coalesce(CAST(t.card3 AS VARCHAR), 'na'),
        coalesce(CAST(t.card5 AS VARCHAR), 'na'),
        coalesce(CAST(t.card4 AS VARCHAR), 'na'),
        coalesce(CAST(t.card6 AS VARCHAR), 'na'),
        coalesce(CAST(t.addr1 AS VARCHAR), 'na')
    )                                                       AS card_key,
    (i.TransactionID IS NOT NULL)::INTEGER                  AS has_identity,
    i.DeviceType                                            AS device_type
FROM raw_transaction AS t
LEFT JOIN raw_identity AS i
    ON t.TransactionID = i.TransactionID;
