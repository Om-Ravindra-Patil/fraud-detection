"""Check whether false alerts fall unevenly on particular kinds of payment.

PaySim has no protected characteristics (age, sex, ethnicity), so a direct fairness test is
not possible. Instead we compare error rates across segments that, in real banking data,
can act as proxies for protected groups: payment size, time of day and payment type.
"""

import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import duckdb  # noqa: E402
import joblib  # noqa: E402

from fraud.config import PROJECT_ROOT, load_config  # noqa: E402
from fraud.models import to_lightgbm_frame  # noqa: E402
from fraud.split import load_splits  # noqa: E402

SEGMENT_SQL = """
WITH scored AS (
    SELECT
        *,
        (risk_score >= ?)::INTEGER AS alert,
        CASE
            WHEN hour_of_day BETWEEN 0 AND 6  THEN '1 night (00-06)'
            WHEN hour_of_day BETWEEN 7 AND 20 THEN '2 day (07-20)'
            ELSE '3 evening (21-23)'
        END AS time_band,
        NTILE(5) OVER (ORDER BY amount) AS amount_quintile
    FROM test
),
segments AS (
    SELECT 'payment type' AS segment_by, type AS segment, * FROM scored
    UNION ALL
    SELECT 'time of day', time_band, * FROM scored
    UNION ALL
    SELECT 'amount quintile', 'Q' || amount_quintile, * FROM scored
)
SELECT
    segment_by,
    segment,
    COUNT(*)                                                        AS payments,
    SUM(is_fraud)                                                   AS frauds,
    ROUND(AVG(alert) * 100, 2)                                      AS alert_rate_pct,
    -- Share of genuine customers in this segment who were wrongly flagged
    ROUND(SUM(alert * (1 - is_fraud)) * 100.0 / NULLIF(SUM(1 - is_fraud), 0), 2)
                                                                    AS false_positive_rate_pct,
    -- Share of fraud in this segment that was caught
    ROUND(SUM(alert * is_fraud) * 100.0 / NULLIF(SUM(is_fraud), 0), 1) AS recall_pct
FROM segments
GROUP BY ALL
ORDER BY segment_by, segment
"""


def segment_report(test, threshold: float):
    con = duckdb.connect()
    con.register("test", test)
    return con.execute(SEGMENT_SQL, [threshold]).df()


def main() -> None:
    cfg = load_config()
    decision = json.loads((PROJECT_ROOT / "models" / "decision.json").read_text())
    bundle = joblib.load(PROJECT_ROOT / "models" / f"{decision['model']}.joblib")
    test = load_splits(cfg["paths"]["features_parquet"], cfg["split"])["test"]
    test["risk_score"] = bundle["model"].predict_proba(
        to_lightgbm_frame(test, bundle["columns"]))[:, 1]

    report = segment_report(test, decision["threshold"])
    out = PROJECT_ROOT / "reports" / "fairness_segments_test.csv"
    report.to_csv(out, index=False)
    print(report.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
