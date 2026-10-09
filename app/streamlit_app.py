"""Fraud alert review dashboard.

Reads the bundle built by `make app-data` (app/data/ and models/fraud_model.txt), so it runs
without the raw data, the training stack or an LLM. Run with: make app
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from fraud.notes import template_note  # noqa: E402
from fraud.reasons import FEATURE_LABELS  # noqa: E402
from fraud.serving import Scorer  # noqa: E402

DATA = ROOT / "app" / "data"
RAISES, LOWERS = "#eb6834", "#2a78d6"
BANDS = ["very high", "high", "elevated"]

st.set_page_config(page_title="Fraud Alert Review", layout="wide")


@st.cache_data
def load_alerts() -> pd.DataFrame:
    alerts = pd.read_parquet(DATA / "alerts.parquet")
    alerts["reasons"] = alerts["reasons"].apply(json.loads)
    return alerts


@st.cache_data
def load_summary() -> dict:
    return json.loads((DATA / "summary.json").read_text())


@st.cache_data
def load_drift() -> dict:
    return json.loads((DATA / "drift.json").read_text())


@st.cache_resource
def load_scorer() -> Scorer:
    return Scorer(ROOT / "models")


def reasons_chart(reasons: list[dict]) -> alt.Chart:
    df = pd.DataFrame(reasons)
    order = df["fact"].tolist()
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("shap:Q", title="Contribution to risk (log-odds)"),
            y=alt.Y("fact:N", sort=order, title=None, axis=alt.Axis(labelLimit=420),
                    scale=alt.Scale(paddingInner=0.35)),
            color=alt.Color("direction:N", title=None,
                            scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                            range=[RAISES, LOWERS]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip("fact:N", title="Fact"),
                     alt.Tooltip("direction:N", title="Effect"),
                     alt.Tooltip("shap:Q", title="SHAP value", format="+.3f")],
        )
    )


def chart_height(n_bars: int, legend: bool = True) -> int:
    """Pixel height for a horizontal bar chart: room per bar plus axis (and legend)."""
    return 44 * n_bars + (110 if legend else 70)


def note_caption(row) -> str:
    if row["note_source"] == "llm":
        return (f"Written by {row['note_model']} running locally, then checked: no invented "
                "numbers or guesses about the fraud type.")
    return "Template note built from the same reasons (no LLM was used for this alert)."


summary = load_summary()
alerts = load_alerts()
main = summary["results"]["lightgbm_realistic_none"]
threshold = summary["threshold"]

st.title("Fraud alert review")
st.caption(
    f"Synthetic PaySim payments, test period (days {summary['test_days'][0]} to "
    f"{summary['test_days'][1]}). A payment becomes an alert when its risk score is at least "
    f"{threshold:.4f}, the cut-off chosen for a team that can review "
    f"{summary['daily_alert_capacity']} alerts a day."
)

queue_tab, score_tab, model_tab, monitor_tab = st.tabs(
    ["Alert queue", "Score a payment", "How the model works", "Monitoring"])

# ---------------------------------------------------------------- Alert queue
with queue_tab:
    k = st.columns(5)
    k[0].metric("Alerts per day", f"{main['alerts_per_day']:.0f}",
                help="Average number of alerts raised per day in the test period")
    k[1].metric("Alerts that were fraud", f"{main['precision']:.1%}", help="Precision")
    k[2].metric("Frauds caught", f"{main['recall']:.1%}", help="Recall")
    k[3].metric("Fraud money caught", f"{main['fraud_money_caught']:.1%}")
    k[4].metric("PR-AUC", f"{main['pr_auc']:.3f}",
                help="Area under the precision-recall curve on the test period")

    f = st.columns([2, 2, 2, 2])
    bands = f[0].multiselect("Risk level", BANDS, default=BANDS)
    types = f[1].multiselect("Payment type", ["TRANSFER", "CASH_OUT"],
                             default=["TRANSFER", "CASH_OUT"])
    outcome = f[2].selectbox("Known outcome", ["All", "Fraud", "Genuine"],
                             help="Shown because this is historical data. An analyst would "
                                  "not know this when reviewing.")
    llm_only = f[3].toggle("Only LLM-written notes", value=False)

    view = alerts[alerts["risk_band"].isin(bands) & alerts["type"].isin(types)]
    if outcome != "All":
        view = view[view["actual_fraud"] == (1 if outcome == "Fraud" else 0)]
    if llm_only:
        view = view[view["note_source"] == "llm"]
    view = view.reset_index(drop=True)

    st.caption(f"Showing {len(view):,} of {len(alerts):,} alerts, highest risk first. "
               "Click a row to see why it was flagged.")
    table = view[["transaction_id", "day", "hour_of_day", "type", "amount", "risk_score",
                  "risk_band", "note_source"]]
    event = st.dataframe(
        table, hide_index=True, width="stretch", height=300,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "transaction_id": st.column_config.NumberColumn("Transaction", format="%d"),
            "day": st.column_config.NumberColumn("Day"),
            "hour_of_day": st.column_config.NumberColumn("Hour"),
            "type": "Type",
            "amount": st.column_config.NumberColumn("Amount", format="%,.0f"),
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0.0, max_value=1.0, format="%.3f"),
            "risk_band": "Risk level",
            "note_source": "Note",
        },
    )

    if view.empty:
        st.info("No alerts match these filters.")
    else:
        selected = event.selection.rows[0] if event.selection.rows else 0
        row = view.iloc[selected]
        st.subheader(f"Transaction {row['transaction_id']}")
        left, right = st.columns([1, 2])
        with left:
            st.metric("Risk score", f"{row['risk_score']:.3f}",
                      help=f"Alert threshold is {threshold:.4f}")
            st.write(f"**Risk level:** {row['risk_band']}")
            st.write(f"**Payment:** {'transfer' if row['type'] == 'TRANSFER' else 'cash-out'} "
                     f"of {row['amount']:,.0f}")
            st.write(f"**When:** day {row['day']}, {row['hour_of_day']:02d}:00")
            st.write(f"**Known outcome:** {'fraud' if row['actual_fraud'] else 'genuine'}")
        with right:
            st.markdown("**Analyst note**")
            st.info(row["note"])
            st.caption(note_caption(row))
        st.markdown("**Why the model flagged it** (top five reasons, strongest first)")
        st.altair_chart(reasons_chart(row["reasons"]), width="stretch",
                        height=chart_height(len(row["reasons"])))

# ---------------------------------------------------------------- Score a payment
with score_tab:
    st.write("Enter a payment to score it live with the packaged model. Notes here use the "
             "template, because LLM notes are written offline for cost and privacy.")
    with st.form("score"):
        c = st.columns(3)
        ptype = c[0].selectbox("Payment type", ["TRANSFER", "CASH_OUT"])
        amount = c[1].number_input("Amount", min_value=0.0, value=250_000.0, step=10_000.0)
        hour = c[2].number_input("Hour of day", min_value=0, max_value=23, value=2)
        c = st.columns(2)
        orig = c[0].number_input("Sender's balance before the payment", min_value=0.0,
                                 value=250_000.0, step=10_000.0)
        dest = c[1].number_input("Receiving account's balance before the payment",
                                 min_value=0.0, value=0.0, step=10_000.0)
        st.markdown("**Receiving account's history**")
        c = st.columns(4)
        n24 = c[0].number_input("Payments in last 24h", min_value=0, value=0)
        amt24 = c[1].number_input("Money received in last 24h", min_value=0.0, value=0.0,
                                  step=10_000.0)
        prior = c[2].number_input("Earlier payments in total", min_value=0, value=0)
        never = c[3].checkbox("Never received money before", value=True)
        since = c[3].number_input("Hours since it last received money", min_value=0, value=24,
                                  disabled=never)
        submitted = st.form_submit_button("Score payment", type="primary")

    if submitted:
        payment = {"type": ptype, "amount": amount, "old_balance_orig": orig,
                   "old_balance_dest": dest, "step": hour, "dest_txn_count_24h": n24,
                   "dest_amount_sum_24h": amt24, "dest_prior_txn_count": prior,
                   "hours_since_prev_dest_txn": None if never else since}
        result = load_scorer().explain_payment(payment)
        c = st.columns(3)
        c[0].metric("Risk score", f"{result['risk_score']:.3f}")
        c[1].metric("Alert?", "Yes" if result["alert"] else "No",
                    help=f"Alert when the score is at least {threshold:.4f}")
        st.info(template_note({**result, "type": ptype, "amount": amount}, threshold))
        st.altair_chart(reasons_chart(result["reasons"]), width="stretch",
                        height=chart_height(len(result["reasons"])))

# ---------------------------------------------------------------- How the model works
with model_tab:
    st.markdown("#### What drives the risk score overall")
    imp = pd.DataFrame({"feature": list(summary["global_importance"]),
                        "importance": list(summary["global_importance"].values())})
    imp["label"] = imp["feature"].map(FEATURE_LABELS).fillna(imp["feature"])
    st.altair_chart(
        alt.Chart(imp).mark_bar(color=LOWERS, cornerRadiusEnd=4).encode(
            x=alt.X("importance:Q", title="Mean |SHAP value|"),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320),
                    scale=alt.Scale(paddingInner=0.35)),
            tooltip=[alt.Tooltip("label:N", title="Feature"),
                     alt.Tooltip("importance:Q", format=".3f")],
        ),
        width="stretch", height=chart_height(len(imp), legend=False),
    )

    st.markdown("#### Test period results at about the same alert volume")
    names = {"lightgbm_realistic_none": "LightGBM (chosen)",
             "logreg_realistic_balanced": "Logistic regression (baseline)",
             "rule_isFlaggedFraud": "Existing rule (isFlaggedFraud)"}
    res = pd.DataFrame([{"Model": label, **summary["results"][key]}
                        for key, label in names.items()])
    st.dataframe(
        res[["Model", "alerts_per_day", "precision", "recall", "fraud_money_caught"]],
        hide_index=True, width="stretch",
        column_config={
            "alerts_per_day": st.column_config.NumberColumn("Alerts per day", format="%.0f"),
            "precision": st.column_config.NumberColumn("Precision", format="percent"),
            "recall": st.column_config.NumberColumn("Recall", format="percent"),
            "fraud_money_caught": st.column_config.NumberColumn("Fraud money caught",
                                                                format="percent"),
        },
    )

    st.markdown("#### Who gets wrongly flagged")
    st.write("Share of genuine payments flagged, by segment. Night-time payments are flagged "
             "about 8 times more often than daytime ones; see docs/responsible_ai.md.")
    fair = pd.DataFrame(summary["fairness"])
    fair["segment"] = (fair["segment"].str.replace(r"^\d ", "", regex=True)
                       .replace({"Q1": "Q1 (smallest)", "Q5": "Q5 (largest)"}))
    st.dataframe(
        fair[["segment_by", "segment", "payments", "false_positive_rate_pct", "recall_pct"]],
        hide_index=True, width="stretch",
        column_config={
            "segment_by": "Segment by", "segment": "Segment", "payments": "Payments",
            "false_positive_rate_pct": st.column_config.NumberColumn(
                "Genuine payments flagged (%)", format="%.1f"),
            "recall_pct": st.column_config.NumberColumn("Frauds caught (%)", format="%.1f"),
        },
    )

    st.caption("Data: PaySim synthetic mobile money dataset (E. A. Lopez-Rojas, Kaggle), "
               "licence CC BY-SA 4.0. No real customer data is used.")

# ---------------------------------------------------------------- Monitoring
with monitor_tab:
    drift = load_drift()
    th = drift["thresholds"]
    st.write(
        "Each feature and the model's risk score are compared with the training period "
        "(days 0 to 19) using the Population Stability Index (PSI). Below "
        f"{th['watch']} is stable, {th['watch']} to {th['drift']} is worth watching, and above "
        f"{th['drift']} is drift. The reference profile is saved with the model, so this check "
        "does not need the training data."
    )
    daily = pd.DataFrame(drift["daily"])
    st.markdown("#### Day by day")
    psi_line = alt.Chart(daily).mark_line(point=True, color=LOWERS).encode(
        x=alt.X("day:Q", title="Day", scale=alt.Scale(zero=False)),
        y=alt.Y("psi_risk_score:Q", title="PSI of the risk score"),
        tooltip=["day", "payments", alt.Tooltip("psi_risk_score:Q", format=".3f")],
    )
    rules = alt.Chart(pd.DataFrame({"y": [th["watch"], th["drift"]]})).mark_rule(
        strokeDash=[4, 4], color="#8a8986").encode(y="y:Q")
    st.altair_chart(psi_line + rules, width="stretch", height=240)
    st.altair_chart(
        alt.Chart(daily).mark_line(point=True, color=LOWERS).encode(
            x=alt.X("day:Q", title="Day", scale=alt.Scale(zero=False)),
            y=alt.Y("alert_rate:Q", title="Alert rate", axis=alt.Axis(format="%")),
            tooltip=["day", "payments", alt.Tooltip("alert_rate:Q", format=".1%")],
        ),
        width="stretch", height=200,
    )
    st.caption("Genuine payment volume collapses on day 17. The risk score's PSI jumps past the "
               "drift line and the alert rate rises from 1.2% to 6.6% that same day, before any "
               "fraud outcomes would be known.")

    st.markdown("#### By feature")
    table = pd.DataFrame(drift["by_period"])
    table["feature"] = table["feature"].map({**FEATURE_LABELS, "risk_score": "Risk score"})
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "feature": "Feature",
            "psi_valid": st.column_config.NumberColumn("PSI, days 20 to 24", format="%.3f"),
            "psi_test": st.column_config.NumberColumn("PSI, days 25 to 29", format="%.3f"),
            "status_test": "Status (days 25 to 29)",
        },
    )
