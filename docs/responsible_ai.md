# Responsible AI: explanations, privacy and fairness

## Explaining the model's decisions

Every alert comes with a reason. We use SHAP (TreeExplainer), which splits a payment's risk
score into a contribution from each feature. The contributions add up exactly to the model's
output, and `src/fraud/explain.py` checks that they do every time it runs.

**Overall** (`reports/figures/shap_global_importance.png`), the score is driven mostly by:

1. Whether the payment empties the sender's account
2. Payment type (transfers are riskier than cash-outs)
3. Payment amount
4. Whether the receiving account held no money beforehand
5. Hour of day

**For each flagged payment**, the five features that moved its score most are turned into
plain sentences, such as "Payment empties the sender's account" or "First payment ever seen
into this receiving account". These feed the dashboard and the analyst notes.

What SHAP does not do: it explains what the model learned, not why fraud happened. A feature
can matter to the model because of a quirk in the data (see "hour of day" below).

## Privacy

**This project uses only synthetic data.** PaySim contains no real customers.

If this approach were used on real customer data, these controls would apply:

- **Data minimisation.** The model uses 11 behavioural features. It does not use names,
  addresses, dates of birth or any protected characteristic. Account IDs are used in SQL
  to count earlier payments but are never model inputs.
- **Nothing identifying goes to the LLM.** The analyst-note prompt contains only the risk
  score, the payment type and amount, and the SHAP reasons. Account IDs and transaction IDs
  are left out, and a unit test checks this.
- **Lawful basis.** Under UK GDPR, fraud prevention is usually processed under legitimate
  interests or legal obligation. A Data Protection Impact Assessment would be needed before
  using a model like this on live customer data.
- **Human in the loop.** The model raises alerts for an analyst; it does not block payments
  on its own. This matters under UK GDPR Article 22, which restricts decisions with
  significant effects that are based solely on automated processing.
- **Retention and access.** Scores, explanations and notes are personal data once they relate
  to a real customer, so they need the same retention limits and access controls as the
  transactions themselves.

## Fairness

### What we could and could not test

PaySim has no protected characteristics (age, sex, ethnicity, disability), so we cannot test
directly whether the model treats protected groups differently. In real data, though, some
features act as **proxies** for protected groups. Payment size can track income, and the time
of day a person pays can track their job and age.

So we checked whether genuine customers are wrongly flagged more often in some segments than
others (`reports/fairness_segments_test.csv`, test period, at the chosen threshold):

| Segment | Genuine customers wrongly flagged |
|---|---|
| Night (00:00 to 06:59) | **19.0%** |
| Day (07:00 to 20:59) | 2.4% |
| Evening (21:00 to 23:59) | 3.2% |
| Transfers | 6.4% |
| Cash-outs | 1.5% |
| Smallest 20% of payments | 2.1% |
| Largest 20% of payments | 4.0% |

### The main finding: night-time payments

**Genuine customers paying at night are wrongly flagged 8 times more often than those paying
during the day.** In PaySim this happens because fraud is spread evenly around the clock while
genuine activity almost stops at night, so the model learns that "night" means "risky". At a
real bank, the people most affected would be night-shift workers, such as nurses, carers and
warehouse staff. Under the FCA's Consumer Duty, a firm has to avoid causing foreseeable harm to
groups of customers, and holding their payments far more often would be that kind of harm.

We tested the obvious fix, removing hour of day, on the validation period:

| | With hour of day (current) | Without hour of day |
|---|---|---|
| PR-AUC | 0.642 | 0.521 |
| Frauds caught at about 500 alerts a day | 78.3% | 71.5% |
| Fraud money caught | 94.1% | 94.6% |
| Night-time genuine customers wrongly flagged | 16.4% | **2.1%** |
| Daytime genuine customers wrongly flagged | 1.4% | 1.8% |

Removing the feature costs about 7 points of recall, but catches the same share of fraud
money and removes almost all of the night-time gap. **Our recommendation is to drop hour of day
before any real use**, or to keep it only with a per-segment threshold and ongoing monitoring.
We kept it in the reported model because the test period had already been scored and changing
the model afterwards would bias the test result.

### Other risks

- **Transfers are flagged four times as often as cash-outs.** This reflects the higher fraud
  rate on transfers (6.7% vs 2.0% in test), but it should be monitored.
- **New receiving accounts score higher.** That fits money-mule behaviour, but it also means
  someone paying a new landlord or a new business is more likely to be held.
- **Feedback loops.** If only alerted payments are investigated, the labels the model is
  retrained on come mostly from its own alerts, which can lock in its blind spots. A small
  random sample of unflagged payments should be reviewed too.

### What we would monitor in production

- False alert rate by time of day, payment type and amount band, every week
- Complaint and appeal rates for held payments, by segment
- Any segment whose false alert rate drifts above twice the overall rate, as a trigger for
  review
