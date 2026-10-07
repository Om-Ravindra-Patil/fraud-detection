# Choosing when to raise an alert

## The question

The model gives every TRANSFER and CASH_OUT payment a risk score between 0 and 1. A fraud
analyst cannot look at every payment, so we need a cut-off: payments scoring above it become
alerts, the rest go through.

Where the cut-off sits is a business decision, not a modelling one, because the two kinds of
mistake cost different amounts:

- **A missed fraud** loses the money that was stolen. In PaySim the median fraud is about
  470,000 units, so each one is expensive.
- **An alert** costs analyst time, and if the payment turns out to be genuine, it also
  annoys a real customer whose payment was held. We set this at 1,000 units per alert,
  about 0.6% of a typical genuine payment.

The total cost of a cut-off is the money lost to fraud we did not catch, plus the review
cost of every alert we raised. The chosen cut-off is the one with the lowest total cost.

## What the numbers said

On cost alone, the answer was "raise lots of alerts". Because a missed fraud costs hundreds
of times more than a review, the cheapest cut-off on the validation period (days 20 to 24)
raised about 1,670 alerts a day, roughly 9% of all payments.

No real team can review that many. So we added a second rule: **stay within what the team can
review, which we set at 500 alerts a day** (for example 10 analysts reviewing 50 each).
Within that limit, the cheapest cut-off is a risk score of **0.0184**.

The scores look small because fraud is rare: the model rarely gives any payment a high score.
What matters is the ranking. A score of 0.0184 puts a payment in roughly the riskiest 3% of
validation payments.

## How sure are we about the cost figures?

The review cost is a guess, so we re-ran the choice with review costs from 100 to 20,000 units
(`reports/threshold_sensitivity_validation.csv`). With the team limited to 500 alerts a day,
the chosen cut-off stays at 0.0184 every time. The team's capacity decides the answer, not
the exact review cost. If the team could handle 1,000 alerts a day, the cut-off would drop
and catch more fraud. That is a useful message for a manager: here, more analyst capacity
directly buys more fraud caught.

## Result on the test period (days 25 to 29)

The cut-off was frozen using validation data, then applied once to the test period.

| | LightGBM (chosen) | Logistic regression | Existing rule | No alerts |
|---|---|---|---|---|
| Alerts per day | 425 | 384 | 1 | 0 |
| Alerts that were real fraud (precision) | 48.8% | 37.7% | 100% | n/a |
| Frauds caught (recall) | 77.6% (1,037 of 1,336) | 54.2% | 0.4% | 0% |
| Fraud money caught | 95.6% | 65.8% | 0.9% | 0% |
| Total cost (millions) | 101.9 | 770.6 | 2,225.0 | 2,244.6 |

In plain English:

- About **one in every two alerts is real fraud**. An analyst working the queue spends half
  their time on genuine customers, which is the price of catching most of the fraud.
- The model **catches 78% of fraud cases but 96% of the money**. The frauds it misses are
  mostly the smaller ones, because large payments that empty an account score highest.
- Compared with raising no alerts at all, total cost falls by **95.5%**. Compared with the
  logistic regression baseline at the same alert budget, it falls by **87%**.
- PaySim's existing rule is almost never wrong, but it only catches 5 of 1,336 frauds.

## Limitations

- **The test period has far more fraud than a real bank sees.** Genuine activity collapses in
  the later days of the simulation while fraud stays steady, so 3.1% of test payments are
  fraud. At a real bank's fraud rate, the same cut-off would give a lower precision. This is
  also why PR-AUC is higher on test (0.725) than on validation (0.642).
- **Costs are in PaySim's unnamed currency** and assume nothing is recovered from a missed
  fraud. Both are settings in `configs/config.yaml`.
- **Alerts are ranked by risk score.** Ranking by risk score multiplied by amount caught
  slightly more money on validation (96.8% vs 94.4%) but fewer fraud cases (74% vs 79%).
  We chose the simpler ranking; a bank focused purely on losses might choose the other.
