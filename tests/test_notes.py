import json

import pandas as pd
import pytest

from fraud.notes import (
    SYSTEM_PROMPT,
    check_note,
    facts_text,
    generate_notes,
    template_note,
    write_note,
)

REASONS = [
    {"feature": "empties_orig_account", "fact": "Payment empties the sender's account",
     "shap": 2.1, "direction": "raises risk"},
    {"feature": "log_amount", "fact": "Amount is 250,000", "shap": 1.2,
     "direction": "raises risk"},
    {"feature": "dest_prior_txn_count", "fact": "Receiving account has 12 earlier payment(s)",
     "shap": -0.4, "direction": "lowers risk"},
]
RECORD = {"transaction_id": 6272926, "risk_score": 0.734, "type": "TRANSFER",
          "amount": 250_000.0, "reasons": json.dumps(REASONS)}
THRESHOLD = 0.0184


def test_prompt_never_contains_identifiers():
    sent = SYSTEM_PROMPT + facts_text(RECORD, THRESHOLD)
    assert str(RECORD["transaction_id"]) not in sent
    for pattern in ("nameOrig", "nameDest", "name_orig", "name_dest"):
        assert pattern not in sent


def test_facts_include_score_amount_and_reasons():
    facts = facts_text(RECORD, THRESHOLD)
    assert "0.734" in facts and "250,000" in facts and "transfer" in facts
    assert "Payment empties the sender's account (raises risk)" in facts


def test_template_note_uses_only_raising_reasons():
    note = template_note(RECORD, THRESHOLD)
    assert note.startswith("Risk is very high (score 0.734)")
    assert "empties the sender's account" in note
    assert "12 earlier" not in note  # a risk-lowering reason is not presented as a concern


def test_check_note_accepts_grounded_numbers():
    facts = facts_text(RECORD, THRESHOLD)
    assert check_note("Very high risk (0.734): a transfer of 250,000 empties the account.",
                      facts) is None


@pytest.mark.parametrize("note, problem", [
    ("", "empty"),
    ("word " * 100, "too long"),
    ("Risk is high; the customer lost 400,000 last week.", "numbers not in the facts"),
    ("Looks like money laundering through a new account.", "speculation not in the facts"),
])
def test_check_note_rejects_bad_notes(note, problem):
    assert problem in check_note(note, facts_text(RECORD, THRESHOLD))


def test_bad_llm_output_falls_back_to_template():
    result = write_note(RECORD, THRESHOLD, llm=lambda s, u: "Score 0.99, definitely fraud.")
    assert result["source"] == "template"
    assert "numbers not in the facts" in result["fallback_reason"]


def test_llm_error_falls_back_to_template():
    def broken(system, user):
        raise TimeoutError

    result = write_note(RECORD, THRESHOLD, llm=broken)
    assert result["source"] == "template"
    assert result["fallback_reason"] == "llm error: TimeoutError"


def test_good_llm_output_is_kept_and_sees_only_facts():
    seen = {}

    def fake(system, user):
        seen["user"] = user
        return "Very high risk: the transfer empties the sender's account. Call the customer."

    result = write_note(RECORD, THRESHOLD, llm=fake)
    assert result["source"] == "llm"
    assert str(RECORD["transaction_id"]) not in seen["user"]


def test_generate_notes_spreads_along_the_queue():
    flagged = pd.DataFrame([{**RECORD, "transaction_id": i, "risk_score": i / 10}
                            for i in range(1, 10)])
    notes = generate_notes(flagged, THRESHOLD, llm=None, max_notes=3, model_name="x")
    assert notes["transaction_id"].tolist() == [9, 5, 1]  # top, middle and bottom
    assert set(notes["source"]) == {"template"}


def test_speculation_is_rejected():
    facts = facts_text(RECORD, THRESHOLD)
    assert "speculation" in check_note("High risk, likely a phishing victim.", facts)


def test_llm_note_is_tidied_to_one_paragraph():
    result = write_note(RECORD, THRESHOLD, llm=lambda s, u: "High risk.\n\nCheck  the payee.")
    assert result["note"] == "High risk. Check the payee."
