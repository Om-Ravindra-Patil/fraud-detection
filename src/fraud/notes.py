"""Turn a flagged transaction's SHAP reasons into a short note for a fraud analyst.

Design choices:
- Notes are generated once, in a batch, and saved. The dashboard never calls the LLM.
- The prompt holds only the risk score, payment type, amount and SHAP reasons. Account IDs
  and transaction IDs are never sent.
- Guardrail: if the LLM writes a number that is not in the facts it was given, or the note
  is empty or too long, the note is replaced with a deterministic template note.
"""

import json
import re
import time
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from fraud.config import PROJECT_ROOT, load_config

PROMPT_VERSION = "v1"
MAX_WORDS = 90

SYSTEM_PROMPT = """You write short notes for bank fraud analysts reviewing flagged payments.

Rules:
- Use only the facts given. Do not add numbers, names or details that are not in the facts.
- Write 2 or 3 plain sentences, under 70 words in total.
- First sentence: how risky the payment is and the main reason.
- Then mention the next one or two reasons.
- Finish with one practical check the analyst could make.
- Do not say the payment is definitely fraud. The analyst decides.
- Do not guess what kind of fraud or scam this is, or anything about the customer's situation."""


def risk_band(score: float, threshold: float) -> str:
    if score >= 0.5:
        return "very high"
    if score >= 5 * threshold:
        return "high"
    return "elevated"


def facts_text(record: dict, threshold: float) -> str:
    """The only information the LLM sees about a transaction."""
    reasons = json.loads(record["reasons"]) if isinstance(record["reasons"], str) \
        else record["reasons"]
    lines = [
        f"Risk score: {record['risk_score']:.3f} (alert threshold {threshold:.3f}, "
        f"risk level {risk_band(record['risk_score'], threshold)})",
        f"Payment type: {'transfer' if record['type'] == 'TRANSFER' else 'cash-out'}",
        f"Amount: {record['amount']:,.0f}",
        "Reasons from the model, strongest first:",
    ]
    lines += [f"- {r['fact']} ({r['direction']})" for r in reasons]
    return "\n".join(lines)


def template_note(record: dict, threshold: float) -> str:
    """Deterministic fallback note. Always available, no LLM needed."""
    reasons = json.loads(record["reasons"]) if isinstance(record["reasons"], str) \
        else record["reasons"]
    raising = [r["fact"] for r in reasons if r["direction"] == "raises risk"][:3]
    band = risk_band(record["risk_score"], threshold)
    main = raising[0] if raising else "a combination of factors"
    note = f"Risk is {band} (score {record['risk_score']:.3f}). Main reason: {main.lower()}."
    if len(raising) > 1:
        note += " Also: " + "; ".join(r.lower() for r in raising[1:]) + "."
    note += " Check whether the customer recognises this payment and the receiving account."
    return note


# Fraud-type labels the model might guess at. None of them appear in the facts we send.
SPECULATIVE_TERMS = ("phishing", "scam", "money laundering", "laundering", "mule", "identity theft",
                     "account takeover", "hacked", "stolen card")

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def check_note(note: str, facts: str) -> str | None:
    """Return a reason the note is unacceptable, or None if it passes."""
    if not note or not note.strip():
        return "empty"
    if len(note.split()) > MAX_WORDS:
        return "too long"
    invented = _numbers(note) - _numbers(facts)
    if invented:
        return f"numbers not in the facts: {sorted(invented)}"
    guessed = [t for t in SPECULATIVE_TERMS if t in note.lower() and t not in facts.lower()]
    if guessed:
        return f"speculation not in the facts: {guessed}"
    return None


def tidy(note: str) -> str:
    """One paragraph, single spaces."""
    return " ".join(note.split())


def ollama_client(url: str, model: str, temperature: float, timeout: float) -> Callable:
    """Return a function that sends (system, user) to a local Ollama server."""

    def call(system: str, user: str) -> str:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": temperature},
        }).encode()
        req = urllib.request.Request(f"{url}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["message"]["content"].strip()

    return call


def write_note(record: dict, threshold: float, llm: Callable | None) -> dict:
    facts = facts_text(record, threshold)
    if llm is not None:
        try:
            note = tidy(llm(SYSTEM_PROMPT, f"Facts:\n{facts}\n\nWrite the note."))
            problem = check_note(note, facts)
            if problem is None:
                return {"note": note, "source": "llm", "fallback_reason": None}
        except Exception as exc:  # LLM down or timed out: fall back, never fail the batch
            problem = f"llm error: {type(exc).__name__}"
    else:
        problem = "no llm configured"
    return {"note": template_note(record, threshold), "source": "template",
            "fallback_reason": problem}


def generate_notes(flagged: pd.DataFrame, threshold: float, llm: Callable | None,
                   max_notes: int, model_name: str) -> pd.DataFrame:
    """Notes for max_notes alerts spread evenly along the queue, from highest risk to the
    threshold, so the dashboard shows clear-cut and borderline cases, not only the top."""
    ranked = flagged.sort_values("risk_score", ascending=False).reset_index(drop=True)
    picks = np.unique(np.linspace(0, len(ranked) - 1, min(max_notes, len(ranked))).round()
                      .astype(int))
    queue = ranked.iloc[picks]
    rows = []
    for i, record in enumerate(queue.to_dict("records"), start=1):
        result = write_note(record, threshold, llm)
        rows.append({"transaction_id": record["transaction_id"], **result,
                     "model": model_name if result["source"] == "llm" else "template",
                     "prompt_version": PROMPT_VERSION,
                     "generated_at": datetime.now(UTC).isoformat(timespec="seconds")})
        if i % 25 == 0:
            print(f"  {i}/{len(queue)} notes")
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    ncfg = cfg["notes"]
    decision = json.loads((PROJECT_ROOT / "models" / "decision.json").read_text())
    processed = PROJECT_ROOT / "data" / "processed"
    flagged = pd.read_parquet(processed / "flagged_test.parquet")

    llm = ollama_client(ncfg["ollama_url"], ncfg["model"], ncfg["temperature"],
                        ncfg["timeout_seconds"])
    try:
        llm("Reply with OK.", "OK?")
    except Exception as exc:
        print(f"Ollama is not reachable at {ncfg['ollama_url']} ({type(exc).__name__}). "
              "Start it with 'brew services start ollama' and pull the model with "
              f"'ollama pull {ncfg['model']}'. Writing template notes instead.")
        llm = None

    start = time.time()
    notes = generate_notes(flagged, decision["threshold"], llm, ncfg["max_notes"], ncfg["model"])
    out = processed / "analyst_notes.parquet"
    notes.to_parquet(out, index=False)

    counts = notes["source"].value_counts().to_dict()
    print(f"\n{len(notes)} notes in {time.time() - start:.0f}s: {counts}. Saved to {out}")
    rejected = notes[notes["source"] == "template"]["fallback_reason"].value_counts()
    if len(rejected):
        print("Fallback reasons:\n" + rejected.to_string())
    for _, row in notes.head(3).iterrows():
        print(f"\n[{row['transaction_id']}] ({row['source']}) {row['note']}")


if __name__ == "__main__":
    main()
