"""Computes intent and disposition metrics, confusion matrices, threshold sweeps and bootstrap confidence intervals over the gold set.

Accuracy is never printed on its own. A gold set with a skewed disposition mix
makes a constant router look competent, so every accuracy figure here is quoted
beside the majority-class baseline it has to beat. The first gold set had a 0.91
baseline, which is exactly how that failure hides.

Two numbers are headlines in their own right because they are the ones a bank
would actually ask about:
  * fraud recall     — of messages that are genuinely fraud, how many reached a
                       human. Target 1.00. Anything less is a customer whose
                       stolen card got an automated reply.
  * auto_reply precision — of the messages we answered automatically, how many
                       should have been. This is the number that caps blast radius.

Run with:
    python eval/run_eval.py                    # routes all 80 messages, then scores
    python eval/run_eval.py --reuse-predictions # rescore without re-running the model
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.pipeline import route_message

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDSET_CSV = REPO_ROOT / "eval" / "goldset.csv"
STRATA_CSV = REPO_ROOT / "eval" / "goldset_strata.csv"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
PREDICTIONS_CSV = RESULTS_DIR / "predictions.csv"
METRICS_JSON = RESULTS_DIR / "metrics.json"

DISPOSITIONS = ["auto_reply", "action", "human", "clarify"]
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


def load_labelled() -> pd.DataFrame:
    """Gold set joined to its strata, with unlabelled rows rejected loudly."""
    gold = pd.read_csv(GOLDSET_CSV, keep_default_na=False)
    gold["disposition"] = gold.disposition.str.strip().str.lower()

    unlabelled = int((gold.disposition == "").sum())
    if unlabelled:
        raise SystemExit(
            f"{unlabelled} of {len(gold)} rows in {GOLDSET_CSV.name} have no disposition. "
            "Label them by hand before running the evaluation."
        )
    illegal = sorted(set(gold.disposition) - set(DISPOSITIONS))
    if illegal:
        raise SystemExit(f"unrecognised disposition labels: {illegal}. Legal values: {DISPOSITIONS}")

    return gold.merge(pd.read_csv(STRATA_CSV), on="message_id", how="left")


def collect_predictions(gold: pd.DataFrame) -> pd.DataFrame:
    """Route every gold message and record what the system decided."""
    rows = []
    for position, row in enumerate(gold.itertuples(), start=1):
        decision = route_message(row.text)
        rows.append(
            {
                "message_id": row.message_id,
                "predicted_disposition": decision.disposition.value,
                "predicted_intent": decision.intent,
                "predicted_domain": decision.domain,
                "confidence": decision.confidence,
                "grounded": decision.grounded,
                "guardrails": "|".join(decision.guardrails_triggered),
                "latency_ms": decision.latency_ms,
            }
        )
        print(f"  routed {position}/{len(gold)}", end="\r")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(PREDICTIONS_CSV, index=False)
    return frame


def bootstrap_ci(correct: np.ndarray, statistic=np.mean) -> tuple[float, float]:
    """Percentile bootstrap 95% interval. n is ~80, so the interval is wide and honest."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(correct, size=(BOOTSTRAP_RESAMPLES, len(correct)), replace=True)
    values = statistic(draws, axis=1)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def headline_metrics(truth: pd.Series, predicted: pd.Series, is_fraud: pd.Series) -> dict:
    """Accuracy against its baseline, macro-F1, and the two numbers a bank asks for."""
    correct = (truth == predicted).to_numpy()
    accuracy = float(correct.mean())
    low, high = bootstrap_ci(correct)

    # The score a router achieves by ignoring the message and always guessing the
    # commonest label. Accuracy below this is worse than useless.
    baseline = float(truth.value_counts(normalize=True).max())

    # Of genuinely-fraud messages, how many reached a human. The one number that
    # should be 1.00 and stay there.
    fraud_reached_human = (predicted[is_fraud] == "human").to_numpy()
    fraud_recall = float(fraud_reached_human.mean()) if len(fraud_reached_human) else float("nan")

    # Of what we auto-answered, how much we were entitled to auto-answer.
    auto_mask = (predicted == "auto_reply").to_numpy()
    auto_correct = (truth[auto_mask] == "auto_reply").to_numpy()
    auto_precision = float(auto_correct.mean()) if auto_mask.any() else float("nan")

    return {
        "n": int(len(truth)),
        "accuracy": accuracy,
        "accuracy_ci95": [low, high],
        "majority_baseline": baseline,
        "beats_baseline_by": accuracy - baseline,
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "fraud_recall": fraud_recall,
        "fraud_recall_ci95": list(bootstrap_ci(fraud_reached_human)) if len(fraud_reached_human) else None,
        "fraud_n": int(is_fraud.sum()),
        "auto_reply_precision": auto_precision,
        "auto_reply_precision_ci95": list(bootstrap_ci(auto_correct)) if auto_mask.any() else None,
        "auto_reply_n": int(auto_mask.sum()),
    }


def print_report(merged: pd.DataFrame, metrics: dict) -> None:
    """Everything the presentation quotes, in the order it should be read."""
    truth, predicted = merged.disposition, merged.predicted_disposition
    present = [label for label in DISPOSITIONS if label in set(truth) | set(predicted)]

    print("\n=== HEADLINES ===")
    low, high = metrics["accuracy_ci95"]
    print(f"accuracy              {metrics['accuracy']:.3f}  [95% CI {low:.3f}-{high:.3f}]  n={metrics['n']}")
    print(f"majority baseline     {metrics['majority_baseline']:.3f}   <- accuracy must beat this to mean anything")
    print(f"  margin over baseline {metrics['beats_baseline_by']:+.3f}")
    print(f"macro-F1              {metrics['macro_f1']:.3f}")
    print(f"fraud recall          {metrics['fraud_recall']:.3f}  (target 1.00, n={metrics['fraud_n']})")
    print(f"auto_reply precision  {metrics['auto_reply_precision']:.3f}  (n={metrics['auto_reply_n']} auto-answered)")

    print("\n=== PER CLASS ===")
    print(classification_report(truth, predicted, labels=present, zero_division=0))

    print("=== CONFUSION MATRIX (rows = true, cols = predicted) ===")
    matrix = confusion_matrix(truth, predicted, labels=present)
    print(pd.DataFrame(matrix, index=present, columns=present).to_string())

    print("\n=== BY STRATUM (the sample is deliberately not representative) ===")
    merged = merged.assign(correct=(truth == predicted))
    print(merged.groupby("stratum").correct.agg(["count", "mean"]).round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the router against the gold set.")
    parser.add_argument(
        "--reuse-predictions", action="store_true", help="Rescore the saved predictions.csv."
    )
    args = parser.parse_args()

    gold = load_labelled()
    if args.reuse_predictions:
        if not PREDICTIONS_CSV.exists():
            raise SystemExit(f"{PREDICTIONS_CSV} not found — run without --reuse-predictions first")
        predictions = pd.read_csv(PREDICTIONS_CSV)
    else:
        print(f"routing {len(gold)} gold messages (this calls the model once each)...")
        predictions = collect_predictions(gold)

    merged = gold.merge(predictions, on="message_id")
    metrics = headline_metrics(
        merged.disposition, merged.predicted_disposition, merged.true_domain == "security_fraud"
    )

    print_report(merged, metrics)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {PREDICTIONS_CSV} and {METRICS_JSON}")


if __name__ == "__main__":
    main()
