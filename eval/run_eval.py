"""Computes intent and disposition metrics, confusion matrices, threshold sweeps and bootstrap confidence intervals over the gold set.

One script, one command, every number in the presentation:

    python eval/run_eval.py                     # full run, ~25 minutes
    python eval/run_eval.py --reuse-predictions # rescore cached runs in seconds

Accuracy is never printed alone. The gold set is 78% one class, so a router that
answers "human" unconditionally scores 0.775 — every accuracy figure here is
quoted beside that baseline, and macro-F1 and balanced accuracy are the headlines
because their baselines do not move with the class balance.

Outputs, all under eval/results/:
    metrics.json                       every headline number with its interval
    predictions.csv, predictions_nogate.csv    per-message decisions, cached
    per_class_disposition.csv, per_class_domain.csv
    threshold_sweep.csv, retrieval_ablation.csv, abstention_ablation.csv
    confusion_disposition.png, confusion_domain.png
    precision_recall.png, retrieval_ablation.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Sibling modules, not a package: running `python eval/run_eval.py` puts eval/ on
# sys.path, and `eval` is a builtin name best left unshadowed.
import metrics as M
from ablations import abstention_ablation, retrieval_ablation, threshold_sweep
from plots import plot_ablation, plot_confusion, plot_precision_recall
from src.agents.rag_agent import RETRIEVAL_THRESHOLD, retrieve_context
from src.pipeline import route_message
from src.taxonomy import Domain

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDSET_CSV = REPO_ROOT / "eval" / "goldset.csv"
STRATA_CSV = REPO_ROOT / "eval" / "goldset_strata.csv"
RESULTS = REPO_ROOT / "eval" / "results"

DISPOSITIONS = ["auto_reply", "action", "human", "clarify"]
DOMAINS = [domain.value for domain in Domain]


def load_labelled() -> pd.DataFrame:
    """Gold set joined to its strata, with unlabelled or illegal rows rejected loudly."""
    gold = pd.read_csv(GOLDSET_CSV, keep_default_na=False)
    gold["disposition"] = gold.disposition.str.strip().str.lower()

    unlabelled = int((gold.disposition == "").sum())
    if unlabelled:
        raise SystemExit(f"{unlabelled} of {len(gold)} rows have no disposition — label them first.")
    illegal = sorted(set(gold.disposition) - set(DISPOSITIONS))
    if illegal:
        raise SystemExit(f"unrecognised labels {illegal}; legal values are {DISPOSITIONS}")

    return gold.merge(pd.read_csv(STRATA_CSV), on="message_id", how="left")


def collect_predictions(gold: pd.DataFrame, path: Path, label: str) -> pd.DataFrame:
    """Route every gold message once and cache what the system decided."""
    rows = []
    for position, row in enumerate(gold.itertuples(), start=1):
        decision = route_message(row.text)
        rows.append(
            {
                "message_id": row.message_id,
                "predicted_disposition": decision.disposition.value,
                "predicted_intent": decision.intent or "",
                "predicted_domain": decision.domain,
                "confidence": decision.confidence,
                "grounded": decision.grounded,
                "guardrails": "|".join(decision.guardrails_triggered),
                "latency_ms": decision.latency_ms,
            }
        )
        print(f"  [{label}] routed {position}/{len(gold)}", end="\r")
    print()

    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def gate_scores(gold: pd.DataFrame) -> pd.DataFrame:
    """Top-1 dense cosine per message. No LLM call, so the sweep is free to recompute."""
    scores = []
    for row in gold.itertuples():
        articles, _ = retrieve_context(row.text)
        scores.append(articles[0].dense_score if articles else 0.0)
    return pd.DataFrame({"message_id": gold.message_id, "top1_cosine": scores})


def headline_block(merged: pd.DataFrame) -> dict:
    """Every number quoted on a slide, each with a 10,000-resample interval."""
    truth = merged.disposition.to_numpy()
    predicted = merged.predicted_disposition.to_numpy()
    domain_truth = merged.true_domain.to_numpy()
    domain_predicted = merged.predicted_domain.to_numpy()
    intent_truth = merged.true_intent.to_numpy()
    intent_predicted = merged.predicted_intent.to_numpy()
    n = len(merged)

    is_fraud = domain_truth == "security_fraud"
    fraud_to_human = (predicted[is_fraud] == "human").astype(float)
    auto_mask = predicted == "auto_reply"
    auto_hits = (truth[auto_mask] == "auto_reply").astype(float)

    metrics: dict = {"n": n, "majority_baseline": round(M.majority_baseline(truth), 4)}
    metrics |= M.with_ci("disposition_accuracy", float((truth == predicted).mean()),
                         lambda idx: float((truth[idx] == predicted[idx]).mean()), n)
    metrics |= M.with_ci("domain_accuracy", float((domain_truth == domain_predicted).mean()),
                         lambda idx: float((domain_truth[idx] == domain_predicted[idx]).mean()), n)
    metrics |= M.with_ci("intent_accuracy", float((intent_truth == intent_predicted).mean()),
                         lambda idx: float((intent_truth[idx] == intent_predicted[idx]).mean()), n)
    metrics |= M.with_ci("macro_f1_disposition", M.macro_f1(truth, predicted, DISPOSITIONS),
                         lambda idx: M.macro_f1(truth[idx], predicted[idx], DISPOSITIONS), n)
    metrics |= M.with_ci("balanced_accuracy_disposition", M.balanced_accuracy(truth, predicted, DISPOSITIONS),
                         lambda idx: M.balanced_accuracy(truth[idx], predicted[idx], DISPOSITIONS), n)
    metrics |= M.with_ci("fraud_recall", float(fraud_to_human.mean()) if len(fraud_to_human) else 0.0,
                         lambda idx: float(fraud_to_human[idx].mean()), len(fraud_to_human))
    metrics |= M.with_ci("auto_reply_precision", float(auto_hits.mean()) if len(auto_hits) else 0.0,
                         lambda idx: float(auto_hits[idx].mean()), max(len(auto_hits), 1))

    # A perfect score has zero variance, so its bootstrap interval collapses to a
    # point and claims a certainty the evidence does not support. Record the
    # rule-of-three bound instead, and mark the interval as degenerate.
    for name, observations in [("fraud_recall", fraud_to_human), ("auto_reply_precision", auto_hits)]:
        if M.is_degenerate(observations):
            metrics[f"{name}_ci_degenerate"] = True
            metrics[f"{name}_rule_of_three_upper_failure_rate"] = round(
                M.rule_of_three_upper(len(observations)), 4)
    return metrics


def print_report(merged: pd.DataFrame, metrics: dict, sweep: pd.DataFrame,
                 retrieval: pd.DataFrame, abstention: pd.DataFrame) -> None:
    """The whole evaluation, in the order it should be read aloud."""
    def show(name: str, note: str = "") -> None:
        low, high = metrics[f"{name}_ci95"]
        if metrics.get(f"{name}_ci_degenerate"):
            bound = metrics[f"{name}_rule_of_three_upper_failure_rate"]
            note = f"<- no variance; true failure rate could still be up to {bound:.0%} (rule of three)"
        print(f"  {name:32} {metrics[name]:.3f}  [95% CI {low:.3f}-{high:.3f}]  n={metrics[f'{name}_n']}  {note}")

    print(f"\n=== HEADLINES (n={metrics['n']}, bootstrap {M.BOOTSTRAP_RESAMPLES:,} resamples) ===")
    print(f"  {'majority_baseline':32} {metrics['majority_baseline']:.3f}   <- accuracy must beat this")
    show("disposition_accuracy")
    show("macro_f1_disposition", "<- headline: baseline-independent")
    show("balanced_accuracy_disposition", f"<- chance = {1/len(DISPOSITIONS):.2f}")
    show("domain_accuracy")
    show("intent_accuracy")
    show("fraud_recall", "<- target 1.00")
    show("auto_reply_precision")

    print("\n=== PER CLASS: disposition ===")
    print(M.per_class_table(merged.disposition.to_numpy(), merged.predicted_disposition.to_numpy(), DISPOSITIONS).to_string(index=False))

    print("\n=== CONFUSION: disposition (rows true, cols predicted) ===")
    print(M.confusion_counts(merged.disposition.to_numpy(), merged.predicted_disposition.to_numpy(), DISPOSITIONS).to_string())

    print("\n=== THRESHOLD SWEEP (auto_reply) ===")
    print(sweep.to_string(index=False))
    print("\n=== ABLATION: retrieval mode, rate-matched ===")
    print(retrieval.to_string(index=False))
    print("\n=== ABLATION: abstention gate ===")
    print(abstention.to_string(index=False))
    print("\n=== BY STRATUM (the sample is deliberately not representative) ===")
    correct = merged.disposition == merged.predicted_disposition
    print(merged.assign(correct=correct).groupby("stratum").correct.agg(["count", "mean"]).round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the router against the gold set.")
    parser.add_argument("--reuse-predictions", action="store_true", help="Use cached routing runs.")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    gold = load_labelled()
    predictions_path = RESULTS / "predictions.csv"
    if args.reuse_predictions and predictions_path.exists():
        predictions = pd.read_csv(predictions_path, keep_default_na=False)
    else:
        print(f"routing {len(gold)} gold messages...")
        predictions = collect_predictions(gold, predictions_path, "gate on")

    merged = gold.merge(predictions, on="message_id").merge(gate_scores(gold), on="message_id")
    metrics = headline_block(merged)
    sweep = threshold_sweep(merged)
    retrieval = retrieval_ablation(merged)
    abstention = abstention_ablation(
        gold, predictions, args.reuse_predictions, RESULTS, collect_predictions
    )

    print_report(merged, metrics, sweep, retrieval, abstention)

    truth, predicted = merged.disposition.to_numpy(), merged.predicted_disposition.to_numpy()
    M.per_class_table(truth, predicted, DISPOSITIONS).to_csv(RESULTS / "per_class_disposition.csv", index=False)
    M.per_class_table(merged.true_domain.to_numpy(), merged.predicted_domain.to_numpy(), DOMAINS).to_csv(
        RESULTS / "per_class_domain.csv", index=False)
    sweep.to_csv(RESULTS / "threshold_sweep.csv", index=False)
    retrieval.to_csv(RESULTS / "retrieval_ablation.csv", index=False)
    abstention.to_csv(RESULTS / "abstention_ablation.csv", index=False)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_confusion(M.confusion_counts(truth, predicted, DISPOSITIONS), "Disposition", RESULTS / "confusion_disposition.png")
    plot_confusion(M.confusion_counts(merged.true_domain.to_numpy(), merged.predicted_domain.to_numpy(), DOMAINS),
                   "Domain", RESULTS / "confusion_domain.png")
    plot_precision_recall(sweep, RETRIEVAL_THRESHOLD, RESULTS / "precision_recall.png")
    plot_ablation(retrieval, RESULTS / "retrieval_ablation.png")
    print(f"\nwrote 8 CSV/JSON files and 4 PNGs to {RESULTS}")


if __name__ == "__main__":
    main()
