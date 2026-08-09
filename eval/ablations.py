"""The three ablations PLAN §11 calls for: the retrieval-threshold sweep, dense vs BM25 vs hybrid, and the abstention gate on versus off.

Split out of run_eval.py because they are one coherent idea — "what changes if I
change one component?" — and because each answers a question a panel will ask.
run_eval.py stays the single entry point; nothing here is run directly.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.agents import rag_agent
from src.agents.rag_agent import RETRIEVAL_THRESHOLD
from src.retrieval.hybrid import dense_ranking, search, sparse_ranking
from src.schemas import Disposition
from src.taxonomy import TAXONOMY

SWEEP_THRESHOLDS = [0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76]


def threshold_sweep(merged: pd.DataFrame) -> pd.DataFrame:
    """Auto-reply precision and recall as the retrieval gate moves.

    Intent resolution is held fixed at what the system actually predicted, and only
    the gate is varied. The confidence gate and the egress guard are unchanged
    across thresholds, so this isolates the one knob being chosen.
    """
    allows_auto = np.array(
        [
            bool(intent) and TAXONOMY[intent].disposition is Disposition.AUTO_REPLY
            for intent in merged.predicted_intent
        ]
    )
    is_auto = (merged.disposition == "auto_reply").to_numpy()

    rows = []
    for threshold in SWEEP_THRESHOLDS:
        predicted_auto = allows_auto & (merged.top1_cosine.to_numpy() >= threshold)
        true_positive = int((predicted_auto & is_auto).sum())
        rows.append(
            {
                "threshold": threshold,
                "precision": true_positive / predicted_auto.sum() if predicted_auto.any() else 0.0,
                "recall": true_positive / is_auto.sum() if is_auto.any() else 0.0,
                "n_auto_replied": int(predicted_auto.sum()),
            }
        )
    return pd.DataFrame(rows).round(4)


def retrieval_ablation(merged: pd.DataFrame) -> pd.DataFrame:
    """Dense vs BM25 vs hybrid at finding the messages a human said were answerable.

    The three scores are on incompatible scales, so they are rate-matched: each
    mode flags exactly as many messages as the production gate does, and we then
    ask how many of those the labeller agreed were auto-answerable. Comparing them
    at equal volume is the only way the comparison means anything.
    """
    dense_top, bm25_top, hybrid_top = [], [], []
    for row in merged.itertuples():
        dense = dense_ranking(row.text, 1)
        sparse = sparse_ranking(row.text)
        fused = search(row.text, top_k=1)
        dense_top.append(dense[0][1] if dense else 0.0)
        bm25_top.append(sparse[0][1] if sparse else 0.0)
        hybrid_top.append(fused[0].dense_score if fused else 0.0)

    budget = int((np.array(dense_top) >= RETRIEVAL_THRESHOLD).sum())
    is_auto = (merged.disposition == "auto_reply").to_numpy()

    rows = []
    for mode, scores in [("dense", dense_top), ("bm25", bm25_top), ("hybrid", hybrid_top)]:
        flagged = np.zeros(len(merged), dtype=bool)
        flagged[np.argsort(-np.array(scores))[:budget]] = True
        true_positive = int((flagged & is_auto).sum())
        rows.append(
            {
                "retriever": mode,
                "flagged": budget,
                "correct": true_positive,
                "precision": true_positive / budget if budget else 0.0,
                "recall": true_positive / is_auto.sum() if is_auto.any() else 0.0,
            }
        )
    return pd.DataFrame(rows).round(4)


def abstention_ablation(
    gold: pd.DataFrame,
    baseline: pd.DataFrame,
    reuse: bool,
    results_dir: Path,
    collect: Callable[[pd.DataFrame, Path, str], pd.DataFrame],
) -> pd.DataFrame:
    """With the retrieval gate versus without it — the hallucination-exposure number.

    Disabling the gate makes the RAG agent attempt an answer for every message the
    taxonomy permits, including those with no grounding at all. The interesting
    column is how many messages a human said needed a person, and the ungated
    system answered anyway.
    """
    path = results_dir / "predictions_nogate.csv"
    if reuse and path.exists():
        ungated = pd.read_csv(path, keep_default_na=False)
    else:
        original = rag_agent.RETRIEVAL_THRESHOLD
        # -1.0 is below every possible cosine, so the gate can never reject.
        rag_agent.RETRIEVAL_THRESHOLD = -1.0
        try:
            ungated = collect(gold, path, "gate off")
        finally:
            rag_agent.RETRIEVAL_THRESHOLD = original

    rows = []
    for name, predictions in [("gate on", baseline), ("gate off", ungated)]:
        joined = gold.merge(predictions, on="message_id")
        auto = (joined.predicted_disposition == "auto_reply").to_numpy()
        should_be_human = (joined.disposition != "auto_reply").to_numpy()
        rows.append(
            {
                "variant": name,
                "auto_replied": int(auto.sum()),
                "auto_reply_precision": float((joined.disposition.to_numpy()[auto] == "auto_reply").mean()) if auto.any() else 0.0,
                "answered_but_needed_human": int((auto & should_be_human).sum()),
                "disposition_accuracy": float((joined.disposition == joined.predicted_disposition).mean()),
            }
        )
    return pd.DataFrame(rows).round(4)
