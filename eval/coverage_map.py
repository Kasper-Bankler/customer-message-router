"""Measures which of the 77 intents the 30-article FAQ can actually ground, by comparing each intent's training-example centroid against every article, and writes eval/results/coverage.{csv,png}.

This is the chart the whole case rests on: it shows that most intents have no
grounding at all, which is why abstention is the designed behaviour rather than a
failure. Run with `python eval/coverage_map.py`. It measures coverage only; it never routes a message or calls an LLM.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from src.banking77 import load_banking77
from src.retrieval import load_faq_corpus
from src.taxonomy import TAXONOMY, Domain

matplotlib.use("Agg")  # No display in a terminal session; write straight to file.
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_YAML = REPO_ROOT / "config" / "settings.yaml"
RESULTS_DIR = REPO_ROOT / "eval" / "results"


def intent_centroids(model: SentenceTransformer, query_prefix: str) -> tuple[list[str], np.ndarray]:
    """Embed every training message and average per intent, giving one vector per intent.

    The centroid is the average of many short customer phrasings, so it represents
    the intent as customers write it rather than as the dataset labels it.
    """
    train = load_banking77("train")
    texts = [query_prefix + text for text in train.text]
    print(f"embedding {len(texts)} training messages across {train.intent.nunique()} intents...")
    vectors = model.encode(texts, batch_size=128, normalize_embeddings=True)

    intents = sorted(train.intent.unique())
    centroids = []
    for intent in intents:
        rows = vectors[(train.intent == intent).to_numpy()]
        centroid = rows.mean(axis=0)
        centroids.append(centroid / np.linalg.norm(centroid))  # Re-normalise: a mean of
        # unit vectors is not itself a unit vector, and cosine needs one.
    return intents, np.vstack(centroids)


def build_coverage_table(threshold: float, model_name: str, query_prefix: str) -> pd.DataFrame:
    """One row per intent: its closest FAQ article, the similarity, and the verdict."""
    model = SentenceTransformer(model_name)
    articles = load_faq_corpus()
    article_vectors = model.encode(
        [article.embedding_text() for article in articles], normalize_embeddings=True
    )
    intents, centroids = intent_centroids(model, query_prefix)

    similarities = centroids @ article_vectors.T  # Both sides unit-normalised, so this is cosine.
    best_index = similarities.argmax(axis=1)
    best_score = similarities.max(axis=1)

    # Every taxonomy field rides along, so this one file is also the hand-review
    # sheet: the routing call and the evidence for it sit on the same row.
    policies = [TAXONOMY[intent] for intent in intents]
    return pd.DataFrame(
        {
            "intent": intents,
            "domain": [policy.domain.value for policy in policies],
            "disposition": [policy.disposition.value for policy in policies],
            "risk_tier": [policy.risk_tier.value for policy in policies],
            "sla_minutes": [policy.sla_minutes for policy in policies],
            "autoreply_allowed": [policy.autoreply_allowed for policy in policies],
            "faq_covered": best_score >= threshold,
            "similarity": best_score.round(4),
            "best_doc_id": [articles[i].doc_id for i in best_index],
            "best_doc_title": [articles[i].title for i in best_index],
            "rationale": [policy.rationale for policy in policies],
        }
    ).sort_values(["domain", "disposition", "intent"], ignore_index=True)


def plot_coverage_by_domain(table: pd.DataFrame, threshold: float, output_path: Path) -> None:
    """Stacked bar per domain: how many of its intents the FAQ can ground, and how many it cannot."""
    counts = table.groupby(["domain", "faq_covered"]).size().unstack(fill_value=0)
    for column in (True, False):  # A domain with no covered intents still needs the column.
        if column not in counts:
            counts[column] = 0
    # Reindex over every domain so lending_products appears as an empty bar: it has
    # five FAQ articles and zero Banking77 intents, and that hole is worth showing.
    counts = counts.reindex([domain.value for domain in Domain], fill_value=0)
    counts = counts.sort_values(True, ascending=False)

    figure, axes = plt.subplots(figsize=(9, 5))
    axes.bar(counts.index, counts[True], label="FAQ grounding available", color="#1f6f4a")
    axes.bar(
        counts.index,
        counts[False],
        bottom=counts[True],
        label="No grounding — must abstain",
        color="#b34a3c",
    )
    for x, (yes, no) in enumerate(zip(counts[True], counts[False])):
        axes.text(x, yes + no + 0.2, f"{yes}/{yes + no}", ha="center", fontsize=9)

    covered = int(table.faq_covered.sum())
    axes.set_title(
        f"FAQ coverage of Banking77 intents by domain\n"
        f"{covered} of {len(table)} intents have grounding at cosine ≥ {threshold}"
    )
    axes.set_ylabel("intents")
    axes.tick_params(axis="x", rotation=30)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    settings = yaml.safe_load(SETTINGS_YAML.read_text(encoding="utf-8"))
    threshold = settings["coverage"]["similarity_threshold"]

    table = build_coverage_table(
        threshold=threshold,
        model_name=settings["embeddings"]["model"],
        query_prefix=settings["embeddings"]["query_prefix"],
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "coverage.csv", index=False)
    plot_coverage_by_domain(table, threshold, RESULTS_DIR / "coverage.png")

    covered = int(table.faq_covered.sum())
    print(f"\n{covered} of {len(table)} intents covered at threshold {threshold}")

    # The rows either side of the cut are the ones worth arguing about, so print them
    # rather than making someone open the CSV to find out how close the call was.
    band = table[table.similarity.between(threshold - 0.02, threshold + 0.02)].sort_values(
        "similarity", ascending=False
    )
    print(f"\nwithin 0.02 of the threshold — hand-check these {len(band)}:")
    print(band[["intent", "best_doc_id", "similarity", "faq_covered"]].to_string(index=False))
    print(f"\nwrote {RESULTS_DIR / 'coverage.csv'} and {RESULTS_DIR / 'coverage.png'}")


if __name__ == "__main__":
    main()
