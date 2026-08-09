"""Samples 80 Banking77 test messages stratified by FAQ coverage, for hand-labelling of disposition into goldset.csv.

Banking77 gives intent labels for free. It does not give dispositions — that label
does not exist anywhere, and it is the one actually being evaluated.

The first attempt at this sampled 100 messages by oversampling fraud and
uncovered intents, and produced a set that labelled 91% HUMAN. A gold set with a
0.91 majority baseline cannot demonstrate anything: a router that answers "human"
unconditionally scores 91%. The strata below fix that by making the
auto_reply-eligible pool the largest single stratum, so the evaluation can
actually distinguish a good router from a constant one.

`goldset_strata.csv` records which stratum each message came from, so
`run_eval.py` can report per stratum rather than quoting one number from a
deliberately non-representative sample.

Run with: python eval/build_goldset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.banking77 import load_banking77
from src.taxonomy import TAXONOMY, Domain

REPO_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_CSV = REPO_ROOT / "eval" / "results" / "coverage.csv"
GOLDSET_CSV = REPO_ROOT / "eval" / "goldset.csv"
STRATA_CSV = REPO_ROOT / "eval" / "goldset_strata.csv"

# Arbitrary value; only its fixity matters. Rerunning must reproduce the same
# sample, or hand-applied labels stop matching the messages they were made for.
SEED = 42

# The near-miss band from the coverage analysis: intents whose best article scored
# just under the 0.77 threshold. These are the rows where the retrieval gate's
# calibration actually gets tested.
NEAR_MISS_LOW, NEAR_MISS_HIGH = 0.75, 0.77

QUOTAS: dict[str, int] = {
    # The auto_reply-eligible pool, and the largest stratum by design. Without it
    # the gold set has almost nothing the system is allowed to answer, and
    # auto_reply precision cannot be measured at all.
    "faq_covered": 30,
    # Where a wrong answer is most expensive, and where forced-HUMAN needs evidence.
    "security_fraud": 20,
    # The thesis: most intents have no grounding and abstention is correct.
    "uncovered": 20,
    # Just under the retrieval threshold, so gate calibration is observable.
    "near_miss": 10,
}


def load_frame() -> pd.DataFrame:
    """The test split, annotated with domain, FAQ coverage and best-article similarity."""
    if not COVERAGE_CSV.exists():
        raise FileNotFoundError(f"{COVERAGE_CSV} is missing — run eval/coverage_map.py first")

    frame = load_banking77("test").reset_index(drop=True)
    coverage = pd.read_csv(COVERAGE_CSV).set_index("intent")

    frame["message_id"] = [f"gold_{index:04d}" for index in frame.index]
    frame["true_domain"] = [TAXONOMY[intent].domain.value for intent in frame.intent]
    frame["is_fraud"] = [TAXONOMY[intent].domain is Domain.SECURITY_FRAUD for intent in frame.intent]
    frame["faq_covered"] = [bool(coverage.faq_covered[intent]) for intent in frame.intent]
    frame["similarity"] = [float(coverage.similarity[intent]) for intent in frame.intent]
    return frame


def spread_across_intents(pool: pd.DataFrame, quota: int, rng: np.random.Generator) -> pd.DataFrame:
    """Take `quota` rows, round-robin by intent, so no one intent dominates a stratum.

    Sampling a stratum uniformly would happily return thirty messages from three
    chatty intents. Round-robin gives the widest intent coverage the quota allows.
    """
    shuffled = {
        intent: group.iloc[rng.permutation(len(group))]
        for intent, group in pool.groupby("intent", sort=True)
    }
    order = [list(shuffled)[i] for i in rng.permutation(len(shuffled))]

    picked: list[pd.DataFrame] = []
    taken = 0
    depth = 0
    while taken < quota:
        for intent in order:
            group = shuffled[intent]
            if depth < len(group):
                picked.append(group.iloc[[depth]])
                taken += 1
                if taken == quota:
                    break
        depth += 1
    return pd.concat(picked)


def assign_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Draw each quota from its pool. Strata partition the sample; nothing is counted twice.

    Fraud intents are excluded from the other three pools rather than deduplicated
    afterwards. A covered fraud intent is not auto_reply-eligible — policy forces
    it to a human regardless — so it belongs in the fraud stratum and nowhere else.
    """
    rng = np.random.default_rng(SEED)
    non_fraud = frame[~frame.is_fraud]
    near_miss_band = non_fraud.similarity.between(NEAR_MISS_LOW, NEAR_MISS_HIGH, inclusive="left")

    pools = {
        "faq_covered": non_fraud[non_fraud.faq_covered],
        "security_fraud": frame[frame.is_fraud],
        "uncovered": non_fraud[~non_fraud.faq_covered & (non_fraud.similarity < NEAR_MISS_LOW)],
        "near_miss": non_fraud[near_miss_band],
    }

    picked: list[pd.DataFrame] = []
    for stratum, quota in QUOTAS.items():
        pool = pools[stratum]
        if len(pool) < quota:
            raise ValueError(f"stratum {stratum!r} wants {quota}, only {len(pool)} available")
        chosen = spread_across_intents(pool, quota, rng).copy()
        chosen["stratum"] = stratum
        picked.append(chosen)

    return pd.concat(picked).sort_values(["stratum", "true_domain", "intent"], ignore_index=True)


def carry_over_labels(sample: pd.DataFrame) -> pd.Series:
    """Reuse any disposition already hand-labelled for a message that survives the resample.

    Labelling is the expensive part of this exercise. A message that appears in
    both samples has the same correct answer in both, so re-deciding it is waste.
    """
    if not GOLDSET_CSV.exists():
        return pd.Series([""] * len(sample), index=sample.index)

    previous = pd.read_csv(GOLDSET_CSV, keep_default_na=False)
    if "disposition" not in previous.columns:
        return pd.Series([""] * len(sample), index=sample.index)

    # Whitespace crept into hand-typed labels last time; normalise on the way in.
    labels = {
        row.message_id: str(row.disposition).strip()
        for row in previous.itertuples()
        if str(row.disposition).strip()
    }
    return sample.message_id.map(labels).fillna("")


def main() -> None:
    frame = load_frame()
    sample = assign_strata(frame)
    sample["disposition"] = carry_over_labels(sample)

    # The labelling file carries no signal about what the system believes. Coverage,
    # similarity and stratum are held back: a labeller who can see that an intent has
    # no grounding will label it HUMAN, and the evaluation then measures agreement
    # with the system rather than correctness.
    sample[["message_id", "text", "intent", "true_domain", "disposition"]].rename(
        columns={"intent": "true_intent"}
    ).to_csv(GOLDSET_CSV, index=False)
    sample[["message_id", "stratum", "faq_covered", "similarity"]].to_csv(STRATA_CSV, index=False)

    carried = int((sample.disposition != "").sum())
    print(f"wrote {GOLDSET_CSV} ({len(sample)} rows, {carried} labels carried over)")
    print(f"wrote {STRATA_CSV} (held back from the labelling file on purpose)\n")
    print(sample.groupby(["stratum", "true_domain"]).size().to_string())
    print(f"\nintents represented: {sample.intent.nunique()} of 77")

    # Not the labels — the taxonomy's own view, as an advance check that the set is
    # not degenerate again. The hand labels decide the real balance.
    predicted = pd.Series([TAXONOMY[i].disposition.value for i in sample.intent])
    print(f"\ntaxonomy-predicted disposition mix (a preview, not the labels):")
    print(predicted.value_counts().to_string())
    print(f"predicted majority baseline: {predicted.value_counts(normalize=True).max():.2f}")


if __name__ == "__main__":
    main()
