"""Two-stage intent resolution: Stage A embedding kNN over Banking77 train, Stage B LLM adjudication over the top-5 candidates.

Stage A is cheap, deterministic and gives a calibrated similarity to threshold on.
Stage B sees five options, not seventy-seven, so the decision space is small enough
that the model is accurate and its choice is auditable. Neither stage decides what
happens next — that is the orchestrator's job, from config.
"""

import json
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field

from src.banking77 import CACHE_DIR, load_banking77
from src.llm import LLMClient
from src.retrieval.index import MODEL_NAME, QUERY_PREFIX, embed_query, get_model
from src.schemas import IntentCandidate, IntentResult, SanitisedMessage, TokenCost
from src.taxonomy import TAXONOMY

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SETTINGS_YAML = REPO_ROOT / "config" / "settings.yaml"

_settings = yaml.safe_load(SETTINGS_YAML.read_text(encoding="utf-8"))["intent"]
KNN_NEIGHBOURS: int = _settings["knn_neighbours"]
CANDIDATE_COUNT: int = _settings["candidate_count"]
AGREEMENT_PENALTY: float = _settings["agreement_penalty"]

VECTORS_NPY = CACHE_DIR / "train_vectors.npy"
VECTORS_META = CACHE_DIR / "train_vectors.meta.json"

# Set when the message falls outside all 77 intents. Not a domain in taxonomy.yaml,
# deliberately: it means "no row applies", which is exactly why it routes to a human.
OUT_OF_TAXONOMY = "out_of_taxonomy"

_vectors: np.ndarray | None = None
_labels: list[str] = []


class AdjudicatorVerdict(BaseModel):
    """What the LLM is allowed to say. Note there is no disposition field, by design."""

    chosen_intent: str | None = Field(
        description="Exactly one of the candidate intent names, or null if none of them fit."
    )
    reasoning: str = Field(description="One sentence explaining the choice.")


def build_vector_cache() -> tuple[np.ndarray, list[str]]:
    """Embed all ~10k training messages once and store them next to the dataset.

    Ten thousand embeddings take about thirteen seconds. Doing that on every run
    would dominate the latency budget for no benefit, since the training set never
    changes between runs.
    """
    train = load_banking77("train")
    # The same query prefix goes on both sides here. A stored training example and
    # an inbound customer message are the same kind of object — a short question —
    # so this is a query-to-query comparison, and both sides must be treated alike.
    # Prefixing only one side is the silent failure this project keeps guarding against.
    vectors = get_model().encode(
        [QUERY_PREFIX + text for text in train.text], batch_size=128, normalize_embeddings=True
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_NPY, vectors)
    VECTORS_META.write_text(json.dumps({"model": MODEL_NAME, "rows": len(train)}), encoding="utf-8")
    return vectors, list(train.intent)


def load_vectors() -> tuple[np.ndarray, list[str]]:
    """Return cached training vectors and their labels, rebuilding when stale."""
    global _vectors, _labels
    if _vectors is not None:
        return _vectors, _labels

    train = load_banking77("train")
    stamp = json.loads(VECTORS_META.read_text(encoding="utf-8")) if VECTORS_META.exists() else {}
    if not VECTORS_NPY.exists() or stamp.get("model") != MODEL_NAME or stamp.get("rows") != len(train):
        _vectors, _labels = build_vector_cache()
    else:
        _vectors, _labels = np.load(VECTORS_NPY), list(train.intent)
    return _vectors, _labels


def knn_candidates(text: str) -> list[IntentCandidate]:
    """Stage A. Return the top distinct intents near this message, best similarity first.

    Similarity is to the single nearest training example of that intent, which is
    what `IntentCandidate.similarity` documents. Grouping by intent after the
    neighbour scan stops one crowded intent occupying the whole shortlist.
    """
    vectors, labels = load_vectors()
    similarities = vectors @ np.array(embed_query(text))

    best_per_intent: dict[str, float] = {}
    for position in np.argsort(-similarities)[:KNN_NEIGHBOURS]:
        intent = labels[position]
        if intent not in best_per_intent:
            best_per_intent[intent] = float(similarities[position])

    ranked = sorted(best_per_intent.items(), key=lambda pair: pair[1], reverse=True)
    return [
        IntentCandidate(intent=intent, similarity=round(similarity, 4))
        for intent, similarity in ranked[:CANDIDATE_COUNT]
    ]


def build_prompt(text: str, candidates: list[IntentCandidate]) -> str:
    """The adjudicator prompt. Note what it does not ask for: any routing decision."""
    options = "\n".join(f"- {candidate.intent}" for candidate in candidates)
    return (
        "You classify a retail bank customer message into one intent.\n\n"
        f"Customer message:\n{text}\n\n"
        f"Candidate intents:\n{options}\n\n"
        "Choose the single candidate that best matches the message. If none of them "
        "fits, return null for chosen_intent. Copy the chosen name exactly as written "
        "above. Do not invent an intent that is not listed."
    )


def adjudicate(
    text: str, candidates: list[IntentCandidate], llm: LLMClient
) -> tuple[AdjudicatorVerdict, TokenCost]:
    """Stage B. Ask the model to pick one candidate, and hold it to the shortlist.

    A name outside the shortlist is coerced to None rather than trusted. The model
    chooses among options it was given; it cannot introduce one.
    """
    verdict, cost = llm.complete_structured(build_prompt(text, candidates), AdjudicatorVerdict)
    allowed = {candidate.intent for candidate in candidates}
    if verdict.chosen_intent is not None and verdict.chosen_intent not in allowed:
        verdict = AdjudicatorVerdict(
            chosen_intent=None,
            reasoning=f"adjudicator returned {verdict.chosen_intent!r}, which was not a candidate",
        )
    return verdict, cost


def score_confidence(chosen: str | None, candidates: list[IntentCandidate]) -> float:
    """Combine the two stages into one number the policy matrix can threshold on.

    Agreement between kNN and the adjudicator is the strong signal, so the top-1
    candidate keeps its raw similarity. A lower-ranked pick means the stages
    disagreed, which is weaker evidence, so it is discounted. Clamped because
    `IntentResult.confidence` is bounded and cosine is not.
    """
    if chosen is None:
        return 0.0
    for rank, candidate in enumerate(candidates):
        if candidate.intent == chosen:
            raw = candidate.similarity if rank == 0 else candidate.similarity * AGREEMENT_PENALTY
            return max(0.0, min(1.0, raw))
    return 0.0


def resolve_intent(message: SanitisedMessage, llm: LLMClient) -> tuple[IntentResult, TokenCost]:
    """Run both stages and assemble the audit record."""
    candidates = knn_candidates(message.text_redacted)
    verdict, cost = adjudicate(message.text_redacted, candidates, llm)
    chosen = verdict.chosen_intent

    return (
        IntentResult(
            intent=chosen,
            domain=TAXONOMY[chosen].domain.value if chosen else OUT_OF_TAXONOMY,
            candidates=candidates,
            confidence=score_confidence(chosen, candidates),
            rationale=verdict.reasoning,
        ),
        cost,
    )
