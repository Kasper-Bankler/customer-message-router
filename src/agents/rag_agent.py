"""Grounded reply generation over the FAQ corpus, behind three abstention gates: retrieval, generation, verification.

Two of the three live here. The retrieval gate runs before any LLM call, so an
unanswerable question costs nothing and cannot be hallucinated at. The generation
gate gives the model exactly one escape hatch and shows it an example of using
that hatch correctly, because models abstain far more readily when shown an
abstention than when merely permitted one. The third gate, verification, is
deterministic and lives in `src/egress.py`.

Abstention is the designed behaviour. Nineteen of seventy-seven intents have any
FAQ grounding at all, so refusing is the correct answer most of the time.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.llm import LLMClient
from src.retrieval import RetrievedArticle, search
from src.schemas import Citation

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SETTINGS_YAML = REPO_ROOT / "config" / "settings.yaml"

_settings = yaml.safe_load(SETTINGS_YAML.read_text(encoding="utf-8"))["rag"]
RETRIEVAL_THRESHOLD: float = _settings["retrieval_threshold"]
MAX_CONTEXT_ARTICLES: int = _settings["max_context_articles"]
DISCLOSURE: str = _settings["disclosure"]

# The single permitted escape hatch. One exact token, so detecting it is a string
# comparison rather than an interpretation of what the model meant.
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


class RagDraft(BaseModel):
    """What the model is allowed to return. Note it cannot choose a disposition."""

    answer: str = Field(
        description=f"A 2-4 sentence answer, or exactly {INSUFFICIENT_CONTEXT} if the "
        "articles do not fully answer the question."
    )
    cited_doc_ids: list[str] = Field(
        default_factory=list, description="doc_ids of the articles the answer relies on."
    )


class RagOutcome(BaseModel):
    """Result of the RAG attempt, including the case where no attempt was made."""

    reply_text: str | None = Field(default=None, description="Draft reply, disclosure appended.")
    citations: list[Citation] = Field(default_factory=list)
    sources: list[str] = Field(
        default_factory=list, description="Raw text of every article shown to the model."
    )
    abstained: bool = Field(description="True when no reply was produced.")
    reason: str | None = Field(default=None, description="Guardrail name when abstained.")


def build_prompt(question: str, articles: list[RetrievedArticle]) -> str:
    """The grounding prompt, including one worked example of refusing correctly.

    The refusal example is doing real work. A prompt that merely permits abstention
    gets abstention rarely; a prompt that demonstrates it gets it reliably.
    """
    context = "\n\n".join(f"[{a.doc_id}] {a.title}\n{a.content}" for a in articles)
    return (
        "You answer retail bank customer questions using ONLY the FAQ articles provided.\n"
        "Rules:\n"
        "- 2 to 4 sentences. No preamble, no greeting.\n"
        "- Use only facts stated in the articles. Never add a phone number, amount, "
        "link or detail that is not written there.\n"
        f"- If the articles do not fully answer the question, reply with exactly "
        f"{INSUFFICIENT_CONTEXT} and cite nothing.\n\n"
        "- Always list the doc_id of every article you used in cited_doc_ids.\n\n"
        "Example of correctly refusing:\n"
        "  Articles: [doc_010] What fees apply for using my card abroad? "
        "Currency exchange fee: 1.5% of the transaction amount.\n"
        "  Question: What exchange rate will I get today?\n"
        f"  answer: {INSUFFICIENT_CONTEXT}\n"
        "  cited_doc_ids: []\n"
        "  (The article gives a fee, never a rate, so the question is not answered.)\n\n"
        # A second example is needed, not optional. With only the refusal shown, the
        # model copies its empty cited_doc_ids into every answer and every grounded
        # reply then fails the citation-validity check. Observed, then fixed.
        "Example of correctly answering:\n"
        "  Articles: [doc_027] How do I change my PIN code? Use any ATM that supports "
        "PIN changes, or the mobile app under Cards > PIN settings.\n"
        "  Question: Can I change my PIN at a cash machine?\n"
        "  answer: Yes, you can change your PIN at any ATM that supports PIN changes. "
        "You can also do it in the mobile app under Cards, then PIN settings.\n"
        "  cited_doc_ids: [\"doc_027\"]\n\n"
        f"FAQ articles:\n{context}\n\n"
        f"Question: {question}\n"
    )


def retrieve_context(question: str) -> tuple[list[RetrievedArticle], bool]:
    """Gate one. Returns the articles and whether retrieval cleared the threshold.

    Thresholds the top result's dense cosine rather than its fused RRF score. RRF
    is computed from ranks, so it says how much the two retrievers agreed, not how
    relevant the winner is — see the comment in settings.yaml.
    """
    articles = search(question, top_k=MAX_CONTEXT_ARTICLES)
    if not articles or articles[0].dense_score < RETRIEVAL_THRESHOLD:
        return articles, False
    return articles, True


def generate_reply(question: str, llm: LLMClient) -> tuple[RagOutcome, object]:
    """Run both gates and return either a grounded draft or an honest abstention."""
    articles, passed = retrieve_context(question)
    if not passed:
        # No LLM call at all. The cheapest possible answer to an unanswerable
        # question, and it removes an entire class of hallucination.
        return RagOutcome(abstained=True, reason="retrieval_gate"), None

    draft, cost = llm.complete_structured(build_prompt(question, articles), RagDraft)
    if draft.answer.strip().upper().startswith(INSUFFICIENT_CONTEXT):
        return RagOutcome(abstained=True, reason="generation_gate"), cost

    retrieved_ids = {article.doc_id for article in articles}
    cited = [doc_id for doc_id in draft.cited_doc_ids if doc_id in retrieved_ids]
    if not cited:
        # A grounded answer that cites nothing retrieved is not grounded. This is
        # citation validity, checked here because it needs the retrieval result.
        return RagOutcome(abstained=True, reason="citation_invalid"), cost

    by_id = {article.doc_id: article for article in articles}
    return (
        RagOutcome(
            reply_text=f"{draft.answer.strip()}\n\n{DISCLOSURE}",
            citations=[
                Citation(doc_id=doc_id, title=by_id[doc_id].title, relevance=by_id[doc_id].rrf_score)
                for doc_id in cited
            ],
            # Every article the model could see, so the egress check verifies against
            # exactly what was in front of it — not just the ones it chose to cite.
            sources=[f"{article.title}\n{article.content}" for article in articles],
            abstained=False,
        ),
        cost,
    )
