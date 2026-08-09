"""Unit tests for the tool registry and the RAG abstention gates.

The registry tests exist to hold one claim: nothing executes, ever. The RAG tests
use a stub LLM so the generation gate is verifiable without Ollama running, and
without depending on whether a particular model felt like refusing today.
"""

from pydantic import BaseModel

from src.agents.intent_resolver import AdjudicatorVerdict
from src.agents.rag_agent import INSUFFICIENT_CONTEXT, RagDraft, generate_reply
from src.llm import LLMClient
from src.pipeline import route_message
from src.schemas import Disposition, TokenCost
from src.tools import INTENT_TO_TOOL, REGISTRY, propose_action


class StubLLM(LLMClient):
    """Returns a scripted RagDraft. Records whether it was called at all."""

    def __init__(self, draft: RagDraft) -> None:
        self.draft = draft
        self.calls = 0

    def complete_structured(self, prompt: str, schema: type[BaseModel]):  # type: ignore[override]
        self.calls += 1
        return self.draft, TokenCost(prompt_tokens=1, completion_tokens=1)


class ScriptedLLM(LLMClient):
    """Answers each schema with its correct type, so the whole pipeline can run offline."""

    def __init__(self, intent: str, answer: str, cited: list[str]) -> None:
        self.intent, self.answer, self.cited = intent, answer, cited

    def complete_structured(self, prompt: str, schema: type[BaseModel]):  # type: ignore[override]
        if schema is AdjudicatorVerdict:
            return AdjudicatorVerdict(chosen_intent=self.intent, reasoning="scripted"), TokenCost()
        return RagDraft(answer=self.answer, cited_doc_ids=self.cited), TokenCost()


def test_an_invented_phone_number_escalates_end_to_end() -> None:
    """The whole chain, not just the regex: invented specific in, human handoff out.

    A hallucinated hotline number is the worst output this system could produce, so
    the assertion is that nothing reaches the customer — `reply_text` is None — and
    the draft survives only as something a person must edit.
    """
    llm = ScriptedLLM(
        intent="transfer_timing",
        answer="Call our hotline on +45 33 44 55 66 to sort this out.",
        cited=["doc_028"],
    )
    decision = route_message("how long does an international transfer take?", llm)

    assert decision.disposition is Disposition.HUMAN
    assert decision.grounded is False  # generated, then rejected
    assert decision.reply_text is None  # nothing was sent
    assert "invented_specifics" in decision.guardrails_triggered
    assert "invented_phone" in decision.guardrails_triggered
    assert decision.escalation is not None
    assert "+45 33 44 55 66" in (decision.escalation.suggested_reply or "")


def test_a_correctly_grounded_reply_survives_the_egress_guard() -> None:
    """The mirror case: a reply whose only specifics come from the source is sent."""
    llm = ScriptedLLM(
        intent="transfer_timing",
        answer="Transfers to EU countries usually take 1-2 business days.",
        cited=["doc_028"],
    )
    decision = route_message("how long does an international transfer take?", llm)

    assert decision.disposition is Disposition.AUTO_REPLY
    assert decision.grounded is True
    assert decision.reply_text is not None


def test_registry_has_four_tools_and_every_executor_is_a_dry_run() -> None:
    assert len(REGISTRY) == 4
    for tool in REGISTRY.values():
        receipt = tool.executor(tool.name, tool.args_schema.model_construct())
        assert receipt.startswith("DRY RUN:")
        assert "Nothing was executed" in receipt


def test_money_and_credential_tools_all_require_human_approval() -> None:
    """Assumption 6: nothing irreversible or credential-changing is ever auto-confirmed."""
    for name in ("block_card", "request_card_limit_change", "update_contact_details"):
        assert REGISTRY[name].requires_human_approval


def test_proposed_action_is_always_a_proposal_with_a_simulated_receipt() -> None:
    action = propose_action("getting_spare_card")
    assert action is not None
    assert action.dry_run_receipt is not None and "DRY RUN" in action.dry_run_receipt


def test_an_intent_with_no_registered_tool_proposes_nothing() -> None:
    """No tool is better than the wrong tool. The pipeline turns this into a handoff."""
    assert propose_action("exchange_rate") is None
    assert "exchange_rate" not in INTENT_TO_TOOL


def test_retrieval_gate_abstains_without_calling_the_model_at_all() -> None:
    """Gate one saves the call entirely — that is the point, not a side effect."""
    llm = StubLLM(RagDraft(answer="this should never be reached", cited_doc_ids=["doc_001"]))
    outcome, cost = generate_reply("xyzzy plugh frobnicate quux", llm)

    assert outcome.abstained and outcome.reason == "retrieval_gate"
    assert llm.calls == 0
    assert cost is None


def test_generation_gate_honours_the_escape_hatch() -> None:
    """A model that says INSUFFICIENT_CONTEXT must abstain, even on strong retrieval."""
    llm = StubLLM(RagDraft(answer=INSUFFICIENT_CONTEXT, cited_doc_ids=[]))
    outcome, _ = generate_reply("how long does an international transfer take?", llm)

    assert llm.calls == 1  # retrieval passed, so the model was consulted
    assert outcome.abstained and outcome.reason == "generation_gate"
    assert outcome.reply_text is None


def test_a_citation_outside_the_retrieved_set_is_rejected() -> None:
    """Citing an article that was never retrieved is not grounding, it is decoration."""
    llm = StubLLM(RagDraft(answer="Transfers take 1-2 days.", cited_doc_ids=["doc_999"]))
    outcome, _ = generate_reply("how long does an international transfer take?", llm)

    assert outcome.abstained and outcome.reason == "citation_invalid"


def test_a_grounded_answer_carries_citations_and_the_disclosure() -> None:
    llm = StubLLM(RagDraft(answer="Transfers take 1-2 business days.", cited_doc_ids=["doc_028"]))
    outcome, _ = generate_reply("how long does an international transfer take?", llm)

    assert not outcome.abstained
    assert outcome.reply_text is not None
    assert "generated automatically" in outcome.reply_text
    assert [c.doc_id for c in outcome.citations] == ["doc_028"]
    assert outcome.sources  # the egress guard needs these to verify against
