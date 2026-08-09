"""Unit tests for the policy matrix, especially that a fraud signal forces HUMAN regardless of model confidence.

These run without Ollama, because `decide_disposition` is a pure function of an
`IntentResult`. That is itself part of the design: the safety property is testable
without the component that could violate it being present.
"""

import pytest

from src.agents.orchestrator import MIN_CONFIDENCE, decide_disposition
from src.schemas import Disposition, IntentCandidate, IntentResult
from src.taxonomy import TAXONOMY, Domain

CONFIDENCE_SWEEP = [0.0, 0.25, 0.5, MIN_CONFIDENCE, 0.75, 0.9, 1.0]

FRAUD_INTENTS = [
    intent for intent, policy in TAXONOMY.items() if policy.domain is Domain.SECURITY_FRAUD
]
AUTOREPLY_INTENTS = [
    intent for intent, policy in TAXONOMY.items() if policy.disposition is Disposition.AUTO_REPLY
]


def make_result(intent: str | None, confidence: float, **extra: object) -> IntentResult:
    """A well-formed IntentResult, with `extra` letting a test smuggle in stray keys."""
    payload = {
        "intent": intent,
        "domain": TAXONOMY[intent].domain.value if intent else "out_of_taxonomy",
        "candidates": [IntentCandidate(intent=intent or "unknown", similarity=confidence)],
        "confidence": confidence,
        "rationale": "test fixture",
    }
    return IntentResult(**(payload | extra))


@pytest.mark.parametrize("intent", FRAUD_INTENTS)
@pytest.mark.parametrize("confidence", CONFIDENCE_SWEEP)
def test_every_fraud_intent_routes_to_human(intent: str, confidence: float) -> None:
    """No confidence value, high or low, lets a security_fraud intent leave the human lane."""
    assert decide_disposition(make_result(intent, confidence)).disposition is Disposition.HUMAN


def test_fabricated_disposition_field_cannot_influence_the_outcome() -> None:
    """The model's output shape has no disposition field; a smuggled one must be inert.

    Two things are asserted. First that Pydantic drops the unknown key outright, so
    it never even reaches an attribute. Second that the outcome is identical with
    and without it — the orchestrator never reads such a field in the first place.
    """
    honest = make_result("compromised_card", 1.0)
    smuggled = make_result("compromised_card", 1.0, disposition="auto_reply")

    assert not hasattr(smuggled, "disposition")
    assert decide_disposition(smuggled) == decide_disposition(honest)
    assert decide_disposition(smuggled).disposition is Disposition.HUMAN


def test_fabricated_confidence_cannot_lift_a_human_row() -> None:
    """Maximum confidence is not a promotion mechanism. Rules only ever downgrade."""
    for intent, policy in TAXONOMY.items():
        if policy.disposition is Disposition.HUMAN:
            assert decide_disposition(make_result(intent, 1.0)).disposition is Disposition.HUMAN


@pytest.mark.parametrize("intent", sorted(TAXONOMY))
def test_autoreply_requires_the_taxonomy_to_allow_it(intent: str) -> None:
    """An intent whose row forbids auto-reply can never be auto-replied to."""
    if TAXONOMY[intent].autoreply_allowed:
        return
    for confidence in CONFIDENCE_SWEEP:
        outcome = decide_disposition(make_result(intent, confidence))
        assert outcome.disposition is not Disposition.AUTO_REPLY


@pytest.mark.parametrize("intent", sorted(TAXONOMY))
def test_policy_never_upgrades_a_row(intent: str) -> None:
    """The outcome is always either the taxonomy's own disposition, or HUMAN."""
    for confidence in CONFIDENCE_SWEEP:
        outcome = decide_disposition(make_result(intent, confidence))
        assert outcome.disposition in (TAXONOMY[intent].disposition, Disposition.HUMAN)


def test_low_confidence_downgrades_an_otherwise_automatable_intent() -> None:
    """Below the threshold, even a clean auto_reply row goes to a human."""
    intent = AUTOREPLY_INTENTS[0]
    assert decide_disposition(make_result(intent, 1.0)).disposition is Disposition.AUTO_REPLY
    outcome = decide_disposition(make_result(intent, MIN_CONFIDENCE - 0.01))
    assert outcome.disposition is Disposition.HUMAN
    assert "low_confidence" in outcome.guardrails_triggered


def test_fraud_override_catches_a_weakened_taxonomy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backstop must work on the day the invariant it duplicates is removed.

    `fraud_override` is unreachable while `check_safety_invariants` holds, since no
    security_fraud row can be non-HUMAN. That makes it dead code today and the
    only line standing between a fraud message and an automated reply tomorrow, if
    someone relaxes the invariant. So the test forges exactly that future.
    """
    weakened = TAXONOMY["compromised_card"].model_copy(
        update={"disposition": Disposition.AUTO_REPLY, "autoreply_allowed": True}
    )
    monkeypatch.setitem(TAXONOMY, "compromised_card", weakened)

    outcome = decide_disposition(make_result("compromised_card", 1.0))
    assert outcome.disposition is Disposition.HUMAN
    assert "fraud_override" in outcome.guardrails_triggered


def test_unresolved_intent_routes_to_human() -> None:
    """"None of these" is not a reason to guess."""
    outcome = decide_disposition(make_result(None, 0.0))
    assert outcome.disposition is Disposition.HUMAN
    assert "out_of_taxonomy" in outcome.guardrails_triggered
