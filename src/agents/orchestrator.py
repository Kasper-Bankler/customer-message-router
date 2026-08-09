"""The deterministic policy matrix: maps intent, confidence and risk tier to a Disposition. No LLM call happens here, by design.

This module is the whole safety argument. `decide_disposition` reads exactly two
fields off the `IntentResult` — `intent` and `confidence` — and looks everything
else up in the frozen `taxonomy.yaml`. There is therefore no code path by which a
model-authored field can reach a disposition, because none is ever read. A
fabricated `disposition` key on the input is not defended against; it is simply
never consulted.

Every rule below can only downgrade. Nothing in this file makes a message *more*
automatable than its taxonomy row already allows.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.schemas import Disposition, Escalation, IntentResult, RiskTier
from src.taxonomy import TAXONOMY, Domain

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SETTINGS_YAML = REPO_ROOT / "config" / "settings.yaml"

_settings = yaml.safe_load(SETTINGS_YAML.read_text(encoding="utf-8"))["orchestrator"]
MIN_CONFIDENCE: float = _settings["min_confidence"]
OUT_OF_TAXONOMY_RISK = RiskTier(_settings["out_of_taxonomy_risk"])
OUT_OF_TAXONOMY_SLA: int = _settings["out_of_taxonomy_sla_minutes"]

# Queue priority by risk tier. Assumption 5 requires fraud to be elevated at
# minimum; critical goes further because money may be leaving the account now.
PRIORITY_BY_RISK: dict[RiskTier, str] = {
    RiskTier.CRITICAL: "urgent",
    RiskTier.HIGH: "elevated",
    RiskTier.MEDIUM: "standard",
    RiskTier.LOW: "standard",
}


class DispositionOutcome(BaseModel):
    """What policy decided, and which rules fired to get there."""

    disposition: Disposition = Field(description="The lane this message is routed to.")
    risk_tier: RiskTier = Field(description="Risk tier from the taxonomy row, or the default.")
    sla_minutes: int = Field(ge=0, description="Response target from the taxonomy row.")
    guardrails_triggered: list[str] = Field(
        default_factory=list, description="Named rules that fired, in the order they applied."
    )


def decide_disposition(result: IntentResult) -> DispositionOutcome:
    """Apply the policy matrix. Reads only `result.intent` and `result.confidence`."""
    intent = result.intent
    confidence = result.confidence

    if intent is None:
        # No taxonomy row applies, so there is nothing authorising an automated
        # answer. Fail closed rather than guess at what the customer meant.
        return DispositionOutcome(
            disposition=Disposition.HUMAN,
            risk_tier=OUT_OF_TAXONOMY_RISK,
            sla_minutes=OUT_OF_TAXONOMY_SLA,
            guardrails_triggered=["out_of_taxonomy"],
        )

    policy = TAXONOMY[intent]
    disposition = policy.disposition
    triggered: list[str] = []

    # Redundant with the taxonomy's own invariant, and kept anyway: the same
    # defence-in-depth argument `autoreply_allowed` makes. One edit should not be
    # able to open the automated path for a fraud message.
    if policy.domain is Domain.SECURITY_FRAUD and disposition is not Disposition.HUMAN:
        disposition = Disposition.HUMAN
        triggered.append("fraud_override")

    if not policy.autoreply_allowed and disposition is Disposition.AUTO_REPLY:
        disposition = Disposition.HUMAN
        triggered.append("autoreply_not_allowed")

    if confidence < MIN_CONFIDENCE and disposition is not Disposition.HUMAN:
        disposition = Disposition.HUMAN
        triggered.append("low_confidence")

    return DispositionOutcome(
        disposition=disposition,
        risk_tier=policy.risk_tier,
        sla_minutes=policy.sla_minutes,
        guardrails_triggered=triggered,
    )


def build_escalation(result: IntentResult, outcome: DispositionOutcome) -> Escalation:
    """Assemble the handoff payload for the HUMAN lane.

    The summary is built from the resolver's rationale and the routing facts, never
    from the raw message, so nothing here can carry a PII value into a queue.
    """
    return Escalation(
        # The domain is the queue. A separate queue-name mapping would be a second
        # place to keep in sync with the taxonomy for no gain at this size.
        queue=result.domain,
        priority=PRIORITY_BY_RISK[outcome.risk_tier],
        sla_minutes=outcome.sla_minutes,
        summary=(
            f"Intent: {result.intent or 'unresolved'} (confidence {result.confidence:.2f}).\n"
            f"Domain: {result.domain}, risk {outcome.risk_tier.value}.\n"
            f"Routed to a human because: {', '.join(outcome.guardrails_triggered) or 'taxonomy policy'}."
        ),
    )
