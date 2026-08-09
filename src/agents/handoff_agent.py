"""Builds the Escalation payload for a human agent: queue, priority, SLA and a PII-free summary.

Moved here from the orchestrator, where it had been sitting since Session 3. The
orchestrator's job is to decide the lane; assembling the handoff is stage 4c's,
and keeping the two apart means the policy matrix stays a pure function.

The summary is built from routing facts and the resolver's rationale, never from
the message. Nothing in this file can carry a PII value into a queue.
"""

from src.agents.orchestrator import DispositionOutcome
from src.schemas import Escalation, IntentResult, RiskTier

# Queue priority by risk tier. Assumption 5 requires a fraud signal to be elevated
# at minimum; critical goes further because money may be leaving the account now.
PRIORITY_BY_RISK: dict[RiskTier, str] = {
    RiskTier.CRITICAL: "urgent",
    RiskTier.HIGH: "elevated",
    RiskTier.MEDIUM: "standard",
    RiskTier.LOW: "standard",
}


def build_summary(result: IntentResult, outcome: DispositionOutcome) -> str:
    """Three lines: what it is, how confident we are, and why a human is seeing it."""
    reasons = ", ".join(outcome.guardrails_triggered) or "taxonomy policy"
    return (
        f"Intent: {result.intent or 'unresolved'} (confidence {result.confidence:.2f}).\n"
        f"Domain: {result.domain}, risk {outcome.risk_tier.value}.\n"
        f"Routed to a human because: {reasons}."
    )


def build_escalation(
    result: IntentResult, outcome: DispositionOutcome, suggested_reply: str | None = None
) -> Escalation:
    """Assemble the handoff payload for the HUMAN lane.

    `suggested_reply` is passed when a draft was generated and then rejected by the
    egress guard. The draft is wrong to send unedited, which is precisely why it
    goes to a person rather than to the customer — but it still saves them typing.
    """
    return Escalation(
        # The domain is the queue. A separate queue-name mapping would be a second
        # thing to keep in sync with the taxonomy, for no gain at this size.
        queue=result.domain,
        priority=PRIORITY_BY_RISK[outcome.risk_tier],
        sla_minutes=outcome.sla_minutes,
        summary=build_summary(result, outcome),
        suggested_reply=suggested_reply,
    )
