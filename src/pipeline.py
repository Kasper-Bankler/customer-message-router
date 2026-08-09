"""Wires the six stages together as plain function calls, no framework: ingress, intent resolution, orchestrator, one agent, egress guard, trace. Explicit control flow is the point, per the complexity budget in CLAUDE.md.

Session 3 wires stages 1-3 and stubs the rest. Reading this file top to bottom
tells you the whole control flow, which is the property a framework would take away.
"""

import time
import uuid

from src.agents.intent_resolver import resolve_intent
from src.agents.orchestrator import build_escalation, decide_disposition
from src.llm import LLMClient, OllamaClient
from src.schemas import (
    Disposition,
    InboundMessage,
    RiskTier,
    RoutingDecision,
    SanitisedMessage,
    TokenCost,
)


def sanitise(message: InboundMessage) -> SanitisedMessage:
    """Placeholder ingress. Session 4 replaces this with real PII and injection screening.

    It exists now so the contract downstream is already correct: everything after
    this point reads `text_redacted` and never `InboundMessage.text`. When the real
    guard lands, no caller changes.
    """
    return SanitisedMessage(
        message_id=message.message_id,
        text_redacted=message.text,
        detected_language="en",  # Language detection lands with the rest of ingress.
        pii_found=[],
        injection_score=0.0,
    )


def stub_reply(intent: str | None) -> str:
    """Stands in for the RAG agent until Session 4. Never shown to a customer."""
    return f"[STUB REPLY — no RAG yet. Intent resolved as {intent!r}.]"


def route_message(text: str, llm: LLMClient | None = None) -> RoutingDecision:
    """One customer message in, one RoutingDecision out.

    Any exception from any stage routes to HUMAN. A router that drops messages or
    guesses when a component fails is worse than one that queues them.
    """
    started = time.perf_counter()
    llm = llm or OllamaClient()
    inbound = InboundMessage(message_id=str(uuid.uuid4()), text=text)
    trace_id = str(uuid.uuid4())

    try:
        sanitised = sanitise(inbound)
        intent_result, cost = resolve_intent(sanitised, llm)
        outcome = decide_disposition(intent_result)
    except Exception as error:  # noqa: BLE001 — fail closed is the whole point
        return _failed_closed(inbound, trace_id, error, started)

    decision = RoutingDecision(
        message_id=inbound.message_id,
        trace_id=trace_id,
        domain=intent_result.domain,
        intent=intent_result.intent,
        disposition=outcome.disposition,
        risk_tier=outcome.risk_tier,
        confidence=intent_result.confidence,
        guardrails_triggered=outcome.guardrails_triggered,
        latency_ms=_elapsed_ms(started),
        token_cost=cost,
    )

    if outcome.disposition is Disposition.HUMAN:
        decision.escalation = build_escalation(intent_result, outcome)
    elif outcome.disposition is Disposition.AUTO_REPLY:
        decision.reply_text = stub_reply(intent_result.intent)

    return decision


def _failed_closed(
    inbound: InboundMessage, trace_id: str, error: Exception, started: float
) -> RoutingDecision:
    """Build the HUMAN decision used when any stage raises."""
    return RoutingDecision(
        message_id=inbound.message_id,
        trace_id=trace_id,
        domain="pipeline_error",
        intent=None,
        disposition=Disposition.HUMAN,
        risk_tier=RiskTier.HIGH,
        confidence=0.0,
        # The exception type, never its message: a traceback can carry the raw text
        # that triggered it, and that text may contain PII.
        guardrails_triggered=["pipeline_error", type(error).__name__],
        latency_ms=_elapsed_ms(started),
        token_cost=TokenCost(),
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
