"""Wires the six stages together as plain function calls, no framework: ingress, intent resolution, orchestrator, one agent, egress guard, trace. Explicit control flow is the point, per the complexity budget in CLAUDE.md.

Session 3 wires stages 1-3 and stubs the rest. Reading this file top to bottom
tells you the whole control flow, which is the property a framework would take away.
"""

import time
import uuid

from src.agents.intent_resolver import resolve_intent
from src.agents.orchestrator import build_escalation, decide_disposition
from src.ingress import sanitise
from src.llm import LLMClient, OllamaClient
from src.schemas import (
    Disposition,
    Escalation,
    InboundMessage,
    RiskTier,
    RoutingDecision,
    SanitisedMessage,
    TokenCost,
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

        # Two ingress verdicts short-circuit before any LLM call. Distress first:
        # a person in crisis reaches a trained human without waiting on a model
        # that might classify their message as a routine balance query.
        if sanitised.vulnerability_flags:
            return _ingress_escalation(sanitised, trace_id, started, reason="vulnerability_detected")
        if sanitised.blocked:
            return _ingress_escalation(sanitised, trace_id, started, reason=sanitised.block_reason)

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
        # Redaction is a guardrail that fired, so it is recorded on every path it
        # runs on, not only the ones that escalate.
        guardrails_triggered=(["pii_redacted"] if sanitised.pii_found else [])
        + outcome.guardrails_triggered,
        latency_ms=_elapsed_ms(started),
        token_cost=cost,
    )

    if outcome.disposition is Disposition.HUMAN:
        decision.escalation = build_escalation(intent_result, outcome)
    elif outcome.disposition is Disposition.AUTO_REPLY:
        decision.reply_text = stub_reply(intent_result.intent)

    return decision


def _ingress_escalation(
    sanitised: SanitisedMessage, trace_id: str, started: float, reason: str | None
) -> RoutingDecision:
    """Route straight to a human on an ingress verdict, without resolving intent.

    No LLM call is made. For a blocked message that is the point — refused input
    must not reach a model. For a distress signal it is a bonus: the routing is
    correct regardless of what the message is nominally about, so spending a model
    call to discover the topic would only add latency to an urgent handoff.
    """
    distressed = bool(sanitised.vulnerability_flags)
    guardrails = ([reason] if reason else []) + sanitised.vulnerability_flags + (
        ["pii_redacted"] if sanitised.pii_found else []
    )

    return RoutingDecision(
        message_id=sanitised.message_id,
        trace_id=trace_id,
        domain="vulnerable_customer" if distressed else "security_fraud",
        intent=None,
        disposition=Disposition.HUMAN,
        risk_tier=RiskTier.HIGH,
        confidence=0.0,
        escalation=Escalation(
            queue="vulnerable_customer" if distressed else "security_fraud",
            # Assumption 5's wording, applied to distress as well: elevated at a
            # minimum, never the standard queue.
            priority="elevated",
            sla_minutes=15 if distressed else 30,
            summary=(
                f"Ingress routed this message to a human before intent resolution.\n"
                f"Reason: {reason}.\n"
                f"PII types redacted: {sanitised.pii_found or 'none'}."
            ),
        ),
        guardrails_triggered=guardrails,
        latency_ms=_elapsed_ms(started),
        token_cost=TokenCost(),
    )


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
