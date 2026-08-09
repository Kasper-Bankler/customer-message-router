"""Wires the six stages together as plain function calls, no framework: ingress, intent resolution, orchestrator, one agent, egress guard, trace. Explicit control flow is the point, per the complexity budget in CLAUDE.md.

Reading this file top to bottom tells you the whole control flow, which is the
property a framework would take away. Every stage is wrapped: any exception routes
to HUMAN with the failing stage named in `guardrails_triggered`. A router that
drops messages or guesses when a component fails is worse than one that queues them.
"""

import time
import uuid

from src.agents.handoff_agent import build_escalation
from src.agents.intent_resolver import resolve_intent
from src.agents.orchestrator import DispositionOutcome, decide_disposition
from src.agents.rag_agent import RagOutcome, generate_reply
from src.egress import verify_reply
from src.ingress import sanitise
from src.llm import LLMClient, OllamaClient
from src.schemas import (
    Disposition,
    Escalation,
    InboundMessage,
    IntentResult,
    RiskTier,
    RoutingDecision,
    SanitisedMessage,
    TokenCost,
)
from src.tools import propose_action
from src.trace import Trace


def route_message(text: str, llm: LLMClient | None = None) -> RoutingDecision:
    """One customer message in, one RoutingDecision out."""
    started = time.perf_counter()
    llm = llm or OllamaClient()
    inbound = InboundMessage(message_id=str(uuid.uuid4()), text=text)
    trace_id = str(uuid.uuid4())

    trace = Trace.begin(trace_id, inbound.message_id, text)

    try:
        mark = time.perf_counter()
        sanitised = sanitise(inbound)
        trace.record_ingress(mark, sanitised)
    except Exception as error:  # noqa: BLE001 — fail closed is the whole point
        return _failed_closed(inbound, trace_id, error, started, "ingress", trace)

    # Two ingress verdicts short-circuit before any LLM call. Distress first: a
    # person in crisis reaches a trained human without waiting on a model that
    # might classify their message as a routine balance query.
    if sanitised.vulnerability_flags:
        return _ingress_escalation(sanitised, trace_id, started, "vulnerability_detected", trace)
    if sanitised.blocked:
        return _ingress_escalation(sanitised, trace_id, started, sanitised.block_reason, trace)

    try:
        mark = time.perf_counter()
        intent_result, cost = resolve_intent(sanitised, llm)
        trace.record_intent(mark, intent_result, cost)

        mark = time.perf_counter()
        outcome = decide_disposition(intent_result)
        trace.record_policy(mark, outcome)
    except Exception as error:  # noqa: BLE001
        return _failed_closed(inbound, trace_id, error, started, "intent_resolution", trace)

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
        token_cost=cost,
        latency_ms=_elapsed_ms(started),
    )

    try:
        _apply_disposition(decision, sanitised, intent_result, outcome, llm, trace)
    except Exception as error:  # noqa: BLE001
        return _failed_closed(inbound, trace_id, error, started, "agent", trace)

    decision.latency_ms = _elapsed_ms(started)
    trace.finish(decision)
    return decision


def _apply_disposition(
    decision: RoutingDecision,
    sanitised: SanitisedMessage,
    intent_result: IntentResult,
    outcome: DispositionOutcome,
    llm: LLMClient,
    trace: Trace,
) -> None:
    """Run the one agent this disposition calls for, mutating `decision` in place."""
    if outcome.disposition is Disposition.AUTO_REPLY:
        _run_rag(decision, sanitised, intent_result, outcome, llm, trace)
    elif outcome.disposition is Disposition.ACTION:
        decision.proposed_action = propose_action(intent_result.intent or "")
        if decision.proposed_action is None:
            # The taxonomy authorised an action but no tool is registered for this
            # intent. Inventing one is not an option, so a human takes it.
            _downgrade_to_human(decision, intent_result, outcome, "no_tool_registered")
    elif outcome.disposition is Disposition.HUMAN:
        decision.escalation = build_escalation(intent_result, outcome)


def _run_rag(
    decision: RoutingDecision,
    sanitised: SanitisedMessage,
    intent_result: IntentResult,
    outcome: DispositionOutcome,
    llm: LLMClient,
    trace: Trace,
) -> None:
    """Generate a grounded reply, or downgrade to HUMAN at whichever gate stopped it."""
    mark = time.perf_counter()
    result, cost = generate_reply(sanitised.text_redacted, llm)
    trace.record_rag(mark, result, cost)
    if cost is not None:
        decision.token_cost = _add_cost(decision.token_cost, cost)

    if result.abstained or result.reply_text is None:
        # grounded stays None: no reply was generated, which is not the same as a
        # reply that was generated and failed.
        _downgrade_to_human(decision, intent_result, outcome, result.reason or "rag_abstained")
        return

    mark = time.perf_counter()
    verdict = verify_reply(result.reply_text, result.sources)
    trace.record_egress(mark, verdict)
    if not verdict.passed:
        decision.grounded = False
        _downgrade_to_human(
            decision, intent_result, outcome, "invented_specifics", suggested_reply=result.reply_text
        )
        decision.guardrails_triggered += [f"invented_{category}" for category in verdict.invented]
        return

    decision.reply_text = result.reply_text
    decision.citations = result.citations
    decision.grounded = True


def _downgrade_to_human(
    decision: RoutingDecision,
    intent_result: IntentResult,
    outcome: DispositionOutcome,
    reason: str,
    suggested_reply: str | None = None,
) -> None:
    """Move a decision into the human lane after an agent declined to answer."""
    decision.disposition = Disposition.HUMAN
    decision.reply_text = None
    decision.citations = []
    decision.guardrails_triggered = decision.guardrails_triggered + [reason]
    downgraded = outcome.model_copy(
        update={"guardrails_triggered": outcome.guardrails_triggered + [reason]}
    )
    decision.escalation = build_escalation(intent_result, downgraded, suggested_reply)


def _add_cost(left: TokenCost, right: TokenCost) -> TokenCost:
    return TokenCost(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        usd=left.usd + right.usd,
    )


def _ingress_escalation(
    sanitised: SanitisedMessage, trace_id: str, started: float, reason: str | None, trace: Trace
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

    decision = RoutingDecision(
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
    trace.finish(decision)
    return decision


def _failed_closed(
    inbound: InboundMessage, trace_id: str, error: Exception, started: float, stage: str, trace: Trace
) -> RoutingDecision:
    """Build the HUMAN decision used when a stage raises, naming the stage that failed."""
    decision = RoutingDecision(
        message_id=inbound.message_id,
        trace_id=trace_id,
        domain="pipeline_error",
        intent=None,
        disposition=Disposition.HUMAN,
        risk_tier=RiskTier.HIGH,
        confidence=0.0,
        # The exception type, never its message: a traceback can carry the raw text
        # that triggered it, and that text may contain PII.
        guardrails_triggered=[f"{stage}_failed", type(error).__name__],
        latency_ms=_elapsed_ms(started),
        token_cost=TokenCost(),
    )
    trace.finish(decision)
    return decision


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
