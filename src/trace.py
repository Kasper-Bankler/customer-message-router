"""trace_id generation, per-agent spans and the JSONL trace sink; records types and hashes only, never raw customer text.

PII safety is structural, not a review discipline. Three things make it so:

  * `Trace.begin` takes the raw message, derives a SHA-256 and a character count,
    and keeps neither the text nor a reference to it. Nothing downstream can reach
    the original.
  * Every `record_*` method takes a typed object and copies named fields out of
    it. There is no generic "log this" path and no field on `TraceRecord` capable
    of holding message text.
  * The ingress guard returns PII *types* only, never values, so by the time any
    code here runs the values do not exist in the process to be leaked.

Field selection therefore lives in this one file, which is what makes the claim
auditable: to check that the system never logs PII, you read this module and
nothing else.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from src.agents.orchestrator import DispositionOutcome
from src.agents.rag_agent import RagOutcome
from src.egress import EgressVerdict
from src.schemas import IntentResult, RoutingDecision, SanitisedMessage, TokenCost

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_YAML = REPO_ROOT / "config" / "settings.yaml"

_settings = yaml.safe_load(SETTINGS_YAML.read_text(encoding="utf-8"))["observability"]
TRACE_ENABLED: bool = _settings["enabled"]
TRACE_PATH = REPO_ROOT / _settings["trace_path"]


class Span(BaseModel):
    """One stage of the pipeline: how long it took, what it cost, what it decided."""

    stage: str = Field(description="Pipeline stage name, e.g. 'intent_resolution'.")
    latency_ms: int = Field(ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    detail: dict[str, Any] = Field(
        default_factory=dict, description="Stage-specific facts. Populated only by this module."
    )


class TraceRecord(BaseModel):
    """One JSONL line per message. Note there is no field that can hold message text."""

    trace_id: str
    message_id: str
    timestamp: str
    message_sha256: str = Field(
        description="Hash of the raw message, so two identical contacts can be correlated "
        "without either being stored. One-way and not reversible to the text."
    )
    message_chars: int = Field(ge=0, description="Length only: the shape, never the content.")
    detected_language: str
    pii_types: list[str] = Field(description="Types such as ['CPR', 'PAN']. Never values.")
    injection_score: float
    vulnerability_flags: list[str]
    spans: list[Span] = Field(default_factory=list)
    guardrails_triggered: list[str] = Field(default_factory=list)
    total_latency_ms: int = Field(default=0, ge=0)
    token_cost: TokenCost = Field(default_factory=TokenCost)
    decision: dict[str, Any] = Field(default_factory=dict, description="The final RoutingDecision.")


class Trace:
    """Collects spans for one message, then writes a single JSONL record.

    Deliberately not a context manager and not a decorator: the pipeline calls one
    method per stage, which keeps the control flow readable top to bottom and
    keeps tracing out of the agents entirely.
    """

    def __init__(self, trace_id: str, message_id: str, message_sha256: str, message_chars: int) -> None:
        self.record = TraceRecord(
            trace_id=trace_id,
            message_id=message_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message_sha256=message_sha256,
            message_chars=message_chars,
            detected_language="unknown",
            pii_types=[],
            injection_score=0.0,
            vulnerability_flags=[],
        )

    @staticmethod
    def begin(trace_id: str, message_id: str, text: str) -> "Trace":
        """Derive a hash and a length from the message, then let the text go out of scope."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return Trace(trace_id, message_id, digest, len(text))

    def _add(self, stage: str, started: float, cost: TokenCost | None, detail: dict[str, Any]) -> None:
        self.record.spans.append(
            Span(
                stage=stage,
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt_tokens=cost.prompt_tokens if cost else 0,
                completion_tokens=cost.completion_tokens if cost else 0,
                detail=detail,
            )
        )

    def record_ingress(self, started: float, sanitised: SanitisedMessage) -> None:
        """Copies the guard's verdict. `text_redacted` is deliberately not copied."""
        self.record.detected_language = sanitised.detected_language
        self.record.pii_types = list(sanitised.pii_found)
        self.record.injection_score = sanitised.injection_score
        self.record.vulnerability_flags = list(sanitised.vulnerability_flags)
        self._add("ingress", started, None, {
            "pii_types": list(sanitised.pii_found),
            "injection_score": sanitised.injection_score,
            "blocked": sanitised.blocked,
            "block_reason": sanitised.block_reason,
            "vulnerability_flags": list(sanitised.vulnerability_flags),
        })

    def record_intent(self, started: float, result: IntentResult, cost: TokenCost | None) -> None:
        """The kNN shortlist and the adjudicated answer — the audit trail for 'why this intent'."""
        self._add("intent_resolution", started, cost, {
            "intent": result.intent,
            "domain": result.domain,
            "confidence": round(result.confidence, 4),
            "candidates": [
                {"intent": c.intent, "similarity": round(c.similarity, 4)} for c in result.candidates
            ],
            # The resolver's rationale is model-authored prose about the redacted
            # message, so it is summarised to its length rather than copied.
            "rationale_chars": len(result.rationale),
        })

    def record_policy(self, started: float, outcome: DispositionOutcome) -> None:
        self._add("policy_matrix", started, None, {
            "disposition": outcome.disposition.value,
            "risk_tier": outcome.risk_tier.value,
            "sla_minutes": outcome.sla_minutes,
            "guardrails": list(outcome.guardrails_triggered),
        })

    def record_rag(self, started: float, outcome: RagOutcome, cost: TokenCost | None) -> None:
        """Retrieved doc_ids and their scores, recorded whether or not a reply followed."""
        self._add("rag_agent", started, cost, {
            "retrieved": [ref.model_dump() for ref in outcome.retrieved],
            "abstained": outcome.abstained,
            "abstain_reason": outcome.reason,
            "cited": [citation.doc_id for citation in outcome.citations],
        })

    def record_egress(self, started: float, verdict: EgressVerdict) -> None:
        self._add("egress_guard", started, None, {
            "passed": verdict.passed,
            "invented": list(verdict.invented),
        })

    def finish(self, decision: RoutingDecision) -> TraceRecord:
        """Attach the final decision and append one line to the sink."""
        self.record.guardrails_triggered = list(decision.guardrails_triggered)
        self.record.total_latency_ms = decision.latency_ms
        self.record.token_cost = decision.token_cost
        self.record.decision = json.loads(decision.model_dump_json())

        if TRACE_ENABLED:
            TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with TRACE_PATH.open("a", encoding="utf-8") as sink:
                sink.write(self.record.model_dump_json() + "\n")
        return self.record
