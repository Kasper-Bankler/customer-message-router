"""All Pydantic models used as I/O contracts between agents. No logic lives here."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Disposition(str, Enum):
    """What the system is permitted to do about a message.

    Chosen by the deterministic policy matrix in the orchestrator, never by an LLM.
    """

    AUTO_REPLY = "auto_reply"  # RAG answers, sent to customer
    ACTION = "action"  # tool proposed, needs approval
    HUMAN = "human"  # queued for an agent
    CLARIFY = "clarify"  # ask the customer one question


class RiskTier(str, Enum):
    """Blast radius if this message is handled wrongly. Drives SLA and approval gates."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InboundMessage(BaseModel):
    """A raw customer contact entering the pipeline, before any sanitisation.

    `text` is unredacted and must never reach a log, a trace record, or an LLM
    prompt. Only `SanitisedMessage.text_redacted` may travel downstream.
    """

    message_id: str = Field(
        description="Stable identifier for this contact; the join key across all trace records."
    )
    text: str = Field(description="Raw customer message. May contain PII. Never log this field.")
    channel: Literal["secure_inbox", "chat", "email"] = Field(
        default="secure_inbox",
        description="Origin channel. Written channels only; voice is out of scope.",
    )
    locale: str | None = Field(
        default=None,
        description="Customer's declared locale (e.g. 'da-DK') if the channel supplies one.",
    )
    customer_ref: str | None = Field(
        default=None,
        description="Pseudonymous customer handle. Stub for real identity; unused in the prototype.",
    )


class SanitisedMessage(BaseModel):
    """Output of the ingress guard: the only representation of the message allowed downstream."""

    message_id: str = Field(description="Carried through from the InboundMessage.")
    text_redacted: str = Field(
        description="Message with PII replaced by typed placeholders such as <CPR> or <PAN>, "
        "so the router still sees the shape of the message without the values."
    )
    detected_language: str = Field(
        description="ISO 639-1 code detected from the raw text, e.g. 'en' or 'da'."
    )
    pii_found: list[str] = Field(
        default_factory=list,
        description="PII types detected, e.g. ['CPR', 'PAN']. Types only, never values (GDPR log hygiene).",
    )
    injection_score: float = Field(
        ge=0,
        le=1,
        description="Prompt-injection likelihood from the ingress screen. 0 is benign, 1 is a certain attack.",
    )
    blocked: bool = Field(
        default=False,
        description="True when ingress refused the message outright; no LLM call is made downstream.",
    )
    block_reason: str | None = Field(
        default=None,
        description="Why ingress blocked, e.g. 'prompt_injection'. Set if and only if blocked is True.",
    )
    vulnerability_flags: list[str] = Field(
        default_factory=list,
        description="Distress signal categories detected, e.g. ['bereavement']. Categories only, "
        "never the matched phrase. Any entry routes the message straight to a human.",
    )


class IntentCandidate(BaseModel):
    """One Banking77 intent proposed by the Stage A embedding kNN, with its raw score."""

    intent: str = Field(description="Banking77 intent label, e.g. 'lost_or_stolen_card'.")
    similarity: float = Field(
        description="Cosine similarity to the nearest training example for this intent. Not bounded to [0,1]."
    )


class IntentResult(BaseModel):
    """Output of the two-stage intent resolver: kNN shortlist plus LLM adjudication."""

    intent: str | None = Field(
        description="Adjudicated Banking77 intent, or None when the message falls outside the taxonomy."
    )
    domain: str = Field(
        description="Business domain from config/taxonomy.yaml, e.g. 'security_fraud'."
    )
    candidates: list[IntentCandidate] = Field(
        description="Stage A shortlist the adjudicator chose from. Kept for the audit log."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Combined confidence from kNN similarity and adjudicator agreement. Thresholded by the policy matrix.",
    )
    rationale: str = Field(description="One line explaining the choice, for the audit log.")


class Citation(BaseModel):
    """A FAQ article cited as grounding for a generated reply."""

    doc_id: str = Field(description="FAQ corpus identifier, e.g. 'doc_028'.")
    title: str = Field(description="Article title as it appears in the corpus.")
    relevance: float = Field(description="Fused retrieval score for this article against the query.")


class ProposedAction(BaseModel):
    """A tool invocation the action agent proposes. Never executed in this prototype."""

    tool_name: str = Field(description="Registry key of the proposed tool, e.g. 'block_card'.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments, already validated against that tool's Pydantic args_schema in the registry.",
    )
    risk: RiskTier = Field(description="Risk tier declared by the tool, not inferred by the LLM.")
    reversible: bool = Field(description="Whether the operation can be undone without a human.")
    requires_auth_level: Literal["none", "basic", "strong"] = Field(
        description="Customer authentication needed before execution; 'strong' means MitID step-up."
    )
    requires_human_approval: bool = Field(
        description="True for anything irreversible or money-moving. Decided by policy, not by the model."
    )
    idempotency_key: str | None = Field(
        default=None, description="Replay guard for tools that declare one is required."
    )
    dry_run_receipt: str | None = Field(
        default=None, description="Simulated receipt from the dry-run executor. No real system is touched."
    )


class Escalation(BaseModel):
    """Handoff payload for a human agent when the disposition is HUMAN."""

    queue: str = Field(description="Target agent queue, e.g. 'fraud_desk'.")
    priority: Literal["standard", "elevated", "urgent"] = Field(
        description="Queue priority. Fraud signals are elevated or higher regardless of model confidence."
    )
    sla_minutes: int = Field(ge=0, description="Response target in minutes, taken from the taxonomy.")
    summary: str = Field(
        description="Three-line brief for the agent, derived from redacted text only. Must contain no PII values."
    )
    suggested_reply: str | None = Field(
        default=None,
        description="Draft reply for the human agent to edit and send. Never reaches the customer without a human.",
    )


class TokenCost(BaseModel):
    """LLM spend for one message, aggregated across every agent hop."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    usd: float = Field(default=0.0, ge=0, description="Estimated cost at the configured model's rate.")


class RoutingDecision(BaseModel):
    """The system's final output for one message, and the unit of audit.

    Every field needed to reconstruct and defend the decision is present here, so
    the trace log alone answers "why was this message routed this way?".
    """

    message_id: str = Field(description="Carried through from the InboundMessage.")
    trace_id: str = Field(description="Correlates this decision with its per-agent spans in the trace sink.")
    domain: str = Field(description="Business domain that owns this message.")
    intent: str | None = Field(description="Resolved Banking77 intent, or None when out of taxonomy.")
    disposition: Disposition = Field(description="What the system decided to do. Set by deterministic policy.")
    risk_tier: RiskTier = Field(description="Risk tier applied to this message.")
    confidence: float = Field(ge=0, le=1, description="Intent confidence that the policy matrix acted on.")
    reply_text: str | None = Field(
        default=None,
        description="Customer-facing reply including the AI disclosure line. None unless disposition is AUTO_REPLY.",
    )
    clarifying_question: str | None = Field(
        default=None,
        description="The single question to put back to the customer. Set only when disposition is CLARIFY.",
    )
    citations: list[Citation] = Field(
        default_factory=list, description="FAQ articles grounding reply_text. Empty means ungrounded."
    )
    grounded: bool | None = Field(
        default=None,
        description="Verifier verdict on whether every claim in reply_text is supported. None when no reply was generated.",
    )
    proposed_action: ProposedAction | None = Field(
        default=None, description="Set when disposition is ACTION. Always a proposal, never an execution."
    )
    escalation: Escalation | None = Field(
        default=None, description="Set when disposition is HUMAN."
    )
    guardrails_triggered: list[str] = Field(
        default_factory=list,
        description="Names of every guardrail that fired, e.g. ['pii_redacted', 'fraud_override'].",
    )
    latency_ms: int = Field(ge=0, description="Wall-clock time from ingress to this decision.")
    token_cost: TokenCost = Field(default_factory=TokenCost, description="LLM spend for this message.")
