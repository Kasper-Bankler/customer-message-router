"""Reads config/taxonomy.yaml into typed policy rows and validates it at import time, so a typo in the routing table fails loudly here instead of silently misrouting a customer."""

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from src.banking77 import INTENT_NAMES
from src.schemas import Disposition, RiskTier

REPO_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_YAML = REPO_ROOT / "config" / "taxonomy.yaml"

# The five rungs documented in the header of taxonomy.yaml. Restricting SLAs to a
# fixed ladder means a stray "45" is a validation error rather than a sixth tier
# nobody staffed a queue for.
ALLOWED_SLA_MINUTES = (15, 30, 60, 120, 240)


class Domain(str, Enum):
    """The seven business domains from PLAN.md §6, one queue owner each.

    `LENDING_PRODUCTS` has no Banking77 intent — the dataset simply contains no
    lending messages — but the FAQ corpus has five lending articles, so the domain
    is real even though the taxonomy cannot reach it. Keeping it here documents
    that gap instead of hiding it.
    """

    CARDS_ISSUING = "cards_issuing"
    PAYMENTS_TRANSFERS = "payments_transfers"
    TOPUP_BALANCE = "topup_balance"
    FX_INTERNATIONAL = "fx_international"
    ACCOUNT_SERVICING = "account_servicing"
    SECURITY_FRAUD = "security_fraud"
    LENDING_PRODUCTS = "lending_products"


class IntentPolicy(BaseModel):
    """One taxonomy row: what this intent is about and what we are allowed to do."""

    intent: str = Field(description="Banking77 intent name, exactly as the dataset spells it.")
    domain: Domain = Field(description="Owning business domain.")
    disposition: Disposition = Field(
        description="Best lane available to this intent. A ceiling, not a guarantee: "
        "abstention can downgrade it, nothing upgrades it."
    )
    risk_tier: RiskTier = Field(description="Blast radius if this message is handled wrongly.")
    sla_minutes: int = Field(description="Response target once a human holds the message.")
    autoreply_allowed: bool = Field(
        description="Independent second switch on the auto-reply path, redundant with disposition."
    )
    rationale: str = Field(description="Why this row is what it is. Written for hand-review.")

    @field_validator("sla_minutes")
    @classmethod
    def sla_must_be_on_the_ladder(cls, value: int) -> int:
        if value not in ALLOWED_SLA_MINUTES:
            raise ValueError(f"sla_minutes must be one of {ALLOWED_SLA_MINUTES}, got {value}")
        return value


def check_safety_invariants(policies: dict[str, IntentPolicy]) -> None:
    """Assert the rules the routing table must never break, whoever edits it.

    These are the properties a reviewer would check by eye across 77 rows. Checking
    them in code means the file cannot drift out of policy between sessions.
    """
    for intent, policy in policies.items():
        if policy.domain is Domain.SECURITY_FRAUD:
            if policy.disposition is not Disposition.HUMAN:
                raise ValueError(f"{intent}: security_fraud must route to human")
            if policy.autoreply_allowed:
                raise ValueError(f"{intent}: security_fraud must not allow auto-reply")

        # The two auto-reply switches must agree, in both directions.
        if policy.disposition is Disposition.AUTO_REPLY and not policy.autoreply_allowed:
            raise ValueError(f"{intent}: disposition is auto_reply but autoreply_allowed is false")
        if policy.autoreply_allowed and policy.disposition is not Disposition.AUTO_REPLY:
            raise ValueError(f"{intent}: autoreply_allowed is true but disposition is not auto_reply")

        if policy.risk_tier is RiskTier.CRITICAL and policy.autoreply_allowed:
            raise ValueError(f"{intent}: critical risk must not allow auto-reply")


def load_taxonomy() -> dict[str, IntentPolicy]:
    """Parse taxonomy.yaml, check it covers exactly the 77 canonical intents, return it.

    Raises rather than returning a partial table: a routing policy with a hole in
    it is more dangerous than no routing policy at all.
    """
    raw = yaml.safe_load(TAXONOMY_YAML.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{TAXONOMY_YAML} must be a mapping of intent name to policy")

    declared = set(raw)
    canonical = set(INTENT_NAMES)
    missing = sorted(canonical - declared)
    unknown = sorted(declared - canonical)
    if missing or unknown:
        raise ValueError(
            f"taxonomy.yaml does not match Banking77: missing {missing}, unknown {unknown}"
        )

    policies = {intent: IntentPolicy(intent=intent, **row) for intent, row in raw.items()}
    check_safety_invariants(policies)
    return policies


# Loaded once, at import, so a broken taxonomy stops the process at start-up rather
# than on the first customer message that happens to hit the broken row.
TAXONOMY: dict[str, IntentPolicy] = load_taxonomy()


def policy_for(intent: str) -> IntentPolicy:
    """Look up one row. Unknown intents raise — the orchestrator must fail closed."""
    if intent not in TAXONOMY:
        raise KeyError(f"{intent!r} is not a Banking77 intent in the taxonomy")
    return TAXONOMY[intent]
