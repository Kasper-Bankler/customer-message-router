"""Guards the routing table: that INTENT_NAMES still matches the real dataset, that taxonomy.yaml covers it exactly, and that the safety invariants actually reject a bad edit."""

import pytest
from pydantic import ValidationError

import src.taxonomy
from src.banking77 import INTENT_NAMES, load_banking77
from src.schemas import Disposition, RiskTier
from src.taxonomy import TAXONOMY, Domain, IntentPolicy, check_safety_invariants, policy_for


def make_policy(**overrides: object) -> IntentPolicy:
    """A valid low-risk row, so each test can break exactly one field and nothing else."""
    defaults = {
        "intent": "test_intent",
        "domain": Domain.CARDS_ISSUING,
        "disposition": Disposition.AUTO_REPLY,
        "risk_tier": RiskTier.LOW,
        "sla_minutes": 240,
        "autoreply_allowed": True,
        "rationale": "Fixture row.",
    }
    return IntentPolicy(**(defaults | overrides))


def test_intent_names_matches_the_live_dataset() -> None:
    """INTENT_NAMES is hardcoded, so it has to be checked against the CSVs we actually load.

    This is the test that catches the two odd labels silently changing:
    'Refund_not_showing_up' is capitalised and 'reverted_card_payment?' ends in '?'.
    """
    labels = set(load_banking77("train").intent) | set(load_banking77("test").intent)
    assert labels == set(INTENT_NAMES)
    assert len(INTENT_NAMES) == 77


def test_taxonomy_covers_every_intent_exactly_once() -> None:
    assert set(TAXONOMY) == set(INTENT_NAMES)


def test_policy_for_rejects_an_unknown_intent() -> None:
    with pytest.raises(KeyError):
        policy_for("intent_that_does_not_exist")


def write_broken_copy(tmp_path, monkeypatch, find: str, replace: str) -> None:
    """Point the loader at a copy of the real taxonomy with one edit applied."""
    broken = src.taxonomy.TAXONOMY_YAML.read_text(encoding="utf-8").replace(find, replace)
    path = tmp_path / "taxonomy.yaml"
    path.write_text(broken, encoding="utf-8")
    monkeypatch.setattr(src.taxonomy, "TAXONOMY_YAML", path)


def test_loader_rejects_a_renamed_intent(tmp_path, monkeypatch) -> None:
    """A typo in an intent key must be caught, not silently leave that intent unroutable."""
    write_broken_copy(tmp_path, monkeypatch, "age_limit:\n", "age_limt:\n")
    with pytest.raises(ValueError, match="missing \\['age_limit'\\]"):
        src.taxonomy.load_taxonomy()


def test_loader_rejects_an_illegal_enum_value(tmp_path, monkeypatch) -> None:
    write_broken_copy(tmp_path, monkeypatch, "disposition: auto_reply", "disposition: autoreply")
    with pytest.raises(ValidationError):
        src.taxonomy.load_taxonomy()


def test_security_fraud_cannot_be_auto_replied() -> None:
    """The invariant that matters most: no edit can open the auto-reply path for fraud."""
    with pytest.raises(ValueError, match="security_fraud must route to human"):
        check_safety_invariants(
            {"x": make_policy(domain=Domain.SECURITY_FRAUD, autoreply_allowed=False)}
        )


def test_the_two_autoreply_switches_must_agree() -> None:
    with pytest.raises(ValueError, match="autoreply_allowed is false"):
        check_safety_invariants({"x": make_policy(autoreply_allowed=False)})
    with pytest.raises(ValueError, match="disposition is not auto_reply"):
        check_safety_invariants({"x": make_policy(disposition=Disposition.HUMAN)})


def test_critical_risk_cannot_be_auto_replied() -> None:
    with pytest.raises(ValueError, match="critical risk"):
        check_safety_invariants({"x": make_policy(risk_tier=RiskTier.CRITICAL)})


def test_sla_must_be_on_the_documented_ladder() -> None:
    with pytest.raises(ValueError, match="sla_minutes must be one of"):
        make_policy(sla_minutes=45)


def test_every_fraud_row_is_forced_to_a_human() -> None:
    """The invariant checker runs at import; this asserts the shipped file actually has fraud rows."""
    fraud = [p for p in TAXONOMY.values() if p.domain is Domain.SECURITY_FRAUD]
    assert len(fraud) == 7
    assert all(p.disposition is Disposition.HUMAN and not p.autoreply_allowed for p in fraud)
