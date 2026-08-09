"""Unit tests for the output guard: every specific in a reply must be traceable to a source article.

The worst thing this system could do is put a plausible, invented emergency
hotline number in front of someone whose card has just been stolen. These tests
exist to make that impossible, so they are written as attacks rather than as
coverage.
"""

import pytest

from src.egress import find_unsupported, normalise, verify_reply

# Real text from doc_001, the lost-or-stolen-card article.
SOURCE = [
    "How do I block my credit card if it is lost or stolen?\n"
    "You can block your card via the mobile banking app under Cards > Block card, "
    "or our 24/7 emergency hotline at +45 70 20 70 20."
]
FEE_SOURCE = [
    "What fees apply for using my card abroad?\n"
    "Currency exchange fee: 1.5% of the transaction amount. "
    "ATM withdrawals: Up to DKK 30 per withdrawal depending on card type."
]


def test_invented_phone_number_is_caught_and_escalates() -> None:
    """The headline case: a number that appears nowhere in the source must not be sent."""
    reply = "Please call our emergency line on +45 33 44 55 66 to block your card immediately."

    verdict = verify_reply(reply, SOURCE)
    assert not verdict.passed
    assert "phone" in verdict.invented


def test_the_real_phone_number_from_the_source_passes() -> None:
    """False positives matter equally: a correctly copied number must not escalate."""
    reply = "You can block the card in the app, or call the 24/7 hotline on +45 70 20 70 20."

    verdict = verify_reply(reply, SOURCE)
    assert verdict.passed and verdict.invented == []


@pytest.mark.parametrize(
    "reformatted",
    ["+4570207020", "+45 70207020", "45-70-20-70-20", "(+45) 70 20 70 20"],
)
def test_reformatting_a_real_number_is_not_an_invention(reformatted: str) -> None:
    """Formatting differences are not hallucinations, and must not trigger an escalation."""
    assert find_unsupported(f"Call {reformatted} to block the card.", SOURCE) == []


def test_invented_dkk_amount_is_caught() -> None:
    reply = "Each withdrawal abroad costs DKK 95, according to our fee schedule."
    assert "dkk_amount" in find_unsupported(reply, FEE_SOURCE)


def test_real_dkk_amount_passes() -> None:
    reply = "ATM withdrawals abroad cost up to DKK 30 per withdrawal, depending on your card type."
    assert find_unsupported(reply, FEE_SOURCE) == []


def test_invented_url_is_caught() -> None:
    reply = "Full details are at https://danskebank.dk/not-a-real-page for your card type."
    assert "url" in find_unsupported(reply, FEE_SOURCE)


def test_invented_iban_is_caught() -> None:
    reply = "Transfer the amount to DK5000400440116243 and we will process it."
    assert "iban" in find_unsupported(reply, FEE_SOURCE)


def test_a_reply_with_no_specifics_at_all_passes() -> None:
    """Most grounded replies contain no phone number, IBAN, URL or amount."""
    reply = (
        "You can block your card yourself in the mobile banking app under Cards, then "
        "Block card. Ordering a replacement is available in the same place."
    )
    assert verify_reply(reply, SOURCE).passed


def test_normalise_strips_only_formatting() -> None:
    assert normalise("+45 70-20 (70).20") == "+45702070.20".replace(".", "")
    assert normalise("DKK 30") == "dkk30"


def test_verdict_reports_categories_never_the_invented_value() -> None:
    """A log line must not repeat the fabricated number back into the trace."""
    reply = "Call +45 33 44 55 66 now."
    verdict = verify_reply(reply, SOURCE)

    assert verdict.invented == ["phone"]
    assert "33 44 55 66" not in " ".join(verdict.invented)
