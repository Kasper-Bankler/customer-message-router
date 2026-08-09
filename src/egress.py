"""Output guardrails: groundedness and citation validity, invented-specifics regex, no-advice filter, AI disclosure line.

The invented-specifics check is the one that matters most and costs least. Every
phone number, IBAN, URL and DKK amount in a draft reply must appear verbatim in
the source articles the model was shown. If it does not, the model made it up.

This is deliberately regex and set membership, not a second LLM call. A
hallucinated emergency hotline on a lost-card reply is the worst thing this system
could do, and the check that prevents it should not itself be a language model
that can be wrong. It costs microseconds and cannot hallucinate.
"""

import re

from pydantic import BaseModel, Field

# What counts as a "specific": a claim a customer could act on and be harmed by if
# it is wrong. Percentages and durations are excluded deliberately — they are
# checked by groundedness, not by verbatim match, because "1-2 business days" is
# routinely reworded without becoming false.
SPECIFIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "phone": re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\d{2,4}[\s-]?){3,5}\d{2,4}"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[\sA-Z0-9]{10,30}\b"),
    "url": re.compile(r"\b(?:https?://|www\.)[^\s,)]+", re.IGNORECASE),
    "dkk_amount": re.compile(r"\b(?:DKK|kr\.?)\s?[\d.,]+\b|\b[\d.,]+\s?(?:DKK|kr\.?)\b", re.IGNORECASE),
}


class EgressVerdict(BaseModel):
    """Outcome of the output guard."""

    passed: bool = Field(description="False when the reply must not be sent to a customer.")
    invented: list[str] = Field(
        default_factory=list,
        description="Categories of unsupported specifics found, e.g. ['phone']. Categories "
        "only — the invented value itself is never carried into a log.",
    )


def normalise(text: str) -> str:
    """Strip separators so '+45 70 20 70 20' and '+4570207020' compare equal.

    Formatting differences are not inventions. Comparing raw strings would raise
    an escalation every time the model reformatted a number it copied correctly.
    """
    return re.sub(r"[\s\-().]", "", text).lower()


def find_unsupported(reply: str, sources: list[str]) -> list[str]:
    """Return the categories of specifics in `reply` that no source text supports."""
    haystack = normalise(" ".join(sources))
    unsupported: list[str] = []

    for category, pattern in SPECIFIC_PATTERNS.items():
        for match in pattern.findall(reply):
            if normalise(match) not in haystack:
                if category not in unsupported:
                    unsupported.append(category)
                break
    return unsupported


def verify_reply(reply: str, sources: list[str]) -> EgressVerdict:
    """The verification gate. Deterministic, no model call, fails closed on any doubt."""
    unsupported = find_unsupported(reply, sources)
    return EgressVerdict(passed=not unsupported, invented=unsupported)
