"""Input guardrails: PII detection and redaction (CPR, PAN/Luhn, IBAN), prompt-injection screening, language detection, abuse and vulnerability flags.

Everything here is deterministic and cheap, and runs before the first LLM call.
That ordering is the point: a PII value that never enters a prompt cannot be
echoed back, logged, or learned from, and a distress signal caught here reaches a
human without waiting on a model that might route it wrong.

`pii_found` and `vulnerability_flags` carry category names only. No matched text
is ever returned, logged, or included in an exception message.
"""

import re
from pathlib import Path

import yaml
from langdetect import DetectorFactory, LangDetectException, detect

from src.schemas import InboundMessage, SanitisedMessage

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_YAML = REPO_ROOT / "config" / "settings.yaml"

_settings = yaml.safe_load(SETTINGS_YAML.read_text(encoding="utf-8"))["ingress"]
INJECTION_SCORE_PER_MATCH: float = _settings["injection_score_per_match"]
INJECTION_BLOCK_THRESHOLD: float = _settings["injection_block_threshold"]
MIN_CHARS_FOR_LANGUAGE: int = _settings["min_chars_for_language_detection"]

# langdetect is randomised by default and will return different answers for the
# same short string across runs. A router whose behaviour changes between
# identical inputs is not auditable.
DetectorFactory.seed = 0

# Order matters and is load-bearing. Email first because addresses contain digits
# that later patterns would otherwise claim. IBAN before the bare-digit patterns
# because its DK prefix makes it unambiguous. CPR before PAN and phone because its
# six-then-four shape is a subset of what those would match.
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("IBAN", re.compile(r"\bDK\d{2}[\s]?(?:\d{4}[\s]?){3}\d{2}\b", re.IGNORECASE)),
    ("CPR", re.compile(r"\b(\d{2})(\d{2})\d{2}-\d{4}\b")),
    ("PAN", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    ("PHONE", re.compile(r"(?:\+45[\s-]?)?\b\d{2}[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}\b")),
]

# Prompt-injection heuristics. Deliberately regex, not a model: at prototype scale
# this is honest and inspectable. The production answer is a dedicated classifier
# (Azure AI Content Safety Prompt Shields), because regex cannot survive
# paraphrase, encoding tricks, or a language it was not written for.
INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(previous|prior|earlier|above) (instruction|prompt|rule|direction)",
        r"disregard (all |any )?(previous|prior|the above)",
        r"(reveal|show|print|repeat|output) (me )?(your|the) (system )?(prompt|instruction)",
        r"developer mode|jailbreak|\bDAN mode\b",
        r"you are now (a|an|in)\b",
        r"pretend (to be|you are)|act as if you",
        r"(new|updated) instructions?:",
        r"</?(system|assistant|instruction)>",
    )
]

# Distress signals. Phrases are specific rather than single words on purpose:
# "died" alone fires on "my card died", which is a plausible banking message.
# The false-positive rate against all 10,003 Banking77 training messages is
# measured rather than assumed — see DECISIONS.md.
VULNERABILITY_PATTERNS: dict[str, list[str]] = {
    "bereavement": [
        r"passed away", r"\bdeceased\b", r"\bbereave", r"\bfuneral\b", r"\bmy late (husband|wife|partner|mother|father|son|daughter)\b",
        r"\b(he|she|they|husband|wife|partner|mother|father) (has |have )?died\b", r"death certificate", r"estate of the late",
    ],
    "debt_distress": [
        r"can(no|')?t afford", r"\bbailiff", r"\bevicted\b", r"\bhomeless\b", r"struggling to (pay|cope)",
        r"behind on (my )?payments", r"\bin serious debt\b", r"debt collector", r"cannot pay my", r"\brepossess",
    ],
    "self_harm": [
        r"kill myself", r"end my life", r"\bsuicide\b", r"\bsuicidal\b", r"self[- ]harm",
        r"want to die", r"no reason to (live|go on)", r"\bharm myself\b",
    ],
}


def luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum. A 16-digit number failing this is almost certainly not a card."""
    total = 0
    for index, character in enumerate(reversed(digits)):
        value = int(character)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def plausible_cpr(match: re.Match[str]) -> bool:
    """Reject impossible dates so ordinary hyphenated numbers are not called a CPR.

    Only the day and month are checked. Validating the modulus-11 control digit
    would reject the many genuine CPR numbers issued since 2007 that no longer
    satisfy it, which is the wrong error to make in a redaction guard.
    """
    day, month = int(match.group(1)), int(match.group(2))
    return 1 <= day <= 31 and 1 <= month <= 12


def redact_one_type(text: str, pii_type: str, pattern: re.Pattern[str], found: list[str]) -> str:
    """Replace every match of one PII pattern, appending the type to `found` if any matched.

    Taking `pii_type` as a parameter is what makes the inner function safe: it
    closes over an argument rather than a loop variable, so no default-argument
    binding trick is needed to pin the value.
    """

    def replace(match: re.Match[str]) -> str:
        if pii_type == "CPR" and not plausible_cpr(match):
            return match.group(0)
        if pii_type == "PAN" and not luhn_valid(re.sub(r"\D", "", match.group(0))):
            return match.group(0)
        if pii_type not in found:
            found.append(pii_type)
        return f"<{pii_type}>"

    return pattern.sub(replace, text)


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace every PII value with a typed placeholder. Returns the text and the types found.

    The placeholder keeps the shape of the message visible to the router — "my
    <PAN> was declined" still classifies correctly — while the value itself never
    travels past this function.
    """
    found: list[str] = []
    for pii_type, pattern in PII_PATTERNS:
        text = redact_one_type(text, pii_type, pattern, found)
    return text, found


def injection_score(text: str) -> float:
    """Heuristic likelihood that this message is trying to steer the model, in [0, 1]."""
    matches = sum(1 for pattern in INJECTION_PATTERNS if pattern.search(text))
    return min(1.0, matches * INJECTION_SCORE_PER_MATCH)


def vulnerability_flags(text: str) -> list[str]:
    """Distress categories present in the message. Category names only, never the phrase."""
    return [
        category
        for category, patterns in VULNERABILITY_PATTERNS.items()
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    ]


def detect_language(text: str) -> str:
    """ISO 639-1 code, or 'unknown'.

    langdetect is unreliable on short strings, which Banking77 messages mostly are.
    That limitation is documented in ASSUMPTIONS.md and measured in the adversarial
    suite rather than papered over with a heavier dependency.
    """
    if len(text) < MIN_CHARS_FOR_LANGUAGE:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def sanitise(message: InboundMessage) -> SanitisedMessage:
    """The first stage. Redact, screen, flag — then everything downstream uses the result.

    Redaction runs first so that language detection, injection screening and
    distress detection all operate on text that no longer contains PII values.
    """
    redacted, pii_found = redact_pii(message.text)
    score = injection_score(redacted)

    return SanitisedMessage(
        message_id=message.message_id,
        text_redacted=redacted,
        detected_language=detect_language(redacted),
        pii_found=pii_found,
        injection_score=score,
        blocked=score >= INJECTION_BLOCK_THRESHOLD,
        block_reason="prompt_injection" if score >= INJECTION_BLOCK_THRESHOLD else None,
        vulnerability_flags=vulnerability_flags(redacted),
    )
