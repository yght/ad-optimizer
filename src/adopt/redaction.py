"""Strip identifiers out of text before it is sent to a model.

Campaign briefs arrive from clients as whatever they had lying around, and
that regularly includes a customer list pasted into the bottom of a document.
None of that needs to reach an inference API to write ad copy.

This is a reduction in exposure, not a guarantee. It catches the shapes that
actually turn up. It will not catch a name, and it is not a substitute for
telling clients not to paste customer data into a brief.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# North American and international shapes. Deliberately not trying to match
# every possible format - a loose phone pattern eats order numbers and prices.
PHONE = re.compile(
    r"(?<![\w-])(?:\+?1[\s.-]?)?"
    r"(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?![\w-])"
)

CREDIT_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# Provider key shapes. Worth catching because a key pasted into a brief and
# then sent to an API is a key in somebody else's logs.
API_KEY = re.compile(
    r"\b(?:sk-ant-[\w-]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36})\b"
)

# Canadian postal codes and US ZIP+4.
POSTAL = re.compile(r"\b(?:[A-Z]\d[A-Z][ -]?\d[A-Z]\d|\d{5}-\d{4})\b")


def luhn_ok(digits: str) -> bool:
    """Luhn check, so a 16-digit order number is not mistaken for a card."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


@dataclass(frozen=True)
class Redaction:
    text: str
    counts: dict[str, int]

    @property
    def redacted_anything(self) -> bool:
        return bool(self.counts)


def redact(text: str) -> Redaction:
    """Replace identifiers with typed placeholders.

    Placeholders are typed rather than blanked so the model can still tell
    that a phone number was there - copy referring to "call us" should still
    make sense.
    """
    counts: dict[str, int] = {}

    def swap(pattern: re.Pattern[str], label: str, value: str) -> str:
        def replace(match: re.Match[str]) -> str:
            counts[label] = counts.get(label, 0) + 1
            return f"[{label}]"

        return pattern.sub(replace, value)

    result = text
    # Keys first: an API key can contain something that looks like other
    # patterns, and a partly-redacted key is still a leaked key.
    result = swap(API_KEY, "API_KEY", result)
    result = swap(EMAIL, "EMAIL", result)

    def card_replace(match: re.Match[str]) -> str:
        digits = re.sub(r"[ -]", "", match.group())
        if len(digits) < 13 or not luhn_ok(digits):
            return match.group()
        counts["CARD"] = counts.get("CARD", 0) + 1
        return "[CARD]"

    result = CREDIT_CARD.sub(card_replace, result)
    result = swap(PHONE, "PHONE", result)
    result = swap(POSTAL, "POSTAL_CODE", result)

    return Redaction(result, counts)
