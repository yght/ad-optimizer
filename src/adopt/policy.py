"""Screening ad copy against advertising policy before it is submitted.

A rejected ad is not just a rejected ad. Enough of them and the account gets
a strike, and enough strikes suspends the account - which on a client's
account is a phone call you do not want to make.

A deterministic pass over terms that are always a problem. No model call, no
latency, no cost, and it catches most of what actually gets rejected. The
judgement calls need a model and come next.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

class Verdict(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    BLOCK = "block"


@dataclass(frozen=True)
class Finding:
    code: str
    verdict: Verdict
    excerpt: str
    reason: str
    source: str
    """'rules' or 'model' - worth keeping, because the two disagree often
    enough that it matters which one flagged something."""


# Terms that are a problem in most verticals regardless of context. Kept
# deliberately short: a long list produces false positives, and an over-eager
# screener that blocks good copy gets switched off by the people using it.
ALWAYS_PROBLEMATIC = [
    (r"\bguarantee[ds]?\b", "UNSUPPORTABLE_GUARANTEE",
     "Guarantees need documented terms behind them"),
    (r"\bcure[sd]?\b", "HEALTH_CLAIM",
     "Claiming to cure anything is a health claim"),
    (r"\brisk[- ]free\b", "UNSUPPORTABLE_GUARANTEE",
     "'Risk-free' is a guarantee and is treated as one"),
    (r"\bmiracle\b", "HEALTH_CLAIM", "Miracle claims are rejected on sight"),
    (r"\bFDA[- ]approved\b", "REGULATORY_CLAIM",
     "Regulatory approval claims need to be true and specific"),
    (r"\bno credit check\b", "LENDING_CLAIM",
     "Restricted in financial services advertising"),
    (r"\b(?:100|99)%\s+(?:effective|guaranteed|success)\b", "UNSUPPORTABLE_GUARANTEE",
     "An absolute efficacy claim"),
    (r"\bclick here\b", "LOW_QUALITY",
     "Not a policy violation but consistently underperforms"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), c, r) for p, c, r in ALWAYS_PROBLEMATIC]

BLOCKING_CODES = {
    "HEALTH_CLAIM", "REGULATORY_CLAIM", "LENDING_CLAIM", "UNSUPPORTABLE_GUARANTEE"
}


def screen_rules(texts: list[str]) -> list[Finding]:
    """Stage one. No model, no cost."""
    findings: list[Finding] = []

    for text in texts:
        for pattern, code, reason in _COMPILED:
            match = pattern.search(text)
            if match:
                findings.append(Finding(
                    code=code,
                    verdict=Verdict.BLOCK if code in BLOCKING_CODES else Verdict.REVIEW,
                    excerpt=match.group(),
                    reason=reason,
                    source="rules",
                ))

    return findings
