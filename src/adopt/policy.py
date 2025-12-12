"""Screening ad copy against advertising policy before it is submitted.

A rejected ad is not just a rejected ad. Enough of them and the account gets
a strike, and enough strikes suspends the account - which on a client's
account is a phone call you do not want to make.

Two stages, cheapest first:

1. A deterministic pass over terms that are always a problem. No model call,
   no latency, no cost, and it catches most of what actually gets rejected.
2. A model pass for the judgement calls - implied claims, misleading framing,
   the "before and after" pattern in health and finance.

Stage one runs first because a model call to tell us that "guaranteed weight
loss" is a problem is money spent on something a regex knew already.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import anthropic

from .costs import Usage

MODEL = "claude-opus-5"


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


SYSTEM = """You review advertising copy against platform policy.

You are the second stage. A deterministic filter has already caught explicit
banned terms, so you are looking for the things a keyword list cannot see:

- Claims that are implied rather than stated
- Comparative claims with no named comparison
- Copy that reads as a personal health or financial outcome
- Urgency that misrepresents genuine availability
- Copy whose meaning changes when the landing page does not match

Return "pass" when copy is fine. Most copy is fine, and a reviewer that flags
everything is one that gets ignored. Flag "review" when it depends on
substantiation the advertiser may well have. Reserve "block" for copy that
will be rejected however it is substantiated."""


@dataclass
class ScreenResult:
    findings: list[Finding]
    usage: Usage | None
    model_consulted: bool

    @property
    def verdict(self) -> Verdict:
        if any(f.verdict is Verdict.BLOCK for f in self.findings):
            return Verdict.BLOCK
        if any(f.verdict is Verdict.REVIEW for f in self.findings):
            return Verdict.REVIEW
        return Verdict.PASS


SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["pass", "review", "block"]},
                    "excerpt": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["code", "verdict", "excerpt", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}


def screen(
    texts: list[str],
    vertical: str,
    client: anthropic.Anthropic | None = None,
    always_consult_model: bool = False,
) -> ScreenResult:
    """Screen copy. Stage one always; stage two unless stage one already blocked.

    Once the rules have found something that will definitely be rejected,
    a model call adds nothing - the copy is going back to be rewritten either
    way. `always_consult_model` overrides that for audit runs where the full
    picture matters more than the cost.
    """
    findings = screen_rules(texts)

    already_blocked = any(f.verdict is Verdict.BLOCK for f in findings)
    if already_blocked and not always_consult_model:
        return ScreenResult(findings, usage=None, model_consulted=False)

    client = client or anthropic.Anthropic()

    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts))

    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=[
            {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}
        ],
        thinking={"type": "adaptive"},
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        # Policy screening is exactly the kind of prompt a safety classifier
        # may decline, since it is full of the language it is screening for.
        # Falling back keeps the pipeline moving instead of failing the batch.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{
            "role": "user",
            "content": f"Vertical: {vertical}\n\nCopy to review:\n{numbered}",
        }],
    )

    usage = Usage(
        model=MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
    )

    if response.stop_reason == "refusal":
        # The whole fallback chain declined. Treat as needing a human rather
        # than as a pass - failing open on a policy screener is the wrong
        # direction to fail.
        findings.append(Finding(
            code="SCREENING_UNAVAILABLE",
            verdict=Verdict.REVIEW,
            excerpt="",
            reason="Automated screening declined this copy; needs a human reviewer",
            source="model",
        ))
        return ScreenResult(findings, usage, model_consulted=True)

    import json
    text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)

    for item in parsed.get("findings", []):
        if item["verdict"] == "pass":
            continue
        findings.append(Finding(
            code=item["code"],
            verdict=Verdict(item["verdict"]),
            excerpt=item["excerpt"],
            reason=item["reason"],
            source="model",
        ))

    return ScreenResult(findings, usage, model_consulted=True)
