"""Generating ad copy with Claude, and refusing to trust it about lengths.

The model is good at writing headlines and bad at counting characters, which
is exactly the wrong way round for an ad platform that rejects anything one
character over. So nothing generated here is returned to the caller until it
has been through `specs.check_group`, and anything that fails goes back to the
model with the specific failures attached.

Copy is validated before it is returned. Repairing it automatically comes
next; for now the caller gets the issues and decides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel, Field

from .costs import Usage
from .redaction import redact
from .specs import SPECS, AssetType, Issue, Platform, check_group, is_publishable

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"


class AdCopy(BaseModel):
    """What we ask the model for. The field descriptions are part of the
    prompt as far as the model is concerned, so the limits are stated here
    as well as enforced afterwards."""

    headlines: list[str] = Field(
        description="Short headlines, each at most 30 characters including spaces",
        min_length=3,
        max_length=15,
    )
    descriptions: list[str] = Field(
        description="Descriptions, each at most 90 characters including spaces",
        min_length=2,
        max_length=4,
    )
    primary_text: str | None = Field(
        default=None,
        description=(
            "The main body text. Required for Meta and LinkedIn placements, "
            "which lead with it; leave null for Google and Microsoft search ads, "
            "which have no such field."
        ),
    )
    rationale: str = Field(
        description="One sentence on the angle these take and who they are aimed at"
    )


@dataclass
class CopyResult:
    copy: AdCopy
    issues: list[Issue]
    usage: list[Usage]
    repair_rounds: int
    redacted: dict[str, int]

    @property
    def publishable(self) -> bool:
        return is_publishable(self.issues)


SYSTEM = """You write search and social ad copy.

Rules that matter more than cleverness:
- Never invent a claim, a statistic, a price, or an award. If the brief does
  not say it, you do not have it.
- No superlatives you cannot support. "Fastest" is a claim; "fast" is not.
- Write for the person searching, not for the brand's self-image.
- Every headline must stand alone. The platform mixes and matches them, so
  headlines that only make sense in sequence will be shown out of order.

Character limits are counted by the platform, including spaces and
punctuation. Copy that runs over is rejected outright, so stay clearly under
the limit rather than exactly at it."""


def _required_assets(platform: Platform) -> set[AssetType]:
    """Which asset types this platform actually has.

    Search and social differ structurally, not just in length limits: Meta and
    LinkedIn lead with a body of primary text that Google search ads have no
    field for at all.
    """
    return {
        asset_type
        for asset_type, spec in SPECS.get(platform, {}).items()
        if spec.min_count > 0
    }


def _brief_prompt(brief: str, platform: Platform, count: int) -> str:
    wanted = _required_assets(platform)

    asks = [f"{count} headlines"]
    if AssetType.DESCRIPTION in wanted:
        asks.append("at least 2 descriptions")
    if AssetType.PRIMARY_TEXT in wanted:
        asks.append("one piece of primary text, which is the body people read first")

    return (
        f"Write ad copy for this campaign, targeting {platform.value}.\n\n"
        f"Brief:\n{brief}\n\n"
        f"Give {', and '.join(asks)}. Vary the angle between headlines - a set "
        f"that says the same thing five ways gives the platform nothing to "
        f"optimise with."
    )


def _required_assets(platform: Platform) -> set[AssetType]:
    """Which asset types this platform actually has.

    Search and social differ structurally, not just in length limits: Meta and
    LinkedIn lead with a body of primary text that Google search ads have no
    field for at all.
    """
    return {
        asset_type
        for asset_type, spec in SPECS.get(platform, {}).items()
        if spec.min_count > 0
    }


def _brief_prompt(brief: str, platform: Platform, count: int) -> str:
    wanted = _required_assets(platform)

    asks = [f"{count} headlines"]
    if AssetType.DESCRIPTION in wanted:
        asks.append("at least 2 descriptions")
    if AssetType.PRIMARY_TEXT in wanted:
        asks.append("one piece of primary text, which is the body people read first")

    return (
        f"Write ad copy for this campaign, targeting {platform.value}.\n\n"
        f"Brief:\n{brief}\n\n"
        f"Give {', and '.join(asks)}. Vary the angle between headlines - a set "
        f"that says the same thing five ways gives the platform nothing to "
        f"optimise with."
    )


def _repair_prompt(issues: list[Issue]) -> str:
    """Feed the validator's output back verbatim.

    Being specific matters. 'Some headlines are too long' produces another
    round of guesses; 'headline 3 is 34 characters, limit is 30' gets fixed.
    """
    lines = []
    for issue in issues:
        if not issue.fatal:
            continue
        where = f"{issue.asset_type.value} {issue.index}" if issue.index >= 0 else issue.asset_type.value
        lines.append(f"- {where}: {issue.message}")

    return (
        "The platform rejected some of that copy:\n\n"
        + "\n".join(lines)
        + "\n\nRewrite only the assets listed. Leave the others exactly as they "
        "were - they passed, and rewriting them risks breaking them."
    )


def generate(
    brief: str,
    platform: Platform,
    client: anthropic.Anthropic | None = None,
    count: int = 5,
    effort: str = "medium",
) -> CopyResult:
    """Generate ad copy that actually fits the platform.

    `effort` is medium by default. Ad copy is not a reasoning problem and
    high effort produces the same headlines for more money; raise it if the
    brief is genuinely complicated.
    """
    client = client or anthropic.Anthropic()

    # The brief comes from a client and regularly has a customer list pasted
    # at the bottom of it.
    cleaned = redact(brief)
    if cleaned.redacted_anything:
        log.info("redacted before sending: %s", cleaned.counts)

    messages: list[anthropic.types.MessageParam] = [
        {"role": "user", "content": _brief_prompt(cleaned.text, platform, count)}
    ]

    usages: list[Usage] = []
    result: AdCopy | None = None
    issues: list[Issue] = []

    if True:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=4096,
            system=[
                # The system prompt never changes, so it is worth caching -
                # a campaign generates dozens of these in a batch.
                {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=messages,
            output_format=AdCopy,
        )

        usages.append(_usage_from(response))

        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise CopyRefused(
                f"The model declined this brief (category: {category}). "
                f"Check the brief for regulated claims."
            )

        result = response.parsed_output
        issues = check_group(_assets_for(result, platform), platform)

        return CopyResult(result, issues, usages, 0, cleaned.counts)

    raise AssertionError("unreachable")


def _assets_for(copy: AdCopy, platform: Platform) -> dict[AssetType, list[str]]:
    """Lay the generated copy out as the platform's asset types.

    Only the types the platform has - handing a Google asset group a
    primary_text it has no field for produces a spurious rejection.
    """
    available = SPECS.get(platform, {})
    assets: dict[AssetType, list[str]] = {}

    if AssetType.HEADLINE in available:
        assets[AssetType.HEADLINE] = copy.headlines
    if AssetType.DESCRIPTION in available:
        assets[AssetType.DESCRIPTION] = copy.descriptions
    if AssetType.PRIMARY_TEXT in available and copy.primary_text:
        assets[AssetType.PRIMARY_TEXT] = [copy.primary_text]

    return assets


def _usage_from(response) -> Usage:
    usage = response.usage
    return Usage(
        model=MODEL,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


class CopyRefused(RuntimeError):
    """The model declined the brief. Usually a regulated vertical - health
    claims, lending rates, that sort of thing - and usually correct to
    decline."""
