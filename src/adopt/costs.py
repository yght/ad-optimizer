"""What a run cost.

Token spend on a campaign generation run is small per call and not small
across a month of them. Every model call in this package returns its usage,
and this turns that into money so it can go on a dashboard next to the ad
spend it is supposed to be improving.

Prices are per million tokens, in USD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PER_MILLION: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# A cache read is a tenth of the input price; writing to the cache carries a
# 25% premium over the base input rate.
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


class UnknownModel(KeyError):
    pass


@dataclass(frozen=True)
class Usage:
    """One call's token counts, as the API reports them."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def cost_usd(self) -> float:
        try:
            input_rate, output_rate = PER_MILLION[self.model]
        except KeyError as exc:
            raise UnknownModel(
                f"No price for {self.model!r}. Add it rather than guessing - "
                f"a silently wrong rate is worse than a crash."
            ) from exc

        million = 1_000_000
        return (
            self.input_tokens / million * input_rate
            + self.output_tokens / million * output_rate
            + self.cache_read_tokens / million * input_rate * CACHE_READ_MULTIPLIER
            + self.cache_write_tokens / million * input_rate * CACHE_WRITE_MULTIPLIER
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass
class Ledger:
    """Running total across a batch of calls."""

    entries: list[Usage] = field(default_factory=list)

    def add(self, usage: Usage) -> None:
        self.entries.append(usage)

    def total_usd(self) -> float:
        return sum(entry.cost_usd() for entry in self.entries)

    def by_model(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for entry in self.entries:
            totals[entry.model] = totals.get(entry.model, 0.0) + entry.cost_usd()
        return totals

    def cache_savings_usd(self) -> float:
        """What the cache saved against paying full input price for those
        tokens. Worth reporting - it is usually the difference between this
        being cheap and being an argument with finance."""
        saved = 0.0
        for entry in self.entries:
            if not entry.cache_read_tokens:
                continue
            input_rate, _ = PER_MILLION[entry.model]
            full = entry.cache_read_tokens / 1_000_000 * input_rate
            paid = full * CACHE_READ_MULTIPLIER
            saved += full - paid
        return saved
