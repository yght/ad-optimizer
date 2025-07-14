"""Traffic allocation across ad variants.

An A/B test splits traffic evenly until it reaches significance, which means
it keeps spending on the losing variant right up until the end. A bandit
shifts spend towards what is winning while it is still learning, which on a
campaign with real money on it is worth a great deal.

Thompson sampling, specifically: model each variant's conversion rate as a
Beta posterior, draw one sample from each, and give the impression to
whichever sample came out highest. Variants that might be good get explored
in proportion to the probability that they actually are.

Everything here is pure and takes its randomness as a parameter, so the tests
are deterministic and a disputed allocation can be replayed exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Variant:
    """One creative, and what it has done so far.

    `conversions` is whatever the campaign optimises for - a click, a signup,
    a purchase. The maths does not care which, but mixing two of them inside
    one experiment quietly produces nonsense, so the caller is responsible for
    consistency.
    """

    id: str
    impressions: int = 0
    conversions: int = 0

    def __post_init__(self) -> None:
        if self.impressions < 0 or self.conversions < 0:
            raise ValueError(f"{self.id}: counts cannot be negative")
        if self.conversions > self.impressions:
            raise ValueError(
                f"{self.id}: {self.conversions} conversions from "
                f"{self.impressions} impressions is not possible"
            )

    @property
    def observed_rate(self) -> float:
        """Naive conversion rate. Useful for reporting, dangerous for deciding.

        A variant that converted the only impression it ever got has an
        observed rate of 100%, which is why nothing in this module makes a
        decision on this number alone.
        """
        if self.impressions == 0:
            return 0.0
        return self.conversions / self.impressions


@dataclass(frozen=True)
class Priors:
    """Beta prior applied to every variant.

    Beta(1, 1) is uniform - it says we know nothing, which is honest for a
    brand new creative and slow to move away from.

    A campaign with history should pass something informative. If the account
    historically converts at 2%, Beta(2, 98) encodes that belief with the
    weight of a hundred impressions, and a new variant then has to actually
    earn its way above it.
    """

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Beta parameters must be positive")


def posterior(variant: Variant, priors: Priors = Priors()) -> tuple[float, float]:
    """Beta posterior parameters for a variant.

    Conjugacy makes this addition rather than integration: alpha gains the
    successes, beta gains the failures.
    """
    successes = variant.conversions
    failures = variant.impressions - variant.conversions
    return (priors.alpha + successes, priors.beta + failures)


def posterior_mean(variant: Variant, priors: Priors = Priors()) -> float:
    """Where we actually think the rate is, prior included.

    This is the number to show a human, not `observed_rate` - it does not
    swing to 100% on a single lucky impression.
    """
    a, b = posterior(variant, priors)
    return a / (a + b)


def sample_rate(variant: Variant, rng: random.Random, priors: Priors = Priors()) -> float:
    """One draw from a variant's posterior."""
    a, b = posterior(variant, priors)
    return rng.betavariate(a, b)


def choose(
    variants: Sequence[Variant],
    rng: random.Random,
    priors: Priors = Priors(),
) -> str:
    """Pick one variant to serve the next impression.

    Ties break on the first variant in order rather than randomly, so a
    replay with the same seed and the same inputs gives the same answer.
    """
    if not variants:
        raise ValueError("No variants to choose from")

    best_id = variants[0].id
    best_draw = -1.0

    for variant in variants:
        draw = sample_rate(variant, rng, priors)
        if draw > best_draw:
            best_draw = draw
            best_id = variant.id

    return best_id


def allocation(
    variants: Sequence[Variant],
    rng: random.Random,
    draws: int = 10_000,
    priors: Priors = Priors(),
) -> dict[str, float]:
    """Share of traffic each variant should get, summing to 1.

    Estimated by sampling rather than solved in closed form - the integral has
    no general analytic solution past two variants, and Monte Carlo with ten
    thousand draws is accurate to well within the noise of the campaign
    itself.
    """
    if not variants:
        return {}

    wins = {v.id: 0 for v in variants}

    for _ in range(draws):
        wins[choose(variants, rng, priors)] += 1

    return {vid: count / draws for vid, count in wins.items()}


def probability_best(
    variants: Sequence[Variant],
    rng: random.Random,
    draws: int = 10_000,
    priors: Priors = Priors(),
) -> dict[str, float]:
    """Probability each variant is the best one.

    Identical computation to `allocation` - under Thompson sampling the share
    of traffic a variant earns *is* the probability it is best. They are named
    separately because they answer different questions and the callers want
    different things.
    """
    return allocation(variants, rng, draws, priors)


@dataclass(frozen=True)
class Guardrails:
    """Limits on what the sampler is allowed to do on its own.

    Pure Thompson sampling will starve a variant to near-zero traffic on the
    strength of a handful of impressions, and starved variants never recover
    because they never get the data that would redeem them.
    """

    min_impressions: int = 1_000
    """No variant may be paused before this many impressions, whatever the
    posterior says. Roughly a day of traffic on the campaigns this ran on."""

    explore_floor: float = 0.05
    """Every live variant keeps at least this share. Costs a little on a
    settled test and prevents the starvation problem entirely."""

    win_threshold: float = 0.95
    """Probability-of-best needed to call it and stop."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.explore_floor < 1.0:
            raise ValueError("explore_floor must be in [0, 1)")
        if not 0.5 < self.win_threshold <= 1.0:
            raise ValueError("win_threshold must be in (0.5, 1]")


def apply_floor(shares: dict[str, float], floor: float) -> dict[str, float]:
    """Lift every share to at least `floor`, then renormalise.

    Taken proportionally from the variants that are above the floor, so the
    ordering among the leaders is preserved.
    """
    if not shares:
        return {}

    if floor * len(shares) >= 1.0:
        # The floor cannot be satisfied - fall back to an even split rather
        # than producing shares that do not sum to one.
        even = 1.0 / len(shares)
        return {k: even for k in shares}

    below = {k: v for k, v in shares.items() if v < floor}
    if not below:
        return dict(shares)

    lifted = len(below) * floor
    remaining = 1.0 - lifted
    above_total = sum(v for k, v in shares.items() if k not in below)

    result = {}
    for key, value in shares.items():
        if key in below:
            result[key] = floor
        elif above_total > 0:
            result[key] = remaining * (value / above_total)
        else:
            result[key] = remaining / max(1, len(shares) - len(below))

    return result


@dataclass(frozen=True)
class Decision:
    """What to do with the experiment right now."""

    shares: dict[str, float]
    winner: str | None
    can_stop: bool
    reason: str


def decide(
    variants: Sequence[Variant],
    rng: random.Random,
    guardrails: Guardrails = Guardrails(),
    priors: Priors = Priors(),
    draws: int = 10_000,
) -> Decision:
    """Allocation plus a recommendation, with the guardrails applied."""
    if not variants:
        return Decision({}, None, False, "no variants")

    if len(variants) == 1:
        only = variants[0].id
        return Decision({only: 1.0}, only, False, "only one variant")

    probabilities = probability_best(variants, rng, draws, priors)
    leader = max(probabilities, key=lambda k: probabilities[k])
    total_impressions = sum(v.impressions for v in variants)
    least_seen = min(v.impressions for v in variants)

    shares = apply_floor(probabilities, guardrails.explore_floor)

    if least_seen < guardrails.min_impressions:
        return Decision(
            shares,
            None,
            False,
            f"still learning: {least_seen} impressions on the least-served variant, "
            f"need {guardrails.min_impressions}",
        )

    if probabilities[leader] >= guardrails.win_threshold:
        return Decision(
            shares,
            leader,
            True,
            f"{leader} is best with probability {probabilities[leader]:.1%} "
            f"after {total_impressions:,} impressions",
        )

    return Decision(
        shares,
        None,
        False,
        f"no clear winner: leader {leader} at {probabilities[leader]:.1%}, "
        f"below the {guardrails.win_threshold:.0%} threshold",
    )


def record(variants: Sequence[Variant], variant_id: str, converted: bool) -> list[Variant]:
    """Return the variant list with one impression recorded.

    Returns a new list; nothing is mutated, so an event that turns out to be
    a duplicate can simply not be applied.
    """
    updated = []
    found = False

    for variant in variants:
        if variant.id == variant_id:
            found = True
            updated.append(
                replace(
                    variant,
                    impressions=variant.impressions + 1,
                    conversions=variant.conversions + (1 if converted else 0),
                )
            )
        else:
            updated.append(variant)

    if not found:
        raise KeyError(f"No variant with id {variant_id!r}")

    return updated
