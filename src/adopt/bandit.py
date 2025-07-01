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
