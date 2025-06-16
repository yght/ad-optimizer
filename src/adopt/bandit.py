"""Traffic allocation across ad variants.

An A/B test splits traffic evenly until it reaches significance, which means
it keeps spending on the losing variant right up until the end. A bandit
shifts spend towards what is winning while it is still learning, which on a
campaign with real money on it is worth a great deal.

Starting with the model: a Beta posterior per variant. The allocator goes on
top of this.
"""

from __future__ import annotations

from dataclasses import dataclass


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
