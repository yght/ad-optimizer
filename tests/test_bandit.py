"""Tests for the Thompson sampling allocator.

Every test seeds its own Random, so a failure here is reproducible rather
than something that shows up one run in twenty.
"""

import random

import pytest

from adopt.bandit import (
    Decision,
    Guardrails,
    Priors,
    Variant,
    allocation,
    apply_floor,
    choose,
    decide,
    posterior,
    posterior_mean,
    probability_best,
    record,
)


def rng():
    return random.Random(20260215)


class TestVariant:
    def test_rejects_more_conversions_than_impressions(self):
        with pytest.raises(ValueError, match="not possible"):
            Variant("a", impressions=10, conversions=11)

    def test_rejects_negative_counts(self):
        with pytest.raises(ValueError, match="negative"):
            Variant("a", impressions=-1)

    def test_observed_rate_of_an_unserved_variant_is_zero_not_an_error(self):
        assert Variant("a").observed_rate == 0.0

    def test_observed_rate_is_naive_on_purpose(self):
        # One impression, one conversion, 100%. This is exactly why nothing
        # decides on this number.
        assert Variant("a", 1, 1).observed_rate == 1.0


class TestPosterior:
    def test_conjugacy_is_addition(self):
        v = Variant("a", impressions=100, conversions=10)
        assert posterior(v) == (11.0, 91.0)  # 1+10, 1+90

    def test_an_unserved_variant_returns_the_prior(self):
        assert posterior(Variant("a")) == (1.0, 1.0)

    def test_posterior_mean_is_pulled_towards_the_prior(self):
        # Observed 100%, but on one impression the posterior says 67%.
        v = Variant("a", impressions=1, conversions=1)
        assert v.observed_rate == 1.0
        assert posterior_mean(v) == pytest.approx(2 / 3)

    def test_the_prior_matters_less_as_evidence_accumulates(self):
        # A uniform prior pulls towards 0.5, so use a rate that is not 0.5 -
        # at exactly 50% the prior and the data agree and there is nothing
        # to observe.
        light = posterior_mean(Variant("a", 10, 2))
        heavy = posterior_mean(Variant("a", 10_000, 2_000))
        assert abs(heavy - 0.2) < abs(light - 0.2)

    def test_an_informative_prior_holds_a_new_variant_down(self):
        # The account converts at 2% historically. A variant with 1 conversion
        # from 2 impressions should not be believed to convert at 50%.
        account = Priors(alpha=2, beta=98)
        assert posterior_mean(Variant("new", 2, 1), account) < 0.05

    def test_rejects_a_non_positive_prior(self):
        with pytest.raises(ValueError, match="positive"):
            Priors(alpha=0, beta=1)


class TestChoose:
    def test_raises_on_an_empty_field(self):
        with pytest.raises(ValueError, match="No variants"):
            choose([], rng())

    def test_a_single_variant_always_wins(self):
        assert choose([Variant("only", 10, 1)], rng()) == "only"

    def test_is_reproducible_for_a_given_seed(self):
        variants = [Variant("a", 100, 10), Variant("b", 100, 12)]
        first = [choose(variants, random.Random(7)) for _ in range(5)]
        second = [choose(variants, random.Random(7)) for _ in range(5)]
        assert first == second


class TestAllocation:
    def test_shares_sum_to_one(self):
        variants = [Variant("a", 500, 50), Variant("b", 500, 40), Variant("c", 500, 30)]
        shares = allocation(variants, rng(), draws=2000)
        assert sum(shares.values()) == pytest.approx(1.0)

    def test_the_better_variant_gets_more_traffic(self):
        variants = [Variant("good", 2000, 200), Variant("bad", 2000, 100)]
        shares = allocation(variants, rng(), draws=3000)
        assert shares["good"] > shares["bad"]

    def test_a_clear_winner_takes_almost_everything(self):
        variants = [Variant("good", 5000, 500), Variant("bad", 5000, 150)]
        shares = allocation(variants, rng(), draws=3000)
        assert shares["good"] > 0.99

    def test_indistinguishable_variants_split_evenly(self):
        variants = [Variant("a", 1000, 100), Variant("b", 1000, 100)]
        shares = allocation(variants, rng(), draws=4000)
        assert shares["a"] == pytest.approx(0.5, abs=0.05)

    def test_uncertainty_earns_traffic(self):
        # 'unknown' has a worse observed rate but has barely been tried, so
        # it should still get a meaningful share. This is the entire point of
        # the approach over a greedy split.
        variants = [Variant("known", 5000, 250), Variant("unknown", 20, 0)]
        shares = allocation(variants, rng(), draws=3000)
        assert shares["unknown"] > 0.02

    def test_an_empty_field_allocates_nothing(self):
        assert allocation([], rng()) == {}


class TestFloor:
    @pytest.mark.parametrize("shares", [
        {"a": 0.0, "b": 0.05, "c": 0.95},
        {"a": 0.0, "b": 0.051, "c": 0.949},
        {"a": 0.0, "b": 0.01, "c": 0.052, "d": 0.938},
    ])
    def test_redistribution_never_starves_another_variant(self, shares):
        original = dict(shares)
        lifted = apply_floor(shares, 0.05)
        assert min(lifted.values()) >= 0.05
        assert sum(lifted.values()) == pytest.approx(1.0)
        assert shares == original

    def test_lifts_a_starved_variant(self):
        lifted = apply_floor({"a": 0.99, "b": 0.01}, 0.05)
        assert lifted["b"] == pytest.approx(0.05)
        assert sum(lifted.values()) == pytest.approx(1.0)

    def test_leaves_shares_alone_when_all_are_above(self):
        shares = {"a": 0.6, "b": 0.4}
        assert apply_floor(shares, 0.05) == shares

    def test_preserves_the_order_of_the_leaders(self):
        lifted = apply_floor({"a": 0.7, "b": 0.29, "c": 0.01}, 0.05)
        assert lifted["a"] > lifted["b"] > lifted["c"]

    def test_falls_back_to_an_even_split_when_the_floor_cannot_be_met(self):
        # Four variants cannot all have 30%.
        lifted = apply_floor({k: 0.25 for k in "abcd"}, 0.30)
        assert sum(lifted.values()) == pytest.approx(1.0)
        assert all(v == pytest.approx(0.25) for v in lifted.values())

    def test_handles_nothing(self):
        assert apply_floor({}, 0.05) == {}


class TestDecide:
    def test_will_not_call_a_winner_before_the_minimum_impressions(self):
        # Overwhelming evidence, not enough of it.
        variants = [Variant("a", 100, 50), Variant("b", 100, 1)]
        decision = decide(variants, rng(), Guardrails(min_impressions=1000), draws=2000)
        assert decision.winner is None
        assert decision.can_stop is False
        assert "still learning" in decision.reason

    def test_calls_a_winner_once_there_is_enough_data(self):
        variants = [Variant("a", 5000, 500), Variant("b", 5000, 150)]
        decision = decide(variants, rng(), Guardrails(min_impressions=1000), draws=2000)
        assert decision.winner == "a"
        assert decision.can_stop is True

    def test_declines_to_call_a_close_race(self):
        variants = [Variant("a", 5000, 250), Variant("b", 5000, 245)]
        decision = decide(variants, rng(), Guardrails(min_impressions=1000), draws=3000)
        assert decision.winner is None
        assert "no clear winner" in decision.reason

    def test_keeps_the_loser_alive_at_the_floor_even_after_deciding(self):
        variants = [Variant("a", 5000, 500), Variant("b", 5000, 150)]
        decision = decide(variants, rng(), Guardrails(min_impressions=1000), draws=2000)
        assert decision.shares["b"] == pytest.approx(0.05)

    def test_a_single_variant_is_not_an_experiment(self):
        decision = decide([Variant("only", 10_000, 500)], rng())
        assert decision.shares == {"only": 1.0}
        assert decision.can_stop is False

    def test_no_variants_is_handled_rather_than_thrown(self):
        decision = decide([], rng())
        assert decision.winner is None
        assert decision.reason == "no variants"

    def test_guardrails_reject_a_nonsensical_threshold(self):
        with pytest.raises(ValueError):
            Guardrails(win_threshold=0.4)
        with pytest.raises(ValueError):
            Guardrails(explore_floor=1.5)


class TestRecord:
    def test_records_an_impression_without_a_conversion(self):
        updated = record([Variant("a", 10, 2)], "a", converted=False)
        assert updated[0].impressions == 11
        assert updated[0].conversions == 2

    def test_records_a_conversion(self):
        updated = record([Variant("a", 10, 2)], "a", converted=True)
        assert updated[0] == Variant("a", 11, 3)

    def test_leaves_the_other_variants_untouched(self):
        variants = [Variant("a", 10, 2), Variant("b", 20, 5)]
        updated = record(variants, "a", converted=True)
        assert updated[1] is variants[1]

    def test_does_not_mutate_the_input(self):
        variants = [Variant("a", 10, 2)]
        record(variants, "a", converted=True)
        assert variants[0].impressions == 10

    def test_an_unknown_variant_is_an_error_not_a_silent_no_op(self):
        with pytest.raises(KeyError):
            record([Variant("a", 10, 2)], "ghost", converted=True)


class TestConvergence:
    def test_the_sampler_finds_the_better_creative_from_scratch(self):
        """End to end: two unknown creatives, simulated traffic, and the
        allocator should end up serving the better one almost exclusively."""
        truth = {"a": 0.05, "b": 0.02}
        variants = [Variant("a"), Variant("b")]
        world = random.Random(99)
        sampler = random.Random(1234)

        for _ in range(4000):
            chosen = choose(variants, sampler)
            converted = world.random() < truth[chosen]
            variants = record(variants, chosen, converted)

        served = {v.id: v.impressions for v in variants}
        assert served["a"] > served["b"] * 3

        final = probability_best(variants, random.Random(5), draws=3000)
        assert final["a"] > 0.95
