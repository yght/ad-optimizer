import pytest

from adopt.costs import Ledger, Usage, UnknownModel


class TestUsage:
    def test_prices_a_plain_call(self):
        # 1M in, 1M out on Opus 5 is $5 + $25.
        usage = Usage("claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000)
        assert usage.cost_usd() == pytest.approx(30.00)

    def test_prices_a_realistic_call(self):
        usage = Usage("claude-opus-5", input_tokens=3_000, output_tokens=800)
        assert usage.cost_usd() == pytest.approx(3_000 / 1e6 * 5 + 800 / 1e6 * 25)

    def test_cache_reads_are_a_tenth_of_input(self):
        cached = Usage("claude-opus-5", cache_read_tokens=1_000_000)
        assert cached.cost_usd() == pytest.approx(0.50)

    def test_cache_writes_carry_a_premium(self):
        written = Usage("claude-opus-5", cache_write_tokens=1_000_000)
        assert written.cost_usd() == pytest.approx(6.25)

    def test_a_cheaper_model_is_cheaper(self):
        args = dict(input_tokens=100_000, output_tokens=10_000)
        assert Usage("claude-haiku-4-5", **args).cost_usd() < Usage("claude-opus-5", **args).cost_usd()

    def test_an_unknown_model_raises_rather_than_guessing(self):
        with pytest.raises(UnknownModel, match="Add it rather than guessing"):
            Usage("some-future-model", input_tokens=1000).cost_usd()

    def test_an_empty_call_costs_nothing(self):
        assert Usage("claude-opus-5").cost_usd() == 0.0

    def test_total_tokens_counts_every_bucket(self):
        usage = Usage("claude-opus-5", 1, 2, 3, 4)
        assert usage.total_tokens == 10


class TestLedger:
    def test_sums_across_calls(self):
        ledger = Ledger()
        ledger.add(Usage("claude-opus-5", input_tokens=1_000_000))
        ledger.add(Usage("claude-opus-5", output_tokens=1_000_000))
        assert ledger.total_usd() == pytest.approx(30.00)

    def test_splits_by_model(self):
        ledger = Ledger()
        ledger.add(Usage("claude-opus-5", input_tokens=1_000_000))
        ledger.add(Usage("claude-haiku-4-5", input_tokens=1_000_000))
        assert ledger.by_model() == {
            "claude-opus-5": pytest.approx(5.0),
            "claude-haiku-4-5": pytest.approx(1.0),
        }

    def test_reports_what_the_cache_saved(self):
        ledger = Ledger()
        ledger.add(Usage("claude-opus-5", cache_read_tokens=1_000_000))
        # Full price would be $5; we paid $0.50.
        assert ledger.cache_savings_usd() == pytest.approx(4.50)

    def test_no_savings_when_nothing_was_cached(self):
        ledger = Ledger()
        ledger.add(Usage("claude-opus-5", input_tokens=1_000_000))
        assert ledger.cache_savings_usd() == 0.0

    def test_an_empty_ledger_is_free(self):
        assert Ledger().total_usd() == 0.0
