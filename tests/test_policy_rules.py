"""Stage one only. The model stage is exercised in test_copy_repair.py with
a stub client; these cover the deterministic filter, which is what actually
runs on every piece of copy."""

from adopt.policy import Verdict, screen_rules


def codes(findings):
    return {f.code for f in findings}


class TestRules:
    def test_passes_ordinary_copy(self):
        assert screen_rules(["Fast hosting for growing teams", "Move in an afternoon"]) == []

    def test_blocks_a_guarantee(self):
        findings = screen_rules(["Guaranteed results in 30 days"])
        assert "UNSUPPORTABLE_GUARANTEE" in codes(findings)
        assert findings[0].verdict is Verdict.BLOCK

    def test_catches_the_inflected_forms(self):
        for text in ("We guarantee it", "Guaranteed savings", "Guarantees results"):
            assert screen_rules([text]), text

    def test_blocks_a_health_claim(self):
        assert "HEALTH_CLAIM" in codes(screen_rules(["Cures back pain fast"]))

    def test_blocks_risk_free_framing(self):
        assert "UNSUPPORTABLE_GUARANTEE" in codes(screen_rules(["Risk-free trial"]))
        assert "UNSUPPORTABLE_GUARANTEE" in codes(screen_rules(["Risk free trial"]))

    def test_blocks_a_lending_claim(self):
        assert "LENDING_CLAIM" in codes(screen_rules(["No credit check required"]))

    def test_flags_click_here_without_blocking(self):
        findings = screen_rules(["Click here to learn more"])
        assert findings[0].verdict is Verdict.REVIEW

    def test_is_case_insensitive(self):
        assert screen_rules(["GUARANTEED savings"])

    def test_does_not_fire_on_a_substring_inside_another_word(self):
        # 'secured' contains 'cure'. Word boundaries matter, and a screener
        # that blocks 'secured hosting' gets turned off.
        assert screen_rules(["Secured hosting for teams"]) == []

    def test_does_not_fire_on_accuracy_or_obscure(self):
        assert screen_rules(["Accurate reporting", "No obscure fees"]) == []

    def test_reports_the_matching_excerpt(self):
        finding = screen_rules(["Totally risk-free today"])[0]
        assert finding.excerpt.lower() == "risk-free"

    def test_marks_the_source_so_the_two_stages_are_distinguishable(self):
        assert screen_rules(["Guaranteed"])[0].source == "rules"

    def test_screens_every_string_it_is_given(self):
        findings = screen_rules(["Fine copy", "Guaranteed results", "Also fine"])
        assert len(findings) == 1

    def test_handles_nothing(self):
        assert screen_rules([]) == []
