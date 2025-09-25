import pytest

from adopt.specs import (
    AssetType,
    Platform,
    check_group,
    check_text,
    counted_length,
    is_publishable,
)


def codes(issues):
    return {i.code for i in issues}


class TestCounting:
    def test_ascii_counts_one_each(self):
        assert counted_length("hello", Platform.GOOGLE) == 5

    def test_google_counts_cjk_as_two(self):
        # Fifteen Japanese characters is exactly the 30-char headline limit.
        assert counted_length("青" * 15, Platform.GOOGLE) == 30

    def test_microsoft_matches_google(self):
        assert counted_length("青" * 15, Platform.MICROSOFT) == 30

    def test_meta_counts_code_points(self):
        assert counted_length("青" * 15, Platform.META) == 15

    def test_full_width_latin_also_counts_double(self):
        assert counted_length("Ａ" * 10, Platform.GOOGLE) == 20


class TestHeadlines:
    def test_accepts_a_normal_headline(self):
        assert check_text("Free shipping on all orders", Platform.GOOGLE, AssetType.HEADLINE) == []

    def test_rejects_one_character_over(self):
        issues = check_text("a" * 31, Platform.GOOGLE, AssetType.HEADLINE)
        assert "TOO_LONG" in codes(issues)
        assert not is_publishable(issues)

    def test_accepts_exactly_the_limit(self):
        assert check_text("a" * 30, Platform.GOOGLE, AssetType.HEADLINE) == []

    def test_rejects_cjk_that_passes_a_naive_length_check(self):
        # len() says 20, which would pass. The platform says 40.
        headline = "青" * 20
        assert len(headline) == 20
        assert "TOO_LONG" in codes(check_text(headline, Platform.GOOGLE, AssetType.HEADLINE))

    def test_the_same_cjk_headline_is_fine_on_meta(self):
        assert check_text("青" * 20, Platform.META, AssetType.HEADLINE) == []

    def test_rejects_an_empty_headline(self):
        assert "EMPTY" in codes(check_text("   ", Platform.GOOGLE, AssetType.HEADLINE))

    def test_flags_surrounding_whitespace_without_rejecting(self):
        issues = check_text(" Free shipping ", Platform.GOOGLE, AssetType.HEADLINE)
        assert "WHITESPACE" in codes(issues)
        assert is_publishable(issues)


class TestGooglePolicy:
    def test_rejects_emoji(self):
        assert "EMOJI" in codes(check_text("Save now 🎉", Platform.GOOGLE, AssetType.HEADLINE))

    def test_allows_emoji_on_meta(self):
        assert check_text("Save now 🎉", Platform.META, AssetType.HEADLINE) == []

    def test_rejects_repeated_punctuation(self):
        assert "REPEATED_PUNCTUATION" in codes(
            check_text("Buy now!!", Platform.GOOGLE, AssetType.HEADLINE))

    def test_allows_a_single_exclamation(self):
        assert check_text("Buy now!", Platform.GOOGLE, AssetType.HEADLINE) == []

    def test_rejects_a_shouted_word(self):
        assert "ALL_CAPS" in codes(
            check_text("SAVE on everything", Platform.GOOGLE, AssetType.HEADLINE))

    def test_allows_known_acronyms(self):
        # These are how the words are written, not shouting.
        for text in ("HIPAA compliant hosting", "Fast CRM for SaaS teams", "GDPR ready"):
            assert check_text(text, Platform.GOOGLE, AssetType.HEADLINE) == [], text

    def test_allows_short_capitalised_words(self):
        assert check_text("Try our API now", Platform.GOOGLE, AssetType.HEADLINE) == []


class TestTruncation:
    def test_warns_without_rejecting_past_the_recommended_length(self):
        issues = check_text("x" * 200, Platform.LINKEDIN, AssetType.PRIMARY_TEXT)
        assert "TRUNCATION_RISK" in codes(issues)
        assert is_publishable(issues)

    def test_rejects_past_the_hard_limit(self):
        issues = check_text("x" * 700, Platform.LINKEDIN, AssetType.PRIMARY_TEXT)
        assert "TOO_LONG" in codes(issues)
        assert not is_publishable(issues)

    def test_google_headlines_have_no_truncation_tier(self):
        # The limit is the limit; there is no soft warning band.
        assert check_text("x" * 30, Platform.GOOGLE, AssetType.HEADLINE) == []


class TestAssetGroups:
    def valid_google(self):
        return {
            AssetType.HEADLINE: ["Fast hosting", "Free migration", "24/7 support"],
            AssetType.DESCRIPTION: ["Move your site in an afternoon.", "No downtime, no fuss."],
        }

    def test_accepts_a_valid_group(self):
        assert check_group(self.valid_google(), Platform.GOOGLE) == []

    def test_rejects_too_few_headlines(self):
        assets = self.valid_google()
        assets[AssetType.HEADLINE] = ["Only one"]
        assert "TOO_FEW" in codes(check_group(assets, Platform.GOOGLE))

    def test_rejects_too_many_headlines(self):
        assets = self.valid_google()
        assets[AssetType.HEADLINE] = [f"Headline {i}" for i in range(16)]
        assert "TOO_MANY" in codes(check_group(assets, Platform.GOOGLE))

    def test_flags_duplicate_headlines_without_rejecting(self):
        assets = self.valid_google()
        assets[AssetType.HEADLINE] = ["Fast hosting", "Fast hosting", "24/7 support"]
        issues = check_group(assets, Platform.GOOGLE)
        assert "DUPLICATE" in codes(issues)
        assert is_publishable(issues)

    def test_duplicate_detection_ignores_case_and_spacing(self):
        assets = self.valid_google()
        assets[AssetType.HEADLINE] = ["Fast hosting", "  fast HOSTING ", "24/7 support"]
        assert "DUPLICATE" in codes(check_group(assets, Platform.GOOGLE))

    def test_reports_the_index_of_the_offending_asset(self):
        assets = self.valid_google()
        assets[AssetType.HEADLINE][1] = "x" * 40
        issue = next(i for i in check_group(assets, Platform.GOOGLE) if i.code == "TOO_LONG")
        assert issue.index == 1

    def test_an_asset_the_platform_does_not_have_is_an_error(self):
        issues = check_text("hi", Platform.LINKEDIN, AssetType.PATH)
        assert "UNSUPPORTED_ASSET" in codes(issues)
