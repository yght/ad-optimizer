"""The repair loop, without touching the network.

The point of this suite is that the interesting behaviour of an LLM-backed
function is not the LLM. It is what happens around it: whether the validator
is trusted over the model, whether repairs are targeted, whether the loop
terminates. All of that is testable with a stub, and none of it needs an API
key or costs anything to run in CI.
"""

from dataclasses import dataclass

import pytest

from adopt.copy import AdCopy, CopyRefused, MAX_REPAIR_ROUNDS, generate
from adopt.specs import Platform


@dataclass
class StubUsage:
    input_tokens: int = 1000
    output_tokens: int = 500
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class StubResponse:
    parsed_output: AdCopy
    usage: StubUsage
    stop_reason: str = "end_turn"
    stop_details: object = None


class StubMessages:
    """Returns a scripted sequence of responses, and records what it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("stub ran out of scripted responses")
        return self.script.pop(0)


class StubClient:
    def __init__(self, script):
        self.messages = StubMessages(script)


def copy_with(headlines, descriptions=None, primary_text="Body copy for social."):
    return AdCopy(
        headlines=headlines,
        descriptions=descriptions or ["A description that fits.", "And a second one."],
        primary_text=primary_text,
        rationale="Test copy.",
    )


GOOD = ["Fast hosting", "Free migration", "24/7 support"]
TOO_LONG = ["Fast hosting", "x" * 40, "24/7 support"]


class TestFirstPass:
    def test_returns_immediately_when_the_copy_is_valid(self):
        client = StubClient([StubResponse(copy_with(GOOD), StubUsage())])
        result = generate("Sell hosting", Platform.GOOGLE, client=client)

        assert result.publishable
        assert result.repair_rounds == 0
        assert len(client.messages.calls) == 1

    def test_records_usage_for_the_ledger(self):
        client = StubClient([StubResponse(copy_with(GOOD), StubUsage(2000, 800))])
        result = generate("Sell hosting", Platform.GOOGLE, client=client)

        assert len(result.usage) == 1
        assert result.usage[0].input_tokens == 2000
        assert result.usage[0].cost_usd() > 0


class TestRepair:
    def test_sends_the_copy_back_when_a_headline_is_too_long(self):
        client = StubClient([
            StubResponse(copy_with(TOO_LONG), StubUsage()),
            StubResponse(copy_with(GOOD), StubUsage()),
        ])
        result = generate("Sell hosting", Platform.GOOGLE, client=client)

        assert result.publishable
        assert result.repair_rounds == 1
        assert len(client.messages.calls) == 2

    def test_the_repair_prompt_names_the_specific_failure(self):
        client = StubClient([
            StubResponse(copy_with(TOO_LONG), StubUsage()),
            StubResponse(copy_with(GOOD), StubUsage()),
        ])
        generate("Sell hosting", Platform.GOOGLE, client=client)

        repair_messages = client.messages.calls[1]["messages"]
        last = repair_messages[-1]["content"]

        assert "headline 1" in last
        assert "40 characters" in last
        assert "limit is 30" in last

    def test_the_repair_turn_carries_the_previous_attempt(self):
        client = StubClient([
            StubResponse(copy_with(TOO_LONG), StubUsage()),
            StubResponse(copy_with(GOOD), StubUsage()),
        ])
        generate("Sell hosting", Platform.GOOGLE, client=client)

        messages = client.messages.calls[1]["messages"]
        assert messages[1]["role"] == "assistant"
        assert "Fast hosting" in messages[1]["content"]

    def test_gives_up_after_the_round_limit_rather_than_looping(self):
        # A model that never fixes it. This must terminate.
        client = StubClient([
            StubResponse(copy_with(TOO_LONG), StubUsage())
            for _ in range(MAX_REPAIR_ROUNDS + 1)
        ])
        result = generate("Sell hosting", Platform.GOOGLE, client=client)

        assert not result.publishable
        assert result.repair_rounds == MAX_REPAIR_ROUNDS
        assert len(client.messages.calls) == MAX_REPAIR_ROUNDS + 1

    def test_returns_the_failed_copy_with_its_issues_attached(self):
        client = StubClient([
            StubResponse(copy_with(TOO_LONG), StubUsage())
            for _ in range(MAX_REPAIR_ROUNDS + 1)
        ])
        result = generate("Sell hosting", Platform.GOOGLE, client=client)

        # The caller may still want the headlines that passed.
        assert result.copy.headlines[0] == "Fast hosting"
        assert any(i.code == "TOO_LONG" for i in result.issues)

    def test_a_non_fatal_issue_does_not_trigger_a_repair(self):
        # Duplicate headlines are wasteful, not rejected. Spending another
        # model call on that is not worth it.
        client = StubClient([
            StubResponse(copy_with(["Fast hosting", "Fast hosting", "Support"]), StubUsage())
        ])
        result = generate("Sell hosting", Platform.GOOGLE, client=client)

        assert result.publishable
        assert result.repair_rounds == 0
        assert any(i.code == "DUPLICATE" for i in result.issues)


class TestPlatformDifferences:
    def test_the_same_copy_can_pass_one_platform_and_fail_another(self):
        # Emoji are fine on Meta and rejected outright on Google search.
        emoji = copy_with(["Save now 🎉", "Free migration", "24/7 support"])

        meta = StubClient([StubResponse(emoji, StubUsage())])
        assert generate("brief", Platform.META, client=meta).publishable

        google = StubClient([StubResponse(emoji, StubUsage()) for _ in range(3)])
        assert not generate("brief", Platform.GOOGLE, client=google).publishable

    def test_primary_text_is_only_asked_for_where_the_platform_has_one(self):
        google = StubClient([StubResponse(copy_with(GOOD), StubUsage())])
        generate("brief", Platform.GOOGLE, client=google)
        assert "primary text" not in google.messages.calls[0]["messages"][0]["content"]

        meta = StubClient([StubResponse(copy_with(GOOD), StubUsage())])
        generate("brief", Platform.META, client=meta)
        assert "primary text" in meta.messages.calls[0]["messages"][0]["content"]

    def test_google_is_not_penalised_for_omitting_primary_text(self):
        client = StubClient([StubResponse(copy_with(GOOD, primary_text=None), StubUsage())])
        assert generate("brief", Platform.GOOGLE, client=client).publishable


class TestRedaction:
    def test_the_brief_is_redacted_before_it_is_sent(self):
        client = StubClient([StubResponse(copy_with(GOOD), StubUsage())])
        result = generate(
            "Target our list: jane@example.com, call 416-555-0142",
            Platform.GOOGLE,
            client=client,
        )

        sent = client.messages.calls[0]["messages"][0]["content"]
        assert "jane@example.com" not in sent
        assert "416-555-0142" not in sent
        assert "[EMAIL]" in sent
        assert result.redacted == {"EMAIL": 1, "PHONE": 1}

    def test_a_clean_brief_reports_no_redactions(self):
        client = StubClient([StubResponse(copy_with(GOOD), StubUsage())])
        result = generate("Sell hosting to developers", Platform.GOOGLE, client=client)
        assert result.redacted == {}


class TestRefusal:
    def test_a_refusal_raises_rather_than_returning_empty_copy(self):
        @dataclass
        class Details:
            category: str = "regulated_advice"
            explanation: str = ""

        client = StubClient([
            StubResponse(copy_with(GOOD), StubUsage(), stop_reason="refusal",
                         stop_details=Details())
        ])

        with pytest.raises(CopyRefused, match="regulated_advice"):
            generate("Write copy for a payday loan", Platform.GOOGLE, client=client)


class TestRequestShape:
    def test_asks_for_adaptive_thinking_and_a_cached_system_prompt(self):
        client = StubClient([StubResponse(copy_with(GOOD), StubUsage())])
        generate("Sell hosting", Platform.GOOGLE, client=client)

        call = client.messages.calls[0]
        assert call["model"] == "claude-opus-5"
        assert call["thinking"] == {"type": "adaptive"}
        assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert call["output_format"] is AdCopy

    def test_effort_is_configurable(self):
        client = StubClient([StubResponse(copy_with(GOOD), StubUsage())])
        generate("Sell hosting", Platform.GOOGLE, client=client, effort="high")
        assert client.messages.calls[0]["output_config"]["effort"] == "high"
