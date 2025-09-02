from adopt.redaction import redact


class TestRedaction:
    def test_removes_an_email(self):
        result = redact("Contact jane.doe+ads@example.co.uk for details")
        assert "jane.doe" not in result.text
        assert "[EMAIL]" in result.text
        assert result.counts["EMAIL"] == 1

    def test_removes_phone_numbers_in_several_shapes(self):
        for number in ("416-555-0142", "(416) 555-0142", "+1 416 555 0142", "416.555.0142"):
            result = redact(f"Call {number} today")
            assert "[PHONE]" in result.text, number
            assert "555" not in result.text, number

    def test_removes_a_card_number(self):
        # A real Visa test number, which passes Luhn.
        result = redact("Card on file 4111 1111 1111 1111")
        assert "[CARD]" in result.text

    def test_leaves_an_order_number_alone(self):
        # Sixteen digits that fail Luhn are not a card.
        result = redact("Order 1234567890123456 shipped")
        assert "[CARD]" not in result.text
        assert "1234567890123456" in result.text

    def test_removes_provider_keys(self):
        for key in ("sk-ant-api03-" + "x" * 30, "AKIA" + "A" * 16, "ghp_" + "a" * 36):
            result = redact(f"key is {key}")
            assert key not in result.text
            assert "[API_KEY]" in result.text

    def test_removes_postal_codes(self):
        assert "[POSTAL_CODE]" in redact("Ship to M5V 3A8 please").text

    def test_keeps_placeholders_typed_so_the_copy_still_makes_sense(self):
        # "Call [PHONE]" still reads as a phone number to the model, which
        # matters when it is writing "call us today" copy.
        result = redact("Call 416-555-0142 now")
        assert result.text == "Call [PHONE] now"

    def test_leaves_ordinary_marketing_copy_untouched(self):
        brief = "Launch the spring sale. 20% off. Target homeowners aged 35-54."
        result = redact(brief)
        assert result.text == brief
        assert not result.redacted_anything

    def test_does_not_eat_prices_or_percentages(self):
        result = redact("Save $1,499 or 25% on the annual plan")
        assert "1,499" in result.text
        assert "25%" in result.text

    def test_counts_multiple_hits(self):
        result = redact("a@b.com and c@d.com and 416-555-0142")
        assert result.counts == {"EMAIL": 2, "PHONE": 1}

    def test_handles_empty_input(self):
        result = redact("")
        assert result.text == ""
        assert not result.redacted_anything
