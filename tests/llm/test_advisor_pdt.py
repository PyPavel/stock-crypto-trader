from unittest.mock import MagicMock
from trader.llm.advisor import LLMAdvisor, PDT_SECTION_TIGHT, PDT_SECTION_NORMAL


def make_advisor() -> LLMAdvisor:
    advisor = LLMAdvisor(api_key="test-key", model="test-model")
    advisor._client = MagicMock()
    return advisor


def _mock_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.choices = [MagicMock(message=MagicMock(content=content))]
    return msg


HEADLINES = ["Strong earnings beat", "Analyst upgrade"]
BUY_RESPONSE = '{"judgment":"buy","conviction":0.8,"reason":"good"}'
AVOID_RESPONSE = '{"judgment":"avoid","conviction":0.0,"reason":"pdt"}'


# ------------------------------------------------------------------ #
#  pdt_remaining=None — normal operation, no PDT section in prompt
# ------------------------------------------------------------------ #

def test_no_pdt_context_sends_no_pdt_section():
    advisor = make_advisor()
    advisor._client.chat.completions.create.return_value = _mock_response(BUY_RESPONSE)

    advisor.advise("AAPL", 0.3, "bullish", 0.4, HEADLINES, pdt_remaining=None)

    prompt = advisor._client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "PDT" not in prompt
    assert "day-trade" not in prompt.lower()


# ------------------------------------------------------------------ #
#  pdt_remaining=3 — full budget, no PDT section
# ------------------------------------------------------------------ #

def test_full_budget_sends_no_pdt_section():
    advisor = make_advisor()
    advisor._client.chat.completions.create.return_value = _mock_response(BUY_RESPONSE)

    advisor.advise("AAPL", 0.3, "bullish", 0.4, HEADLINES, pdt_remaining=3)

    prompt = advisor._client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "PDT" not in prompt


# ------------------------------------------------------------------ #
#  pdt_remaining=2 — normal PDT note injected
# ------------------------------------------------------------------ #

def test_two_remaining_injects_normal_pdt_note():
    advisor = make_advisor()
    advisor._client.chat.completions.create.return_value = _mock_response(BUY_RESPONSE)

    advisor.advise("AAPL", 0.3, "bullish", 0.4, HEADLINES, pdt_remaining=2)

    prompt = advisor._client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "2 day-trades remaining" in prompt
    assert "Prefer positions with multi-day" in prompt


# ------------------------------------------------------------------ #
#  pdt_remaining=1 — tight constraint injected
# ------------------------------------------------------------------ #

def test_one_remaining_injects_tight_constraint():
    advisor = make_advisor()
    advisor._client.chat.completions.create.return_value = _mock_response(BUY_RESPONSE)

    advisor.advise("AAPL", 0.3, "bullish", 0.4, HEADLINES, pdt_remaining=1)

    prompt = advisor._client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "1 day-trade(s) remaining" in prompt
    assert "overnight" in prompt.lower()


# ------------------------------------------------------------------ #
#  pdt_remaining=0 — short-circuit, return avoid without calling LLM
# ------------------------------------------------------------------ #

def test_zero_remaining_returns_avoid_without_llm_call():
    advisor = make_advisor()

    result = advisor.advise("AAPL", 0.3, "bullish", 0.4, HEADLINES, pdt_remaining=0)

    advisor._client.chat.completions.create.assert_not_called()
    assert result["judgment"] == "avoid"
    assert result["conviction"] == 0.0


# ------------------------------------------------------------------ #
#  Prompt no longer says "crypto trader" (stock-aware label fix)
# ------------------------------------------------------------------ #

def test_prompt_says_equity_trader_not_crypto():
    advisor = make_advisor()
    advisor._client.chat.completions.create.return_value = _mock_response(BUY_RESPONSE)

    advisor.advise("AAPL", 0.3, "bullish", 0.4, HEADLINES)

    prompt = advisor._client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "crypto trader" not in prompt.lower()
    assert "equity trader" in prompt.lower()
