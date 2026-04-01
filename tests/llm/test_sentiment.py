from unittest.mock import MagicMock, patch
from trader.llm.sentiment import SentimentAnalyzer


def _make_analyzer():
    """Return a SentimentAnalyzer with a mocked OpenAI client."""
    analyzer = SentimentAnalyzer(model="mimo-v2-flash", api_key="test-key")
    return analyzer


def _mock_response(content: str) -> MagicMock:
    """Build a fake openai chat.completions.create() return value."""
    message = MagicMock()
    message.choices = [MagicMock(message=MagicMock(content=content))]
    return message


def test_bullish_text_positive_score():
    analyzer = _make_analyzer()
    analyzer._client = MagicMock()
    analyzer._client.chat.completions.create.return_value = _mock_response(
        '{"sentiment": "bullish", "confidence": 0.85}'
    )
    score = analyzer.score_texts(["Bitcoin breaks all-time high"])
    assert score > 0


def test_bearish_text_negative_score():
    analyzer = _make_analyzer()
    analyzer._client = MagicMock()
    analyzer._client.chat.completions.create.return_value = _mock_response(
        '{"sentiment": "bearish", "confidence": 0.90}'
    )
    score = analyzer.score_texts(["Crypto market crashes"])
    assert score < 0


def test_returns_zero_on_empty_input():
    analyzer = _make_analyzer()
    assert analyzer.score_texts([]) == 0.0


def test_returns_zero_on_api_failure():
    analyzer = _make_analyzer()
    analyzer._client = MagicMock()
    analyzer._client.chat.completions.create.side_effect = Exception("connection refused")
    score = analyzer.score_texts(["some text"])
    assert score == 0.0
