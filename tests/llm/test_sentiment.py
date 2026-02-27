from unittest.mock import patch, MagicMock
from trader.llm.sentiment import SentimentAnalyzer


def test_bullish_text_positive_score():
    with patch("trader.llm.sentiment.ollama.chat") as mock_chat:
        mock_chat.return_value = MagicMock(
            message=MagicMock(content='{"sentiment": "bullish", "confidence": 0.85}')
        )
        analyzer = SentimentAnalyzer(model="mistral", base_url="http://localhost:11434")
        score = analyzer.score_texts(["Bitcoin breaks all-time high"])
    assert score > 0


def test_bearish_text_negative_score():
    with patch("trader.llm.sentiment.ollama.chat") as mock_chat:
        mock_chat.return_value = MagicMock(
            message=MagicMock(content='{"sentiment": "bearish", "confidence": 0.90}')
        )
        analyzer = SentimentAnalyzer(model="mistral", base_url="http://localhost:11434")
        score = analyzer.score_texts(["Crypto market crashes"])
    assert score < 0


def test_returns_zero_on_empty_input():
    analyzer = SentimentAnalyzer(model="mistral", base_url="http://localhost:11434")
    assert analyzer.score_texts([]) == 0.0


def test_returns_zero_on_ollama_failure():
    with patch("trader.llm.sentiment.ollama.chat") as mock_chat:
        mock_chat.side_effect = Exception("connection refused")
        analyzer = SentimentAnalyzer(model="mistral", base_url="http://localhost:11434")
        score = analyzer.score_texts(["some text"])
    assert score == 0.0
