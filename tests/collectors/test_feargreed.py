from unittest.mock import patch, MagicMock
from trader.collectors.feargreed import FearGreedCollector


def test_extreme_fear_gives_positive_score():
    mock_data = {"data": [{"value": "10", "value_classification": "Extreme Fear"}]}
    with patch("trader.collectors.feargreed.requests.get") as mock_get:
        mock_get.return_value = MagicMock(json=lambda: mock_data)
        mock_get.return_value.raise_for_status = MagicMock()
        score = FearGreedCollector().score()
    assert score > 0  # fear = buy signal


def test_extreme_greed_gives_negative_score():
    mock_data = {"data": [{"value": "90", "value_classification": "Extreme Greed"}]}
    with patch("trader.collectors.feargreed.requests.get") as mock_get:
        mock_get.return_value = MagicMock(json=lambda: mock_data)
        mock_get.return_value.raise_for_status = MagicMock()
        score = FearGreedCollector().score()
    assert score < 0  # greed = sell signal


def test_neutral_gives_zero():
    mock_data = {"data": [{"value": "50", "value_classification": "Neutral"}]}
    with patch("trader.collectors.feargreed.requests.get") as mock_get:
        mock_get.return_value = MagicMock(json=lambda: mock_data)
        mock_get.return_value.raise_for_status = MagicMock()
        score = FearGreedCollector().score()
    assert score == 0.0


def test_returns_none_on_error():
    with patch("trader.collectors.feargreed.requests.get") as mock_get:
        mock_get.side_effect = Exception("timeout")
        score = FearGreedCollector().score()
    assert score is None
