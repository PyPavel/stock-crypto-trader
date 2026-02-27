from unittest.mock import patch, MagicMock
from trader.collectors.coingecko import CoinGeckoCollector


def test_positive_market_gives_positive_score():
    mock_data = {"data": {"market_cap_change_percentage_24h_usd": 5.0}}
    with patch("trader.collectors.coingecko.requests.get") as mock_get:
        mock_get.return_value = MagicMock(json=lambda: mock_data)
        mock_get.return_value.raise_for_status = MagicMock()
        score = CoinGeckoCollector().score(["BTC/USD"])
    assert score > 0


def test_negative_market_gives_negative_score():
    mock_data = {"data": {"market_cap_change_percentage_24h_usd": -5.0}}
    with patch("trader.collectors.coingecko.requests.get") as mock_get:
        mock_get.return_value = MagicMock(json=lambda: mock_data)
        mock_get.return_value.raise_for_status = MagicMock()
        score = CoinGeckoCollector().score(["BTC/USD"])
    assert score < 0


def test_score_capped_at_bounds():
    mock_data = {"data": {"market_cap_change_percentage_24h_usd": 50.0}}
    with patch("trader.collectors.coingecko.requests.get") as mock_get:
        mock_get.return_value = MagicMock(json=lambda: mock_data)
        mock_get.return_value.raise_for_status = MagicMock()
        score = CoinGeckoCollector().score(["BTC/USD"])
    assert score == 1.0


def test_returns_none_on_error():
    with patch("trader.collectors.coingecko.requests.get") as mock_get:
        mock_get.side_effect = Exception("rate limited")
        score = CoinGeckoCollector().score(["BTC/USD"])
    assert score is None
