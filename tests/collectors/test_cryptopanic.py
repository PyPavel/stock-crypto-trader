from unittest.mock import patch, MagicMock
from trader.collectors.cryptopanic import CryptoPanicCollector


def test_fetch_returns_headlines():
    mock_response = {
        "results": [
            {"title": "Bitcoin hits new high", "currencies": [{"code": "BTC"}]},
            {"title": "ETH upgrade successful", "currencies": [{"code": "ETH"}]},
        ]
    }
    with patch("trader.collectors.cryptopanic.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: mock_response
        )
        mock_get.return_value.raise_for_status = MagicMock()
        collector = CryptoPanicCollector(api_key="")
        headlines = collector.fetch(symbols=["BTC/USD"])
    assert len(headlines) >= 1
    assert any("Bitcoin" in h for h in headlines)


def test_fetch_empty_on_error():
    with patch("trader.collectors.cryptopanic.requests.get") as mock_get:
        mock_get.side_effect = Exception("network error")
        collector = CryptoPanicCollector(api_key="")
        headlines = collector.fetch(symbols=["BTC/USD"])
    assert headlines == []
