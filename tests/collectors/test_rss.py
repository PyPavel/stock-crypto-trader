from unittest.mock import patch, MagicMock
from trader.collectors.rss import RSSCollector

RSS_RESPONSE = b"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <item><title>Bitcoin surges to new highs</title></item>
    <item><title>Ethereum upgrade goes live</title></item>
    <item><title>Stock market news unrelated</title></item>
  </channel>
</rss>"""


def test_fetch_returns_relevant_headlines():
    with patch("trader.collectors.rss.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.content = RSS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        collector = RSSCollector(feeds=["http://fake.rss/feed"])
        headlines = collector.fetch(symbols=["BTC/USD"])
    assert any("Bitcoin" in h for h in headlines)


def test_fetch_returns_empty_on_error():
    with patch("trader.collectors.rss.requests.get") as mock_get:
        mock_get.side_effect = Exception("timeout")
        collector = RSSCollector(feeds=["http://fake.rss/feed"])
        result = collector.fetch(symbols=["BTC/USD"])
    assert result == []


def test_filters_irrelevant_articles():
    with patch("trader.collectors.rss.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.content = RSS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        collector = RSSCollector(feeds=["http://fake.rss/feed"])
        headlines = collector.fetch(symbols=["BTC/USD"])
    # "Stock market news unrelated" should not appear (no crypto keywords)
    assert not any("Stock market" in h for h in headlines)
