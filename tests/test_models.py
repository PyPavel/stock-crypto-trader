from datetime import datetime, timezone
from trader.models import Candle, Order, Trade, SentimentScore, Signal


def test_candle_creation():
    c = Candle(
        symbol="BTC/USD",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=40000.0, high=41000.0, low=39000.0, close=40500.0, volume=100.0,
    )
    assert c.symbol == "BTC/USD"
    assert c.close == 40500.0


def test_signal_score_bounded():
    s = Signal(symbol="BTC/USD", score=0.75, reason="RSI oversold")
    assert -1.0 <= s.score <= 1.0


def test_sentiment_score():
    ss = SentimentScore(symbol="BTC/USD", score=0.4, source="reddit", items_analyzed=10)
    assert ss.source == "reddit"


def test_order_sides():
    o = Order(symbol="BTC/USD", side="buy", amount=0.001, price=40000.0, mode="paper")
    assert o.side in ("buy", "sell")
