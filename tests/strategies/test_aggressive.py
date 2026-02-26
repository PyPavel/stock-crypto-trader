from trader.strategies.aggressive import AggressiveStrategy
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig


def decide(tech, sent, position=0.0):
    s = AggressiveStrategy(risk=RiskConfig())
    return s.decide("BTC/USD",
                    Signal(symbol="BTC/USD", score=tech, reason=""),
                    SentimentScore(symbol="BTC/USD", score=sent, source="combined", items_analyzed=5),
                    capital=100.0, position=position)


def test_low_threshold_triggers_buy():
    assert decide(0.2, 0.1)["action"] == "buy"


def test_uses_large_position():
    result = decide(0.9, 0.9)
    assert result["usd_amount"] >= 30.0
