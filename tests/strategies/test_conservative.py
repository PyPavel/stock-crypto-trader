from trader.strategies.conservative import ConservativeStrategy
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig


def decide(tech, sent, position=0.0):
    s = ConservativeStrategy(risk=RiskConfig())
    return s.decide("BTC/USD",
                    Signal(symbol="BTC/USD", score=tech, reason=""),
                    SentimentScore(symbol="BTC/USD", score=sent, source="combined", items_analyzed=5),
                    capital=100.0, position=position)


def test_requires_strong_signal_to_buy():
    assert decide(0.5, 0.3)["action"] == "hold"


def test_strong_signal_buys():
    assert decide(0.9, 0.8)["action"] == "buy"


def test_buy_uses_small_position():
    result = decide(0.9, 0.9)
    assert result["usd_amount"] <= 10.0
