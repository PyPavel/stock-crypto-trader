from trader.strategies.conservative import ConservativeStrategy
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig


def decide(tech, sent, position=0.0, strategy=None):
    if strategy is None:
        strategy = ConservativeStrategy(risk=RiskConfig())
    return strategy.decide("BTC/USD",
                           Signal(symbol="BTC/USD", score=tech, reason=""),
                           SentimentScore(symbol="BTC/USD", score=sent, source="combined", items_analyzed=5),
                           capital=100.0, position=position), strategy


def test_requires_strong_signal_to_buy():
    result, _ = decide(0.5, 0.3)
    assert result["action"] == "hold"


def test_strong_signal_buys():
    s = ConservativeStrategy(risk=RiskConfig())
    # Build persistence: 2 consecutive strong signals
    decide(0.9, 0.8, strategy=s)
    result, _ = decide(0.9, 0.8, strategy=s)
    assert result["action"] == "buy"


def test_buy_uses_small_position():
    s = ConservativeStrategy(risk=RiskConfig())
    decide(0.9, 0.9, strategy=s)
    result, _ = decide(0.9, 0.9, strategy=s)
    assert result["usd_amount"] <= 10.0
