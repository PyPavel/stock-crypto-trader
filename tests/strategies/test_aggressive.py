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
    # 0.3 * 0.55 + 0.3 * 0.45 = 0.30, which is >= 0.25 threshold
    assert decide(0.3, 0.3)["action"] == "buy"


def test_uses_large_position():
    result = decide(0.9, 0.9)
    # RiskConfig default max_position_pct is 0.20, so 100 * 0.20 = 20.0
    assert result["usd_amount"] >= 20.0
