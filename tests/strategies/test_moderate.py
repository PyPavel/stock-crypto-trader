from trader.strategies.moderate import ModerateStrategy
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig


def make_decision(tech_score, sentiment_score, strategy=None):
    if strategy is None:
        strategy = ModerateStrategy(risk=RiskConfig())
    tech_signal = Signal(symbol="BTC/USD", score=tech_score, reason="test")
    sentiment = SentimentScore(symbol="BTC/USD", score=sentiment_score,
                               source="combined", items_analyzed=5)
    return strategy.decide("BTC/USD", tech_signal, sentiment, capital=100.0, position=0.0), strategy


def test_strong_buy_signal_returns_buy():
    strategy = ModerateStrategy(risk=RiskConfig())
    # Need 2 consecutive signals above threshold for persistence
    make_decision(tech_score=0.8, sentiment_score=0.6, strategy=strategy)
    decision, _ = make_decision(tech_score=0.8, sentiment_score=0.6, strategy=strategy)
    assert decision["action"] == "buy"
    assert decision.get("amount", 0) > 0 or decision.get("usd_amount", 0) > 0


def test_strong_sell_signal_with_position():
    strategy = ModerateStrategy(risk=RiskConfig())
    tech_signal = Signal(symbol="BTC/USD", score=-0.8, reason="test")
    sentiment = SentimentScore(symbol="BTC/USD", score=-0.6, source="combined", items_analyzed=5)
    # Build persistence: first call records signal
    strategy.decide("BTC/USD", tech_signal, sentiment, capital=100.0, position=20.0)
    decision = strategy.decide("BTC/USD", tech_signal, sentiment, capital=100.0, position=20.0)
    assert decision["action"] == "sell"


def test_neutral_signal_returns_hold():
    decision, _ = make_decision(tech_score=0.1, sentiment_score=0.0)
    assert decision["action"] == "hold"


def test_buy_amount_respects_max_position_pct():
    strategy = ModerateStrategy(risk=RiskConfig(max_position_pct=0.20))
    tech_signal = Signal(symbol="BTC/USD", score=0.9, reason="test")
    sentiment = SentimentScore(symbol="BTC/USD", score=0.9, source="combined", items_analyzed=5)
    # Build persistence
    strategy.decide("BTC/USD", tech_signal, sentiment, capital=100.0, position=0.0)
    decision = strategy.decide("BTC/USD", tech_signal, sentiment, capital=100.0, position=0.0)
    assert decision.get("usd_amount", 0) <= 20.0
