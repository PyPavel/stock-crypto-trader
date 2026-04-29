from trader.strategies.moderate import ModerateStrategy
from trader.strategies.registry import get_strategy
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


# ------------------------------------------------------------------ #
#  Config-driven persistence and threshold
# ------------------------------------------------------------------ #

def _signal(score):
    tech = Signal(symbol="X", score=score, reason="test", trend_bullish=True)
    sent = SentimentScore(symbol="X", score=score, source="combined", items_analyzed=1)
    return tech, sent


def test_config_persistence_cycles_4_requires_four_signals_before_buy():
    """With persistence_cycles=4, three signals should still be HOLD."""
    strategy = get_strategy("moderate", RiskConfig(persistence_cycles=4, buy_score_threshold=0.30))
    tech, sent = _signal(0.9)
    for _ in range(3):
        d = strategy.decide("X", tech, sent, capital=100.0, position=0.0)
    assert d["action"] == "hold"


def test_config_persistence_cycles_4_buys_on_fourth_signal():
    """With persistence_cycles=4, four consecutive signals should trigger a buy."""
    strategy = get_strategy("moderate", RiskConfig(persistence_cycles=4, buy_score_threshold=0.30))
    tech, sent = _signal(0.9)
    for _ in range(4):
        d = strategy.decide("X", tech, sent, capital=100.0, position=0.0)
    assert d["action"] == "buy"


def test_config_buy_score_threshold_blocks_score_below_threshold():
    """Score below the configured threshold must not trigger a buy."""
    strategy = get_strategy("moderate", RiskConfig(persistence_cycles=2, buy_score_threshold=0.40))
    tech, sent = _signal(0.35)  # below threshold of 0.40
    for _ in range(4):  # plenty of cycles
        d = strategy.decide("X", tech, sent, capital=100.0, position=0.0)
    assert d["action"] == "hold"


def test_config_rotation_min_score_delta_prevents_low_delta_rotation():
    """Engine should not rotate when score improvement < rotation_min_score_delta."""
    from trader.core.engine import TradingEngine
    from trader.config import Config
    from unittest.mock import MagicMock

    cfg = Config(
        exchange="coinbase", mode="paper", strategy="moderate",
        capital=1000.0, pairs=["AAPL"], cycle_interval=300,
    )
    cfg.risk = RiskConfig(rotation_min_score_delta=0.25)

    engine = TradingEngine(
        config=cfg,
        adapter=MagicMock(),
        sentiment_analyzer=MagicMock(),
        collectors=[],
        db_path=":memory:",
    )

    # Seed a position for WEAK at score 0.30
    engine.portfolio.positions["WEAK"] = {"amount": 1.0, "entry_price": 100.0, "side": "buy"}
    engine._current_scores = {"WEAK": 0.30}

    # NEW_SYM scores 0.45 — delta = 0.15, below the 0.25 threshold
    rotated = engine._try_position_rotation("NEW_SYM", 0.45, {"WEAK": 100.0})
    assert rotated is False  # should NOT rotate
