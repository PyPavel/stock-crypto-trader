# tests/core/test_engine_telegram.py
from unittest.mock import MagicMock, patch
from trader.core.engine import TradingEngine
from trader.notifications.telegram import TelegramNotifier


def _make_engine(notifier=None):
    from trader.config import Config, RiskConfig
    cfg = Config(
        exchange="coinbase", mode="paper", strategy="moderate",
        capital=10000.0, pairs=["BTC-USD"], cycle_interval=3600,
    )
    adapter = MagicMock()
    adapter.get_price.return_value = 50000.0
    adapter.get_candles.return_value = []
    adapter.place_order.return_value = MagicMock(id="o1", amount=0.001, price=50000.0)

    sentiment = MagicMock()
    sentiment.score_texts.return_value = 0.6

    engine = TradingEngine(
        config=cfg,
        adapter=adapter,
        sentiment_analyzer=sentiment,
        collectors=[],
        notifier=notifier,
    )
    return engine


def test_engine_accepts_notifier():
    notifier = TelegramNotifier(bot_token="", chat_id="")
    engine = _make_engine(notifier=notifier)
    assert engine._notifier is notifier


def test_engine_notifier_defaults_to_none():
    engine = _make_engine()
    assert engine._notifier is None


def test_notifier_called_on_buy():
    notifier = MagicMock()
    engine = _make_engine(notifier=notifier)

    # Patch strategy to always return buy
    engine._strategy.decide = MagicMock(return_value={
        "action": "buy", "usd_amount": 500.0, "reason": "bullish signal"
    })
    engine._signals.score_with_trend = MagicMock(return_value={"score": 0.8, "trend_bullish": True})
    engine._signals.atr = MagicMock(return_value=None)

    prices = {}
    engine._process_symbol("BTC-USD", prices)

    notifier.send.assert_called_once()
    msg = notifier.send.call_args[0][0]
    assert "BUY" in msg
    assert "BTC-USD" in msg


def test_notifier_called_on_sell():
    notifier = MagicMock()
    engine = _make_engine(notifier=notifier)

    # Seed an open position
    engine.portfolio.positions["BTC-USD"] = {
        "amount": 0.01, "entry_price": 50000.0, "side": "buy"
    }
    engine._peak_prices["BTC-USD"] = 50000.0

    # Patch strategy to always return sell
    engine._strategy.decide = MagicMock(return_value={
        "action": "sell", "usd_amount": 500.0, "reason": "bearish signal"
    })
    engine._signals.score_with_trend = MagicMock(return_value={"score": 0.2, "trend_bullish": False})
    engine._signals.atr = MagicMock(return_value=None)

    prices = {}
    engine._process_symbol("BTC-USD", prices)

    notifier.send.assert_called_once()
    msg = notifier.send.call_args[0][0]
    assert "SELL" in msg
    assert "BTC-USD" in msg


def test_notifier_not_called_on_hold():
    notifier = MagicMock()
    engine = _make_engine(notifier=notifier)

    engine._strategy.decide = MagicMock(return_value={
        "action": "hold", "usd_amount": 0.0, "reason": "neutral"
    })
    engine._signals.score_with_trend = MagicMock(return_value={"score": 0.5, "trend_bullish": True})
    engine._signals.atr = MagicMock(return_value=None)

    prices = {}
    engine._process_symbol("BTC-USD", prices)

    notifier.send.assert_not_called()


def test_notifier_called_on_stop_loss():
    from trader.config import RiskConfig
    notifier = MagicMock()
    engine = _make_engine(notifier=notifier)

    # Set stop_loss_pct to 5%
    engine._risk.risk.stop_loss_pct = 0.05

    # Seed a position with entry at 50000; current price 40000 (20% drop, triggers 5% stop)
    engine.portfolio.positions["BTC-USD"] = {
        "amount": 0.01, "entry_price": 50000.0, "side": "buy"
    }
    engine._peak_prices["BTC-USD"] = 50000.0

    # Mock adapter to return a price that triggers stop-loss
    engine._adapter.get_price.return_value = 40000.0
    engine._signals.score_with_trend = MagicMock(return_value={"score": 0.5, "trend_bullish": True})
    engine._signals.atr = MagicMock(return_value=None)

    prices = {}
    engine._process_symbol("BTC-USD", prices)

    notifier.send.assert_called_once()
    msg = notifier.send.call_args[0][0]
    assert "SELL" in msg
    assert "BTC-USD" in msg
