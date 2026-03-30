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
