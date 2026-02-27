from unittest.mock import MagicMock
from datetime import datetime, timezone, timedelta
from trader.core.engine import TradingEngine
from trader.config import Config, RiskConfig
from trader.models import Candle


def make_candles(n=50):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        Candle("BTC/USD", base + timedelta(hours=i),
               40000.0, 41000.0, 39000.0, 40000.0 + i * 10, 100.0)
        for i in range(n)
    ]


def make_engine(tmp_path, mode="paper"):
    cfg = Config(exchange="coinbase", mode=mode, strategy="moderate",
                 capital=100.0, pairs=["BTC/USD"], cycle_interval=60)

    adapter = MagicMock()
    adapter.get_candles.return_value = make_candles()
    adapter.get_price.return_value = 40500.0
    adapter.get_balance.return_value = {"USD": 100.0}

    sentiment_analyzer = MagicMock()
    sentiment_analyzer.score_texts.return_value = 0.3

    news_collector = MagicMock()
    news_collector.fetch.return_value = ["BTC is bullish"]

    return TradingEngine(
        config=cfg,
        adapter=adapter,
        sentiment_analyzer=sentiment_analyzer,
        collectors=[news_collector],
        db_path=str(tmp_path / "test.db"),
    )


def test_run_cycle_completes_without_error(tmp_path):
    engine = make_engine(tmp_path)
    engine.run_cycle()  # should not raise


def test_run_cycle_in_paper_mode_no_live_orders(tmp_path):
    engine = make_engine(tmp_path, mode="paper")
    engine.run_cycle()
    trades = engine.portfolio.get_trades()
    for t in trades:
        assert t.mode == "paper"


def test_engine_handles_collector_failure(tmp_path):
    engine = make_engine(tmp_path)
    engine._collectors[0].fetch.side_effect = Exception("network error")
    engine.run_cycle()  # should not raise, sentiment falls back to 0
