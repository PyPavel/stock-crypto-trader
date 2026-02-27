from datetime import datetime, timezone, timedelta
from trader.core.backtest import Backtester
from trader.models import Candle
from trader.config import RiskConfig
from trader.strategies.moderate import ModerateStrategy


def make_candles(closes):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        Candle("BTC/USD", base + timedelta(hours=i),
               c, c * 1.01, c * 0.99, c, 100.0)
        for i, c in enumerate(closes)
    ]


def test_backtest_returns_results():
    strategy = ModerateStrategy(risk=RiskConfig())
    bt = Backtester(strategy=strategy, starting_capital=100.0)
    closes = [40000 - i * 100 for i in range(50)] + [36000 + i * 150 for i in range(50)]
    results = bt.run(symbol="BTC/USD", candles=make_candles(closes))

    assert "final_value" in results
    assert "num_trades" in results
    assert "win_rate" in results
    assert "max_drawdown" in results
    assert results["final_value"] > 0


def test_backtest_win_rate_between_0_and_1():
    strategy = ModerateStrategy(risk=RiskConfig())
    bt = Backtester(strategy=strategy, starting_capital=100.0)
    closes = [40000 + (i % 10) * 100 for i in range(100)]
    results = bt.run(symbol="BTC/USD", candles=make_candles(closes))
    assert 0.0 <= results["win_rate"] <= 1.0
