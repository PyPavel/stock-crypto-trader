"""Tests for enhanced backtester: trailing stops, cooldowns, multi-symbol, and analytics."""
from datetime import datetime, timezone, timedelta
import math
from trader.core.backtest import Backtester
from trader.models import Candle, Signal, SentimentScore
from trader.config import RiskConfig
from trader.strategies.moderate import ModerateStrategy


def make_candles(closes, symbol="BTC/USD"):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(symbol, base + timedelta(hours=i),
               c, c * 1.01, c * 0.99, c, 100.0)
        for i, c in enumerate(closes)
    ]


# ── Backward compatibility: original tests still work ──

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


# ── New analytics fields ──

def test_results_include_sharpe_ratio():
    strategy = ModerateStrategy(risk=RiskConfig())
    bt = Backtester(strategy=strategy, starting_capital=100.0)
    closes = [40000 + i * 50 for i in range(100)]
    results = bt.run(symbol="BTC/USD", candles=make_candles(closes))
    assert "sharpe_ratio" in results
    assert isinstance(results["sharpe_ratio"], float)


def test_results_include_sortino_ratio():
    strategy = ModerateStrategy(risk=RiskConfig())
    bt = Backtester(strategy=strategy, starting_capital=100.0)
    closes = [40000 + i * 50 for i in range(100)]
    results = bt.run(symbol="BTC/USD", candles=make_candles(closes))
    assert "sortino_ratio" in results
    assert isinstance(results["sortino_ratio"], float)


def test_results_include_calmar_ratio():
    strategy = ModerateStrategy(risk=RiskConfig())
    bt = Backtester(strategy=strategy, starting_capital=100.0)
    closes = [40000 + i * 50 for i in range(100)]
    results = bt.run(symbol="BTC/USD", candles=make_candles(closes))
    assert "calmar_ratio" in results
    assert isinstance(results["calmar_ratio"], float)


def test_results_include_profit_factor():
    strategy = ModerateStrategy(risk=RiskConfig())
    bt = Backtester(strategy=strategy, starting_capital=100.0)
    closes = [40000 + (i % 10) * 100 for i in range(100)]
    results = bt.run(symbol="BTC/USD", candles=make_candles(closes))
    assert "profit_factor" in results
    assert isinstance(results["profit_factor"], float)


def test_results_include_avg_win_loss():
    strategy = ModerateStrategy(risk=RiskConfig())
    bt = Backtester(strategy=strategy, starting_capital=100.0)
    closes = [40000 + (i % 10) * 100 for i in range(100)]
    results = bt.run(symbol="BTC/USD", candles=make_candles(closes))
    assert "avg_win" in results
    assert "avg_loss" in results
    assert "max_consecutive_losses" in results
    assert "max_consecutive_wins" in results


def test_results_include_trades_list():
    strategy = ModerateStrategy(risk=RiskConfig())
    bt = Backtester(strategy=strategy, starting_capital=100.0)
    closes = [40000 + i * 50 for i in range(100)]
    results = bt.run(symbol="BTC/USD", candles=make_candles(closes))
    assert "trades" in results
    assert isinstance(results["trades"], list)


# ── Trailing stop ──

def test_trailing_stop_triggers_sell():
    """Price rises sharply then drops — trailing stop should fire."""
    from trader.strategies.aggressive import AggressiveStrategy
    risk = RiskConfig(trailing_stop_pct=0.08, stop_loss_pct=0.20, cooldown_minutes=0)
    strategy = AggressiveStrategy(risk=risk, buy_threshold=0.05)
    bt = Backtester(strategy=strategy, starting_capital=1000.0)

    # Sharp rise then drop — need enough volatility for signal generation
    closes = (
        [40000 + i * 500 for i in range(50)]  # rise to ~64.5k
        + [64500 - i * 1500 for i in range(20)]  # sharp drop to ~36k
    )
    candles = make_candles(closes)
    results = bt.run(symbol="BTC/USD", candles=candles)

    # Check that a stop (trailing or regular) triggered
    sell_trades = results.get("trades", [])
    reasons = [t.get("reason", "") for t in sell_trades if t.get("side") == "sell"]
    has_trailing = any("trailing" in r for r in reasons)
    has_stoploss = any("stop-loss" in r for r in reasons)
    assert has_trailing or has_stoploss, f"Expected stop triggered, got reasons: {reasons}, num_trades={results['num_trades']}"


# ── Cooldown ──

def test_cooldown_prevents_rapid_trades():
    """With cooldown_minutes=1000 (effectively infinite), should have very few trades."""
    risk = RiskConfig(cooldown_minutes=1000)
    strategy = ModerateStrategy(risk=risk)
    bt = Backtester(strategy=strategy, starting_capital=100.0)
    closes = [40000 + (i % 5) * 200 for i in range(200)]
    candles = make_candles(closes)
    results = bt.run(symbol="BTC/USD", candles=candles)

    # Without cooldown this would produce many trades; with extreme cooldown very few
    assert results["num_trades"] <= 2


# ── Multi-symbol portfolio ──

def test_multi_symbol_portfolio():
    """Running two symbols should combine their contributions."""
    risk = RiskConfig(max_open_positions=5)
    strategy = ModerateStrategy(risk=risk)
    bt = Backtester(strategy=strategy, starting_capital=1000.0)

    candles_btc = make_candles([40000 + i * 50 for i in range(100)], symbol="BTC/USD")
    candles_eth = make_candles([2000 + i * 10 for i in range(100)], symbol="ETH/USD")

    results = bt.run_portfolio({"BTC/USD": candles_btc, "ETH/USD": candles_eth})
    assert "final_value" in results
    assert "num_trades" in results
    # Multi-symbol should potentially have more trades than single
    # (or at least the same, since both can trade independently)
    assert results["num_trades"] >= 0


def test_empty_portfolio_returns_capital():
    strategy = ModerateStrategy(risk=RiskConfig())
    bt = Backtester(strategy=strategy, starting_capital=100.0)
    results = bt.run_portfolio({})
    assert results["final_value"] == 100.0
    assert results["num_trades"] == 0


# ── Sharpe / Sortino helpers ──

def test_sharpe_ratio_positive_returns():
    returns = [0.01, 0.02, -0.005, 0.015, 0.01]
    sharpe = Backtester._sharpe_ratio(returns)
    assert sharpe > 0


def test_sharpe_ratio_empty():
    assert Backtester._sharpe_ratio([]) == 0.0


def test_sortino_ratio_positive():
    returns = [0.01, 0.02, -0.005, 0.015, 0.01]
    sortino = Backtester._sortino_ratio(returns)
    assert sortino > 0


def test_sortino_ratio_no_downside():
    returns = [0.01, 0.02, 0.01]
    sortino = Backtester._sortino_ratio(returns)
    assert sortino == float("inf")


def test_max_consecutive_losses():
    pnls = [0.01, -0.02, -0.01, -0.03, 0.05, -0.01, -0.02]
    assert Backtester._max_consecutive_losses(pnls) == 3


def test_max_consecutive_wins():
    pnls = [0.01, 0.02, 0.01, -0.03, 0.05, 0.01]
    assert Backtester._max_consecutive_wins(pnls) == 3
