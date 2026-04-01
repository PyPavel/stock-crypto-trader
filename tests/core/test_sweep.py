"""Tests for parameter sweep, walk-forward optimization, and strategy comparison."""
from datetime import datetime, timezone, timedelta
from trader.core.sweep import (
    parameter_sweep, walk_forward, compare_strategies,
    SweepParam, SweepResult,
)
from trader.models import Candle
from trader.config import RiskConfig
from trader.strategies.moderate import ModerateStrategy
from trader.strategies.conservative import ConservativeStrategy
from trader.strategies.aggressive import AggressiveStrategy


def make_candles(closes, symbol="BTC/USD"):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        Candle(symbol, base + timedelta(hours=i),
               c, c * 1.01, c * 0.99, c, 100.0)
        for i, c in enumerate(closes)
    ]


# ── Parameter Sweep ──

def test_parameter_sweep_returns_results():
    candles = make_candles([40000 + i * 50 for i in range(200)])
    results = parameter_sweep(
        ModerateStrategy,
        {"BTC/USD": candles},
        params=[
            SweepParam("buy_threshold", [0.25, 0.35]),
            SweepParam("sell_threshold", [-0.35, -0.25]),
        ],
        top_n=4,
    )
    assert len(results) <= 4
    assert all(isinstance(r, SweepResult) for r in results)
    for r in results:
        assert "buy_threshold" in r.params
        assert "sell_threshold" in r.params
        assert isinstance(r.sharpe_ratio, float)


def test_parameter_sweep_sorts_by_sharpe():
    candles = make_candles([40000 + i * 50 for i in range(200)])
    results = parameter_sweep(
        ModerateStrategy,
        {"BTC/USD": candles},
        params=[
            SweepParam("buy_threshold", [0.25, 0.35, 0.45]),
        ],
        top_n=3,
        sort_by="sharpe_ratio",
    )
    for i in range(len(results) - 1):
        assert results[i].sharpe_ratio >= results[i + 1].sharpe_ratio


def test_parameter_sweep_can_sort_by_pnl():
    candles = make_candles([40000 + i * 50 for i in range(200)])
    results = parameter_sweep(
        ModerateStrategy,
        {"BTC/USD": candles},
        params=[
            SweepParam("buy_threshold", [0.25, 0.35]),
        ],
        top_n=2,
        sort_by="pnl_pct",
    )
    for i in range(len(results) - 1):
        assert results[i].pnl_pct >= results[i + 1].pnl_pct


def test_parameter_sweep_with_risk_overrides():
    candles = make_candles([40000 + i * 50 for i in range(200)])
    results = parameter_sweep(
        ModerateStrategy,
        {"BTC/USD": candles},
        params=[],
        risk_overrides={"stop_loss_pct": 0.03, "trailing_stop_pct": 0.05},
        top_n=1,
    )
    assert len(results) == 1


def test_parameter_sweep_multi_symbol():
    candles_btc = make_candles([40000 + i * 50 for i in range(200)], "BTC/USD")
    candles_eth = make_candles([2000 + i * 10 for i in range(200)], "ETH/USD")
    results = parameter_sweep(
        ModerateStrategy,
        {"BTC/USD": candles_btc, "ETH/USD": candles_eth},
        params=[SweepParam("buy_threshold", [0.25, 0.35])],
        top_n=2,
    )
    assert len(results) <= 2


# ── Walk-Forward ──

def test_walk_forward_splits_data():
    candles = make_candles([40000 + i * 30 for i in range(200)])
    wf = walk_forward(
        ModerateStrategy,
        {"BTC/USD": candles},
        param_grid=[SweepParam("buy_threshold", [0.25, 0.35, 0.45])],
        train_pct=0.7,
        top_n=3,
    )
    assert wf["train_best"] is not None
    assert isinstance(wf["train_best"], SweepResult)
    assert wf["oos_best"] is not None
    assert isinstance(wf["oos_best"], SweepResult)
    assert len(wf["oos_results"]) <= 3


def test_walk_forward_oos_results_sorted():
    candles = make_candles([40000 + i * 30 for i in range(200)])
    wf = walk_forward(
        ModerateStrategy,
        {"BTC/USD": candles},
        param_grid=[SweepParam("buy_threshold", [0.25, 0.35, 0.45])],
        top_n=3,
    )
    oos = wf["oos_results"]
    for i in range(len(oos) - 1):
        assert oos[i].sharpe_ratio >= oos[i + 1].sharpe_ratio


def test_walk_forward_empty_candles():
    wf = walk_forward(
        ModerateStrategy,
        {},
        param_grid=[SweepParam("buy_threshold", [0.25])],
    )
    assert wf["train_best"] is None


# ── Strategy Comparison ──

def test_compare_strategies_returns_sorted():
    candles = make_candles([40000 + i * 50 for i in range(200)])
    results = compare_strategies(
        [
            (ModerateStrategy, RiskConfig()),
            (ConservativeStrategy, RiskConfig()),
            (AggressiveStrategy, RiskConfig()),
        ],
        {"BTC/USD": candles},
    )
    assert len(results) == 3
    names = [r["strategy"] for r in results]
    assert "ModerateStrategy" in names
    assert "ConservativeStrategy" in names
    assert "AggressiveStrategy" in names
    # Sorted by sharpe descending
    for i in range(len(results) - 1):
        assert results[i].get("sharpe_ratio", 0) >= results[i + 1].get("sharpe_ratio", 0)


def test_compare_strategies_has_analytics():
    candles = make_candles([40000 + i * 50 for i in range(200)])
    results = compare_strategies(
        [(ModerateStrategy, RiskConfig())],
        {"BTC/USD": candles},
    )
    assert len(results) == 1
    r = results[0]
    assert "sharpe_ratio" in r
    assert "sortino_ratio" in r
    assert "calmar_ratio" in r
    assert "profit_factor" in r
    assert "avg_win" in r
    assert "avg_loss" in r
