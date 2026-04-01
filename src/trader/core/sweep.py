"""Parameter sweep — grid search over strategy/threshold combinations."""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from trader.core.backtest import Backtester
from trader.models import Candle
from trader.config import RiskConfig


@dataclass
class SweepParam:
    """A single parameter to sweep over."""
    name: str
    values: list[Any]


@dataclass
class SweepResult:
    params: dict[str, Any]
    final_value: float
    pnl_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    sortino_ratio: float
    calmar_ratio: float


def parameter_sweep(
    strategy_cls,
    candles: dict[str, list[Candle]],
    params: list[SweepParam],
    starting_capital: float = 100.0,
    risk_overrides: dict[str, Any] | None = None,
    top_n: int = 10,
    sort_by: str = "sharpe_ratio",
) -> list[SweepResult]:
    """
    Grid search over parameter combinations.

    Args:
        strategy_cls: Strategy class (e.g., ModerateStrategy)
        candles: dict mapping symbol -> list of Candles
        params: list of SweepParam defining the grid
        starting_capital: initial capital
        risk_overrides: optional dict of RiskConfig field overrides
        top_n: number of top results to return
        sort_by: metric to rank by (default: sharpe_ratio)

    Returns:
        List of SweepResult sorted by sort_by descending, capped at top_n.
    """
    names = [p.name for p in params]
    value_lists = [p.values for p in params]
    results: list[SweepResult] = []

    base_risk_kwargs = risk_overrides or {}

    for combo in product(*value_lists):
        param_dict = dict(zip(names, combo))

        # Build RiskConfig with overrides
        risk_kwargs = dict(base_risk_kwargs)
        risk_field_names = {f.name for f in RiskConfig.__dataclass_fields__.values()}
        for k, v in param_dict.items():
            if k in risk_field_names:
                risk_kwargs[k] = v
        risk = RiskConfig(**risk_kwargs)

        # Build strategy — pass only constructor-known params
        strat_kwargs = {}
        if "buy_threshold" in param_dict:
            strat_kwargs["buy_threshold"] = param_dict["buy_threshold"]
        if "sell_threshold" in param_dict:
            strat_kwargs["sell_threshold"] = param_dict["sell_threshold"]
        if "tech_weight" in param_dict:
            strat_kwargs["tech_weight"] = param_dict["tech_weight"]
        if "sentiment_weight" in param_dict:
            strat_kwargs["sentiment_weight"] = param_dict["sentiment_weight"]

        try:
            strategy = strategy_cls(risk=risk, **strat_kwargs)
        except TypeError:
            strategy = strategy_cls(risk=risk)

        # Apply any non-constructor params by setting attributes
        for k, v in param_dict.items():
            if k not in strat_kwargs and k not in risk_field_names:
                if hasattr(strategy, k):
                    setattr(strategy, k, v)

        bt = Backtester(strategy=strategy, starting_capital=starting_capital)
        if len(candles) == 1:
            sym = next(iter(candles))
            res = bt.run(sym, candles[sym])
        else:
            res = bt.run_portfolio(candles)

        results.append(SweepResult(
            params=param_dict,
            final_value=res["final_value"],
            pnl_pct=res["pnl_pct"],
            sharpe_ratio=res["sharpe_ratio"],
            max_drawdown=res["max_drawdown"],
            win_rate=res["win_rate"],
            profit_factor=res["profit_factor"],
            num_trades=res["num_trades"],
            sortino_ratio=res["sortino_ratio"],
            calmar_ratio=res["calmar_ratio"],
        ))

    # Sort descending by sort_by
    valid_sort_keys = {
        "sharpe_ratio", "pnl_pct", "win_rate", "profit_factor",
        "final_value", "sortino_ratio", "calmar_ratio",
    }
    if sort_by not in valid_sort_keys:
        sort_by = "sharpe_ratio"
    results.sort(key=lambda r: getattr(r, sort_by), reverse=True)
    return results[:top_n]


def walk_forward(
    strategy_cls,
    candles: dict[str, list[Candle]],
    param_grid: list[SweepParam],
    train_pct: float = 0.7,
    starting_capital: float = 100.0,
    risk_overrides: dict[str, Any] | None = None,
    sort_by: str = "sharpe_ratio",
    top_n: int = 5,
) -> dict:
    """
    Walk-forward optimization: optimize on in-sample, validate on out-of-sample.

    Splits candles at train_pct boundary, runs parameter sweep on train set,
    then validates top N parameter sets on the out-of-sample set.

    Returns:
        dict with keys: train_best, oos_results, oos_best
    """
    # Split candles
    train_candles: dict[str, list[Candle]] = {}
    oos_candles: dict[str, list[Candle]] = {}
    for sym, c_list in candles.items():
        split_idx = max(1, int(len(c_list) * train_pct))
        train_candles[sym] = c_list[:split_idx]
        oos_candles[sym] = c_list[split_idx:]

    if not train_candles or all(len(v) == 0 for v in train_candles.values()):
        return {"train_best": None, "oos_results": [], "oos_best": None}

    # Sweep on train
    train_results = parameter_sweep(
        strategy_cls, train_candles, param_grid,
        starting_capital=starting_capital,
        risk_overrides=risk_overrides,
        top_n=top_n, sort_by=sort_by,
    )

    if not train_results:
        return {"train_best": None, "oos_results": [], "oos_best": None}

    # Validate each top train result on OOS
    oos_results = []
    base_risk_kwargs = risk_overrides or {}

    for train_res in train_results:
        param_dict = train_res.params

        risk_kwargs = dict(base_risk_kwargs)
        risk_field_names = {f.name for f in RiskConfig.__dataclass_fields__.values()}
        for k, v in param_dict.items():
            if k in risk_field_names:
                risk_kwargs[k] = v
        risk = RiskConfig(**risk_kwargs)

        strat_kwargs = {}
        for k in ("buy_threshold", "sell_threshold", "tech_weight", "sentiment_weight"):
            if k in param_dict:
                strat_kwargs[k] = param_dict[k]

        try:
            strategy = strategy_cls(risk=risk, **strat_kwargs)
        except TypeError:
            strategy = strategy_cls(risk=risk)

        bt = Backtester(strategy=strategy, starting_capital=starting_capital)
        if len(oos_candles) == 1:
            sym = next(iter(oos_candles))
            res = bt.run(sym, oos_candles[sym])
        else:
            res = bt.run_portfolio(oos_candles)

        oos_results.append(SweepResult(
            params=param_dict,
            final_value=res["final_value"],
            pnl_pct=res["pnl_pct"],
            sharpe_ratio=res["sharpe_ratio"],
            max_drawdown=res["max_drawdown"],
            win_rate=res["win_rate"],
            profit_factor=res["profit_factor"],
            num_trades=res["num_trades"],
            sortino_ratio=res["sortino_ratio"],
            calmar_ratio=res["calmar_ratio"],
        ))

    oos_results.sort(key=lambda r: getattr(r, sort_by), reverse=True)

    return {
        "train_best": train_results[0],
        "oos_results": oos_results,
        "oos_best": oos_results[0] if oos_results else None,
    }


def compare_strategies(
    strategy_configs: list[tuple[type, RiskConfig]],
    candles: dict[str, list[Candle]],
    starting_capital: float = 100.0,
) -> list[dict]:
    """
    Head-to-head comparison of strategy instances.

    Args:
        strategy_configs: list of (StrategyClass, RiskConfig) tuples
        candles: dict of symbol -> candles
        starting_capital: initial capital

    Returns:
        list of dicts with strategy name + all backtest metrics, sorted by sharpe_ratio desc.
    """
    results = []
    for strategy_cls, risk in strategy_configs:
        try:
            strategy = strategy_cls(risk=risk)
        except TypeError:
            continue

        bt = Backtester(strategy=strategy, starting_capital=starting_capital)
        if len(candles) == 1:
            sym = next(iter(candles))
            res = bt.run(sym, candles[sym])
        else:
            res = bt.run_portfolio(candles)

        res["strategy"] = strategy_cls.__name__
        results.append(res)

    results.sort(key=lambda r: r.get("sharpe_ratio", 0), reverse=True)
    return results
