from datetime import datetime, timedelta
from trader.core.signals import SignalGenerator
from trader.core.risk import RiskManager
from trader.models import Candle, SentimentScore, Signal
from trader.strategies.base import Strategy
from trader.config import RiskConfig

WINDOW = 50
FEE_RATE = 0.001   # 0.1% per trade
SLIPPAGE = 0.0005  # 0.05% slippage


class Backtester:
    def __init__(self, strategy: Strategy, starting_capital: float = 100.0):
        self._strategy = strategy
        self._starting_capital = starting_capital
        self._signals = SignalGenerator()
        self._risk = RiskManager(strategy.risk)

    def run(self, symbol: str, candles: list[Candle]) -> dict:
        """Run backtest on a single symbol."""
        return self.run_portfolio({symbol: candles})

    def run_portfolio(self, symbol_candles: dict[str, list[Candle]]) -> dict:
        """Run backtest across multiple symbols with trailing stops and cooldowns."""
        if not symbol_candles:
            return self._empty_results()

        # Sort all candles by timestamp and build a unified timeline
        all_events = []
        for symbol, candles in symbol_candles.items():
            for c in candles:
                all_events.append((c.timestamp, symbol, c))
        all_events.sort(key=lambda x: x[0])

        # Group by timestamp to process bars simultaneously
        timeline: dict[datetime, dict[str, Candle]] = {}
        for ts, sym, candle in all_events:
            if ts not in timeline:
                timeline[ts] = {}
            timeline[ts][sym] = candle

        sorted_timestamps = sorted(timeline.keys())

        cash = self._starting_capital
        positions: dict[str, float] = {}  # symbol -> amount held
        entry_prices: dict[str, float] = {}
        peak_prices: dict[str, float] = {}
        last_trade_time: dict[str, datetime] = {}
        trades: list[dict] = []
        portfolio_values: list[float] = [cash]
        cooldown_minutes = self._strategy.risk.cooldown_minutes

        neutral_sentiments: dict[str, SentimentScore] = {}
        for sym in symbol_candles:
            neutral_sentiments[sym] = SentimentScore(
                symbol=sym, score=0.0, source="backtest", items_analyzed=0
            )

        # Per-symbol candle history window for signal generation
        candle_history: dict[str, list[Candle]] = {s: [] for s in symbol_candles}

        for ts in sorted_timestamps:
            bars = timeline[ts]

            # Update candle history
            for sym, candle in bars.items():
                candle_history[sym].append(candle)

            for sym, candle in bars.items():
                history = candle_history[sym]
                if len(history) < WINDOW:
                    continue

                window = history[-WINDOW:]
                price = candle.close
                pos_amount = positions.get(sym, 0.0)
                position_usd = pos_amount * price

                # Update trailing peak
                if pos_amount > 0:
                    peak_prices[sym] = max(peak_prices.get(sym, price), price)

                # 1. Compute ATR for this symbol
                current_atr = self._signals.atr(window, self._strategy.risk.atr_period) if len(window) >= self._strategy.risk.atr_period + 1 else None

                # 2. Stop-loss check (ATR-adaptive if configured)
                if pos_amount > 0:
                    entry = entry_prices.get(sym, price)
                    if self._risk.check_stop_loss(sym, entry, price, atr=current_atr):
                        proceeds = self._execute_sell(
                            sym, pos_amount, entry, price, trades, "stop-loss triggered"
                        )
                        cash += proceeds
                        positions[sym] = 0.0
                        entry_prices.pop(sym, None)
                        peak_prices.pop(sym, None)
                        last_trade_time[sym] = ts
                        continue

                    # 3. Trailing stop check (ATR-adaptive if configured)
                    peak = peak_prices.get(sym, price)
                    if peak > entry and self._risk.check_trailing_stop(peak, price, atr=current_atr):
                        proceeds = self._execute_sell(
                            sym, pos_amount, entry, price, trades, "trailing-stop triggered"
                        )
                        cash += proceeds
                        positions[sym] = 0.0
                        entry_prices.pop(sym, None)
                        peak_prices.pop(sym, None)
                        last_trade_time[sym] = ts
                        continue

                    # 4. Per-trade loss limit
                    if self._risk.check_trade_loss_limit(entry, price, self._starting_capital):
                        proceeds = self._execute_sell(
                            sym, pos_amount, entry, price, trades, "per-trade-loss-limit triggered"
                        )
                        cash += proceeds
                        positions[sym] = 0.0
                        entry_prices.pop(sym, None)
                        peak_prices.pop(sym, None)
                        last_trade_time[sym] = ts
                        continue

                    # 5. Take-profit (full exit)
                    if self._risk.check_take_profit(entry, price):
                        proceeds = self._execute_sell(
                            sym, pos_amount, entry, price, trades, "take-profit triggered"
                        )
                        cash += proceeds
                        positions[sym] = 0.0
                        entry_prices.pop(sym, None)
                        peak_prices.pop(sym, None)
                        last_trade_time[sym] = ts
                        continue

                    # 6. Partial take-profit
                    if self._risk.check_partial_take_profit(entry, price):
                        partial_usd = self._risk.partial_sell_amount(position_usd)
                        if partial_usd > 0 and partial_usd < position_usd:
                            partial_amount = partial_usd / (price * (1 - SLIPPAGE))
                            proceeds = self._execute_sell(
                                sym, partial_amount, entry, price, trades, "partial-take-profit triggered"
                            )
                            cash += proceeds
                            positions[sym] -= partial_amount
                            last_trade_time[sym] = ts
                            # Don't clear entry_prices/peak_prices — still have remaining position
                            continue

                # 3. Cooldown check
                if sym in last_trade_time and cooldown_minutes > 0:
                    elapsed = (ts - last_trade_time[sym]).total_seconds() / 60
                    if elapsed < cooldown_minutes:
                        continue

                # 4. Signal + strategy
                tech_score = self._signals.score(window)
                tech_result = self._signals.score_with_trend(window)
                tech_signal = Signal(
                    symbol=sym, score=tech_score, reason="backtest",
                    trend_bullish=tech_result["trend_bullish"],
                )

                decision = self._strategy.decide(
                    sym, tech_signal, neutral_sentiments[sym],
                    capital=cash, position=position_usd,
                )

                if decision["action"] == "buy" and pos_amount == 0.0:
                    check = self._risk.validate_buy(
                        sym, decision["usd_amount"], cash, {},
                        self._starting_capital,
                    )
                    if check["allowed"]:
                        fill_price = price * (1 + SLIPPAGE)
                        amount = decision["usd_amount"] / fill_price
                        fee = amount * fill_price * FEE_RATE
                        cash -= amount * fill_price + fee
                        positions[sym] = amount
                        entry_prices[sym] = fill_price
                        peak_prices[sym] = fill_price
                        last_trade_time[sym] = ts
                        trades.append({
                            "side": "buy", "price": fill_price, "fee": fee,
                            "symbol": sym, "timestamp": ts,
                        })

                elif decision["action"] == "sell" and pos_amount > 0.0:
                    proceeds = self._execute_sell(
                        sym, pos_amount, entry_prices.get(sym, price),
                        price, trades, decision.get("reason", ""),
                    )
                    cash += proceeds
                    positions[sym] = 0.0
                    entry_prices.pop(sym, None)
                    peak_prices.pop(sym, None)
                    last_trade_time[sym] = ts

            # Portfolio value at this timestep
            total = cash
            for sym, amount in positions.items():
                if amount > 0 and sym in bars:
                    total += amount * bars[sym].close
            portfolio_values.append(total)

        # Close any open positions at end
        for sym, amount in list(positions.items()):
            if amount > 0:
                candles = symbol_candles[sym]
                final_price = candles[-1].close * (1 - SLIPPAGE)
                fee = amount * final_price * FEE_RATE
                entry = entry_prices.get(sym, final_price)
                pnl = (final_price - entry) / entry
                trades.append({
                    "side": "sell", "price": final_price, "pnl": pnl,
                    "fee": fee, "symbol": sym, "timestamp": candles[-1].timestamp,
                })
                cash += amount * final_price - fee

        return self._compute_results(portfolio_values, trades, cash)

    def _execute_sell(
        self, symbol: str, amount: float, entry_price: float,
        price: float, trades: list[dict], reason: str,
    ) -> float:
        """Record a sell trade and return net proceeds after fees+slippage."""
        fill_price = price * (1 - SLIPPAGE)
        fee = amount * fill_price * FEE_RATE
        pnl = (fill_price - entry_price) / entry_price
        trades.append({
            "side": "sell", "price": fill_price, "pnl": pnl,
            "fee": fee, "symbol": symbol, "reason": reason,
        })
        return amount * fill_price - fee

    def _compute_results(self, portfolio_values: list[float], trades: list[dict], cash: float) -> dict:
        sells = [t for t in trades if t["side"] == "sell"]
        win_rate = (
            sum(1 for t in sells if t.get("pnl", 0) > 0) / len(sells)
        ) if sells else 0.0

        peak = self._starting_capital
        max_drawdown = 0.0
        for v in portfolio_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, dd)

        total_fees = sum(t.get("fee", 0) for t in trades)
        total_pnl = cash - self._starting_capital
        pnl_pct = total_pnl / self._starting_capital

        # --- Risk-adjusted returns ---
        returns = []
        for i in range(1, len(portfolio_values)):
            prev = portfolio_values[i - 1]
            if prev > 0:
                returns.append((portfolio_values[i] - prev) / prev)

        sharpe_ratio = self._sharpe_ratio(returns)
        sortino_ratio = self._sortino_ratio(returns)
        calmar_ratio = self._calmar_ratio(pnl_pct, max_drawdown, len(portfolio_values))

        # --- Trade-level analysis ---
        pnls = [t.get("pnl", 0) for t in sells]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf") if wins else 0.0

        max_consecutive_losses = self._max_consecutive_losses(pnls)
        max_consecutive_wins = self._max_consecutive_wins(pnls)

        return {
            "final_value": cash,
            "num_trades": len(sells),
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "pnl_pct": pnl_pct,
            "total_fees": total_fees,
            "total_pnl": total_pnl,
            # Risk-adjusted
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            # Trade-level
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_consecutive_losses": max_consecutive_losses,
            "max_consecutive_wins": max_consecutive_wins,
            "trades": sells,
        }

    @staticmethod
    def _sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
        if len(returns) < 2:
            return 0.0
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std = variance ** 0.5
        if std == 0:
            return 0.0
        # Annualize assuming hourly bars (8760 periods/year)
        return (mean_ret - risk_free_rate) / std * (8760 ** 0.5)

    @staticmethod
    def _sortino_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
        if len(returns) < 2:
            return 0.0
        mean_ret = sum(returns) / len(returns)
        downside = [r for r in returns if r < risk_free_rate]
        if not downside:
            return float("inf") if mean_ret > risk_free_rate else 0.0
        downside_var = sum((r - risk_free_rate) ** 2 for r in downside) / len(downside)
        downside_std = downside_var ** 0.5
        if downside_std == 0:
            return 0.0
        return (mean_ret - risk_free_rate) / downside_std * (8760 ** 0.5)

    @staticmethod
    def _calmar_ratio(pnl_pct: float, max_drawdown: float, num_points: int) -> float:
        if max_drawdown == 0:
            return float("inf") if pnl_pct > 0 else 0.0
        # Annualize: scale pnl_pct by (8760 / num_points)
        annual_return = pnl_pct * (8760 / max(num_points, 1))
        return annual_return / max_drawdown

    @staticmethod
    def _max_consecutive_losses(pnls: list[float]) -> int:
        max_streak = 0
        streak = 0
        for p in pnls:
            if p <= 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    @staticmethod
    def _max_consecutive_wins(pnls: list[float]) -> int:
        max_streak = 0
        streak = 0
        for p in pnls:
            if p > 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    def _empty_results(self) -> dict:
        return {
            "final_value": self._starting_capital,
            "num_trades": 0, "win_rate": 0.0, "max_drawdown": 0.0,
            "pnl_pct": 0.0, "total_fees": 0.0, "total_pnl": 0.0,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
            "max_consecutive_losses": 0, "max_consecutive_wins": 0,
            "trades": [],
        }
