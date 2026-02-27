from trader.core.signals import SignalGenerator
from trader.core.risk import RiskManager
from trader.models import Candle, SentimentScore, Signal
from trader.strategies.base import Strategy

WINDOW = 50


class Backtester:
    def __init__(self, strategy: Strategy, starting_capital: float = 100.0):
        self._strategy = strategy
        self._starting_capital = starting_capital
        self._signals = SignalGenerator()
        self._risk = RiskManager(strategy.risk)

    def run(self, symbol: str, candles: list[Candle]) -> dict:
        cash = self._starting_capital
        position = 0.0
        entry_price = 0.0
        trades = []
        portfolio_values = [cash]

        neutral_sentiment = SentimentScore(symbol=symbol, score=0.0,
                                           source="backtest", items_analyzed=0)

        for i in range(WINDOW, len(candles)):
            window = candles[i - WINDOW:i]
            current = candles[i]
            price = current.close

            tech_score = self._signals.score(window)
            tech_signal = Signal(symbol=symbol, score=tech_score, reason="backtest")
            position_usd = position * price

            decision = self._strategy.decide(symbol, tech_signal, neutral_sentiment,
                                              capital=cash, position=position_usd)

            if decision["action"] == "buy" and position == 0.0:
                check = self._risk.validate_buy(symbol, decision["usd_amount"], cash, {},
                                                self._starting_capital)
                if check["allowed"]:
                    amount = decision["usd_amount"] / price
                    cash -= amount * price
                    position = amount
                    entry_price = price
                    trades.append({"side": "buy", "price": price})

            elif decision["action"] == "sell" and position > 0.0:
                proceeds = position * price
                pnl = (price - entry_price) / entry_price
                trades.append({"side": "sell", "price": price, "pnl": pnl})
                cash += proceeds
                position = 0.0
                entry_price = 0.0

            portfolio_values.append(cash + position * price)

        # Close any open position at end
        if position > 0.0:
            final_price = candles[-1].close
            pnl = (final_price - entry_price) / entry_price
            trades.append({"side": "sell", "price": final_price, "pnl": pnl})
            cash += position * final_price

        sells = [t for t in trades if t["side"] == "sell"]
        win_rate = (sum(1 for t in sells if t.get("pnl", 0) > 0) / len(sells)) if sells else 0.0

        peak = self._starting_capital
        max_drawdown = 0.0
        for v in portfolio_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, dd)

        return {
            "final_value": cash,
            "num_trades": len(sells),
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "pnl_pct": (cash - self._starting_capital) / self._starting_capital,
        }
