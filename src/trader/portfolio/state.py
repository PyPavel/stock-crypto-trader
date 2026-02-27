import sqlite3
from trader.models import Trade
from trader.portfolio.db import init_db, save_trade, load_trades


class Portfolio:
    def __init__(self, db_path: str, starting_capital: float):
        self.starting_capital = starting_capital
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        init_db(self._conn)

        self.cash = starting_capital
        self.positions: dict[str, dict] = {}  # symbol -> {amount, entry_price}
        for trade in load_trades(self._conn):
            self._apply(trade)

    def record_trade(self, trade: Trade) -> None:
        save_trade(self._conn, trade)
        self._apply(trade)

    def _apply(self, trade: Trade) -> None:
        cost = trade.amount * trade.price + trade.fee
        if trade.side == "buy":
            self.cash -= cost
            pos = self.positions.setdefault(trade.symbol, {"amount": 0.0, "entry_price": 0.0})
            pos["amount"] += trade.amount
            pos["entry_price"] = trade.price
        else:
            self.cash += trade.amount * trade.price - trade.fee
            if trade.symbol in self.positions:
                self.positions[trade.symbol]["amount"] -= trade.amount
                if self.positions[trade.symbol]["amount"] <= 0:
                    del self.positions[trade.symbol]

    def total_value(self, prices: dict[str, float]) -> float:
        holdings = sum(
            pos["amount"] * prices.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        )
        return self.cash + holdings

    def get_trades(self) -> list[Trade]:
        return load_trades(self._conn)
