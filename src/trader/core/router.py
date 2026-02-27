from trader.adapters.base import ExchangeAdapter
from trader.models import Order


class OrderRouter:
    def __init__(self, adapter: ExchangeAdapter, mode: str):
        self._adapter = adapter
        self._mode = mode  # "paper" | "live"

    def execute(self, side: str, symbol: str, usd_amount: float, price: float | None = None) -> Order:
        if price is None:
            price = self._adapter.get_price(symbol)
        amount = usd_amount / price

        if self._mode == "paper":
            return Order(
                symbol=symbol, side=side, amount=amount,
                price=price, mode="paper", status="filled",
            )

        return self._adapter.place_order(side, symbol, amount)
