from trader.adapters.base import ExchangeAdapter
from trader.models import Order


# Paper trading assumptions
DEFAULT_FEE_RATE = 0.001   # 0.1% per trade (typical for Coinbase Pro / Alpaca)
DEFAULT_SLIPPAGE = 0.0005  # 0.05% simulated slippage


class OrderRouter:
    def __init__(self, adapter: ExchangeAdapter, mode: str,
                 fee_rate: float = DEFAULT_FEE_RATE,
                 slippage: float = DEFAULT_SLIPPAGE):
        self._adapter = adapter
        self._mode = mode  # "paper" | "live"
        self._fee_rate = fee_rate
        self._slippage = slippage

    def execute(self, side: str, symbol: str, usd_amount: float, price: float | None = None) -> Order:
        if price is None:
            price = self._adapter.get_price(symbol)
        if price <= 0:
            raise ValueError(f"Invalid price {price} for {symbol}")

        # Apply slippage: buys pay more, sells receive less
        if side == "buy":
            fill_price = price * (1 + self._slippage)
        else:
            fill_price = price * (1 - self._slippage)

        amount = usd_amount / fill_price

        if self._mode == "paper":
            return Order(
                symbol=symbol, side=side, amount=amount,
                price=fill_price, mode="paper", status="filled",
            )

        return self._adapter.place_order(side, symbol, amount)
