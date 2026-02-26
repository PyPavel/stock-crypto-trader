from abc import ABC, abstractmethod
from trader.models import Candle, Order


class ExchangeAdapter(ABC):
    """Implement this interface to add any exchange (crypto or stocks)."""

    @abstractmethod
    def get_candles(self, symbol: str, interval: str, limit: int = 100) -> list[Candle]:
        """Return OHLCV candles. interval: '1m', '5m', '1h', '1d'"""

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Return current market price."""

    @abstractmethod
    def get_balance(self) -> dict[str, float]:
        """Return {currency: amount} balances."""

    @abstractmethod
    def place_order(self, side: str, symbol: str, amount: float) -> Order:
        """Place a market order. side: 'buy' | 'sell'"""

    @abstractmethod
    def get_open_orders(self, symbol: str) -> list[Order]:
        """Return all open orders for symbol."""

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order. Returns True if cancelled."""
