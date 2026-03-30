import logging
from datetime import datetime, timezone
import ccxt
from trader.adapters.base import ExchangeAdapter
from trader.models import Candle, Order

logger = logging.getLogger(__name__)

INTERVAL_MAP = {"1m": "1m", "5m": "5m", "1h": "1h", "1d": "1d"}


class CoinbaseAdapter(ExchangeAdapter):
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self._exchange = ccxt.coinbaseadvanced({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })

    def get_candles(self, symbol: str, interval: str = "1h", limit: int = 100) -> list[Candle]:
        raw = self._exchange.fetch_ohlcv(symbol, INTERVAL_MAP.get(interval, "1h"), limit=limit)
        return [
            Candle(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
                open=row[1], high=row[2], low=row[3], close=row[4], volume=row[5],
            )
            for row in raw
        ]

    def get_price(self, symbol: str) -> float:
        ticker = self._exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    def get_balance(self) -> dict[str, float]:
        raw = self._exchange.fetch_balance()
        return {k: v["free"] for k, v in raw.items() if isinstance(v, dict) and "free" in v}

    def place_order(self, side: str, symbol: str, amount: float) -> Order:
        raw = self._exchange.create_market_order(symbol, side, amount)
        return Order(
            id=raw.get("id", ""),
            symbol=symbol,
            side=side,
            amount=float(raw.get("amount", amount)),
            price=float(raw.get("price", 0.0)),
            mode="live",
            status=raw.get("status", "filled"),
        )

    def get_tradeable_symbols(self) -> set[str]:
        """Return all symbols tradeable on Coinbase Advanced."""
        try:
            markets = self._exchange.load_markets()
            return set(markets.keys())
        except Exception as e:
            logger.warning("Could not load Coinbase markets: %s", e)
            return set()

    def get_open_orders(self, symbol: str) -> list[Order]:
        raw = self._exchange.fetch_open_orders(symbol)
        return [
            Order(id=o["id"], symbol=symbol, side=o["side"],
                  amount=o["amount"], price=o.get("price", 0.0), mode="live")
            for o in raw
        ]

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception:
            return False
