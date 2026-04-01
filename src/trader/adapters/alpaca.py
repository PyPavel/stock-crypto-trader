import logging
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from trader.adapters.base import ExchangeAdapter
from trader.models import Candle, Order

logger = logging.getLogger(__name__)

_NYSE_TZ = ZoneInfo("America/New_York")

# Major NYSE holidays (month, day) — covers early-closed days too
_NYSE_HOLIDAYS_2026 = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
    date(2027, 1, 1),   # New Year's Day 2027
}

INTERVAL_MAP: dict[str, TimeFrame] = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}


class AlpacaAdapter(ExchangeAdapter):
    """Alpaca adapter for US equities (paper or live)."""

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self._paper = paper
        self._trading = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper,
        )
        # Market data client (free tier works without base_url override)
        self._data = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )

    # ------------------------------------------------------------------
    # Market-hours guard
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Return True if NYSE is currently open (09:30–16:00 ET, Mon–Fri)."""
        now_et = datetime.now(_NYSE_TZ)
        if now_et.weekday() >= 5:          # Saturday=5, Sunday=6
            return False
        if now_et.date() in _NYSE_HOLIDAYS_2026:
            return False
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_et < market_close

    # ------------------------------------------------------------------
    # ExchangeAdapter interface
    # ------------------------------------------------------------------

    def get_candles(self, symbol: str, interval: str = "1h", limit: int = 100) -> list[Candle]:
        timeframe = INTERVAL_MAP.get(interval, TimeFrame(1, TimeFrameUnit.Hour))

        # Calculate a start date that gives us roughly `limit` bars
        now = datetime.now(timezone.utc)
        if interval == "1d":
            start = now - timedelta(days=limit + 10)
        elif interval == "1h":
            start = now - timedelta(hours=limit + 10)
        elif interval == "5m":
            start = now - timedelta(minutes=5 * (limit + 10))
        else:
            start = now - timedelta(minutes=limit + 10)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=now,
            limit=limit,
            feed=DataFeed.IEX,  # free tier — IEX feed only
        )
        bars = self._data.get_stock_bars(request)
        raw = bars[symbol] if symbol in bars else []

        return [
            Candle(
                symbol=symbol,
                timestamp=bar.timestamp.astimezone(timezone.utc),
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            )
            for bar in raw
        ][-limit:]

    def get_price(self, symbol: str) -> float:
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        quotes = self._data.get_stock_latest_quote(request)
        quote = quotes[symbol]
        # Use mid-price if both sides available, else ask price
        ask = float(quote.ask_price) if quote.ask_price else 0.0
        bid = float(quote.bid_price) if quote.bid_price else 0.0
        if ask > 0 and bid > 0:
            return (ask + bid) / 2.0
        return ask or bid

    def get_balance(self) -> dict[str, float]:
        account = self._trading.get_account()
        return {
            "USD": float(account.cash),
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
        }

    def place_order(self, side: str, symbol: str, amount: float) -> Order:
        alpaca_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        # amount is USD notional — use notional field for market orders
        request = MarketOrderRequest(
            symbol=symbol,
            notional=round(amount, 2),
            side=alpaca_side,
            time_in_force=TimeInForce.DAY,
        )
        raw = self._trading.submit_order(request)

        filled_price = float(raw.filled_avg_price) if raw.filled_avg_price else 0.0
        filled_qty = float(raw.filled_qty) if raw.filled_qty else (amount / filled_price if filled_price else 0.0)

        return Order(
            id=str(raw.id),
            symbol=symbol,
            side=side,
            amount=filled_qty,
            price=filled_price,
            mode="paper" if self._paper else "live",
            status=str(raw.status),
        )

    def get_open_orders(self, symbol: str) -> list[Order]:
        from alpaca.trading.requests import GetOrdersRequest
        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol],
        )
        raw_orders = self._trading.get_orders(filter=request)
        return [
            Order(
                id=str(o.id),
                symbol=symbol,
                side=str(o.side).split(".")[-1].lower(),
                amount=float(o.qty) if o.qty else 0.0,
                price=float(o.limit_price) if o.limit_price else 0.0,
                mode="paper" if self._paper else "live",
                status=str(o.status),
            )
            for o in raw_orders
        ]

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self._trading.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False
