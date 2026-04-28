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
from alpaca.common.exceptions import APIError

from trader.adapters.base import ExchangeAdapter
from trader.models import Candle, Order

logger = logging.getLogger(__name__)


# Alpaca error code for Pattern Day Trader protection
PDT_ERROR_CODE = 40310100


class PDTRejectedError(Exception):
    """Raised when Alpaca rejects an order due to Pattern Day Trader protection."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        super().__init__(f"PDT protection rejected order for {symbol}")

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
            # IEX only has data during market hours (~6.5h/day, no weekends).
            # timedelta(hours=limit+10) only covers ~4.6 calendar days -> ~20 bars,
            # below MIN_CANDLES=35. Use calendar-day lookback instead.
            calendar_days = limit // 3 + 14  # generous buffer for weekends/holidays
            start = now - timedelta(days=calendar_days)
        elif interval == "5m":
            start = now - timedelta(minutes=5 * (limit + 10))
        else:
            start = now - timedelta(minutes=limit + 10)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=now,
            feed=DataFeed.IEX,  # free tier — IEX feed only
        )
        bars = self._data.get_stock_bars(request)
        raw = bars.data.get(symbol, []) if hasattr(bars, "data") else []  # BarSet.__contains__ is broken

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

    def get_day_trade_count(self) -> int:
        """Return the number of day-trades used in the current 5-rolling-day window."""
        account = self._trading.get_account()
        return int(account.daytrade_count or 0)

    def place_order(self, side: str, symbol: str, amount: float) -> Order:
        """Place a market order. `amount` is in shares (from router)."""
        import time
        alpaca_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        price_est = self.get_price(symbol)

        if side == "sell":
            # Use actual position qty from Alpaca to avoid insufficient-qty errors
            # caused by floating point drift between our portfolio tracking and Alpaca's records
            try:
                position = self._trading.get_open_position(symbol)
                qty = float(position.qty_available or position.qty)
            except Exception:
                qty = amount
            if qty <= 0:
                raise ValueError(f"No position to sell for {symbol}")
            request = MarketOrderRequest(
                symbol=symbol,
                qty=round(qty, 9),
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            # Convert shares to USD notional for fractional-share buy orders
            notional_usd = round(amount * price_est, 2) if price_est > 0 else round(amount, 2)
            if notional_usd < 1.0:
                raise ValueError(f"Order notional ${notional_usd:.2f} too small for {symbol}")
            request = MarketOrderRequest(
                symbol=symbol,
                notional=notional_usd,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
            )
        try:
            raw = self._trading.submit_order(request)
        except APIError as e:
            err = getattr(e, "_error", None)
            code = None
            if isinstance(err, dict):
                code = err.get("code")
            if code == PDT_ERROR_CODE or str(PDT_ERROR_CODE) in str(e):
                raise PDTRejectedError(symbol) from e
            # Insufficient qty: Alpaca's position endpoint can return a slightly
            # higher qty than their order validator allows. Retry once with the
            # exact available qty from the error response.
            if side == "sell" and (code == 40310000 or "insufficient qty" in str(e).lower()):
                available = float(err.get("available", 0)) if isinstance(err, dict) else 0.0
                if available > 0:
                    request = MarketOrderRequest(
                        symbol=symbol,
                        qty=round(available, 9),
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY,
                    )
                    raw = self._trading.submit_order(request)
                else:
                    raise
            else:
                raise

        # Poll for fill — Alpaca market orders may not fill instantly
        order_id = str(raw.id)
        for _ in range(6):  # up to 3 seconds
            if raw.filled_avg_price and float(raw.filled_avg_price) > 0:
                break
            time.sleep(0.5)
            raw = self._trading.get_order_by_id(order_id)

        filled_price = float(raw.filled_avg_price) if raw.filled_avg_price else price_est
        notional_est = notional_usd if side == "buy" else (amount * filled_price)
        filled_qty = float(raw.filled_qty) if raw.filled_qty else (notional_est / filled_price if filled_price else 0.0)

        if filled_qty <= 0:
            logger.warning("Order %s for %s not filled after polling, using estimate", order_id, symbol)
            filled_qty = notional_est / filled_price if filled_price > 0 else 0.0

        return Order(
            id=order_id,
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
