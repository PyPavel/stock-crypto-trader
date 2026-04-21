import logging
import time
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from tastytrade import Session
from tastytrade.account import Account
from tastytrade.order import (
    NewOrder, Leg, OrderAction, OrderTimeInForce, OrderType, OrderStatus
)
from tastytrade.instruments import InstrumentType

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from trader.adapters.base import ExchangeAdapter
from trader.config import TastyTradeConfig
from trader.models import Candle, Order

logger = logging.getLogger(__name__)

_NYSE_TZ = ZoneInfo("America/New_York")

_NYSE_HOLIDAYS_2026 = {
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),
    date(2027, 1, 1),
}

INTERVAL_MAP: dict[str, TimeFrame] = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}


class TastyTradeAdapter(ExchangeAdapter):
    """TastyTrade adapter for US equities. Execution via TastyTrade; market data via Alpaca IEX."""

    # Persists the rotating remember-token across container restarts without
    # requiring env var updates after each auth.
    _TOKEN_FILE = Path("/app/data/tt_remember_token")

    def __init__(self, tastytrade_cfg: TastyTradeConfig, alpaca_data_key: str, alpaca_data_secret: str):
        self._paper = tastytrade_cfg.paper

        # Token priority: file (rotated) > env var > password
        token = self._load_token() or tastytrade_cfg.remember_token
        if token:
            self._session = Session(
                tastytrade_cfg.username,
                remember_token=token,
                remember_me=True,
            )
        else:
            self._session = Session(
                tastytrade_cfg.username,
                password=tastytrade_cfg.password,
                remember_me=True,
            )
        self._save_token(self._session.remember_token)

        accounts = Account.get_accounts(self._session)
        if not accounts:
            raise RuntimeError("TastyTrade: no accounts returned for this session")
        if tastytrade_cfg.account_number:
            self._account = next(
                (a for a in accounts if a.account_number == tastytrade_cfg.account_number),
                None,
            )
            if self._account is None:
                logger.warning(
                    "Account %s not found; falling back to %s",
                    tastytrade_cfg.account_number,
                    accounts[0].account_number,
                )
                self._account = accounts[0]
        else:
            self._account = accounts[0]
        logger.info("TastyTrade connected to account %s (paper=%s)", self._account.account_number, self._paper)

        self._data = StockHistoricalDataClient(
            api_key=alpaca_data_key,
            secret_key=alpaca_data_secret,
        )

    def _load_token(self) -> str:
        try:
            return self._TOKEN_FILE.read_text().strip()
        except OSError:
            return ""

    def _save_token(self, token: str | None) -> None:
        if not token:
            return
        try:
            self._TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._TOKEN_FILE.write_text(token)
        except OSError as e:
            logger.warning("Could not persist remember-token: %s", e)

    def is_market_open(self) -> bool:
        now_et = datetime.now(_NYSE_TZ)
        if now_et.weekday() >= 5:
            return False
        if now_et.date() in _NYSE_HOLIDAYS_2026:
            return False
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_et < market_close

    def get_candles(self, symbol: str, interval: str = "1h", limit: int = 100) -> list[Candle]:
        timeframe = INTERVAL_MAP.get(interval, TimeFrame(1, TimeFrameUnit.Hour))
        now = datetime.now(timezone.utc)
        if interval == "1d":
            start = now - timedelta(days=limit + 10)
        elif interval == "1h":
            calendar_days = limit // 3 + 14
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
            feed=DataFeed.IEX,
        )
        bars = self._data.get_stock_bars(request)
        raw = bars.data.get(symbol, []) if hasattr(bars, "data") else []
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
        ask = float(quote.ask_price) if quote.ask_price else 0.0
        bid = float(quote.bid_price) if quote.bid_price else 0.0
        if ask > 0 and bid > 0:
            return (ask + bid) / 2.0
        return ask or bid

    def get_balance(self) -> dict[str, float]:
        balance = self._account.get_balances(self._session)
        return {
            "USD": float(balance.cash_balance),
            "equity": float(balance.net_liquidating_value),
            "buying_power": float(balance.equity_buying_power),
        }

    def place_order(self, side: str, symbol: str, amount: float) -> Order:
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side!r}")
        price_est = self.get_price(symbol)

        if side == "sell":
            positions = self._account.get_positions(self._session)
            pos = next(
                (p for p in positions
                 if p.symbol == symbol and p.instrument_type == InstrumentType.EQUITY),
                None,
            )
            if pos is None or float(pos.quantity) <= 0:
                raise ValueError(f"No position to sell for {symbol}")
            qty = int(float(pos.quantity))
        else:
            qty = max(1, round(amount))

        action = OrderAction.BUY_TO_OPEN if side == "buy" else OrderAction.SELL_TO_CLOSE
        leg = Leg(
            instrument_type=InstrumentType.EQUITY,
            symbol=symbol,
            quantity=Decimal(str(qty)),
            action=action,
        )
        new_order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.MARKET,
            legs=[leg],
        )
        response = self._account.place_order(self._session, new_order, dry_run=self._paper)

        if self._paper:
            order_id = f"paper-{int(time.time() * 1000)}"
            return Order(
                id=order_id,
                symbol=symbol,
                side=side,
                amount=float(qty),
                price=price_est,
                mode="paper",
                status="paper",
            )

        placed = response.order
        order_id = str(placed.id)
        filled_price = price_est
        for _ in range(6):
            status_val = placed.status.value if hasattr(placed.status, "value") else str(placed.status)
            if status_val.lower() in ("filled", "executed"):
                if placed.price and float(placed.price) > 0:
                    filled_price = float(placed.price)
                break
            time.sleep(0.5)
            try:
                placed = self._account.get_order(self._session, placed.id)
            except Exception:
                break
        else:
            logger.warning("TastyTrade order %s not confirmed filled after polling; using price estimate", order_id)

        final_status = placed.status.value if hasattr(placed.status, "value") else str(placed.status)
        return Order(
            id=order_id,
            symbol=symbol,
            side=side,
            amount=float(qty),
            price=filled_price,
            mode="live",
            status=final_status,
        )

    def get_open_orders(self, symbol: str) -> list[Order]:
        live_orders = self._account.get_live_orders(self._session)
        result = []
        for o in live_orders:
            legs = getattr(o, "legs", [])
            if not legs or legs[0].symbol != symbol:
                continue
            leg = legs[0]
            side = "buy" if leg.action == OrderAction.BUY_TO_OPEN else "sell"
            result.append(Order(
                id=str(o.id),
                symbol=symbol,
                side=side,
                amount=float(leg.quantity),
                price=float(o.price) if o.price else 0.0,
                mode="paper" if self._paper else "live",
                status=str(o.status),
            ))
        return result

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self._account.delete_order(self._session, int(order_id))
            return True
        except Exception as e:
            logger.warning("Failed to cancel TastyTrade order %s: %s", order_id, e)
            return False
