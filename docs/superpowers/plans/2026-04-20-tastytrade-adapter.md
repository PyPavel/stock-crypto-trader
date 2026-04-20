# TastyTrade Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `TastyTradeAdapter` that executes stock trades via TastyTrade (paper sandbox supported) while reusing Alpaca's free data feed for candles and prices.

**Architecture:** Single new adapter class in `src/trader/adapters/tastytrade.py` implementing `ExchangeAdapter`. Order execution uses the `tastytrade` SDK; candles/prices use Alpaca's `StockHistoricalDataClient` (same as `AlpacaAdapter`). Config adds a `TastyTradeConfig` dataclass. `__main__.py` gets an `elif cfg.exchange == "tastytrade":` branch identical in structure to the Alpaca branch.

**Tech Stack:** `tastytrade` PyPI package, `alpaca-py` (already a dependency), Python 3.12+

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Modify | `pyproject.toml` | Add `tastytrade>=8.0` dependency |
| Modify | `src/trader/config.py` | Add `TastyTradeConfig` dataclass + env var overrides |
| Create | `src/trader/adapters/tastytrade.py` | Full `TastyTradeAdapter` implementation |
| Modify | `src/trader/__main__.py` | Wire up `tastytrade` exchange branch |
| Create | `tests/adapters/test_tastytrade.py` | Unit tests (all external calls mocked) |

---

## Task 1: Add `tastytrade` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependency**

In `pyproject.toml`, add `"tastytrade>=8.0"` to the `dependencies` list:

```toml
dependencies = [
    "ccxt>=4.0",
    "pandas-ta>=0.3",
    "pandas>=2.0",
    "numpy>=1.26",
    "praw>=7.7",
    "requests>=2.31",
    "openai>=1.0",
    "fastapi>=0.110",
    "uvicorn>=0.27",
    "apscheduler>=3.10",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "jinja2>=3.1",
    "alpaca-py>=0.20",
    "lightgbm>=4.0",
    "scikit-learn>=1.3",
    "pyarrow>=14.0",
    "tastytrade>=8.0",
]
```

- [ ] **Step 2: Install**

```bash
pip install tastytrade
```

Expected: installs without error, `import tastytrade` works.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add tastytrade dependency"
```

---

## Task 2: Add `TastyTradeConfig` to config

**Files:**
- Modify: `src/trader/config.py`

- [ ] **Step 1: Write failing test**

In `tests/test_config.py`, add to the existing test file (read it first to find the right place):

```python
def test_tastytrade_config_defaults():
    from trader.config import TastyTradeConfig
    cfg = TastyTradeConfig()
    assert cfg.username == ""
    assert cfg.password == ""
    assert cfg.account_number == ""
    assert cfg.paper is True


def test_tastytrade_config_in_main_config(tmp_path):
    from trader.config import load_config
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "exchange: tastytrade\n"
        "mode: paper\n"
        "strategy: moderate\n"
        "capital: 5000\n"
        "pairs: [AAPL]\n"
        "tastytrade:\n"
        "  username: user\n"
        "  password: pass\n"
        "  account_number: ABC123\n"
        "  paper: true\n"
    )
    cfg = load_config(str(config_file))
    assert cfg.tastytrade.username == "user"
    assert cfg.tastytrade.account_number == "ABC123"
    assert cfg.tastytrade.paper is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py::test_tastytrade_config_defaults tests/test_config.py::test_tastytrade_config_in_main_config -v
```

Expected: FAIL with `ImportError: cannot import name 'TastyTradeConfig'`

- [ ] **Step 3: Add `TastyTradeConfig` to `src/trader/config.py`**

After the `AlpacaConfig` dataclass (around line 26), insert:

```python
@dataclass
class TastyTradeConfig:
    username: str = ""
    password: str = ""
    account_number: str = ""   # picks first account if empty
    paper: bool = True
```

In the `Config` dataclass (after the `alpaca` field, around line 119), add:

```python
    tastytrade: TastyTradeConfig = field(default_factory=TastyTradeConfig)
```

In `load_config`, add `"tastytrade"` to the key→class loop (around line 159):

```python
    for key, cls in [
        ("mimo", MimoConfig),
        ("coinbase", CoinbaseConfig),
        ("alpaca", AlpacaConfig),
        ("tastytrade", TastyTradeConfig),
        ("reddit", RedditConfig),
        ("cryptopanic", CryptoPanicConfig),
        ("discord", DiscordConfig),
        ("llm_advisor", LLMAdvisorConfig),
        ("risk", RiskConfig),
        ("ml", MLConfig),
        ("telegram", TelegramConfig),
        ("universe", UniverseConfig),
    ]:
```

After the existing Alpaca env var block (around line 196), add:

```python
    if os.environ.get("TASTYTRADE_USERNAME"):
        cfg.tastytrade.username = os.environ["TASTYTRADE_USERNAME"]
    if os.environ.get("TASTYTRADE_PASSWORD"):
        cfg.tastytrade.password = os.environ["TASTYTRADE_PASSWORD"]
    if os.environ.get("TASTYTRADE_ACCOUNT_NUMBER"):
        cfg.tastytrade.account_number = os.environ["TASTYTRADE_ACCOUNT_NUMBER"]
    if os.environ.get("TASTYTRADE_PAPER") is not None:
        cfg.tastytrade.paper = os.environ["TASTYTRADE_PAPER"].lower() in ("1", "true", "yes")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py::test_tastytrade_config_defaults tests/test_config.py::test_tastytrade_config_in_main_config -v
```

Expected: PASS

- [ ] **Step 5: Run full config test suite to check no regressions**

```bash
pytest tests/test_config.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/trader/config.py tests/test_config.py
git commit -m "feat: add TastyTradeConfig to config"
```

---

## Task 3: Implement `TastyTradeAdapter` — market data methods

**Files:**
- Create: `src/trader/adapters/tastytrade.py`
- Create: `tests/adapters/test_tastytrade.py`

These two methods (`get_candles`, `get_price`) delegate entirely to Alpaca's data client. The logic is identical to `AlpacaAdapter`.

- [ ] **Step 1: Write failing tests**

Create `tests/adapters/test_tastytrade.py`:

```python
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from trader.models import Candle, Order


def make_adapter(paper=True):
    """Build a TastyTradeAdapter with all external clients mocked."""
    with patch("trader.adapters.tastytrade.Session") as mock_session_cls, \
         patch("trader.adapters.tastytrade.Account") as mock_account_cls, \
         patch("trader.adapters.tastytrade.StockHistoricalDataClient") as mock_data_cls:

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_account = MagicMock()
        mock_account_cls.get_accounts.return_value = [mock_account]
        mock_account.account_number = "ABC123"

        mock_data = MagicMock()
        mock_data_cls.return_value = mock_data

        from trader.adapters.tastytrade import TastyTradeAdapter
        from trader.config import TastyTradeConfig, AlpacaConfig

        tt_cfg = TastyTradeConfig(username="u", password="p", account_number="ABC123", paper=paper)
        adapter = TastyTradeAdapter(
            tastytrade_cfg=tt_cfg,
            alpaca_data_key="ak",
            alpaca_data_secret="as",
        )
        adapter._session = mock_session
        adapter._account = mock_account
        adapter._data = mock_data
        return adapter, mock_session, mock_account, mock_data


def test_get_price_returns_midprice():
    adapter, _, _, mock_data = make_adapter()
    mock_quote = MagicMock()
    mock_quote.ask_price = 101.0
    mock_quote.bid_price = 99.0
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

    price = adapter.get_price("AAPL")
    assert price == 100.0


def test_get_price_ask_only():
    adapter, _, _, mock_data = make_adapter()
    mock_quote = MagicMock()
    mock_quote.ask_price = 105.0
    mock_quote.bid_price = 0.0
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

    price = adapter.get_price("AAPL")
    assert price == 105.0


def test_get_candles_returns_candle_objects():
    adapter, _, _, mock_data = make_adapter()
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mock_bar = MagicMock()
    mock_bar.timestamp = ts
    mock_bar.open = 100.0
    mock_bar.high = 110.0
    mock_bar.low = 90.0
    mock_bar.close = 105.0
    mock_bar.volume = 1000.0
    mock_bars = MagicMock()
    mock_bars.data = {"AAPL": [mock_bar]}
    mock_data.get_stock_bars.return_value = mock_bars

    candles = adapter.get_candles("AAPL", "1h", limit=1)
    assert len(candles) == 1
    assert isinstance(candles[0], Candle)
    assert candles[0].symbol == "AAPL"
    assert candles[0].close == 105.0
    assert candles[0].volume == 1000.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/adapters/test_tastytrade.py::test_get_price_returns_midprice tests/adapters/test_tastytrade.py::test_get_candles_returns_candle_objects -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'trader.adapters.tastytrade'`

- [ ] **Step 3: Create `src/trader/adapters/tastytrade.py` with data methods**

```python
import logging
import time
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from tastytrade import Session
from tastytrade.account import Account
from tastytrade.order import (
    NewOrder, NewOrderLeg, OrderAction, OrderTimeInForce, OrderType
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

    def __init__(self, tastytrade_cfg: TastyTradeConfig, alpaca_data_key: str, alpaca_data_secret: str):
        self._paper = tastytrade_cfg.paper
        self._session = Session(
            tastytrade_cfg.username,
            tastytrade_cfg.password,
            is_test_env=tastytrade_cfg.paper,
        )
        accounts = Account.get_accounts(self._session)
        if tastytrade_cfg.account_number:
            self._account = next(
                (a for a in accounts if a.account_number == tastytrade_cfg.account_number),
                accounts[0],
            )
        else:
            self._account = accounts[0]
        logger.info("TastyTrade connected to account %s (paper=%s)", self._account.account_number, self._paper)

        self._data = StockHistoricalDataClient(
            api_key=alpaca_data_key,
            secret_key=alpaca_data_secret,
        )

    # ------------------------------------------------------------------
    # Market-hours guard
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        now_et = datetime.now(_NYSE_TZ)
        if now_et.weekday() >= 5:
            return False
        if now_et.date() in _NYSE_HOLIDAYS_2026:
            return False
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_et < market_close

    # ------------------------------------------------------------------
    # Market data (Alpaca IEX free feed)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Execution (TastyTrade)
    # ------------------------------------------------------------------

    def get_balance(self) -> dict[str, float]:
        raise NotImplementedError("implemented in Task 4")

    def place_order(self, side: str, symbol: str, amount: float) -> Order:
        raise NotImplementedError("implemented in Task 4")

    def get_open_orders(self, symbol: str) -> list[Order]:
        raise NotImplementedError("implemented in Task 4")

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        raise NotImplementedError("implemented in Task 4")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/adapters/test_tastytrade.py::test_get_price_returns_midprice tests/adapters/test_tastytrade.py::test_get_price_ask_only tests/adapters/test_tastytrade.py::test_get_candles_returns_candle_objects -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trader/adapters/tastytrade.py tests/adapters/test_tastytrade.py
git commit -m "feat: add TastyTradeAdapter skeleton with Alpaca data methods"
```

---

## Task 4: Implement execution methods (`get_balance`, `place_order`, `get_open_orders`, `cancel_order`)

**Files:**
- Modify: `src/trader/adapters/tastytrade.py`
- Modify: `tests/adapters/test_tastytrade.py`

**Key TastyTrade constraints:**
- Integer shares only (no fractional). Buys: `max(1, int(amount))` shares. Sells: use actual position qty.
- Order IDs are integers from TastyTrade; store as string in `Order.id`.
- `OrderAction.BUY_TO_OPEN` for buys, `OrderAction.SELL_TO_CLOSE` for sells.

- [ ] **Step 1: Write failing tests — add to `tests/adapters/test_tastytrade.py`**

```python
def test_get_balance():
    adapter, mock_session, mock_account, _ = make_adapter()
    mock_balance = MagicMock()
    mock_balance.cash_balance = Decimal("4500.00")
    mock_balance.net_liquidating_value = Decimal("5200.00")
    mock_balance.equity_buying_power = Decimal("4500.00")
    mock_account.get_balances.return_value = mock_balance

    bal = adapter.get_balance()
    assert bal["USD"] == 4500.0
    assert bal["equity"] == 5200.0
    assert bal["buying_power"] == 4500.0


def test_place_order_buy():
    adapter, mock_session, mock_account, mock_data = make_adapter()
    mock_quote = MagicMock()
    mock_quote.ask_price = 150.0
    mock_quote.bid_price = 149.0
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

    mock_placed = MagicMock()
    mock_placed.order.id = 42
    mock_placed.order.status = "Filled"
    mock_placed.order.filled_price = Decimal("149.50")
    mock_account.place_order.return_value = mock_placed

    # amount=10 shares → buys 10 shares
    order = adapter.place_order("buy", "AAPL", 10.0)
    assert isinstance(order, Order)
    assert order.symbol == "AAPL"
    assert order.side == "buy"
    assert order.mode == "paper"
    assert order.id == "42"
    mock_account.place_order.assert_called_once()


def test_place_order_buy_fractional_rounds_to_int():
    adapter, mock_session, mock_account, mock_data = make_adapter()
    mock_quote = MagicMock()
    mock_quote.ask_price = 150.0
    mock_quote.bid_price = 149.0
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

    mock_placed = MagicMock()
    mock_placed.order.id = 43
    mock_placed.order.status = "Filled"
    mock_placed.order.filled_price = Decimal("149.50")
    mock_account.place_order.return_value = mock_placed

    # amount=0.7 shares → rounds to 1 (minimum)
    order = adapter.place_order("buy", "AAPL", 0.7)
    assert order.amount == 1


def test_place_order_sell_uses_position_qty():
    adapter, mock_session, mock_account, mock_data = make_adapter()
    mock_quote = MagicMock()
    mock_quote.ask_price = 150.0
    mock_quote.bid_price = 149.0
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

    mock_pos = MagicMock()
    mock_pos.symbol = "AAPL"
    mock_pos.quantity = Decimal("5")
    mock_pos.instrument_type = InstrumentType.EQUITY
    mock_account.get_positions.return_value = [mock_pos]

    mock_placed = MagicMock()
    mock_placed.order.id = 44
    mock_placed.order.status = "Filled"
    mock_placed.order.filled_price = Decimal("149.50")
    mock_account.place_order.return_value = mock_placed

    order = adapter.place_order("sell", "AAPL", 3.0)
    assert order.amount == 5.0  # uses actual position qty, not requested amount


def test_place_order_sell_no_position_raises():
    adapter, mock_session, mock_account, mock_data = make_adapter()
    mock_quote = MagicMock()
    mock_quote.ask_price = 150.0
    mock_quote.bid_price = 149.0
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
    mock_account.get_positions.return_value = []

    import pytest
    with pytest.raises(ValueError, match="No position to sell"):
        adapter.place_order("sell", "AAPL", 3.0)


def test_get_open_orders_filters_by_symbol():
    adapter, mock_session, mock_account, _ = make_adapter()
    mock_o1 = MagicMock()
    mock_o1.id = 10
    mock_o1.legs = [MagicMock(symbol="AAPL", action=OrderAction.BUY_TO_OPEN, quantity=Decimal("5"))]
    mock_o1.status = "Live"
    mock_o1.price = Decimal("0")

    mock_o2 = MagicMock()
    mock_o2.id = 11
    mock_o2.legs = [MagicMock(symbol="TSLA", action=OrderAction.BUY_TO_OPEN, quantity=Decimal("2"))]
    mock_o2.status = "Live"
    mock_o2.price = Decimal("0")

    mock_account.get_live_orders.return_value = [mock_o1, mock_o2]

    orders = adapter.get_open_orders("AAPL")
    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"


def test_cancel_order_returns_true():
    adapter, mock_session, mock_account, _ = make_adapter()
    mock_account.delete_order.return_value = None
    result = adapter.cancel_order("42", "AAPL")
    assert result is True
    mock_account.delete_order.assert_called_once_with(mock_session, 42)


def test_cancel_order_returns_false_on_error():
    adapter, mock_session, mock_account, _ = make_adapter()
    mock_account.delete_order.side_effect = Exception("not found")
    result = adapter.cancel_order("99", "AAPL")
    assert result is False
```

You also need this import at the top of the test file (add after existing imports):

```python
from tastytrade.order import OrderAction
from tastytrade.instruments import InstrumentType
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/adapters/test_tastytrade.py::test_get_balance tests/adapters/test_tastytrade.py::test_place_order_buy tests/adapters/test_tastytrade.py::test_cancel_order_returns_true -v
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Replace the `NotImplementedError` stubs in `src/trader/adapters/tastytrade.py` with full implementations**

Replace the four stub methods with:

```python
    def get_balance(self) -> dict[str, float]:
        balance = self._account.get_balances(self._session)
        return {
            "USD": float(balance.cash_balance),
            "equity": float(balance.net_liquidating_value),
            "buying_power": float(balance.equity_buying_power),
        }

    def place_order(self, side: str, symbol: str, amount: float) -> Order:
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
            qty = max(1, int(amount))

        action = OrderAction.BUY_TO_OPEN if side == "buy" else OrderAction.SELL_TO_CLOSE
        leg = NewOrderLeg(
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
        response = self._account.place_order(self._session, new_order, dry_run=False)
        placed = response.order
        order_id = str(placed.id)

        # Poll for fill — TastyTrade market orders may not report filled_price immediately
        filled_price = price_est
        for _ in range(6):
            fp = getattr(placed, "filled_price", None)
            if fp and float(fp) > 0:
                filled_price = float(fp)
                break
            time.sleep(0.5)
            try:
                placed = self._account.get_order(self._session, placed.id)
            except Exception:
                break

        return Order(
            id=order_id,
            symbol=symbol,
            side=side,
            amount=float(qty),
            price=filled_price,
            mode="paper" if self._paper else "live",
            status=str(placed.status),
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
```

- [ ] **Step 4: Run all tastytrade adapter tests**

```bash
pytest tests/adapters/test_tastytrade.py -v
```

Expected: all pass

- [ ] **Step 5: Run full adapter test suite to check no regressions**

```bash
pytest tests/adapters/ -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/trader/adapters/tastytrade.py tests/adapters/test_tastytrade.py
git commit -m "feat: implement TastyTradeAdapter execution methods"
```

---

## Task 5: Wire up `tastytrade` in `__main__.py`

**Files:**
- Modify: `src/trader/__main__.py`

No new tests needed — `__main__.py` wiring is covered by integration testing against real/sandbox endpoints.

- [ ] **Step 1: Add `elif cfg.exchange == "tastytrade":` branch**

In `src/trader/__main__.py`, after the `if cfg.exchange == "alpaca":` block (around line 58), add before the `else:` for coinbase:

```python
    elif cfg.exchange == "tastytrade":
        from trader.adapters.tastytrade import TastyTradeAdapter
        from trader.collectors.stock_news import StockNewsCollector
        from trader.collectors.market_sentiment import MarketSentimentCollector
        from trader.collectors.polymarket import PolymarketCollector
        from trader.collectors.unusual_whales import UnusualWhalesCollector
        from trader.collectors.google_trends import GoogleTrendsCollector
        from trader.collectors.earnings import EarningsCollector

        adapter = TastyTradeAdapter(
            tastytrade_cfg=cfg.tastytrade,
            alpaca_data_key=cfg.alpaca.api_key,
            alpaca_data_secret=cfg.alpaca.api_secret,
        )
        collectors = [
            StockNewsCollector(),
            StockTwitsCollector(),
        ]
        if cfg.discord.bot_token and cfg.discord.stock_channels:
            collectors.append(DiscordCollector(
                bot_token=cfg.discord.bot_token,
                channel_ids=cfg.discord.stock_channels,
                asset_class="stock",
                limit=cfg.discord.limit,
                cache_seconds=cfg.discord.cache_seconds,
            ))
            logger.info("Discord collector enabled for stocks (%d channels)", len(cfg.discord.stock_channels))
        numeric_collectors = [
            MarketSentimentCollector(),
            MacroCollector(asset_class="stock"),
            EarningsCollector(),
            PolymarketCollector(),
            UnusualWhalesCollector(),
            GoogleTrendsCollector(asset_class="stock"),
        ]
```

- [ ] **Step 2: Fix the `label` line** (around line 149) to also recognise `tastytrade` as a stock exchange:

```python
    label = "STOCK" if cfg.exchange in ("alpaca", "tastytrade") else "CRYPTO"
```

- [ ] **Step 3: Fix the `SymbolUniverse` instantiation** (around line 132) to pass tastytrade config where alpaca data keys are needed:

The current code passes `alpaca_cfg=cfg.alpaca if cfg.exchange == "alpaca" else None`. TastyTrade also uses Alpaca for data, so keep passing `cfg.alpaca` for `tastytrade` exchange too:

```python
    universe = SymbolUniverse(
        exchange=cfg.exchange,
        seed_pairs=cfg.pairs,
        universe_config=cfg.universe,
        alpaca_cfg=cfg.alpaca if cfg.exchange in ("alpaca", "tastytrade") else None,
        valid_symbols=valid_symbols,
    )
```

- [ ] **Step 4: Verify the bot starts cleanly in dry-run (import check)**

```bash
python -c "from trader.__main__ import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/trader/__main__.py
git commit -m "feat: wire tastytrade exchange in __main__"
```

---

## Task 6: Add example config and run full test suite

**Files:**
- Create: `config-tastytrade.yaml.example`

- [ ] **Step 1: Create example config**

```yaml
# config-tastytrade.yaml.example
# Copy to config-tastytrade.yaml and fill in credentials.
# Run: python -m trader --config config-tastytrade.yaml --port 8001 --db tastytrade.db

exchange: tastytrade
mode: paper
strategy: moderate
capital: 5000
cycle_interval: 300

pairs:
  - AAPL
  - TSLA
  - NVDA
  - MSFT
  - GOOGL
  - AMZN
  - META

tastytrade:
  paper: true          # set to false for live trading

alpaca:                # required for market data (free IEX feed)
  paper: true          # value ignored for data-only use

telegram:
  bot_token: ""        # set TELEGRAM_BOT_TOKEN env var
  chat_id: ""          # set TELEGRAM_CHAT_ID env var

risk:
  max_position_pct: 0.20
  stop_loss_pct: 0.05
  max_drawdown_pct: 0.15
  max_open_positions: 5
  min_position_usd: 50.0
```

- [ ] **Step 2: Run full test suite**

```bash
pytest -v
```

Expected: all tests pass, no regressions

- [ ] **Step 3: Commit**

```bash
git add config-tastytrade.yaml.example
git commit -m "chore: add tastytrade example config"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Task |
|---|---|
| `TastyTradeAdapter` implementing `ExchangeAdapter` | Task 3, 4 |
| Order execution via TastyTrade SDK | Task 4 |
| Candles/price via Alpaca data | Task 3 |
| Paper mode via TastyTrade sandbox (`is_test_env=True`) | Task 3 |
| `TastyTradeConfig` dataclass | Task 2 |
| Env var overrides | Task 2 |
| `__main__.py` `elif` branch | Task 5 |
| Same collectors as Alpaca branch | Task 5 |
| `is_market_open()` NYSE guard | Task 3 |
| `alpaca` section required for data credentials | Task 5 (SymbolUniverse fix) |
| Multiple instances via separate config files | Task 6 (example config) |
| Integer shares only | Task 4 (`max(1, int(amount))`) |
| Sell uses actual position qty | Task 4 |

All spec requirements covered. No gaps found.
