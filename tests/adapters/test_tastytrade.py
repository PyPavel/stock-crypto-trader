from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

from tastytrade.order import OrderAction
from tastytrade.instruments import InstrumentType

from trader.models import Candle, Order


def make_adapter(paper=True):
    """Build a TastyTradeAdapter with all external clients mocked."""
    # Python 3.14 patch() uses pkgutil._resolve_name which traverses attributes —
    # the module must be in sys.modules before the with-patch context is entered.
    import trader.adapters.tastytrade  # noqa: F401
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
        from trader.config import TastyTradeConfig

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
