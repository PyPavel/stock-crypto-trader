import pytest
from datetime import datetime, timezone, date
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.common.exceptions import APIError

from trader.adapters.alpaca import AlpacaAdapter, PDTRejectedError
from trader.models import Candle, Order

_NYSE_TZ = ZoneInfo("America/New_York")

def make_adapter(paper=True):
    with patch("trader.adapters.alpaca.TradingClient") as mock_trading_cls, \
         patch("trader.adapters.alpaca.StockHistoricalDataClient") as mock_data_cls:
        
        mock_trading = MagicMock()
        mock_trading_cls.return_value = mock_trading
        
        mock_data = MagicMock()
        mock_data_cls.return_value = mock_data
        
        adapter = AlpacaAdapter(api_key="ak", api_secret="as", paper=paper)
        # Re-assign to ensure we use the mocks
        adapter._trading = mock_trading
        adapter._data = mock_data
        return adapter, mock_trading, mock_data

def test_is_market_open_weekday_noon():
    adapter, _, _ = make_adapter()
    # Monday April 27, 2026 12:00 ET
    dt = datetime(2026, 4, 27, 12, 0, 0, tzinfo=_NYSE_TZ)
    with patch("trader.adapters.alpaca.datetime") as mock_datetime:
        mock_datetime.now.return_value = dt
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        assert adapter.is_market_open() is True

def test_is_market_open_weekend():
    adapter, _, _ = make_adapter()
    # Saturday April 25, 2026 12:00 ET
    dt = datetime(2026, 4, 25, 12, 0, 0, tzinfo=_NYSE_TZ)
    with patch("trader.adapters.alpaca.datetime") as mock_datetime:
        mock_datetime.now.return_value = dt
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        assert adapter.is_market_open() is False

def test_is_market_open_holiday():
    adapter, _, _ = make_adapter()
    # New Year's Day 2026
    dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=_NYSE_TZ)
    with patch("trader.adapters.alpaca.datetime") as mock_datetime:
        mock_datetime.now.return_value = dt
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        assert adapter.is_market_open() is False

def test_get_price_midpoint():
    adapter, _, mock_data = make_adapter()
    mock_quote = MagicMock()
    mock_quote.ask_price = 100.10
    mock_quote.bid_price = 99.90
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
    
    price = adapter.get_price("AAPL")
    assert price == 100.0

def test_get_price_ask_only():
    adapter, _, mock_data = make_adapter()
    mock_quote = MagicMock()
    mock_quote.ask_price = 105.0
    mock_quote.bid_price = 0.0
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
    
    price = adapter.get_price("AAPL")
    assert price == 105.0

def test_get_candles():
    adapter, _, mock_data = make_adapter()
    ts = datetime(2026, 4, 27, 10, 0, 0, tzinfo=timezone.utc)
    mock_bar = MagicMock()
    mock_bar.timestamp = ts
    mock_bar.open = 150.0
    mock_bar.high = 155.0
    mock_bar.low = 149.0
    mock_bar.close = 152.0
    mock_bar.volume = 10000.0
    
    mock_bars = MagicMock()
    mock_bars.data = {"AAPL": [mock_bar]}
    mock_data.get_stock_bars.return_value = mock_bars
    
    candles = adapter.get_candles("AAPL", "1h", limit=1)
    assert len(candles) == 1
    assert candles[0].close == 152.0
    assert candles[0].timestamp == ts

def test_get_balance():
    adapter, mock_trading, _ = make_adapter()
    mock_account = MagicMock()
    mock_account.cash = "1000.50"
    mock_account.equity = "5000.75"
    mock_account.buying_power = "2000.00"
    mock_trading.get_account.return_value = mock_account
    
    bal = adapter.get_balance()
    assert bal["USD"] == 1000.50
    assert bal["equity"] == 5000.75
    assert bal["buying_power"] == 2000.00

def test_place_order_buy():
    adapter, mock_trading, mock_data = make_adapter()
    # Mock price
    mock_quote = MagicMock()
    mock_quote.ask_price = 200.0
    mock_quote.bid_price = 199.0
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
    
    # Mock submit order response
    mock_raw_order = MagicMock()
    mock_raw_order.id = "order-123"
    mock_raw_order.status = OrderStatus.FILLED
    mock_raw_order.filled_avg_price = "200.50"
    mock_raw_order.filled_qty = "5.0"
    mock_trading.submit_order.return_value = mock_raw_order
    mock_trading.get_order_by_id.return_value = mock_raw_order
    
    order = adapter.place_order("buy", "AAPL", 5.0)
    assert order.id == "order-123"
    assert order.side == "buy"
    assert order.price == 200.50
    assert order.amount == 5.0

def test_place_order_sell_uses_actual_qty():
    adapter, mock_trading, mock_data = make_adapter()
    # Mock price
    mock_quote = MagicMock()
    mock_quote.ask_price = 200.0
    mock_quote.bid_price = 199.0
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
    
    # Mock position
    mock_position = MagicMock()
    mock_position.qty = "10.0"
    mock_position.qty_available = "10.0"
    mock_trading.get_open_position.return_value = mock_position
    
    # Mock submit order response
    mock_raw_order = MagicMock()
    mock_raw_order.id = "order-456"
    mock_raw_order.status = OrderStatus.FILLED
    mock_raw_order.filled_avg_price = "199.50"
    mock_raw_order.filled_qty = "10.0"
    mock_trading.submit_order.return_value = mock_raw_order
    mock_trading.get_order_by_id.return_value = mock_raw_order
    
    order = adapter.place_order("sell", "AAPL", 5.0)
    # requested 5, but adapter should sell all 10 available
    assert order.amount == 10.0
    assert order.id == "order-456"

def test_place_order_pdt_error():
    adapter, mock_trading, mock_data = make_adapter()
    # Mock price
    mock_quote = MagicMock()
    mock_quote.ask_price = 200.0
    mock_quote.bid_price = 199.0
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
    
    # Mock APIError for PDT
    e = APIError("PDT error")
    e._error = {"code": 40310100, "message": "PDT protection"}
    mock_trading.submit_order.side_effect = e
    
    with pytest.raises(PDTRejectedError):
        adapter.place_order("buy", "AAPL", 1.0)

def test_get_open_orders():
    adapter, mock_trading, _ = make_adapter()
    mock_o = MagicMock()
    mock_o.id = "o-1"
    mock_o.side = OrderSide.BUY
    mock_o.qty = "5"
    mock_o.limit_price = "150.0"
    mock_o.status = OrderStatus.OPEN
    mock_trading.get_orders.return_value = [mock_o]
    
    orders = adapter.get_open_orders("AAPL")
    assert len(orders) == 1
    assert orders[0].id == "o-1"
    assert orders[0].side == "buy"

def test_cancel_order():
    adapter, mock_trading, _ = make_adapter()
    mock_trading.cancel_order_by_id.return_value = None
    
    assert adapter.cancel_order("o-1", "AAPL") is True
    mock_trading.cancel_order_by_id.assert_called_once_with("o-1")
