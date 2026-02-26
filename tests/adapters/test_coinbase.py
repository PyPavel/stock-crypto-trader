from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from trader.adapters.coinbase import CoinbaseAdapter
from trader.models import Candle, Order


def make_adapter():
    with patch("trader.adapters.coinbase.ccxt.coinbaseadvanced") as mock_cls:
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange
        adapter = CoinbaseAdapter(api_key="k", api_secret="s")
        adapter._exchange = mock_exchange
        return adapter, mock_exchange


def test_get_price():
    adapter, exchange = make_adapter()
    exchange.fetch_ticker.return_value = {"last": 42000.0}
    assert adapter.get_price("BTC/USD") == 42000.0
    exchange.fetch_ticker.assert_called_once_with("BTC/USD")


def test_get_candles_returns_candle_objects():
    adapter, exchange = make_adapter()
    ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    exchange.fetch_ohlcv.return_value = [
        [ts, 40000.0, 41000.0, 39000.0, 40500.0, 100.0],
    ]
    candles = adapter.get_candles("BTC/USD", "1h", limit=1)
    assert len(candles) == 1
    assert isinstance(candles[0], Candle)
    assert candles[0].close == 40500.0


def test_get_balance():
    adapter, exchange = make_adapter()
    exchange.fetch_balance.return_value = {"USDT": {"free": 100.0}, "BTC": {"free": 0.001}}
    bal = adapter.get_balance()
    assert bal["USDT"] == 100.0
    assert bal["BTC"] == 0.001


def test_place_order_returns_order():
    adapter, exchange = make_adapter()
    exchange.create_market_order.return_value = {
        "id": "order-123", "price": 42000.0, "amount": 0.001, "status": "filled"
    }
    order = adapter.place_order("buy", "BTC/USD", 0.001)
    assert isinstance(order, Order)
    assert order.side == "buy"
    assert order.amount == 0.001
