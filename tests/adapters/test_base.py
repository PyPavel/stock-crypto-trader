import pytest
from trader.adapters.base import ExchangeAdapter
from trader.models import Candle, Order


class ConcreteAdapter(ExchangeAdapter):
    def get_candles(self, symbol, interval, limit=100):
        return []
    def get_price(self, symbol):
        return 42000.0
    def get_balance(self):
        return {"USDT": 100.0}
    def place_order(self, side, symbol, amount):
        return Order(symbol=symbol, side=side, amount=amount, price=42000.0, mode="paper")
    def get_open_orders(self, symbol):
        return []
    def cancel_order(self, order_id, symbol):
        return True


def test_adapter_interface():
    adapter = ConcreteAdapter()
    assert adapter.get_price("BTC/USD") == 42000.0
    assert adapter.get_balance() == {"USDT": 100.0}


def test_abstract_adapter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ExchangeAdapter()
