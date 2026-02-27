from unittest.mock import MagicMock
from trader.core.router import OrderRouter
from trader.models import Order
from trader.adapters.base import ExchangeAdapter


def make_router(mode):
    adapter = MagicMock(spec=ExchangeAdapter)
    adapter.get_price.return_value = 42000.0
    adapter.place_order.return_value = Order(
        symbol="BTC/USD", side="buy", amount=0.001, price=42000.0, mode="live"
    )
    return OrderRouter(adapter=adapter, mode=mode), adapter


def test_paper_buy_does_not_call_adapter():
    router, adapter = make_router("paper")
    order = router.execute("buy", "BTC/USD", usd_amount=10.0)
    assert order.mode == "paper"
    adapter.place_order.assert_not_called()


def test_paper_order_calculates_amount_from_price():
    router, adapter = make_router("paper")
    order = router.execute("buy", "BTC/USD", usd_amount=42.0)
    assert abs(order.amount - 0.001) < 1e-6


def test_live_buy_calls_adapter():
    router, adapter = make_router("live")
    router.execute("buy", "BTC/USD", usd_amount=42.0)
    adapter.place_order.assert_called_once()


def test_paper_sell_returns_sell_order():
    router, adapter = make_router("paper")
    order = router.execute("sell", "BTC/USD", usd_amount=42.0)
    assert order.side == "sell"
    assert order.mode == "paper"
