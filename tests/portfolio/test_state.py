import pytest
from trader.portfolio.state import Portfolio
from trader.models import Trade
from datetime import datetime, timezone


@pytest.fixture
def portfolio(tmp_path):
    return Portfolio(db_path=str(tmp_path / "test.db"), starting_capital=100.0)


def test_initial_balance(portfolio):
    assert portfolio.cash == 100.0
    assert portfolio.positions == {}


def test_record_buy(portfolio):
    trade = Trade(order_id="1", symbol="BTC/USD", side="buy",
                  amount=0.001, price=40000.0, fee=0.5, mode="paper",
                  timestamp=datetime.now(timezone.utc))
    portfolio.record_trade(trade)
    assert "BTC/USD" in portfolio.positions
    assert portfolio.cash < 100.0


def test_record_sell(portfolio):
    buy = Trade(order_id="1", symbol="BTC/USD", side="buy",
                amount=0.001, price=40000.0, fee=0.5, mode="paper",
                timestamp=datetime.now(timezone.utc))
    portfolio.record_trade(buy)
    sell = Trade(order_id="2", symbol="BTC/USD", side="sell",
                 amount=0.001, price=42000.0, fee=0.5, mode="paper",
                 timestamp=datetime.now(timezone.utc))
    portfolio.record_trade(sell)
    assert portfolio.positions.get("BTC/USD", {}).get("amount", 0.0) <= 0.0


def test_total_value(portfolio):
    assert portfolio.total_value(prices={"BTC/USD": 40000.0}) == 100.0


def test_trades_persisted(tmp_path):
    p = Portfolio(db_path=str(tmp_path / "test.db"), starting_capital=100.0)
    trade = Trade(order_id="1", symbol="BTC/USD", side="buy",
                  amount=0.001, price=40000.0, fee=0.5, mode="paper",
                  timestamp=datetime.now(timezone.utc))
    p.record_trade(trade)
    p2 = Portfolio(db_path=str(tmp_path / "test.db"), starting_capital=100.0)
    assert len(p2.get_trades()) == 1
