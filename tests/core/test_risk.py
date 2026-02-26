from trader.core.risk import RiskManager
from trader.config import RiskConfig


def make_rm():
    return RiskManager(RiskConfig(max_position_pct=0.20, stop_loss_pct=0.05, max_drawdown_pct=0.15))


def test_allows_valid_buy():
    rm = make_rm()
    result = rm.validate_buy("BTC/USD", 20.0, 100.0, {})
    assert result["allowed"] is True


def test_blocks_oversized_buy():
    rm = make_rm()
    result = rm.validate_buy("BTC/USD", 50.0, 100.0, {})
    assert result["allowed"] is False
    assert "position" in result["reason"].lower()


def test_blocks_buy_when_max_drawdown_hit():
    rm = make_rm()
    result = rm.validate_buy("BTC/USD", 10.0, 84.0, {}, starting_capital=100.0)
    assert result["allowed"] is False
    assert "drawdown" in result["reason"].lower()


def test_stop_loss_triggered():
    rm = make_rm()
    assert rm.check_stop_loss("BTC/USD", 40000.0, 37000.0) is True


def test_stop_loss_not_triggered():
    rm = make_rm()
    assert rm.check_stop_loss("BTC/USD", 40000.0, 39000.0) is False
