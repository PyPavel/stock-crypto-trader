from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch
from trader.core.time_gate import TimeWindowGate
from trader.config import TimeGateConfig

_ET = ZoneInfo("America/New_York")


def _gate(enabled=True, buy_start="15:00", buy_end="16:00",
          sell_start="09:30", sell_end="10:30"):
    cfg = TimeGateConfig(enabled=enabled, buy_start=buy_start, buy_end=buy_end,
                         sell_start=sell_start, sell_end=sell_end)
    return TimeWindowGate(cfg)


def _mock_et(hour, minute):
    """Return an ET-aware datetime for the given wall-clock time."""
    return datetime(2026, 4, 29, hour, minute, 0, tzinfo=_ET)


def test_can_buy_inside_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(15, 30)
        assert gate.can_buy() is True


def test_can_buy_at_start_of_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(15, 0)
        assert gate.can_buy() is True


def test_can_buy_at_end_of_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(16, 0)
        assert gate.can_buy() is True


def test_cannot_buy_before_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(10, 0)
        assert gate.can_buy() is False


def test_cannot_buy_after_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(16, 1)
        assert gate.can_buy() is False


def test_can_sell_inside_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(10, 0)
        assert gate.can_sell() is True


def test_cannot_sell_outside_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(14, 0)
        assert gate.can_sell() is False


def test_disabled_gate_always_allows_buy():
    gate = _gate(enabled=False)
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(10, 0)   # outside buy window
        assert gate.can_buy() is True


def test_disabled_gate_always_allows_sell():
    gate = _gate(enabled=False)
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(14, 0)   # outside sell window
        assert gate.can_sell() is True
