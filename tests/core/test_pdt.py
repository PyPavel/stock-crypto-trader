from datetime import date
from unittest.mock import MagicMock

import pytest

from trader.core.pdt import PDTGuard


def make_adapter(daytrade_count: int = 0) -> MagicMock:
    adapter = MagicMock()
    adapter.get_day_trade_count.return_value = daytrade_count
    return adapter


def make_guard(daytrade_count: int = 0) -> PDTGuard:
    guard = PDTGuard(make_adapter(daytrade_count))
    guard.refresh()
    return guard


# ------------------------------------------------------------------ #
#  remaining() / buy_threshold()
# ------------------------------------------------------------------ #

def test_remaining_full_budget():
    guard = make_guard(daytrade_count=0)
    assert guard.remaining() == 3


def test_remaining_partial_budget():
    guard = make_guard(daytrade_count=1)
    assert guard.remaining() == 2


def test_remaining_at_limit():
    guard = make_guard(daytrade_count=3)
    assert guard.remaining() == 0


def test_remaining_over_limit_clamps_to_zero():
    guard = make_guard(daytrade_count=5)
    assert guard.remaining() == 0


def test_buy_threshold_full_budget():
    guard = make_guard(daytrade_count=0)
    assert guard.buy_threshold() == pytest.approx(0.25)


def test_buy_threshold_two_remaining():
    guard = make_guard(daytrade_count=1)
    assert guard.buy_threshold() == pytest.approx(0.35)


def test_buy_threshold_one_remaining():
    guard = make_guard(daytrade_count=2)
    assert guard.buy_threshold() == pytest.approx(0.45)


def test_buy_threshold_none_when_exhausted():
    guard = make_guard(daytrade_count=3)
    assert guard.buy_threshold() is None


# ------------------------------------------------------------------ #
#  record_buy / is_same_day_buy
# ------------------------------------------------------------------ #

def test_symbol_not_same_day_before_buy():
    guard = make_guard()
    assert guard.is_same_day_buy("QCOM") is False


def test_symbol_is_same_day_after_buy():
    guard = make_guard()
    guard.record_buy("QCOM")
    assert guard.is_same_day_buy("QCOM") is True


def test_other_symbol_not_same_day():
    guard = make_guard()
    guard.record_buy("QCOM")
    assert guard.is_same_day_buy("AAPL") is False


# ------------------------------------------------------------------ #
#  daily reset
# ------------------------------------------------------------------ #

def test_today_buys_reset_on_new_day():
    guard = make_guard()
    guard.record_buy("QCOM")
    assert guard.is_same_day_buy("QCOM") is True

    # Simulate a new day by backdating the stored date
    from datetime import timedelta
    guard._today_date = date.today() - timedelta(days=1)
    guard.refresh()

    assert guard.is_same_day_buy("QCOM") is False


# ------------------------------------------------------------------ #
#  can_exit_today
# ------------------------------------------------------------------ #

def test_overnight_hold_can_always_exit():
    guard = make_guard(daytrade_count=3)   # budget exhausted
    # symbol NOT bought today → not a day-trade
    assert guard.can_exit_today("QCOM") is True


def test_same_day_buy_blocked_when_no_budget():
    guard = make_guard(daytrade_count=3)
    guard.record_buy("QCOM")
    assert guard.can_exit_today("QCOM") is False


def test_same_day_buy_allowed_when_budget_remains():
    guard = make_guard(daytrade_count=1)   # 2 remaining
    guard.record_buy("QCOM")
    assert guard.can_exit_today("QCOM") is True
