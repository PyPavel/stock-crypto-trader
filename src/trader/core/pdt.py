import logging
from datetime import date

logger = logging.getLogger(__name__)

# PDT allows 3 day-trades per 5-rolling-trading-day window for accounts < $25k
_DT_BUDGET = 3

# Required combined score to open a new position by remaining day-trade slots.
# None means buys are blocked entirely (budget exhausted).
_BUY_THRESHOLDS: dict[int, float | None] = {
    3: 0.25,
    2: 0.35,
    1: 0.45,
    0: None,
}


class PDTGuard:
    """
    Tracks Pattern Day Trader constraints for sub-$25k Alpaca accounts.

    Responsibilities:
    - Fetches current day-trade count from broker once per cycle
    - Tracks which symbols were bought today (same-day exit = day-trade)
    - Provides buy threshold adjusted by remaining day-trade budget
    - Tells engine whether a protective exit can execute today
    """

    def __init__(self, adapter) -> None:
        self._adapter = adapter
        self._today_buys: set[str] = set()
        self._today_date: date | None = None
        self._cached_dt_count: int = 0

    def refresh(self) -> None:
        """Call once at the start of each trading cycle."""
        today = date.today()
        if self._today_date != today:
            self._today_buys.clear()
            self._today_date = today
        self._cached_dt_count = self._adapter.get_day_trade_count()
        logger.debug(
            "PDT: daytrade_count=%d remaining=%d today_buys=%s",
            self._cached_dt_count, self.remaining(), self._today_buys,
        )

    def remaining(self) -> int:
        """Day-trade slots left this rolling window."""
        return max(0, _DT_BUDGET - self._cached_dt_count)

    def buy_threshold(self) -> float | None:
        """
        Minimum combined score required to open a new position.
        Returns None when budget is exhausted (all buys blocked).
        """
        return _BUY_THRESHOLDS[min(self.remaining(), 3)]

    def record_buy(self, symbol: str) -> None:
        """Call after every successful buy order."""
        self._today_buys.add(symbol)

    def is_same_day_buy(self, symbol: str) -> bool:
        """True if this symbol was opened today (selling it today = day-trade)."""
        return symbol in self._today_buys

    def can_exit_today(self, symbol: str) -> bool:
        """
        True if a protective exit (stop-loss etc.) can execute today.
        Overnight holds are always safe. Same-day buys require remaining budget.
        """
        if not self.is_same_day_buy(symbol):
            return True
        return self.remaining() > 0
