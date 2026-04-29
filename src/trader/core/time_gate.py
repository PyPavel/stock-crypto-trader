import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from trader.config import TimeGateConfig

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


def _parse_hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


class TimeWindowGate:
    """
    Gates buy and sell decisions to specific time windows (US/Eastern).
    When disabled, both can_buy() and can_sell() always return True.
    """

    def __init__(self, config: TimeGateConfig) -> None:
        self._enabled = config.enabled
        self._buy_start = _parse_hhmm(config.buy_start)
        self._buy_end = _parse_hhmm(config.buy_end)
        self._sell_start = _parse_hhmm(config.sell_start)
        self._sell_end = _parse_hhmm(config.sell_end)

    def can_buy(self) -> bool:
        if not self._enabled:
            return True
        return self._in_window(self._buy_start, self._buy_end)

    def can_sell(self) -> bool:
        if not self._enabled:
            return True
        return self._in_window(self._sell_start, self._sell_end)

    def _in_window(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        now = datetime.now(tz=_ET)
        current = (now.hour, now.minute)
        return start <= current <= end
