from abc import ABC, abstractmethod
from collections import deque
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig


class Strategy(ABC):
    """Implement this to add a new trading strategy."""

    def __init__(self, risk: RiskConfig):
        self.risk = risk
        # Signal momentum: keep last N combined scores per symbol
        self._signal_history: dict[str, deque] = {}
        self._history_maxlen = 5

    def _record_signal(self, symbol: str, combined_score: float) -> None:
        """Append combined score to rolling history for this symbol."""
        if symbol not in self._signal_history:
            self._signal_history[symbol] = deque(maxlen=self._history_maxlen)
        self._signal_history[symbol].append(combined_score)

    def _signal_persistent(self, symbol: str, threshold: float, direction: str, min_consecutive: int = 2) -> bool:
        """
        Returns True if the last min_consecutive signals were all above (direction='above')
        or below (direction='below') the threshold. Requires at least min_consecutive entries.
        """
        history = self._signal_history.get(symbol)
        if history is None or len(history) < min_consecutive:
            return False
        recent = list(history)[-min_consecutive:]
        if direction == "above":
            return all(s >= threshold for s in recent)
        elif direction == "below":
            return all(s <= threshold for s in recent)
        return False

    @abstractmethod
    def decide(
        self,
        symbol: str,
        technical: Signal,
        sentiment: SentimentScore,
        capital: float,
        position: float,   # current USD value held in this symbol
    ) -> dict:
        """
        Returns dict with keys:
          action: 'buy' | 'sell' | 'hold'
          usd_amount: float
          reason: str
        """
