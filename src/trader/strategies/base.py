from abc import ABC, abstractmethod
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig


class Strategy(ABC):
    """Implement this to add a new trading strategy."""

    def __init__(self, risk: RiskConfig):
        self.risk = risk

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
