from trader.strategies.base import Strategy
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig

BUY_THRESHOLD = 0.35
SELL_THRESHOLD = -0.35
TECH_WEIGHT = 0.70
SENTIMENT_WEIGHT = 0.30


class ModerateStrategy(Strategy):
    """Balanced risk/reward. Weights technicals 70%, sentiment 30%."""

    def decide(self, symbol, technical, sentiment, capital, position):
        combined = technical.score * TECH_WEIGHT + sentiment.score * SENTIMENT_WEIGHT

        if combined >= BUY_THRESHOLD and position == 0.0:
            usd_amount = min(capital * self.risk.max_position_pct, capital)
            return {
                "action": "buy",
                "usd_amount": usd_amount,
                "reason": f"combined score {combined:.2f}",
            }

        if combined <= SELL_THRESHOLD and position > 0.0:
            return {
                "action": "sell",
                "usd_amount": position,
                "reason": f"combined score {combined:.2f}",
            }

        return {"action": "hold", "usd_amount": 0.0, "reason": f"combined score {combined:.2f}"}
