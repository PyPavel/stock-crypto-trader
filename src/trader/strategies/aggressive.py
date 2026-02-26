from trader.strategies.base import Strategy

BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15
TECH_WEIGHT = 0.55
SENTIMENT_WEIGHT = 0.45
MAX_POSITION_PCT = 0.40


class AggressiveStrategy(Strategy):
    """Low entry threshold, large positions, sentiment has high weight."""

    def decide(self, symbol, technical, sentiment, capital, position):
        combined = technical.score * TECH_WEIGHT + sentiment.score * SENTIMENT_WEIGHT

        if combined >= BUY_THRESHOLD and position == 0.0:
            return {"action": "buy", "usd_amount": capital * MAX_POSITION_PCT,
                    "reason": f"aggressive buy at {combined:.2f}"}

        if combined <= SELL_THRESHOLD and position > 0.0:
            return {"action": "sell", "usd_amount": position,
                    "reason": f"aggressive sell at {combined:.2f}"}

        return {"action": "hold", "usd_amount": 0.0, "reason": f"score {combined:.2f}"}
