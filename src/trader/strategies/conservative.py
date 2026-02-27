from trader.strategies.base import Strategy

BUY_THRESHOLD = 0.65
SELL_THRESHOLD = -0.50
TECH_WEIGHT = 0.80
SENTIMENT_WEIGHT = 0.20
MAX_POSITION_PCT = 0.10


class ConservativeStrategy(Strategy):
    """Tight entry threshold, small positions, sentiment has minimal weight."""

    def decide(self, symbol, technical, sentiment, capital, position):
        combined = technical.score * TECH_WEIGHT + sentiment.score * SENTIMENT_WEIGHT

        if combined >= BUY_THRESHOLD and position == 0.0:
            return {"action": "buy", "usd_amount": min(capital * self.risk.max_position_pct, capital * 0.10),
                    "reason": f"conservative buy at {combined:.2f}"}

        if combined <= SELL_THRESHOLD and position > 0.0:
            return {"action": "sell", "usd_amount": position,
                    "reason": f"conservative sell at {combined:.2f}"}

        return {"action": "hold", "usd_amount": 0.0, "reason": f"score {combined:.2f}"}
