from trader.strategies.base import Strategy

BUY_THRESHOLD = 0.65
SELL_THRESHOLD = -0.50
TECH_WEIGHT = 0.80
SENTIMENT_WEIGHT = 0.20
MAX_POSITION_PCT = 0.10
PERSISTENCE_MIN = 2
SCALE_IN_PCT = 0.50
MIN_SIGNAL_FOR_SCALE = 0.75


class ConservativeStrategy(Strategy):
    """Tight entry threshold, small positions, sentiment has minimal weight.
    Hard trend filter: only opens BUY positions when price is above 50-EMA.
    Requires signal persistence for entries. Supports scaling into positions."""

    def __init__(self, risk, buy_threshold=BUY_THRESHOLD, sell_threshold=SELL_THRESHOLD,
                 tech_weight=TECH_WEIGHT, sentiment_weight=SENTIMENT_WEIGHT):
        super().__init__(risk)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.tech_weight = tech_weight
        self.sentiment_weight = sentiment_weight

    def decide(self, symbol, technical, sentiment, capital, position):
        combined = technical.score * self.tech_weight + sentiment.score * self.sentiment_weight
        self._record_signal(symbol, combined)

        # --- Buy logic ---
        if combined >= self.buy_threshold:
            if not technical.trend_bullish:
                return {
                    "action": "hold",
                    "usd_amount": 0.0,
                    "reason": f"trend filter blocked buy (bearish trend), score={combined:.2f}",
                }

            if position == 0.0:
                if not self._signal_persistent(symbol, self.buy_threshold, "above", PERSISTENCE_MIN):
                    return {
                        "action": "hold",
                        "usd_amount": 0.0,
                        "reason": f"waiting for signal persistence, score={combined:.2f}",
                    }
                return {
                    "action": "buy",
                    "usd_amount": min(capital * self.risk.max_position_pct, capital * 0.10),
                    "reason": f"conservative buy at {combined:.2f} (persistent)",
                }

            elif position > 0 and combined >= MIN_SIGNAL_FOR_SCALE:
                if not self._signal_persistent(symbol, MIN_SIGNAL_FOR_SCALE, "above", PERSISTENCE_MIN):
                    return {
                        "action": "hold",
                        "usd_amount": 0.0,
                        "reason": f"scale-in waiting for persistence, score={combined:.2f}",
                    }
                return {
                    "action": "buy",
                    "usd_amount": min(capital * self.risk.max_position_pct * SCALE_IN_PCT, capital * 0.05),
                    "reason": f"conservative scale-in at {combined:.2f} (persistent)",
                }

        # --- Sell logic: require persistence ---
        if combined <= self.sell_threshold and position > 0.0:
            if not self._signal_persistent(symbol, self.sell_threshold, "below", PERSISTENCE_MIN):
                return {
                    "action": "hold",
                    "usd_amount": 0.0,
                    "reason": f"sell signal not persistent, score={combined:.2f}",
                }
            return {
                "action": "sell",
                "usd_amount": position,
                "reason": f"conservative sell at {combined:.2f} (persistent)",
            }

        return {"action": "hold", "usd_amount": 0.0, "reason": f"score {combined:.2f}"}
