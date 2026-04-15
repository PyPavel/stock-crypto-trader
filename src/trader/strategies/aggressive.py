from trader.strategies.base import Strategy

BUY_THRESHOLD = 0.25
SELL_THRESHOLD = -0.10
TECH_WEIGHT = 0.55
SENTIMENT_WEIGHT = 0.45
MAX_POSITION_PCT = 0.40
PERSISTENCE_MIN = 2
SCALE_IN_PCT = 0.50
MIN_SIGNAL_FOR_SCALE = 0.30


class AggressiveStrategy(Strategy):
    """Low entry threshold, large positions, sentiment has high weight.
    Soft trend filter: still buys in bearish trend but reduces position size by 50%.
    No persistence required for initial entry (aggressive). Supports scaling into positions."""

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

        # --- Buy logic: aggressive acts on first signal (no persistence for initial entry) ---
        if combined >= self.buy_threshold:
            conviction_mult = self.risk.conviction_size_multiplier if abs(combined) >= 0.50 else 1.0
            base_amount = capital * self.risk.max_position_pct * conviction_mult

            if position == 0.0:
                if not technical.trend_bullish:
                    return {
                        "action": "buy",
                        "usd_amount": base_amount * 0.50,
                        "reason": f"aggressive buy (counter-trend, reduced size) at {combined:.2f}",
                    }
                return {
                    "action": "buy",
                    "usd_amount": base_amount,
                    "reason": f"aggressive buy at {combined:.2f}",
                }

            # Scale into existing position: require persistence + higher threshold
            elif position > 0 and combined >= MIN_SIGNAL_FOR_SCALE:
                if not self._signal_persistent(symbol, MIN_SIGNAL_FOR_SCALE, "above", PERSISTENCE_MIN):
                    return {
                        "action": "hold",
                        "usd_amount": 0.0,
                        "reason": f"scale-in waiting for persistence, score={combined:.2f}",
                    }
                scale_amount = base_amount * SCALE_IN_PCT
                if not technical.trend_bullish:
                    scale_amount *= 0.50
                return {
                    "action": "buy",
                    "usd_amount": scale_amount,
                    "reason": f"aggressive scale-in at {combined:.2f} (persistent)",
                }

        # --- Sell logic: require persistence to avoid whipsaws ---
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
                "reason": f"aggressive sell at {combined:.2f} (persistent)",
            }

        return {"action": "hold", "usd_amount": 0.0, "reason": f"score {combined:.2f}"}
