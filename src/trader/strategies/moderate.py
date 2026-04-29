from trader.strategies.base import Strategy
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig

BUY_THRESHOLD = 0.25
SELL_THRESHOLD = -0.25
TECH_WEIGHT = 0.70
SENTIMENT_WEIGHT = 0.30
PERSISTENCE_MIN = 2
SCALE_IN_PCT = 0.50
MIN_SIGNAL_FOR_SCALE = 0.50
RSI_OVERSOLD = 35  # Allow buys in bearish trend when RSI is this low (reversal play)


class ModerateStrategy(Strategy):
    """Balanced risk/reward. Weights technicals 70%, sentiment 30%.
    Hard trend filter: only opens BUY positions when price is above 50-EMA.
    Requires signal persistence for entries. Supports scaling into positions."""

    def __init__(self, risk, buy_threshold=None, sell_threshold=SELL_THRESHOLD,
                 tech_weight=TECH_WEIGHT, sentiment_weight=SENTIMENT_WEIGHT,
                 persistence_min=None):
        super().__init__(risk)
        self.buy_threshold = buy_threshold if buy_threshold is not None else getattr(risk, "buy_score_threshold", BUY_THRESHOLD)
        self.sell_threshold = sell_threshold
        self.tech_weight = tech_weight
        self.sentiment_weight = sentiment_weight
        self._persistence_min = persistence_min if persistence_min is not None else getattr(risk, "persistence_cycles", PERSISTENCE_MIN)

    def decide(self, symbol, technical, sentiment, capital, position):
        combined = technical.score * self.tech_weight + sentiment.score * self.sentiment_weight
        self._record_signal(symbol, combined)

        # --- Buy logic ---
        if combined >= self.buy_threshold:
            # Trend filter: block buys in bearish trend UNLESS RSI signals oversold reversal
            rsi = getattr(technical, "rsi", None)
            oversold_reversal = rsi is not None and rsi <= RSI_OVERSOLD
            if not technical.trend_bullish and not oversold_reversal:
                return {
                    "action": "hold",
                    "usd_amount": 0.0,
                    "reason": f"trend filter blocked buy (bearish trend, RSI={rsi:.0f}), score={combined:.2f}",
                }

            # New position: require signal persistence
            if position == 0.0:
                if not self._signal_persistent(symbol, self.buy_threshold, "above", self._persistence_min):
                    return {
                        "action": "hold",
                        "usd_amount": 0.0,
                        "reason": f"waiting for signal persistence, score={combined:.2f}",
                    }
                usd_amount = min(capital * self.risk.max_position_pct, capital)
                return {
                    "action": "buy",
                    "usd_amount": usd_amount,
                    "reason": f"combined score {combined:.2f} (persistent)",
                }

            # Scale into existing position: higher threshold, smaller size
            elif position > 0 and combined >= MIN_SIGNAL_FOR_SCALE:
                if not self._signal_persistent(symbol, MIN_SIGNAL_FOR_SCALE, "above", self._persistence_min):
                    return {
                        "action": "hold",
                        "usd_amount": 0.0,
                        "reason": f"scale-in waiting for persistence, score={combined:.2f}",
                    }
                usd_amount = min(capital * self.risk.max_position_pct * SCALE_IN_PCT, capital)
                return {
                    "action": "buy",
                    "usd_amount": usd_amount,
                    "reason": f"scale-in score {combined:.2f} (persistent)",
                }

        # --- Sell logic: require persistence for sell signals too ---
        if combined <= self.sell_threshold and position > 0.0:
            if not self._signal_persistent(symbol, self.sell_threshold, "below", self._persistence_min):
                return {
                    "action": "hold",
                    "usd_amount": 0.0,
                    "reason": f"sell signal not persistent, score={combined:.2f}",
                }
            return {
                "action": "sell",
                "usd_amount": position,
                "reason": f"combined score {combined:.2f} (persistent)",
            }

        return {"action": "hold", "usd_amount": 0.0, "reason": f"combined score {combined:.2f}"}
