import requests
import logging

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/"


class FearGreedCollector:
    """
    Crypto Fear & Greed Index from alternative.me.
    Returns score from -1.0 (extreme fear = buy) to +1.0 (extreme greed = sell).
    No API key required.
    """

    def score(self) -> float | None:
        """Return a score in [-1, +1] or None on failure.
        Extreme fear (0-25) → positive score (buy signal).
        Extreme greed (75-100) → negative score (sell signal).
        """
        try:
            response = requests.get(FEAR_GREED_URL, params={"limit": 1}, timeout=10)
            response.raise_for_status()
            data = response.json()
            value = int(data["data"][0]["value"])
            # Map 0-100 to +1 (fear=buy) to -1 (greed=sell)
            # 50 = neutral (0.0), 0 = extreme fear (+1.0), 100 = extreme greed (-1.0)
            score = (50 - value) / 50.0
            label = data["data"][0]["value_classification"]
            logger.info(f"Fear & Greed Index: {value} ({label}) → signal={score:.3f}")
            return score
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
            return None
