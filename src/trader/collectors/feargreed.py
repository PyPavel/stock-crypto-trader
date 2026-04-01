import time
import requests
import logging

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/"
CACHE_TTL = 600  # 10-minute cache


class FearGreedCollector:
    """
    Crypto Fear & Greed Index from alternative.me.
    Returns score from -1.0 (extreme greed = sell) to +1.0 (extreme fear = buy).
    No API key required. Cached for 10 minutes.
    """

    def __init__(self):
        self._cached_score: float | None = None
        self._cache_ts: float = 0.0

    def score(self) -> float | None:
        if self._cached_score is not None and (time.time() - self._cache_ts) < CACHE_TTL:
            return self._cached_score

        try:
            response = requests.get(FEAR_GREED_URL, params={"limit": 1}, timeout=10,
                                    headers={"User-Agent": "trader-bot/1.0"})
            response.raise_for_status()
            data = response.json()
            value = int(data["data"][0]["value"])
            label = data["data"][0]["value_classification"]
            # Map 0-100 to +1 (fear=buy) to -1 (greed=sell)
            score = max(-1.0, min(1.0, (50 - value) / 50.0))
            logger.info(f"Fear & Greed Index: {value} ({label}) → signal={score:.3f}")
            self._cached_score = score
            self._cache_ts = time.time()
            return score
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
            return self._cached_score
