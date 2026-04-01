import time
import requests
import logging

logger = logging.getLogger(__name__)

COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
CACHE_TTL = 900  # seconds — 15 min; global market data changes slowly, reduces 429s


class CoinGeckoCollector:
    """
    Uses CoinGecko free API (no key required) to derive a market signal.
    Signal based on: 24h market cap change % and BTC dominance trend.
    Returns score in [-1, +1]. Result is cached for 15 minutes to avoid rate limits.
    """

    def __init__(self):
        self._cached_score: float | None = None
        self._cache_ts: float = 0.0

    def score(self, symbols: list[str]) -> float | None:
        if self._cached_score is not None and (time.time() - self._cache_ts) < CACHE_TTL:
            return self._cached_score

        try:
            resp = requests.get(COINGECKO_GLOBAL_URL, timeout=10,
                                headers={"User-Agent": "trader-bot/1.0"})
            resp.raise_for_status()
            data = resp.json().get("data", {})

            mcap_change = float(data.get("market_cap_change_percentage_24h_usd", 0))
            score = max(-1.0, min(1.0, mcap_change / 10.0))
            logger.info(f"CoinGecko global 24h mcap change: {mcap_change:.2f}% → signal={score:.3f}")

            self._cached_score = score
            self._cache_ts = time.time()
            return score
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed: {e}")
            return self._cached_score  # return stale cache on failure rather than None
