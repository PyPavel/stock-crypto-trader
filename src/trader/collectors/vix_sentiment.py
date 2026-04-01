import math
import time
import requests
import logging

logger = logging.getLogger(__name__)

_VIX_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=1d"
_CACHE_TTL = 3600  # 1-hour cache — VIX updates intraday but hourly is sufficient


class VIXSentimentCollector:
    """
    Equity market sentiment based on CBOE VIX (volatility index) via Yahoo Finance.
    Free, no API key required.

    Contrarian signal: high VIX = fear = buy opportunity; low VIX = complacency = caution.
    Score range: -1.0 (VIX very low, complacent market) to +1.0 (VIX very high, fearful market).
    Neutral at VIX ≈ 20 (historical average).
    """

    def __init__(self):
        self._cached_score: float | None = None
        self._cache_ts: float = 0.0

    def score(self) -> float | None:
        if self._cached_score is not None and (time.time() - self._cache_ts) < _CACHE_TTL:
            return self._cached_score

        try:
            resp = requests.get(
                _VIX_URL, timeout=10, headers={"User-Agent": "trader-bot/1.0"}
            )
            resp.raise_for_status()
            data = resp.json()
            vix = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
            # tanh maps VIX deviations from 20 to (-1, +1)
            # VIX 20 → 0.0 (neutral), VIX 30 → +0.76 (fearful/bullish), VIX 12 → -0.66 (complacent/bearish)
            score = math.tanh((vix - 20) / 10)
            logger.info("VIX=%.1f → sentiment=%.3f", vix, score)
            self._cached_score = score
            self._cache_ts = time.time()
            return score
        except Exception as e:
            logger.warning("VIX fetch failed: %s", e)
            return self._cached_score
