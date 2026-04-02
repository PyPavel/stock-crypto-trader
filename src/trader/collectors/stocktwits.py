"""
StockTwitsCollector — free public social sentiment from StockTwits.

No API key or auth required for public symbol streams.
Endpoint: https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json

Returns recent message texts for LLM sentiment scoring.
Also provides a pre-computed numeric signal from bull/bear sentiment ratios.
"""

import logging
import time
import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
_CACHE_TTL = 300   # 5 minutes — matches cycle interval
_MAX_MESSAGES = 30


# Map our symbols to StockTwits format (stocks: ticker, crypto: BTC.X format)
def _to_stocktwits_symbol(symbol: str) -> str:
    """Convert exchange symbol to StockTwits format."""
    base = symbol.split("/")[0].upper()
    # Crypto symbols use .X suffix on StockTwits
    _CRYPTO = {
        "BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "LINK", "LTC", "AVAX",
        "SHIB", "DOGE", "NEAR", "HBAR", "UNI", "POL",
    }
    if base in _CRYPTO:
        return f"{base}.X"
    return base


class StockTwitsCollector:
    """
    Fetches public message streams from StockTwits for sentiment analysis.

    Returns message texts (for LLM scoring) via fetch() and a pre-computed
    bull/bear ratio signal via score() for the numeric collector pipeline.
    """

    def __init__(self, cache_seconds: int = _CACHE_TTL):
        self._cache_seconds = cache_seconds
        # {stocktwits_symbol: (fetched_at, messages, numeric_score)}
        self._cache: dict[str, tuple[float, list[str], float]] = {}

    def fetch(self, symbols: list[str], limit: int = _MAX_MESSAGES) -> list[str]:
        """Return message texts for the given symbols (for LLM sentiment pipeline)."""
        results: list[str] = []
        for symbol in symbols:
            st_sym = _to_stocktwits_symbol(symbol)
            messages, _ = self._fetch_symbol(st_sym, limit)
            results.extend(messages)
        return results

    def score(self, symbols: list[str]) -> float | None:
        """
        Return a numeric sentiment signal [-1, +1] based on bull/bear ratio.

        bull_count / total → mapped to [-1, +1] centered at 0.5.
        Returns None if no data available.
        """
        if not symbols:
            return None

        scores = []
        for symbol in symbols:
            st_sym = _to_stocktwits_symbol(symbol)
            _, numeric = self._fetch_symbol(st_sym, _MAX_MESSAGES)
            if numeric is not None:
                scores.append(numeric)

        if not scores:
            return None
        return sum(scores) / len(scores)

    def _fetch_symbol(self, st_symbol: str, limit: int) -> tuple[list[str], float | None]:
        """Fetch messages for one StockTwits symbol, using cache."""
        now = time.time()
        cached = self._cache.get(st_symbol)
        if cached and (now - cached[0]) < self._cache_seconds:
            return cached[1], cached[2]

        try:
            url = _BASE_URL.format(symbol=st_symbol)
            resp = requests.get(
                url,
                params={"limit": min(limit, 30)},  # StockTwits API max per call
                headers={"User-Agent": "trader-bot/1.0"},
                timeout=10,
            )

            if resp.status_code == 429:
                logger.warning("StockTwits rate limited for %s — using stale cache", st_symbol)
                return (cached[1], cached[2]) if cached else ([], None)

            if resp.status_code == 404:
                logger.debug("StockTwits: symbol %s not found", st_symbol)
                return [], None

            resp.raise_for_status()
            data = resp.json()

            messages: list[str] = []
            bull_count = 0
            bear_count = 0

            for msg in data.get("messages", []):
                body = msg.get("body", "").strip()
                if body and len(body) > 10:
                    messages.append(body)
                # StockTwits messages include user sentiment tags
                entities = msg.get("entities", {})
                sentiment = entities.get("sentiment", {})
                if sentiment:
                    basic = sentiment.get("basic", "")
                    if basic == "Bullish":
                        bull_count += 1
                    elif basic == "Bearish":
                        bear_count += 1

            # Compute bull/bear ratio signal
            total_tagged = bull_count + bear_count
            numeric: float | None = None
            if total_tagged >= 3:
                # Map bull_ratio [0,1] → signal [-1,+1] centered at 0.5
                bull_ratio = bull_count / total_tagged
                numeric = (bull_ratio - 0.5) * 2.0
                logger.info(
                    "StockTwits %s: %d msgs, bull=%d bear=%d → signal=%.3f",
                    st_symbol, len(messages), bull_count, bear_count, numeric,
                )
            else:
                logger.info("StockTwits %s: %d msgs (insufficient sentiment tags)", st_symbol, len(messages))

            self._cache[st_symbol] = (now, messages, numeric)
            return messages, numeric

        except Exception as e:
            logger.warning("StockTwits fetch failed for %s: %s", st_symbol, e)
            return (cached[1], cached[2]) if cached else ([], None)
