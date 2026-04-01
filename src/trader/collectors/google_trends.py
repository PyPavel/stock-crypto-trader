import time
import logging
from typing import Mapping

logger = logging.getLogger(__name__)

CACHE_TTL = 1800  # 30 minutes — pytrends has aggressive rate limiting

# Map ticker → full name fallback for crypto
CRYPTO_NAMES: Mapping[str, str] = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "XRP": "XRP",
    "ADA": "Cardano",
    "DOGE": "Dogecoin",
    "DOT": "Polkadot",
    "AVAX": "Avalanche",
    "LINK": "Chainlink",
    "MATIC": "Polygon",
}


class GoogleTrendsCollector:
    """
    Google Trends search interest collector via pytrends.

    Signal logic: if recent search interest is rising (retail FOMO) → bullish.
    Compares last 2 days average vs prior 5 days average over a 7-day window.
    Rising  → positive signal (+1 max).
    Falling → negative signal (-1 min).

    Usable for both crypto (keyword = ticker, fallback to full name) and stocks
    (keyword = "<TICKER> stock").

    Results are cached per-symbol for 30 minutes.
    """

    def __init__(self, asset_class: str = "crypto"):
        """
        Parameters
        ----------
        asset_class : "crypto" | "stock"
            Determines how keywords are constructed.
        """
        if asset_class not in ("crypto", "stock"):
            raise ValueError("asset_class must be 'crypto' or 'stock'")
        self._asset_class = asset_class
        # cache: symbol → (score, timestamp)
        self._cache: dict[str, tuple[float, float]] = {}
        # Global rate-limit backoff: skip all requests until this timestamp
        self._rate_limited_until: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, symbols: list[str] | None = None) -> float | None:
        """Return aggregate signal in [-1, +1] for the given symbols, or None on failure.

        If multiple symbols are provided the scores are averaged.
        Returns stale cache on rate-limit / error.
        """
        if not symbols:
            logger.warning("GoogleTrendsCollector.score() called with no symbols")
            return None

        if time.time() < self._rate_limited_until:
            remaining = int(self._rate_limited_until - time.time())
            logger.debug(f"GoogleTrends rate-limit backoff active ({remaining}s remaining)")
            # Return stale cache average if available
            stale = [v for v, _ in self._cache.values()]
            return sum(stale) / len(stale) if stale else None

        scores = []
        for sym in symbols:
            s = self._score_symbol(sym)
            if s is not None:
                scores.append(s)

        if not scores:
            return None
        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_symbol(self, symbol: str) -> float | None:
        """Return signal for a single symbol, using cache when fresh."""
        upper = symbol.split("/")[0].upper()  # "NEAR/USD" → "NEAR"

        # Return cached value if still fresh
        if upper in self._cache:
            cached_score, cached_ts = self._cache[upper]
            if (time.time() - cached_ts) < CACHE_TTL:
                logger.debug(f"GoogleTrends cache hit for {upper}")
                return cached_score

        keyword = self._build_keyword(upper)
        new_score = self._fetch_trend_score(upper, keyword)

        # If fetch failed but we have a stale cache entry, return it
        if new_score is None and upper in self._cache:
            logger.warning(f"GoogleTrends fetch failed for {upper}; returning stale cache")
            return self._cache[upper][0]

        if new_score is not None:
            self._cache[upper] = (new_score, time.time())

        return new_score

    def _build_keyword(self, symbol: str) -> str:
        """Construct the search keyword for pytrends."""
        if self._asset_class == "stock":
            return f"{symbol} stock"
        # crypto: try ticker first; full-name fallback is handled inside _fetch_trend_score
        return symbol

    def _fetch_trend_score(self, symbol: str, keyword: str) -> float | None:
        """Query pytrends and compute a [-1, +1] trend score."""
        try:
            from pytrends.request import TrendReq  # lazy import — optional dependency
        except ImportError:
            logger.error("pytrends is not installed; GoogleTrendsCollector will not work")
            return None

        try:
            score = self._query_pytrends(keyword, TrendReq)
            if score is None and self._asset_class == "crypto":
                # Fall back to full crypto name
                fallback = CRYPTO_NAMES.get(symbol)
                if fallback and fallback != keyword:
                    logger.info(f"GoogleTrends: retrying {symbol} with keyword '{fallback}'")
                    score = self._query_pytrends(fallback, TrendReq)
            return score
        except Exception as e:
            logger.warning(f"GoogleTrends unexpected error for {symbol}: {e}")
            return None

    _RATE_LIMIT_BACKOFF = 3600  # 1 hour backoff on 429

    def _query_pytrends(self, keyword: str, TrendReq) -> float | None:  # noqa: N803
        """Run a single pytrends query and return a [-1, +1] score, or None."""
        try:
            pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
            pytrends.build_payload([keyword], timeframe="now 7-d")
            df = pytrends.interest_over_time()
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "too many" in msg or "rate" in msg:
                self._rate_limited_until = time.time() + self._RATE_LIMIT_BACKOFF
                logger.info(
                    f"GoogleTrends rate-limited for '{keyword}': {e} "
                    f"— backing off for {self._RATE_LIMIT_BACKOFF // 60}min"
                )
            else:
                logger.warning(f"GoogleTrends query failed for '{keyword}': {e}")
            return None

        if df is None or df.empty or keyword not in df.columns:
            logger.info(f"GoogleTrends: no data returned for '{keyword}'")
            return None

        series = df[keyword].dropna()
        if len(series) < 3:
            logger.info(f"GoogleTrends: insufficient data points for '{keyword}'")
            return None

        # Compare recent 2 data points vs the earlier ones
        recent_avg = float(series.iloc[-2:].mean())
        earlier_avg = float(series.iloc[:-2].mean())

        if earlier_avg == 0:
            # Avoid division by zero; if there was no earlier interest, treat as neutral
            logger.info(f"GoogleTrends '{keyword}': earlier avg is 0, returning neutral")
            return 0.0

        # Relative change, capped at ±100 % → mapped to [-1, +1]
        relative_change = (recent_avg - earlier_avg) / earlier_avg  # unbounded
        score = max(-1.0, min(1.0, relative_change))

        logger.info(
            f"GoogleTrends '{keyword}': earlier={earlier_avg:.1f} recent={recent_avg:.1f} "
            f"rel_change={relative_change:+.3f} → signal={score:+.3f}"
        )
        return score
