import time
import requests
import logging

logger = logging.getLogger(__name__)

# Hyperliquid — decentralized perp DEX, no geo-restrictions, free public API
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
CACHE_TTL = 600  # 10 minutes — funding rates update every hour on Hyperliquid


class FundingRateCollector:
    """
    Hyperliquid perpetual futures funding rates as a crypto market signal.

    Fetches all asset contexts in one request (efficient), then looks up each symbol.
    Hyperliquid funding is per-hour; typical values ~0.000010–0.000125/hr.

    Positive funding rate → longs paying shorts → bearish → negative signal.
    Negative funding rate → shorts paying longs → bullish → positive signal.

    Returns a score in [-1, +1], averaged across the requested symbols.
    Results are cached for 10 minutes.
    """

    def __init__(self):
        self._cache: dict[str, tuple[float, float]] = {}  # cache_key → (score, timestamp)
        # Bulk snapshot: symbol → hourly_funding_rate, refreshed with cache
        self._rates: dict[str, float] = {}
        self._rates_ts: float = 0.0

    def score(self, symbols: list[str] | None = None) -> float | None:
        """Return a funding-rate signal in [-1, +1], or None on failure."""
        if not symbols:
            return None

        tickers = [s.split("/")[0].upper() for s in symbols]
        cache_key = ",".join(sorted(tickers))

        cached_score, cached_ts = self._cache.get(cache_key, (None, 0))
        if cached_score is not None and (time.time() - cached_ts) < CACHE_TTL:
            logger.debug(f"FundingRate: returning cached score={cached_score:.3f}")
            return cached_score

        # Refresh bulk rates if stale
        if (time.time() - self._rates_ts) >= CACHE_TTL:
            self._refresh_rates()

        if not self._rates:
            logger.warning("FundingRate: no rate data available")
            return cached_score

        scores = []
        for ticker in tickers:
            rate = self._rates.get(ticker)
            if rate is None:
                continue
            # Hyperliquid funding is per-hour. Scale: ±0.000125/hr (typical high) → ±1.0
            # i.e. multiply by 8000. Positive rate → bearish → negate.
            signal = max(-1.0, min(1.0, -rate * 8000.0))
            logger.info(f"FundingRate {ticker}: rate={rate:.8f}/hr → signal={signal:.3f}")
            scores.append(signal)

        if not scores:
            logger.debug(f"FundingRate: no data for {tickers}")
            return cached_score

        result = max(-1.0, min(1.0, sum(scores) / len(scores)))
        self._cache[cache_key] = (result, time.time())
        logger.info(f"FundingRate signal for {tickers}: {result:.3f} ({len(scores)} symbols)")
        return result

    def _refresh_rates(self) -> None:
        """Fetch all Hyperliquid asset funding rates in one API call."""
        try:
            resp = requests.post(
                HYPERLIQUID_INFO_URL,
                json={"type": "metaAndAssetCtxs"},
                timeout=10,
                headers={"User-Agent": "trader-bot/1.0", "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            # Response: [meta, assetCtxs]
            # meta["universe"][i]["name"] → assetCtxs[i]["funding"]
            universe = data[0].get("universe", [])
            asset_ctxs = data[1] if len(data) > 1 else []
            self._rates = {}
            for i, asset in enumerate(universe):
                if i >= len(asset_ctxs):
                    break
                name = asset.get("name", "").upper()
                funding = asset_ctxs[i].get("funding")
                if name and funding is not None:
                    self._rates[name] = float(funding)
            self._rates_ts = time.time()
            logger.info(f"FundingRate: loaded {len(self._rates)} Hyperliquid symbols")
        except Exception as e:
            logger.warning(f"FundingRate Hyperliquid fetch failed: {e}")
