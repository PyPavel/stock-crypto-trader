import time
import requests
import logging

logger = logging.getLogger(__name__)

COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
CACHE_TTL = 900  # seconds — 15 min; global market data changes slowly, reduces 429s


class CoinGeckoCollector:
    """
    Uses CoinGecko free API (no key required) to derive two market signals:
      1. 24h market cap change % → overall crypto momentum signal
      2. BTC dominance → altcoin rotation signal (rising dominance = bearish for alts)

    Returns composite score in [-1, +1]. Cached 15 minutes to avoid rate limits.
    """

    def __init__(self):
        self._cached_score: float | None = None
        self._cached_btc_dominance: float | None = None
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
            mcap_signal = max(-1.0, min(1.0, mcap_change / 10.0))

            # BTC dominance signal for altcoins:
            # Rising dominance → capital rotating INTO BTC, OUT of alts → bearish for alts
            # We track this but only apply it when the symbol is NOT BTC
            btc_dominance = float(data.get("market_cap_percentage", {}).get("btc", 50.0))
            self._cached_btc_dominance = btc_dominance

            # Dominance signal: above 55% is alt-bearish, below 45% is alt-bullish
            # For BTC itself, high dominance is neutral/slightly positive
            dominance_signal = max(-1.0, min(1.0, (50.0 - btc_dominance) / 15.0))

            # Check if request is for non-BTC symbols
            is_btc_only = all(s.upper().startswith("BTC") for s in (symbols or []))
            if is_btc_only:
                score = mcap_signal
            else:
                # Blend: 70% mcap momentum, 30% dominance (alts suffer when BTC dominates)
                score = mcap_signal * 0.70 + dominance_signal * 0.30

            logger.info(
                "CoinGecko: mcap_chg=%.2f%% btc_dom=%.1f%% → mcap_sig=%.3f dom_sig=%.3f combined=%.3f",
                mcap_change, btc_dominance, mcap_signal, dominance_signal, score,
            )

            self._cached_score = score
            self._cache_ts = time.time()
            return score
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed: {e}")
            return self._cached_score  # return stale cache on failure rather than None
