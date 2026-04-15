import time
import requests
import logging

logger = logging.getLogger(__name__)

# CBOE delayed options data — free and public, no auth required
CBOE_OPTIONS_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
CACHE_TTL = 900  # 15 minutes


class UnusualWhalesCollector:
    """
    Options flow signal for stocks using CBOE delayed quotes.

    For each symbol: fetches all option contracts and computes total call volume
    vs total put volume. A high call/put ratio is bullish; high put/call is bearish.

    call_put_ratio → signal:
      ratio >= 4.0  → +1.0 (strongly bullish)
      ratio == 1.0  → 0.0  (neutral)
      ratio <= 0.25 → -1.0 (strongly bearish)

    Uses log-scale mapping: signal = tanh(ln(ratio)) clamped to [-1, +1].
    Returns score in [-1, +1]. Cached per symbol for 15 minutes.
    """

    def __init__(self):
        self._cache: dict[str, tuple[float, float]] = {}  # ticker → (score, timestamp)

    def score(self, symbols: list[str] | None = None) -> float | None:
        tickers = [s.split("/")[0].upper() for s in (symbols or [])]
        if not tickers:
            return None

        all_scores = []
        for ticker in tickers:
            s = self._score_ticker(ticker)
            if s is not None:
                all_scores.append(s)

        if not all_scores:
            logger.warning("UnusualWhales/CBOE: no usable options data found")
            return None

        result = max(-1.0, min(1.0, sum(all_scores) / len(all_scores)))
        logger.info(f"Options flow signal for {tickers}: {result:.3f} ({len(all_scores)} tickers)")
        return result

    def _score_ticker(self, ticker: str) -> float | None:
        cached_score, cached_ts = self._cache.get(ticker, (None, 0))
        if cached_score is not None and (time.time() - cached_ts) < CACHE_TTL:
            return cached_score

        try:
            url = CBOE_OPTIONS_URL.format(ticker=ticker)
            resp = requests.get(url, timeout=15, headers={"User-Agent": "trader-bot/1.0"})
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"CBOE options fetch failed for {ticker}: {e}")
            return cached_score  # return stale cache on failure

        try:
            options = data.get("data", {}).get("options", [])
            if not options:
                logger.warning(f"CBOE: no options contracts returned for {ticker}")
                return cached_score

            call_volume = 0
            put_volume = 0
            for contract in options:
                # Option code format: SYMBOL + 6-digit date (YYMMDD) + C/P + 8-digit strike
                # e.g. "AAPL260415C00175000" → type is at position [-9]
                option_code = (contract.get("option") or "").upper()
                option_type = option_code[-9] if len(option_code) >= 9 else ""
                volume = int(contract.get("volume") or 0)
                if option_type == "C":
                    call_volume += volume
                elif option_type == "P":
                    put_volume += volume

            total = call_volume + put_volume
            if total == 0:
                logger.warning(f"CBOE: zero total options volume for {ticker}")
                return cached_score

            # call_put_ratio: >1 bullish, <1 bearish
            # Use log scale: signal = tanh(ln(ratio)), clamped to [-1, +1]
            import math
            ratio = call_volume / max(put_volume, 1)
            signal = math.tanh(math.log(ratio))
            signal = max(-1.0, min(1.0, signal))

            logger.info(
                f"CBOE {ticker}: calls={call_volume} puts={put_volume} "
                f"ratio={ratio:.2f} → signal={signal:.3f}"
            )
            self._cache[ticker] = (signal, time.time())
            return signal

        except Exception as e:
            logger.warning(f"CBOE options parse error for {ticker}: {e}")
            return cached_score
