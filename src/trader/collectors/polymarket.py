import json
import time
import requests
import logging

logger = logging.getLogger(__name__)

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CACHE_TTL = 900  # 15 minutes — prediction prices change slowly

# tag_id=21 is Polymarket's crypto category — use for crypto tickers
CRYPTO_TAG_ID = 21

# Keywords that indicate a bullish-framed question ("Will X go UP / reach Y?")
_BULLISH_KEYWORDS = {"rise", "up", "above", "exceed", "reach", "high", "bull", "gain", "rally"}
# Keywords that indicate a bearish-framed question ("Will X drop / fall below Y?")
_BEARISH_KEYWORDS = {"fall", "drop", "below", "crash", "bear", "down", "lose", "low", "decline"}

# Known crypto tickers — use tag_id=21 for these, keyword search for everything else
_CRYPTO_TICKERS = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "LINK", "LTC", "AVAX",
                   "SHIB", "DOGE", "NEAR", "HBAR", "UNI", "POL", "MATIC"}

# Polymarket questions often use full names, not tickers — match either
_TICKER_ALIASES: dict[str, list[str]] = {
    "BTC": ["btc", "bitcoin"],
    "ETH": ["eth", "ethereum", "ether"],
    "SOL": ["sol", "solana"],
    "XRP": ["xrp", "ripple"],
    "ADA": ["ada", "cardano"],
    "DOGE": ["doge", "dogecoin"],
    "AVAX": ["avax", "avalanche"],
    "LINK": ["link", "chainlink"],
    "DOT": ["dot", "polkadot"],
    "LTC": ["ltc", "litecoin"],
    "SHIB": ["shib", "shiba"],
    "NEAR": ["near"],
    "HBAR": ["hbar", "hedera"],
    "UNI": ["uni", "uniswap"],
    "POL": ["pol", "polygon", "matic"],
}

# Minimum liquidity (USD) — lowered to catch more markets
MIN_LIQUIDITY = 1_000


class PolymarketCollector:
    """
    Uses Polymarket prediction market prices as a sentiment signal.

    For each relevant market found:
      - Bullish-framed question (e.g. "Will BTC rise?"): YES price → positive signal
      - Bearish-framed question (e.g. "Will BTC crash?"): YES price → negative signal

    Returns a score in [-1, +1]. Cached for 15 minutes.
    """

    def __init__(self):
        self._cache: dict[str, tuple[float, float]] = {}  # key → (score, timestamp)

    def score(self, symbols: list[str] | None = None) -> float | None:
        tickers = [s.split("/")[0].upper() for s in (symbols or [])]
        if not tickers:
            return None

        cache_key = ",".join(sorted(tickers))
        cached_score, cached_ts = self._cache.get(cache_key, (None, 0))
        if cached_score is not None and (time.time() - cached_ts) < CACHE_TTL:
            return cached_score

        all_scores = []
        for ticker in tickers:
            s = self._score_ticker(ticker)
            if s is not None:
                all_scores.append(s)

        if not all_scores:
            logger.debug("Polymarket: no usable markets found for %s", tickers)
            return cached_score  # return stale cache on failure

        result = max(-1.0, min(1.0, sum(all_scores) / len(all_scores)))
        self._cache[cache_key] = (result, time.time())
        logger.info(f"Polymarket signal for {tickers}: {result:.3f} ({len(all_scores)} markets)")
        return result

    def _score_ticker(self, ticker: str) -> float | None:
        try:
            if ticker in _CRYPTO_TICKERS:
                # Use crypto tag for known crypto tickers — more reliable than keyword search
                params = {
                    "tag_id": CRYPTO_TAG_ID,
                    "active": "true",
                    "closed": "false",
                    "order": "volume24hr",
                    "ascending": "false",
                    "limit": 20,
                }
            else:
                # Stocks and unknowns — search all active markets, filter by question text
                params = {
                    "active": "true",
                    "closed": "false",
                    "order": "volume24hr",
                    "ascending": "false",
                    "limit": 50,
                }

            resp = requests.get(GAMMA_MARKETS_URL, params=params, timeout=10,
                                headers={"User-Agent": "trader-bot/1.0"})
            resp.raise_for_status()
            data = resp.json()
            markets = data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Polymarket fetch failed for {ticker}: {e}")
            return None

        scores = []
        for m in markets:
            s = self._market_signal(m, ticker)
            if s is not None:
                scores.append(s)

        return sum(scores) / len(scores) if scores else None

    def _market_signal(self, market: dict, ticker: str) -> float | None:
        try:
            liquidity = float(market.get("liquidityNum") or market.get("liquidity") or 0)
        except (TypeError, ValueError):
            liquidity = 0
        if liquidity < MIN_LIQUIDITY:
            return None

        outcome_prices = market.get("outcomePrices")
        if not outcome_prices:
            return None
        try:
            # Gamma API returns outcomePrices as a JSON string, e.g. '["0.65","0.35"]'
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            yes_prob = float(outcome_prices[0])
        except (IndexError, TypeError, ValueError):
            return None

        question = (market.get("question") or "").lower()

        # Match ticker OR any known alias (e.g. "bitcoin" matches BTC)
        aliases = _TICKER_ALIASES.get(ticker.upper(), [ticker.lower()])
        if not any(alias in question for alias in aliases):
            return None

        bullish = any(kw in question for kw in _BULLISH_KEYWORDS)
        bearish = any(kw in question for kw in _BEARISH_KEYWORDS)

        if bullish and not bearish:
            signal = (yes_prob - 0.5) * 2.0
        elif bearish and not bullish:
            signal = (0.5 - yes_prob) * 2.0
        else:
            return None  # ambiguous framing

        return max(-1.0, min(1.0, signal))
