import requests
import logging

logger = logging.getLogger(__name__)

COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

COIN_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
}


class CoinGeckoCollector:
    """
    Uses CoinGecko free API (no key required) to derive a market signal.
    Signal based on: 24h market cap change % and BTC dominance trend.
    Returns score in [-1, +1].
    """

    def score(self, symbols: list[str]) -> float | None:
        try:
            resp = requests.get(COINGECKO_GLOBAL_URL, timeout=10,
                                headers={"User-Agent": "trader-bot/1.0"})
            resp.raise_for_status()
            data = resp.json().get("data", {})

            # 24h market cap change: positive = bullish, negative = bearish
            mcap_change = float(data.get("market_cap_change_percentage_24h_usd", 0))

            # Normalize: ±10% change maps to ±1.0 signal
            score = max(-1.0, min(1.0, mcap_change / 10.0))
            logger.info(f"CoinGecko global 24h mcap change: {mcap_change:.2f}% → signal={score:.3f}")
            return score
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed: {e}")
            return None
