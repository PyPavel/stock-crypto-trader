"""
DiscordCollector — reads recent messages from configured Discord channels.

Requirements:
  1. Create a Discord application + bot at https://discord.com/developers/applications
  2. Copy the bot token → set DISCORD_BOT_TOKEN env var (or discord.bot_token in config)
  3. Enable "Message Content Intent" in the bot settings (Bot → Privileged Gateway Intents)
  4. Invite the bot to each target server with "Read Messages" + "Read Message History" permissions
  5. Copy channel IDs (Discord → Settings → Advanced → Developer Mode on, then right-click channel)

Config example (config.yaml):
  discord:
    bot_token: ''
    crypto_channels:       # IDs of channels used when exchange=coinbase
      - '123456789012345678'
    stock_channels:        # IDs of channels used when exchange=alpaca
      - '987654321098765432'
    limit: 50              # messages to fetch per channel per cycle
    cache_seconds: 300     # don't re-fetch same channel more often than this
"""

import logging
import time
import requests

logger = logging.getLogger(__name__)

_DISCORD_API = "https://discord.com/api/v10"

# Keyword map for filtering relevant messages (crypto)
CRYPTO_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "ADA": ["cardano", "ada"],
    "XRP": ["xrp", "ripple"],
    "DOT": ["polkadot", "dot"],
    "LINK": ["chainlink", "link"],
    "LTC": ["litecoin", "ltc"],
    "AVAX": ["avalanche", "avax"],
    "SHIB": ["shiba", "shib"],
    "DOGE": ["dogecoin", "doge"],
    "NEAR": ["near protocol", "near"],
    "HBAR": ["hedera", "hbar"],
    "UNI": ["uniswap", "uni"],
    "POL": ["polygon", "matic", "pol"],
}

# For stocks we match ticker symbols and company names
STOCK_KEYWORDS = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "GOOGL": ["google", "googl", "alphabet"],
    "AMZN": ["amazon", "amzn"],
    "NVDA": ["nvidia", "nvda"],
    "META": ["meta", "facebook"],
    "TSLA": ["tesla", "tsla"],
    "JPM": ["jpmorgan", "jpm"],
    "V": [" visa ", "$v "],
    "MA": ["mastercard", " ma "],
}

# Generic market terms always included
_CRYPTO_GENERIC = ["crypto", "bitcoin", "altcoin", "defi", "bull", "bear", "pump", "dump"]
_STOCK_GENERIC = ["stock", "market", "nasdaq", "s&p", "earnings", "fed", "rally", "selloff"]


class DiscordCollector:
    """Fetches messages from Discord channels via the bot REST API."""

    def __init__(
        self,
        bot_token: str,
        channel_ids: list[str],
        asset_class: str = "crypto",   # "crypto" or "stock"
        limit: int = 50,
        cache_seconds: int = 300,
    ):
        self._token = bot_token
        self._channel_ids = [str(c) for c in channel_ids if c]
        self._asset_class = asset_class
        self._limit = limit
        self._cache_seconds = cache_seconds
        self._disabled = False  # circuit-breaker on auth failure

        # Per-channel cache: {channel_id: (fetched_at, [messages])}
        self._cache: dict[str, tuple[float, list[str]]] = {}

        self._keyword_map = STOCK_KEYWORDS if asset_class == "stock" else CRYPTO_KEYWORDS
        self._generic = _STOCK_GENERIC if asset_class == "stock" else _CRYPTO_GENERIC

    def fetch(self, symbols: list[str], limit: int | None = None) -> list[str]:
        """Return message texts relevant to the given symbols."""
        if self._disabled or not self._token or not self._channel_ids:
            return []

        # Build keyword set for these symbols
        currencies = {s.split("/")[0].upper() for s in symbols}
        keywords: set[str] = set(self._generic)
        for c in currencies:
            keywords.update(self._keyword_map.get(c, [c.lower()]))

        results: list[str] = []
        for channel_id in self._channel_ids:
            messages = self._fetch_channel(channel_id, limit or self._limit)
            for msg in messages:
                text = msg.lower()
                if any(kw in text for kw in keywords):
                    results.append(msg)

        return results

    def _fetch_channel(self, channel_id: str, limit: int) -> list[str]:
        """Fetch messages from a single channel, using cache if fresh."""
        now = time.time()
        cached_at, cached_msgs = self._cache.get(channel_id, (0.0, []))
        if now - cached_at < self._cache_seconds:
            return cached_msgs

        try:
            resp = requests.get(
                f"{_DISCORD_API}/channels/{channel_id}/messages",
                params={"limit": min(limit, 100)},  # Discord max is 100
                headers={
                    "Authorization": f"Bot {self._token}",
                    "User-Agent": "trader-bot/1.0",
                },
                timeout=10,
            )

            if resp.status_code == 401:
                logger.warning("Discord bot token invalid (401) — disabling collector")
                self._disabled = True
                return []

            if resp.status_code == 403:
                logger.warning(
                    "Discord: no access to channel %s (403) — bot not in server or missing permissions",
                    channel_id,
                )
                return []

            if resp.status_code == 404:
                logger.warning("Discord: channel %s not found (404)", channel_id)
                return []

            resp.raise_for_status()
            data = resp.json()

            messages = []
            for item in data:
                content = item.get("content", "").strip()
                # Skip bot messages, empty content, and very short messages (reactions etc.)
                if content and len(content) > 10 and not item.get("author", {}).get("bot", False):
                    messages.append(content)

            self._cache[channel_id] = (now, messages)
            logger.info("Discord channel %s: fetched %d messages", channel_id, len(messages))
            return messages

        except Exception as e:
            logger.warning("Discord fetch failed for channel %s: %s", channel_id, e)
            return cached_msgs  # return stale cache on error
