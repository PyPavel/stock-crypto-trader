import requests
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# Free crypto RSS feeds — no API key required
DEFAULT_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://beincrypto.com/feed/",
    "https://ambcrypto.com/feed/",
    "https://u.today/rss",
]

COIN_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "ADA": ["cardano", "ada"],
}


class RSSCollector:
    """Fetches headlines from crypto RSS feeds. No API key required."""

    def __init__(self, feeds: list[str] | None = None):
        self._feeds = feeds or DEFAULT_FEEDS

    def fetch(self, symbols: list[str], limit: int = 10) -> list[str]:
        """Return headlines relevant to the given symbols."""
        currencies = {s.split("/")[0].upper() for s in symbols}
        keywords = set()
        for c in currencies:
            keywords.update(COIN_KEYWORDS.get(c, [c.lower()]))
        # Always include generic crypto terms
        keywords.update(["crypto", "cryptocurrency", "blockchain"])

        headlines = []
        for feed_url in self._feeds:
            try:
                resp = requests.get(feed_url, timeout=10,
                                    headers={"User-Agent": "trader-bot/1.0"})
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
                # Handle both RSS and Atom formats
                items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
                for item in items[:limit]:
                    title_el = item.find("title")
                    if title_el is None:
                        title_el = item.find("{http://www.w3.org/2005/Atom}title")
                    if title_el is not None and title_el.text:
                        title = title_el.text.strip()
                        if any(kw in title.lower() for kw in keywords):
                            headlines.append(title)
            except Exception as e:
                logger.warning(f"RSS fetch failed for {feed_url}: {e}")

        return headlines
