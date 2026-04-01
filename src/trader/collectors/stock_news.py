import requests
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# Free stock news RSS feeds — no API key required
GENERAL_FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
]

# Yahoo Finance per-symbol RSS template
YAHOO_FEED_TEMPLATE = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s={symbol}&region=US&lang=en-US"
)


class StockNewsCollector:
    """Fetches stock headlines from free RSS feeds. No API key required."""

    def __init__(self, general_feeds: list[str] | None = None):
        self._general_feeds = general_feeds or GENERAL_FEEDS

    def fetch(self, symbols: list[str], limit: int = 10) -> list[str]:
        """Return headlines relevant to the given stock symbols.

        Fetches per-symbol Yahoo Finance feeds for each ticker, plus general
        business feeds filtered by ticker keyword.
        """
        tickers = [s.split("/")[0].upper() for s in symbols]
        keywords = {t.lower() for t in tickers}

        headlines: list[str] = []

        # Per-symbol Yahoo Finance feeds
        for ticker in tickers:
            url = YAHOO_FEED_TEMPLATE.format(symbol=ticker)
            headlines.extend(self._parse_feed(url, keywords=None, limit=limit))

        # General business feeds filtered by ticker keyword
        for feed_url in self._general_feeds:
            headlines.extend(self._parse_feed(feed_url, keywords=keywords, limit=limit))

        return headlines

    def _parse_feed(
        self,
        feed_url: str,
        keywords: set[str] | None,
        limit: int,
    ) -> list[str]:
        """Fetch and parse an RSS feed, returning matching headlines."""
        results: list[str] = []
        try:
            resp = requests.get(
                feed_url,
                timeout=10,
                headers={"User-Agent": "trader-bot/1.0"},
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            # Support both RSS <item> and Atom <entry>
            items = root.findall(".//item") or root.findall(
                ".//{http://www.w3.org/2005/Atom}entry"
            )
            for item in items[:limit]:
                title_el = item.find("title")
                if title_el is None:
                    title_el = item.find("{http://www.w3.org/2005/Atom}title")
                if title_el is not None and title_el.text:
                    title = title_el.text.strip()
                    if keywords is None or any(kw in title.lower() for kw in keywords):
                        results.append(title)
        except Exception as e:
            logger.warning(f"Stock news RSS fetch failed for {feed_url}: {e}")
        return results
