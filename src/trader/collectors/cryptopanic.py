import requests
import logging

logger = logging.getLogger(__name__)

CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"


class CryptoPanicCollector:
    def __init__(self, api_key: str = ""):
        self._api_key = api_key

    def fetch(self, symbols: list[str], limit: int = 20) -> list[str]:
        currencies = {s.split("/")[0] for s in symbols}
        params = {"public": "true", "kind": "news", "filter": "hot"}
        if self._api_key:
            params["auth_token"] = self._api_key

        try:
            response = requests.get(CRYPTOPANIC_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"CryptoPanic fetch failed: {e}")
            return []

        headlines = []
        for item in data.get("results", [])[:limit]:
            item_currencies = {c["code"] for c in item.get("currencies", [])}
            if not currencies or item_currencies & currencies:
                headlines.append(item["title"])

        return headlines
