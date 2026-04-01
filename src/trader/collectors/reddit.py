import logging
import praw

logger = logging.getLogger(__name__)

DEFAULT_SUBREDDIT_MAP = {
    "BTC": ["bitcoin", "cryptocurrency"],
    "ETH": ["ethtrader", "ethereum"],
    "SOL": ["solana"],
    "ADA": ["cardano"],
}


class RedditCollector:
    def __init__(self, client_id: str, client_secret: str, user_agent: str,
                 subreddit_map: dict | None = None):
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._subreddit_map = subreddit_map or DEFAULT_SUBREDDIT_MAP
        self._disabled = False  # circuit breaker: set True on auth failure
        self._reddit = None  # lazy-initialized PRAW client

    def _get_reddit(self):
        if self._reddit is None:
            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
        return self._reddit

    def fetch(self, symbols: list[str], limit: int = 15) -> list[str]:
        if self._disabled:
            return []

        currencies = {s.split("/")[0] for s in symbols}

        try:
            reddit = self._get_reddit()
        except Exception as e:
            logger.warning(f"Reddit init failed: {e}")
            return []

        titles = []
        for currency in currencies:
            subs = self._subreddit_map.get(currency, [])
            for sub_name in subs:
                try:
                    sub = reddit.subreddit(sub_name)
                    for post in sub.hot(limit=limit):
                        titles.append(post.title)
                except Exception as e:
                    msg = str(e)
                    if "401" in msg or "invalid_grant" in msg or "Unauthorized" in msg:
                        logger.warning(
                            f"Reddit credentials invalid (401) — disabling collector. "
                            f"Update REDDIT_CLIENT_ID/SECRET in .env to re-enable."
                        )
                        self._disabled = True
                        return titles  # return whatever we got before the failure
                    logger.warning(f"Reddit fetch from r/{sub_name} failed: {e}")

        return titles
