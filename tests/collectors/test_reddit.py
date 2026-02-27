from unittest.mock import MagicMock, patch
from trader.collectors.reddit import RedditCollector

SUBREDDITS = {"BTC": ["bitcoin"], "ETH": ["ethtrader"]}


def test_fetch_returns_post_titles():
    mock_post = MagicMock()
    mock_post.title = "Bitcoin looking bullish today"
    mock_subreddit = MagicMock()
    mock_subreddit.hot.return_value = [mock_post]

    with patch("trader.collectors.reddit.praw.Reddit") as mock_reddit_cls:
        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_reddit_cls.return_value = mock_reddit

        collector = RedditCollector(client_id="id", client_secret="secret",
                                    user_agent="test", subreddit_map=SUBREDDITS)
        posts = collector.fetch(symbols=["BTC/USD"])

    assert len(posts) >= 1
    assert "Bitcoin" in posts[0]


def test_fetch_empty_on_error():
    with patch("trader.collectors.reddit.praw.Reddit") as mock_reddit_cls:
        mock_reddit_cls.side_effect = Exception("auth failed")
        collector = RedditCollector(client_id="", client_secret="",
                                    user_agent="test", subreddit_map=SUBREDDITS)
        posts = collector.fetch(symbols=["BTC/USD"])
    assert posts == []
