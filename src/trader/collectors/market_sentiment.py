# Stocks path uses VIX-based sentiment (CBOE volatility index via Yahoo Finance).
# Crypto path uses FearGreedCollector directly — do not use this for crypto.
from trader.collectors.vix_sentiment import VIXSentimentCollector as MarketSentimentCollector

__all__ = ["MarketSentimentCollector"]
