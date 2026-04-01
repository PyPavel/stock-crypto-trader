# Re-export FearGreedCollector as MarketSentimentCollector.
# Both crypto and stock paths use the same alternative.me Fear & Greed API.
from trader.collectors.feargreed import FearGreedCollector as MarketSentimentCollector

__all__ = ["MarketSentimentCollector"]
