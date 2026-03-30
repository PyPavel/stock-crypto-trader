import logging
import argparse
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from trader.config import load_config
from trader.llm.sentiment import SentimentAnalyzer
from trader.core.engine import TradingEngine
from trader.dashboard.api import create_app
from trader.notifications.telegram import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Trader Bot")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--db", default="trader.db")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Starting in %s mode, exchange=%s, strategy=%s, pairs=%s",
                cfg.mode.upper(), cfg.exchange, cfg.strategy, cfg.pairs)

    notifier = TelegramNotifier(
        bot_token=cfg.telegram.bot_token,
        chat_id=cfg.telegram.chat_id,
    )
    if cfg.telegram.bot_token:
        logger.info("Telegram notifications enabled (chat_id=%s)", cfg.telegram.chat_id)
    else:
        logger.info("Telegram notifications disabled (no TELEGRAM_BOT_TOKEN set)")

    sentiment = SentimentAnalyzer(model=cfg.mimo.model, api_key=cfg.mimo.api_key)

    if cfg.exchange == "alpaca":
        from trader.adapters.alpaca import AlpacaAdapter
        from trader.collectors.stock_news import StockNewsCollector
        from trader.collectors.market_sentiment import MarketSentimentCollector
        from trader.collectors.polymarket import PolymarketCollector
        from trader.collectors.unusual_whales import UnusualWhalesCollector
        from trader.collectors.google_trends import GoogleTrendsCollector

        adapter = AlpacaAdapter(
            api_key=cfg.alpaca.api_key,
            api_secret=cfg.alpaca.api_secret,
            paper=cfg.alpaca.paper,
        )
        collectors = [
            StockNewsCollector(),
        ]
        numeric_collectors = [
            MarketSentimentCollector(),  # Fear & Greed for equities
            PolymarketCollector(),       # Prediction market crowd sentiment
            UnusualWhalesCollector(),    # Options flow: call/put ratio signal
            GoogleTrendsCollector(asset_class="stock"),  # Search interest trend signal
        ]
    else:
        # Default: coinbase (existing logic unchanged)
        from trader.adapters.coinbase import CoinbaseAdapter
        from trader.collectors.reddit import RedditCollector
        from trader.collectors.rss import RSSCollector
        from trader.collectors.feargreed import FearGreedCollector
        from trader.collectors.coingecko import CoinGeckoCollector
        from trader.collectors.polymarket import PolymarketCollector
        from trader.collectors.funding_rates import FundingRateCollector
        from trader.collectors.google_trends import GoogleTrendsCollector

        adapter = CoinbaseAdapter(api_key=cfg.coinbase.api_key, api_secret=cfg.coinbase.api_secret)
        collectors = [
            RedditCollector(client_id=cfg.reddit.client_id, client_secret=cfg.reddit.client_secret,
                            user_agent=cfg.reddit.user_agent),
            RSSCollector(),  # 8 free feeds: CoinDesk, CoinTelegraph, Decrypt, Bitcoin Magazine, and more
        ]
        numeric_collectors = [
            FearGreedCollector(),        # Crypto Fear & Greed Index
            CoinGeckoCollector(),        # Global market cap change
            PolymarketCollector(),       # Prediction market crowd sentiment
            FundingRateCollector(),      # Binance perpetual futures funding rates
            GoogleTrendsCollector(),     # Search interest trend signal (crypto)
        ]

    from trader.core.universe import SymbolUniverse
    universe = SymbolUniverse(
        exchange=cfg.exchange,
        seed_pairs=cfg.pairs,
        universe_config=cfg.universe,
        alpaca_cfg=cfg.alpaca if cfg.exchange == "alpaca" else None,
    )
    if cfg.universe.enabled:
        universe.refresh_universe()

    engine = TradingEngine(config=cfg, adapter=adapter, sentiment_analyzer=sentiment,
                           collectors=collectors, numeric_collectors=numeric_collectors,
                           db_path=args.db, notifier=notifier, universe=universe)

    label = "STOCK" if cfg.exchange == "alpaca" else "CRYPTO"
    notifier.send(
        f"[{label}] Trader started\n"
        f"Mode: {cfg.mode.upper()}  Strategy: {cfg.strategy}\n"
        f"Pairs: {', '.join(cfg.pairs[:5])}{'...' if len(cfg.pairs) > 5 else ''}"
    )

    def run_cycle_with_market_check():
        if hasattr(adapter, "is_market_open") and not adapter.is_market_open():
            logger.info("Market is closed — skipping trading cycle")
            return
        engine.run_cycle()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_cycle_with_market_check,
        "interval",
        seconds=cfg.cycle_interval,
        id="trading_cycle",
    )
    if cfg.universe.enabled:
        scheduler.add_job(
            universe.refresh_universe,
            "interval",
            hours=24,
            id="universe_refresh",
        )
    scheduler.start()
    logger.info(f"Scheduler started, cycle every {cfg.cycle_interval}s")

    app = create_app(engine, config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
