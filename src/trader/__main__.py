import logging
import argparse
import uvicorn
from apscheduler import Scheduler
from apscheduler.triggers.interval import IntervalTrigger
from trader.config import load_config
from trader.adapters.coinbase import CoinbaseAdapter
from trader.llm.sentiment import SentimentAnalyzer
from trader.collectors.cryptopanic import CryptoPanicCollector
from trader.collectors.reddit import RedditCollector
from trader.core.engine import TradingEngine
from trader.dashboard.api import create_app

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
    logger.info(f"Starting in {cfg.mode.upper()} mode, strategy={cfg.strategy}, pairs={cfg.pairs}")

    adapter = CoinbaseAdapter(api_key=cfg.coinbase.api_key, api_secret=cfg.coinbase.api_secret)
    sentiment = SentimentAnalyzer(model=cfg.ollama.model, base_url=cfg.ollama.base_url)
    collectors = [
        CryptoPanicCollector(api_key=cfg.cryptopanic.api_key),
        RedditCollector(client_id=cfg.reddit.client_id, client_secret=cfg.reddit.client_secret,
                        user_agent=cfg.reddit.user_agent),
    ]

    engine = TradingEngine(config=cfg, adapter=adapter, sentiment_analyzer=sentiment,
                           collectors=collectors, db_path=args.db)

    scheduler = Scheduler()
    scheduler.add_schedule(
        engine.run_cycle,
        IntervalTrigger(seconds=cfg.cycle_interval),
        id="trading_cycle",
    )
    scheduler.start_in_background()
    logger.info(f"Scheduler started, cycle every {cfg.cycle_interval}s")

    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
