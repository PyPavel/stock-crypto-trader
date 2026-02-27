import inspect
import logging
from trader.config import Config
from trader.adapters.base import ExchangeAdapter
from trader.llm.sentiment import SentimentAnalyzer
from trader.core.signals import SignalGenerator
from trader.core.risk import RiskManager
from trader.core.router import OrderRouter
from trader.portfolio.state import Portfolio
from trader.strategies.registry import get_strategy
from trader.models import SentimentScore, Trade, Signal

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(
        self,
        config: Config,
        adapter: ExchangeAdapter,
        sentiment_analyzer: SentimentAnalyzer,
        collectors: list,
        numeric_collectors: list | None = None,
        db_path: str = "trader.db",
    ):
        self.config = config
        self._adapter = adapter
        self._sentiment = sentiment_analyzer
        self._collectors = collectors
        self._numeric_collectors = numeric_collectors or []
        self._signals = SignalGenerator()
        self._risk = RiskManager(config.risk)
        self._router = OrderRouter(adapter=adapter, mode=config.mode)
        self._strategy = get_strategy(config.strategy, config.risk)
        self.portfolio = Portfolio(db_path=db_path, starting_capital=config.capital)

    def run_cycle(self) -> None:
        logger.info("Starting trading cycle")
        prices = {}

        for symbol in self.config.pairs:
            try:
                self._process_symbol(symbol, prices)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    def _process_symbol(self, symbol: str, prices: dict) -> None:
        # 1. Market data
        candles = self._adapter.get_candles(symbol, "1h", limit=100)
        price = self._adapter.get_price(symbol)
        prices[symbol] = price

        # 2. Sentiment
        texts = []
        for collector in self._collectors:
            try:
                texts.extend(collector.fetch(symbols=[symbol]))
            except Exception as e:
                logger.warning(f"Collector failed: {e}")

        raw_sentiment = self._sentiment.score_texts(texts)

        # Blend in numeric signals (Fear & Greed, CoinGecko, etc.)
        numeric_scores = []
        for nc in self._numeric_collectors:
            try:
                sig = inspect.signature(nc.score)
                params = list(sig.parameters.keys())
                s = nc.score(symbols=[symbol]) if "symbols" in params else nc.score()
                if s is not None:
                    numeric_scores.append(s)
            except Exception as e:
                logger.warning(f"Numeric collector failed: {e}")

        if numeric_scores:
            numeric_avg = sum(numeric_scores) / len(numeric_scores)
            # Weight: 60% text sentiment, 40% numeric signals (or 100% numeric if no texts)
            if texts:
                raw_sentiment = raw_sentiment * 0.60 + numeric_avg * 0.40
            else:
                raw_sentiment = numeric_avg
            logger.info(f"{symbol} numeric_signals={[f'{s:.3f}' for s in numeric_scores]} blended_sentiment={raw_sentiment:.3f}")

        sentiment = SentimentScore(symbol=symbol, score=raw_sentiment,
                                   source="combined", items_analyzed=len(texts))

        # 3. Technical signals
        tech_score = self._signals.score(candles)
        tech_signal = Signal(symbol=symbol, score=tech_score, reason="technical indicators")
        logger.info(f"{symbol} price={price:.2f} tech={tech_score:.3f} sentiment={raw_sentiment:.3f} texts={len(texts)}")

        # 4. Strategy decision
        position_usd = 0.0
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            position_usd = pos["amount"] * price

        # Independent stop-loss check — fires before strategy decision
        if position_usd > 0 and symbol in self.portfolio.positions:
            entry = self.portfolio.positions[symbol]["entry_price"]
            if self._risk.check_stop_loss(symbol, entry, price):
                logger.warning(f"Stop-loss triggered for {symbol}: entry={entry:.2f}, current={price:.2f}")
                order = self._router.execute("sell", symbol, position_usd, price=price)
                trade = Trade(order_id=order.id, symbol=symbol, side="sell",
                              amount=order.amount, price=order.price, fee=0.0,
                              mode=self.config.mode, narrative="stop-loss triggered")
                self.portfolio.record_trade(trade)
                return  # exit after stop-loss, don't run strategy

        decision = self._strategy.decide(
            symbol=symbol,
            technical=tech_signal,
            sentiment=sentiment,
            capital=self.portfolio.cash,
            position=position_usd,
        )

        # 5. Risk check and execution
        if decision["action"] == "buy":
            risk_check = self._risk.validate_buy(
                symbol=symbol,
                usd_amount=decision["usd_amount"],
                capital=self.portfolio.cash,
                positions=self.portfolio.positions,
                starting_capital=self.portfolio.starting_capital,
            )
            if not risk_check["allowed"]:
                logger.info(f"Buy blocked by risk manager: {risk_check['reason']}")
                return

            order = self._router.execute("buy", symbol, decision["usd_amount"], price=price)
            trade = Trade(order_id=order.id, symbol=symbol, side="buy",
                          amount=order.amount, price=order.price, fee=0.0,
                          mode=self.config.mode, narrative=decision["reason"])
            self.portfolio.record_trade(trade)
            logger.info(f"BUY {symbol}: {order.amount:.6f} @ {order.price:.2f}")

        elif decision["action"] == "sell" and position_usd > 0:
            order = self._router.execute("sell", symbol, position_usd, price=price)
            trade = Trade(order_id=order.id, symbol=symbol, side="sell",
                          amount=order.amount, price=order.price, fee=0.0,
                          mode=self.config.mode, narrative=decision["reason"])
            self.portfolio.record_trade(trade)
            logger.info(f"SELL {symbol}: {order.amount:.6f} @ {order.price:.2f}")

        else:
            logger.info(f"HOLD {symbol}: {decision['reason']}")
