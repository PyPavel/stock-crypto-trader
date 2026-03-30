import inspect
import logging
from datetime import datetime, timezone
from trader.config import Config
from trader.adapters.base import ExchangeAdapter
from trader.llm.sentiment import SentimentAnalyzer
from trader.notifications.telegram import TelegramNotifier
from trader.core.signals import SignalGenerator
from trader.core.risk import RiskManager
from trader.core.router import OrderRouter
from trader.portfolio.state import Portfolio
from trader.strategies.registry import get_strategy
from trader.models import SentimentScore, Trade, Signal
from trader.ml.predictor import MLPredictor

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
        notifier=None,
        universe=None,
    ):
        self.config = config
        self._adapter = adapter
        self._sentiment = sentiment_analyzer
        self._collectors = collectors
        self._numeric_collectors = numeric_collectors or []
        self._notifier: TelegramNotifier | None = notifier
        self._universe = universe  # SymbolUniverse | None
        self._signals = SignalGenerator()
        self._risk = RiskManager(config.risk)
        self._router = OrderRouter(adapter=adapter, mode=config.mode)
        self._strategy = get_strategy(config.strategy, config.risk)
        self.portfolio = Portfolio(db_path=db_path, starting_capital=config.capital)

        # ML predictor — only instantiated when enabled in config
        self._ml: MLPredictor | None = None
        if config.ml.enabled:
            self._ml = MLPredictor(config.ml.model_path)

        # Trailing stop: track peak price per open position
        self._peak_prices: dict[str, float] = {}

        # Cooldown: track last trade timestamp per symbol
        self._last_trade_time: dict[str, datetime] = {}

    def run_cycle(self) -> None:
        logger.info("Starting trading cycle")

        # Market-hours guard: skip cycle if adapter reports market is closed
        if hasattr(self._adapter, "is_market_open") and not self._adapter.is_market_open():
            logger.info("Market is closed — skipping cycle")
            return

        # Determine candidate symbols
        if self._universe is not None and self.config.universe.enabled:
            candidates = self._universe.get_candidates()
        else:
            candidates = self.config.pairs

        # Always include symbols with open positions so stop-loss/sell fires regardless
        open_symbols = list(self.portfolio.positions.keys())
        all_symbols = list(dict.fromkeys(open_symbols + list(candidates)))

        prices: dict[str, float] = {}

        # Stage 1: score every symbol
        scored: list[tuple[str, dict]] = []
        for symbol in all_symbols:
            result = self._score_symbol(symbol, prices)
            if result is not None:
                scored.append((symbol, result))

        # Stage 2: determine buy-eligible set — top active_pairs from candidates by signal
        n_active = self.config.universe.active_pairs if self.config.universe.enabled else len(self.config.pairs)
        candidate_set = set(candidates)
        candidate_scored = sorted(
            [(s, r) for s, r in scored if s in candidate_set],
            key=lambda x: abs(x[1]["combined_score"]),
            reverse=True,
        )
        buy_eligible: set[str] = {s for s, _ in candidate_scored[:n_active]}

        # Stage 3: execute decisions
        for symbol, result in scored:
            try:
                self._execute_decisions(symbol, result, prices, can_buy=(symbol in buy_eligible))
            except Exception as e:
                logger.exception("Error executing decisions for %s: %s", symbol, e)

    def _in_cooldown(self, symbol: str) -> bool:
        """Returns True if the symbol is still within its post-trade cooldown period."""
        last = self._last_trade_time.get(symbol)
        if last is None:
            return False
        elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 60
        return elapsed < self.config.risk.cooldown_minutes

    def _record_trade_time(self, symbol: str) -> None:
        self._last_trade_time[symbol] = datetime.now(timezone.utc)

    def _notify(self, side: str, symbol: str, amount: float, price: float, reason: str, prices: dict | None = None) -> None:
        if self._notifier is None:
            return
        label = "CRYPTO" if self.config.exchange != "alpaca" else "STOCK"
        portfolio_value = self.portfolio.total_value(prices or {})
        msg = (
            f"[{label}] {side.upper()} {symbol}\n"
            f"Amount: {amount:.6f}  Price: ${price:,.2f}\n"
            f"Reason: {reason}\n"
            f"Portfolio: ${portfolio_value:,.2f}"
        )
        self._notifier.send(msg)

    def _score_symbol(self, symbol: str, prices: dict) -> dict | None:
        """
        Collect market data and compute all signals for `symbol`.

        Returns a dict with scoring results, or None on unrecoverable error.
        No trades are placed here — side-effect-free.
        """
        try:
            # 1. Market data
            candles = self._adapter.get_candles(symbol, "1h", limit=100)
            price = self._adapter.get_price(symbol)
            prices[symbol] = price

            # 2. Sentiment — collect, deduplicate, score
            texts: list[str] = []
            for collector in self._collectors:
                try:
                    texts.extend(collector.fetch(symbols=[symbol]))
                except Exception as e:
                    logger.warning("Collector failed: %s", e)
            texts = list(dict.fromkeys(texts))

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
                    logger.warning("Numeric collector failed: %s", e)

            if numeric_scores:
                numeric_avg = sum(numeric_scores) / len(numeric_scores)
                if texts:
                    raw_sentiment = raw_sentiment * 0.60 + numeric_avg * 0.40
                else:
                    raw_sentiment = numeric_avg
                logger.info(
                    "%s numeric_signals=%s blended_sentiment=%.3f",
                    symbol, [f"{s:.3f}" for s in numeric_scores], raw_sentiment,
                )

            sentiment = SentimentScore(symbol=symbol, score=raw_sentiment,
                                       source="combined", items_analyzed=len(texts))

            # 3. Technical signals
            sig_result = self._signals.score_with_trend(candles)
            tech_score = sig_result["score"]
            trend_bullish = sig_result["trend_bullish"]
            tech_signal = Signal(
                symbol=symbol,
                score=tech_score,
                reason="technical indicators",
                trend_bullish=trend_bullish,
            )

            # 3b. ML score override
            ml_score = None
            if self._ml is not None:
                ml_score = self._ml.score(candles)
                if ml_score is not None:
                    tech_signal = Signal(symbol=symbol, score=ml_score,
                                        reason="ml predictor",
                                        trend_bullish=tech_signal.trend_bullish)

            # 3c. ATR
            current_atr = self._signals.atr(candles, self.config.risk.atr_period)

            # Combined score used for ranking candidates
            active_score = ml_score if ml_score is not None else tech_score
            combined_score = active_score * 0.6 + raw_sentiment * 0.4

            if ml_score is not None:
                logger.info(
                    "%s price=%.2f tech=%.3f ml=%.3f(active) trend=%s sentiment=%.3f atr=%.2f texts=%d",
                    symbol, price, tech_score, ml_score, "bull" if trend_bullish else "bear",
                    raw_sentiment, current_atr or 0.0, len(texts),
                )
            else:
                logger.info(
                    "%s price=%.2f tech=%.3f trend=%s sentiment=%.3f atr=%.2f texts=%d",
                    symbol, price, tech_score, "bull" if trend_bullish else "bear",
                    raw_sentiment, current_atr or 0.0, len(texts),
                )

            return {
                "price": price,
                "candles": candles,
                "tech_signal": tech_signal,
                "sentiment": sentiment,
                "tech_score": tech_score,
                "ml_score": ml_score,
                "trend_bullish": trend_bullish,
                "raw_sentiment": raw_sentiment,
                "combined_score": combined_score,
                "atr": current_atr,
                "texts": texts,
            }
        except Exception as e:
            logger.exception("Error scoring %s: %s", symbol, e)
            return None

    def _execute_decisions(self, symbol: str, result: dict, prices: dict,
                           can_buy: bool = True) -> None:
        """
        Run stop-loss checks, strategy, risk validation, and order execution
        for `symbol` using the pre-computed `result` from `_score_symbol`.
        """
        price = result["price"]
        tech_signal = result["tech_signal"]
        sentiment = result["sentiment"]
        current_atr = result["atr"]

        # Re-read position from live portfolio (may have changed if another symbol was processed)
        position_usd = 0.0
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            position_usd = pos["amount"] * price

        # 4a. Update trailing peak price
        if position_usd > 0:
            prev_peak = self._peak_prices.get(symbol, price)
            self._peak_prices[symbol] = max(prev_peak, price)
        else:
            self._peak_prices.pop(symbol, None)

        # 4b. Stop-loss and take-profit checks — fire regardless of buy eligibility
        if position_usd > 0 and symbol in self.portfolio.positions:
            entry = self.portfolio.positions[symbol]["entry_price"]

            if self._risk.check_stop_loss(symbol, entry, price, atr=current_atr):
                logger.warning(
                    "Stop-loss triggered for %s: entry=%.2f, current=%.2f",
                    symbol, entry, price,
                )
                self._execute_sell(symbol, position_usd, price, "stop-loss triggered", prices=prices)
                return

            peak = self._peak_prices.get(symbol, price)
            if peak > entry and self._risk.check_trailing_stop(peak, price, atr=current_atr):
                logger.warning(
                    "Trailing stop triggered for %s: peak=%.2f, current=%.2f",
                    symbol, peak, price,
                )
                self._execute_sell(symbol, position_usd, price, "trailing-stop triggered", prices=prices)
                return

            if self._risk.check_trade_loss_limit(entry, price, self.portfolio.starting_capital):
                logger.warning(
                    "Per-trade loss limit triggered for %s: entry=%.2f, current=%.2f",
                    symbol, entry, price,
                )
                self._execute_sell(symbol, position_usd, price, "per-trade-loss-limit triggered", prices=prices)
                return

            if self._risk.check_take_profit(entry, price):
                logger.warning(
                    "Take-profit triggered for %s: entry=%.2f, current=%.2f (+%.1f%%)",
                    symbol, entry, price, (price - entry) / entry * 100,
                )
                self._execute_sell(symbol, position_usd, price, "take-profit triggered", prices=prices)
                return

            if self._risk.check_partial_take_profit(entry, price):
                partial_usd = self._risk.partial_sell_amount(position_usd)
                if partial_usd > 0:
                    logger.warning(
                        "Partial take-profit triggered for %s: entry=%.2f, current=%.2f, selling $%.2f",
                        symbol, entry, price, partial_usd,
                    )
                    self._execute_sell(symbol, partial_usd, price, "partial-take-profit triggered", prices=prices)
                    return

        # 4c. Cooldown check
        if self._in_cooldown(symbol):
            remaining = self.config.risk.cooldown_minutes - (
                (datetime.now(timezone.utc) - self._last_trade_time[symbol]).total_seconds() / 60
            )
            logger.info("COOLDOWN %s: %.0fm remaining, skipping", symbol, remaining)
            return

        # 5. Strategy decision
        decision = self._strategy.decide(
            symbol=symbol,
            technical=tech_signal,
            sentiment=sentiment,
            capital=self.portfolio.cash,
            position=position_usd,
        )

        # Block new buys if symbol didn't rank in top active_pairs
        if decision["action"] == "buy" and not can_buy:
            logger.info("BUY skipped for %s: not in top active pairs this cycle", symbol)
            return

        # 6. Risk check and execution
        if decision["action"] == "buy":
            current_total = self.portfolio.cash
            if self.portfolio.positions and prices:
                for sym, pos in self.portfolio.positions.items():
                    current_total += pos.get("amount", 0.0) * prices.get(sym, 0.0)
            self._risk.reset_daily_tracking(current_total)
            daily_pnl = self._risk.get_daily_pnl(current_total)

            risk_check = self._risk.validate_buy(
                symbol=symbol,
                usd_amount=decision["usd_amount"],
                capital=self.portfolio.cash,
                positions=self.portfolio.positions,
                starting_capital=self.portfolio.starting_capital,
                prices=prices,
                daily_pnl=daily_pnl,
            )

            final_usd = decision["usd_amount"]
            if not risk_check["allowed"]:
                if "exceeds max" in risk_check["reason"]:
                    max_allowed = self._risk.calc_position_size(
                        self.portfolio.cash, price, atr=current_atr,
                    )
                    logger.info(
                        "Capping buy amount: requested $%.2f, capping to max $%.2f",
                        decision['usd_amount'], max_allowed,
                    )
                    final_usd = max_allowed
                else:
                    logger.info("Buy blocked by risk manager: %s", risk_check['reason'])
                    return

            if final_usd <= 0:
                logger.debug("Final buy amount is zero, skipping")
                return

            order = self._router.execute("buy", symbol, final_usd, price=price)
            fee = order.amount * order.price * 0.001
            trade = Trade(order_id=order.id, symbol=symbol, side="buy",
                          amount=order.amount, price=order.price, fee=fee,
                          mode=self.config.mode, narrative=decision["reason"])
            self.portfolio.record_trade(trade)
            self._record_trade_time(symbol)
            self._peak_prices[symbol] = order.price
            logger.info("BUY %s: %.6f @ %.2f", symbol, order.amount, order.price)
            self._notify("buy", symbol, order.amount, order.price, decision["reason"], prices=prices)

        elif decision["action"] == "sell" and position_usd > 0:
            self._execute_sell(symbol, position_usd, price, decision["reason"], prices=prices)

        else:
            logger.info("HOLD %s: %s", symbol, decision['reason'])

    def _execute_sell(self, symbol: str, position_usd: float, price: float, reason: str, prices: dict | None = None) -> None:
        """Execute a sell order, record the trade, and reset tracking state."""
        is_partial = "partial" in reason.lower()
        order = self._router.execute("sell", symbol, position_usd, price=price)
        fee = order.amount * order.price * 0.001
        trade = Trade(order_id=order.id, symbol=symbol, side="sell",
                      amount=order.amount, price=order.price, fee=fee,
                      mode=self.config.mode, narrative=reason)
        self.portfolio.record_trade(trade)
        self._record_trade_time(symbol)
        if not is_partial:
            self._peak_prices.pop(symbol, None)
        logger.info("SELL %s: %.6f @ %.2f (%s)", symbol, order.amount, order.price, reason)
        self._notify("sell", symbol, order.amount, order.price, reason, prices=prices)
