import inspect
import logging
from collections import deque
from datetime import datetime, timezone
from trader.config import Config
from trader.adapters.base import ExchangeAdapter
from trader.llm.sentiment import SentimentAnalyzer
from trader.llm.advisor import LLMAdvisor
from trader.notifications.telegram import TelegramNotifier
from trader.core.signals import SignalGenerator
from trader.core.risk import RiskManager
from trader.core.router import OrderRouter
from trader.core.pdt import PDTGuard
from trader.core.time_gate import TimeWindowGate
from trader.portfolio.state import Portfolio
from trader.portfolio.db import save_signal_history, load_signal_history
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
        advisor: LLMAdvisor | None = None,
        numeric_collectors: list | None = None,
        db_path: str = "trader.db",
        notifier=None,
        universe=None,
        time_gate: "TimeWindowGate | None" = None,
    ):
        self.config = config
        self._adapter = adapter
        self._sentiment = sentiment_analyzer
        self._advisor = advisor
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

        # Partial TP: track symbols that already took partial profit (reset on full exit)
        self._partial_tp_taken: set[str] = set()

        # PDT guard — Alpaca US-equity accounts only (PDT is an SEC rule, not applicable to crypto)
        self._pdt: PDTGuard | None = None
        if config.exchange in ("alpaca", "tastytrade") and hasattr(adapter, "get_day_trade_count"):
            self._pdt = PDTGuard(adapter)
        self._time_gate = time_gate

        # Cycle counter for periodic Telegram summaries
        self._cycle_count: int = 0

        # Dedup signal alerts: track last texts hash per symbol to avoid re-sending identical news
        self._last_alert_hash: dict[str, int] = {}

        # Current cycle scores — used for position rotation decisions
        self._current_scores: dict[str, float] = {}

        # Restore persisted signal history so persistence checks survive restarts
        history = load_signal_history(self.portfolio._conn)
        for sym, scores in history.items():
            self._strategy._signal_history[sym] = deque(
                scores, maxlen=self._strategy._history_maxlen
            )

    def run_cycle(self) -> None:
        logger.info("Starting trading cycle")

        # Market-hours guard: skip cycle if adapter reports market is closed
        if hasattr(self._adapter, "is_market_open") and not self._adapter.is_market_open():
            logger.info("Market is closed — skipping cycle")
            return

        if self._pdt:
            self._pdt.refresh()

        # Determine candidate symbols
        if self._universe is not None and self.config.universe.enabled:
            # Retry refresh if universe is still empty (e.g. started outside market hours)
            if not self._universe._universe:
                logger.info("Universe empty — retrying refresh")
                self._universe.refresh_universe()
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

        # Update current scores for rotation logic
        self._current_scores = {s: r["combined_score"] for s, r in scored}

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

        # Persist signal history for restart recovery
        for sym, hist in self._strategy._signal_history.items():
            save_signal_history(self.portfolio._conn, sym, hist)

        # Signal alerts — strong signals with news context, even when no trade fires
        self._send_signal_alerts(scored, prices)

        # Periodic Telegram summary every 6 cycles (~30 min)
        self._cycle_count += 1
        if self._notifier and self._cycle_count % 6 == 0:
            self._send_cycle_summary(prices)

    def _in_cooldown(self, symbol: str) -> bool:
        """Returns True if the symbol is still within its post-trade cooldown period."""
        last = self._last_trade_time.get(symbol)
        if last is None:
            return False
        elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 60
        return elapsed < self.config.risk.cooldown_minutes

    def _record_trade_time(self, symbol: str) -> None:
        self._last_trade_time[symbol] = datetime.now(timezone.utc)

    def _notify(self, side: str, symbol: str, amount: float, price: float, reason: str,
                prices: dict | None = None, result: dict | None = None) -> None:
        if self._notifier is None:
            return
        label = "CRYPTO" if self.config.exchange != "alpaca" else "STOCK"
        portfolio_value = self.portfolio.total_value(prices or {})

        # Format price with enough decimals for micro-priced tokens
        price_str = f"${price:,.6f}".rstrip("0").rstrip(".") if price < 0.01 else f"${price:,.2f}"

        lines = [f"[{label}] {side.upper()} {symbol}"]
        lines.append(f"Price: {price_str}  Amount: {amount:.4f}")
        lines.append(f"Reason: {reason}")

        # Signal breakdown
        if result:
            tech = result.get("tech_score", 0.0)
            sent = result.get("raw_sentiment", 0.0)
            combo = result.get("combined_score", 0.0)
            trend = "bull" if result.get("trend_bullish") else "bear"
            ml = result.get("ml_score")
            if ml is not None:
                lines.append(f"Signals: tech={tech:+.2f} ml={ml:+.2f} sent={sent:+.2f} => {combo:+.2f} ({trend})")
            else:
                lines.append(f"Signals: tech={tech:+.2f} sent={sent:+.2f} => {combo:+.2f} ({trend})")

            # Top texts that drove the decision (truncated)
            texts = result.get("texts", [])
            for t in texts[:2]:
                snippet = t[:120].replace("\n", " ")
                lines.append(f"  >> {snippet}")

        lines.append(f"Portfolio: ${portfolio_value:,.2f}")
        self._notifier.send("\n".join(lines))

    def _send_signal_alerts(self, scored: list[tuple[str, dict]], prices: dict) -> None:
        """
        Send Telegram alerts for strong signals that weren't traded (due to limits,
        cooldown, or bearish trend). Fires at most 3 alerts per cycle to avoid spam.
        Threshold: |combined_score| >= 0.35 or |raw_sentiment| >= 0.40.
        """
        if self._notifier is None:
            return

        label = "CRYPTO" if self.config.exchange != "alpaca" else "STOCK"
        SCORE_THRESHOLD = 0.35
        SENTIMENT_THRESHOLD = 0.40
        MAX_ALERTS = 3

        # Find symbols with strong signals that we're not executing a trade on right now
        open_syms = set(self.portfolio.positions.keys())
        alerts = []
        for sym, r in scored:
            combo = r.get("combined_score", 0.0)
            sent = r.get("raw_sentiment", 0.0)
            texts = r.get("texts", [])
            # Only alert if signal is strong AND there are actual texts explaining why
            if (abs(combo) >= SCORE_THRESHOLD or abs(sent) >= SENTIMENT_THRESHOLD) and texts:
                alerts.append((sym, r, combo))

        # Sort by abs(score) descending, cap at MAX_ALERTS
        alerts.sort(key=lambda x: abs(x[2]), reverse=True)
        for sym, r, combo in alerts[:MAX_ALERTS]:
            texts = r.get("texts", [])
            texts_hash = hash(tuple(texts[:3]))
            if self._last_alert_hash.get(sym) == texts_hash:
                continue
            self._last_alert_hash[sym] = texts_hash

            tech = r.get("tech_score", 0.0)
            sent = r.get("raw_sentiment", 0.0)
            trend = "bull" if r.get("trend_bullish") else "bear"
            ml = r.get("ml_score")
            price = prices.get(sym, 0.0)
            price_str = f"${price:,.6f}".rstrip("0").rstrip(".") if price < 0.01 else f"${price:,.2f}"
            direction = "BULLISH" if combo > 0 else "BEARISH"
            in_position = sym in open_syms

            lines = [f"[{label}] {direction} SIGNAL: {sym} ({combo:+.2f})"]
            lines.append(f"Price: {price_str}  Trend: {trend}")
            if ml is not None:
                lines.append(f"tech={tech:+.2f} ml={ml:+.2f} sent={sent:+.2f}")
            else:
                lines.append(f"tech={tech:+.2f} sent={sent:+.2f}")
            lines.append(f"Position: {'OPEN' if in_position else 'none'}")

            for t in texts[:3]:
                snippet = t[:140].replace("\n", " ")
                lines.append(f"  >> {snippet}")

            self._notifier.send("\n".join(lines))

    def _send_cycle_summary(self, prices: dict) -> None:
        """Send a brief portfolio status + top signals to Telegram every N cycles."""
        label = "CRYPTO" if self.config.exchange != "alpaca" else "STOCK"
        portfolio_value = self.portfolio.total_value(prices)
        positions = self.portfolio.positions
        cash = self.portfolio.cash

        lines = [f"[{label}] Portfolio update"]
        lines.append(f"Value: ${portfolio_value:,.2f}  Cash: ${cash:,.2f}")
        if positions:
            lines.append(f"Positions ({len(positions)}):")
            for sym, pos in sorted(positions.items()):
                amt = pos["amount"] if isinstance(pos, dict) else pos
                entry = pos.get("entry_price", 0) if isinstance(pos, dict) else 0
                price = prices.get(sym, 0)
                current_val = amt * price
                pnl = current_val - amt * entry if entry else 0
                pnl_pct = (pnl / (amt * entry) * 100) if entry and amt else 0
                lines.append(f"  {sym}: ${current_val:,.2f} (PnL {pnl:+.2f} / {pnl_pct:+.1f}%)")
        else:
            lines.append("No open positions")

        # Top 3 signals this cycle
        if self._current_scores:
            top = sorted(self._current_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            lines.append("Top signals:")
            for sym, score in top:
                arrow = "▲" if score > 0 else "▼"
                lines.append(f"  {arrow} {sym}: {score:+.2f}")

        self._notifier.send("\n".join(lines))

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

            # 2. Technical signals first — cheap, used to gate expensive LLM calls
            sig_result = self._signals.score_with_trend(candles)
            tech_score = sig_result["score"]
            trend_bullish = sig_result["trend_bullish"]
            rsi_value = sig_result.get("rsi")

            # 3. Sentiment — skip LLM for non-held symbols with deeply negative tech score.
            # Even with perfect sentiment (1.0), combined = tech*0.6 + sent*0.4 can't reach
            # the buy threshold (0.35) when tech < -0.15, so the LLM call would be wasted.
            has_position = symbol in self.portfolio.positions
            skip_llm = not has_position and tech_score < -0.15

            texts: list[str] = []
            if not skip_llm:
                for collector in self._collectors:
                    try:
                        texts.extend(collector.fetch(symbols=[symbol]))
                    except Exception as e:
                        logger.warning("Collector failed: %s", e)
                texts = list(dict.fromkeys(texts))

            raw_sentiment = self._sentiment.score_texts(texts) if not skip_llm else 0.0

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
            tech_signal = Signal(
                symbol=symbol,
                score=tech_score,
                reason="technical indicators",
                trend_bullish=trend_bullish,
                rsi=rsi_value,
            )

            # 3b. ML score — blend with tech rather than override
            ml_score = None
            if self._ml is not None:
                ml_score = self._ml.score(candles)

            # 3c. ATR
            current_atr = self._signals.atr(candles, self.config.risk.atr_period)

            # Combined score used for ranking candidates
            if ml_score is not None:
                active_score = ml_score * 0.7 + tech_score * 0.3
                tech_signal = Signal(symbol=symbol, score=active_score,
                                     reason="ml+tech blend",
                                     trend_bullish=tech_signal.trend_bullish,
                                     rsi=tech_signal.rsi)
            else:
                active_score = tech_score
            # Calculate weights
            strat = self._strategy
            tw = getattr(strat, "tech_weight", 0.7)
            sw = getattr(strat, "sentiment_weight", 0.3)

            # If no sentiment texts were found, use the active_score directly 
            # instead of penalizing it with the 0.7 multiplier (tw)
            if not texts:
                combined_score = active_score
            else:
                combined_score = active_score * tw + raw_sentiment * sw

            if ml_score is not None:
                logger.info(
                    "%s price=%.2f tech=%.3f ml=%.3f blend=%.3f(active) trend=%s sentiment=%.3f atr=%.2f texts=%d",
                    symbol, price, tech_score, ml_score, active_score, "bull" if trend_bullish else "bear",
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

        # 4b. Protective exits — fire regardless of cooldown (stop-loss, trailing, loss-limit, full TP)
        if position_usd > 0 and symbol in self.portfolio.positions:
            entry = self.portfolio.positions[symbol]["entry_price"]

            # PDT guard: same-day buys cannot be sold when day-trade budget is exhausted.
            # The position holds overnight; notify once so the loss is visible in Telegram.
            if self._pdt and not self._pdt.can_exit_today(symbol):
                peak = self._peak_prices.get(symbol, price)
                would_exit = (
                    self._risk.check_stop_loss(symbol, entry, price, atr=current_atr)
                    or (peak > entry and self._risk.check_trailing_stop(peak, price, atr=current_atr))
                    or self._risk.check_trade_loss_limit(entry, price, self.portfolio.starting_capital)
                )
                if would_exit:
                    self._notify_pdt_blocked(symbol, entry, price)
                # Skip all protective exits — fall through to strategy/cooldown
            else:
                if self._risk.check_stop_loss(symbol, entry, price, atr=current_atr):
                    logger.warning(
                        "Stop-loss triggered for %s: entry=%.2f, current=%.2f",
                        symbol, entry, price,
                    )
                    self._execute_sell(symbol, position_usd, price, "stop-loss triggered", prices=prices, result=result)
                    return

                peak = self._peak_prices.get(symbol, price)
                if peak > entry and self._risk.check_trailing_stop(peak, price, atr=current_atr):
                    logger.warning(
                        "Trailing stop triggered for %s: peak=%.2f, current=%.2f",
                        symbol, peak, price,
                    )
                    self._execute_sell(symbol, position_usd, price, "trailing-stop triggered", prices=prices, result=result)
                    return

                if self._risk.check_trade_loss_limit(entry, price, self.portfolio.starting_capital):
                    logger.warning(
                        "Per-trade loss limit triggered for %s: entry=%.2f, current=%.2f",
                        symbol, entry, price,
                    )
                    self._execute_sell(symbol, position_usd, price, "per-trade-loss-limit triggered", prices=prices, result=result)
                    return

                if self._risk.check_take_profit(entry, price):
                    logger.warning(
                        "Take-profit triggered for %s: entry=%.2f, current=%.2f (+%.1f%%)",
                        symbol, entry, price, (price - entry) / entry * 100,
                    )
                    self._execute_sell(symbol, position_usd, price, "take-profit triggered", prices=prices, result=result)
                    return

        # 4c. Cooldown check (partial TP also respects cooldown — prevents cascade halving)
        if self._in_cooldown(symbol):
            remaining = self.config.risk.cooldown_minutes - (
                (datetime.now(timezone.utc) - self._last_trade_time[symbol]).total_seconds() / 60
            )
            logger.info("COOLDOWN %s: %.0fm remaining, skipping", symbol, remaining)
            return

        # 4d. Partial take-profit — fires once per position (not per cooldown window)
        if position_usd > 0 and symbol in self.portfolio.positions and symbol not in self._partial_tp_taken:
            entry = self.portfolio.positions[symbol]["entry_price"]
            if self._risk.check_partial_take_profit(entry, price):
                partial_usd = self._risk.partial_sell_amount(position_usd)
                if partial_usd > 0:
                    logger.warning(
                        "Partial take-profit triggered for %s: entry=%.4f, current=%.4f, selling $%.2f",
                        symbol, entry, price, partial_usd,
                    )
                    self._partial_tp_taken.add(symbol)
                    self._execute_sell(symbol, partial_usd, price, "partial-take-profit triggered", prices=prices, result=result)
                    return

        # 5. Strategy decision
        decision = self._strategy.decide(
            symbol=symbol,
            technical=tech_signal,
            sentiment=sentiment,
            capital=self.portfolio.cash,
            position=position_usd,
        )

        # 5b. LLM Advisor Confirmation / Boosting
        if decision["action"] == "buy" and self._advisor:
            trend = "bullish" if result.get("trend_bullish") else "bearish"
            advise = self._advisor.advise(
                symbol=symbol,
                tech_score=result.get("tech_score", 0.0),
                trend=trend,
                sentiment=result.get("raw_sentiment", 0.0),
                headlines=result.get("texts", []),
                pdt_remaining=self._pdt.remaining() if self._pdt else None,
            )
            if advise["judgment"] == "avoid":
                logger.info("LLM Advisor VETO for %s: %s", symbol, advise.get("reason"))
                decision["action"] = "hold"
                decision["reason"] = f"LLM Advisor veto: {advise.get('reason')}"
            elif advise["judgment"] == "buy" and advise.get("conviction", 0) >= 0.8:
                multiplier = self.config.risk.conviction_size_multiplier
                if multiplier > 1.0:
                    logger.info("LLM Advisor BOOST for %s: conviction=%.2f, multiplier=%.2f",
                                symbol, advise["conviction"], multiplier)
                    decision["usd_amount"] *= multiplier
                    decision["reason"] += f" (LLM High Conviction Boost x{multiplier})"

        # 5c. Time window gate — block buys/sells outside configured windows
        if self._time_gate:
            if decision["action"] == "buy" and not self._time_gate.can_buy():
                logger.info("TIME GATE: buy blocked for %s (outside buy window)", symbol)
                return
            if decision["action"] == "sell" and not self._time_gate.can_sell():
                logger.info("TIME GATE: sell blocked for %s (outside sell window)", symbol)
                return

        # Block new buys if symbol didn't rank in top active_pairs
        if decision["action"] == "buy" and not can_buy:
            logger.info("BUY skipped for %s: not in top active pairs this cycle", symbol)
            return

        # PDT: gate buys by remaining day-trade budget and score threshold
        if decision["action"] == "buy" and self._pdt:
            remaining = self._pdt.remaining()
            if remaining == 0:
                logger.info("BUY blocked for %s: PDT budget exhausted", symbol)
                return
            threshold = self._pdt.buy_threshold()
            if threshold is not None and result["combined_score"] < threshold:
                logger.info(
                    "BUY blocked for %s: score %.3f below PDT threshold %.2f (remaining=%d)",
                    symbol, result["combined_score"], threshold, remaining,
                )
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
                elif "max open positions" in risk_check["reason"] and \
                        self._try_position_rotation(symbol, result["combined_score"], prices):
                    # Rotation closed the weakest position — re-validate
                    risk_check = self._risk.validate_buy(
                        symbol=symbol,
                        usd_amount=decision["usd_amount"],
                        capital=self.portfolio.cash,
                        positions=self.portfolio.positions,
                        starting_capital=self.portfolio.starting_capital,
                        prices=prices,
                        daily_pnl=daily_pnl,
                    )
                    if not risk_check["allowed"]:
                        logger.info("Buy blocked after rotation: %s", risk_check["reason"])
                        return
                else:
                    logger.info("Buy blocked by risk manager: %s", risk_check['reason'])
                    return

            if final_usd <= 0:
                logger.debug("Final buy amount is zero, skipping")
                return

            min_pos = getattr(self.config.risk, "min_position_usd", 50.0)
            if final_usd < min_pos:
                logger.info("BUY skipped %s: $%.2f below minimum $%.0f", symbol, final_usd, min_pos)
                return

            order = self._router.execute("buy", symbol, final_usd, price=price)
            fee = order.amount * order.price * 0.001
            trade = Trade(order_id=order.id, symbol=symbol, side="buy",
                          amount=order.amount, price=order.price, fee=fee,
                          mode=self.config.mode, narrative=decision["reason"])
            self.portfolio.record_trade(trade)
            self._record_trade_time(symbol)
            self._peak_prices[symbol] = order.price
            if self._pdt:
                self._pdt.record_buy(symbol)
            logger.info("BUY %s: %.6f @ %.2f", symbol, order.amount, order.price)
            self._notify("buy", symbol, order.amount, order.price, decision["reason"], prices=prices, result=result)

        elif decision["action"] == "sell" and position_usd > 0:
            self._execute_sell(symbol, position_usd, price, decision["reason"], prices=prices, result=result)

        else:
            logger.info("HOLD %s: %s", symbol, decision['reason'])

    def _try_position_rotation(self, new_symbol: str, new_score: float, prices: dict) -> bool:
        """
        Close the weakest open position to make room for a stronger signal.
        Returns True if a position was closed (caller should retry the buy).
        """
        rotation_min_delta = self.config.risk.rotation_min_score_delta
        if not self.portfolio.positions:
            return False

        weakest_sym = min(
            self.portfolio.positions,
            key=lambda s: abs(self._current_scores.get(s, 0.0)),
        )
        weakest_score = abs(self._current_scores.get(weakest_sym, 0.0))

        if abs(new_score) - weakest_score < rotation_min_delta:
            return False

        pos = self.portfolio.positions[weakest_sym]
        price = prices.get(weakest_sym, 0.0)
        if price <= 0:
            return False

        position_usd = pos["amount"] * price
        if position_usd < 1.0:
            # Ghost position (zero/dust amount) — clean it up instead of selling
            logger.warning("Removing ghost position %s (value=$%.4f)", weakest_sym, position_usd)
            del self.portfolio.positions[weakest_sym]
            return True
        logger.info(
            "ROTATION: closing %s (score=%.3f) for %s (score=%.3f, delta=%.3f)",
            weakest_sym, weakest_score, new_symbol, abs(new_score),
            abs(new_score) - weakest_score,
        )
        self._execute_sell(weakest_sym, position_usd, price,
                           f"rotation — replaced by {new_symbol}", prices=prices)
        return True

    def _execute_sell(self, symbol: str, position_usd: float, price: float, reason: str,
                      prices: dict | None = None, result: dict | None = None) -> None:
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
            self._partial_tp_taken.discard(symbol)
        logger.info("SELL %s: %.6f @ %.2f (%s)", symbol, order.amount, order.price, reason)
        self._notify("sell", symbol, order.amount, order.price, reason, prices=prices, result=result)

    def _notify_pdt_blocked(self, symbol: str, entry: float, price: float) -> None:
        """Alert Telegram when a protective exit is blocked by PDT budget exhaustion."""
        loss_pct = (entry - price) / entry * 100
        logger.warning(
            "PDT blocked exit for %s: entry=%.2f, current=%.2f (%.1f%% loss), holding overnight",
            symbol, entry, price, loss_pct,
        )
        if self._notifier:
            remaining = self._pdt.remaining() if self._pdt else 0
            label = "STOCK"
            msg = (
                f"[{label}] PDT BLOCKED EXIT — {symbol}\n"
                f"Stop-loss triggered but PDT budget exhausted ({remaining} day-trades left).\n"
                f"Entry: ${entry:.2f}  Current: ${price:.2f}  Loss: {loss_pct:.1f}%\n"
                f"Holding overnight — will reassess next session."
            )
            self._notifier.send(msg)
