# Trader Agent — Improvement Plan

Generated: 2026-04-01  
Scope: Crypto (Coinbase, aggressive) + Stocks (Alpaca, moderate)  
Current performance: Crypto +0.32% | Stocks +2.17% | Starting capital $800 each

---

## TIER 1 — Fix Silent Failures (Zero-risk, immediate wins)

### 1. Re-wire CryptoPanic collector
- **Domain:** Signal & Data Sources
- **What:** Fully implemented but silently dropped from `__main__.py`. Add it back to the collector list.
- **Why:** Free news signal, no API key required, zero cost.
- **File:** `src/trader/__main__.py`

### 2. Fix aggressive strategy position sizing conflict
- **Domain:** Strategy & Risk Logic
- **What:** `aggressive.py` uses `max_position_pct * 2` (40%) but risk manager validates against 20%, silently capping every buy. Remove the `*2` multiplier or add an explicit `aggressive_max_position_pct` config field.
- **Why:** Every aggressive buy silently fails the size check and gets capped — the strategy is not operating as intended.
- **Files:** `src/trader/strategies/aggressive.py`, `src/trader/config.py`

### 3. Enable daily loss circuit breaker
- **Domain:** Strategy & Risk Logic
- **What:** `max_daily_loss_pct=0` is disabled. Set to 3–5% of starting capital.
- **Why:** No intraday halt means the bot keeps trading on bad days until the 15% portfolio drawdown limit is hit.
- **File:** `config.yaml`, `config-stocks.yaml`

### 4. Enable crypto ML predictor
- **Domain:** ML & LLM Integration
- **What:** `ml.enabled: false` in `config.yaml` but `models/crypto.lgbm` exists and is ready. One-line config change.
- **Why:** ML is already enabled for stocks and the model file exists — crypto is missing free signal.
- **File:** `config.yaml`

### 5. Verify MIMO API base URL
- **Domain:** ML & LLM Integration
- **What:** `sentiment.py` uses `xiaomimimo.com` but project context references `xiaomimiao.com` — different domains. Add a startup health-check ping to surface silent failures.
- **Why:** If the URL is wrong, all LLM sentiment calls silently return `None` and fall back to 0.0 (neutral).
- **File:** `src/trader/llm/sentiment.py`

### 6. Expand RSS and Reddit coin keyword maps
- **Domain:** Signal & Data Sources
- **What:** Only 4 of 15 crypto pairs have coin-specific keywords. Add entries for XRP, DOGE, AVAX, LINK, NEAR, HBAR, UNI, POL, LTC, DOT, SHIB to both `rss.py` and `reddit.py`.
- **Why:** 11 altcoins receive undifferentiated generic crypto headlines, diluting LLM sentiment quality.
- **Files:** `src/trader/collectors/rss.py`, `src/trader/collectors/reddit.py`

---

## TIER 2 — High Impact, Low Complexity

### 7. Enable LLM sentiment
- **Domain:** ML & LLM Integration + Signal & Data Sources
- **What:** Add MIMO API key to `.env`. Infrastructure is fully built and ready.
- **Why:** Sentiment carries 40% weight in combined score. When disabled it falls back to the numeric average of other collectors — enabling it is the single highest-leverage improvement available.
- **File:** `.env` (Vultr server)

### 8. Fix ML as blend, not override
- **Domain:** ML & LLM Integration
- **What:** When `ml_score` is available, it fully replaces `tech_score`. Change to `final = ml_score * 0.7 + tech_score * 0.3`. Make ratio configurable in `MLConfig`.
- **Why:** Completely discarding the technical score on ML availability is lossy — if the model drifts or encounters a regime shift, there is no fallback signal.
- **Files:** `src/trader/core/engine.py`, `src/trader/config.py`

### 9. Enable ATR-adaptive stops for crypto
- **Domain:** Strategy & Risk Logic
- **What:** `use_atr_stops=False` by default. Enable for crypto bot with starting multipliers: 2.5× ATR stop, 4.0× ATR trailing.
- **Why:** Fixed 5%/8% stops are too tight for volatile crypto — observed constant stop-outs on ADA, LINK (trailing), SHIB, HBAR, NEAR (persistent sell pattern).
- **Files:** `config.yaml`, `src/trader/core/risk.py`

### 10. Raise aggressive entry threshold
- **Domain:** Strategy & Risk Logic
- **What:** `BUY_THRESHOLD=0.10` is near noise level. Raise to 0.20–0.25.
- **Why:** Near-zero threshold causes positions to fill up immediately on any faint signal, triggering the "max open positions reached" spam and blocking better opportunities.
- **File:** `src/trader/strategies/aggressive.py`

### 11. Replace stock Market Sentiment signal
- **Domain:** Signal & Data Sources
- **What:** `MarketSentimentCollector` for stocks is a thin alias of the crypto alternative.me Fear & Greed index. Replace with a VIX-based signal from Yahoo Finance (free) or CNN Fear & Greed Index.
- **Why:** The crypto Fear & Greed index is semantically incorrect as an equity sentiment indicator.
- **Files:** `src/trader/collectors/market_sentiment.py`, `config-stocks.yaml`

### 12. Fix momentum scoring formula
- **Domain:** Portfolio & Universe Selection
- **What:** Replace `price_change × raw_volume` with `price_change × volume_percentile_rank` (0–1 normalized within the universe).
- **Why:** Raw volume is scale-biased — BTC's volume in USD dwarfs all altcoins, making the formula effectively re-rank by market cap rather than momentum quality.
- **File:** `src/trader/core/universe.py`

### 13. Fix stocks universe capacity
- **Domain:** Portfolio & Universe Selection
- **What:** Alpaca `/movers` endpoint returns max 100 symbols despite `universe.size=1000` in config. Switch to `/screener/stocks/most_actives` (supports `top=500`) with real dollar volume.
- **Why:** Stock universe is 10× smaller than intended and the volume proxy (`abs(pct_change)`) is meaningless — it just squares the price change.
- **File:** `src/trader/core/universe.py`

---

## TIER 3 — Structural Improvements

### 14. Implement position rotation
- **Domain:** Strategy & Portfolio
- **What:** When `max_open_positions` is full and a new signal arrives with score > weakest current position score + 0.20 (configurable), close the weakest position and open the stronger one.
- **Why:** Currently all buys are hard-blocked when full. High-conviction signals are silently discarded — this is the single biggest alpha-generation gap.
- **Files:** `src/trader/core/engine.py`, `src/trader/core/risk.py`

### 15. Enable correlation groups for crypto by default
- **Domain:** Strategy & Portfolio
- **What:** Pre-populate `correlation_groups` in config with sensible defaults:
  - `eth_ecosystem: [ETH, UNI, POL, LINK]`
  - `layer1: [SOL, AVAX, DOT, NEAR, HBAR]`
  - `meme: [SHIB, DOGE]`
  - Cap each group at 2 simultaneous positions.
- **Why:** Live positions included ETH + UNI + POL (all Ethereum ecosystem) simultaneously — a market rotation out of ETH would stop them all at once, amplifying drawdown.
- **Files:** `config.yaml`, `src/trader/core/risk.py`

### 16. Add minimum cash reserve guard
- **Domain:** Strategy & Risk Logic
- **What:** Add `min_cash_reserve_pct=0.10` to `RiskConfig`. Always keep 10% of capital uninvested.
- **Why:** 5 positions × 20% = 100% deployment with zero buffer. One unexpected event with no cash to respond.
- **Files:** `src/trader/config.py`, `src/trader/core/risk.py`

### 17. Enable take-profit for crypto
- **Domain:** Strategy & Risk Logic
- **What:** Enable `take_profit_pct=0.20` with partial take-profit: sell 50% at 10%, let remainder run with trailing stop.
- **Why:** Currently profits are only captured when the trailing stop fires after a peak-to-trough drop of 8%. On a 12% rally, an 8% pullback captures only ~4% gain. Partial TP locks in gains on big moves.
- **Files:** `config.yaml`, `src/trader/core/risk.py`

### 18. Signal-strength-proportional position sizing
- **Domain:** Portfolio & Universe Selection
- **What:** Introduce two allocation tiers: high-conviction (score ≥ 0.50) gets `max_position_pct × 1.25`, standard gets `max_position_pct × 1.0`. Or implement full Kelly-fraction sizing based on historical win rate per symbol.
- **Why:** A score of 0.11 (barely above threshold) gets the same dollar allocation as a score of 0.95. No differentiation by conviction.
- **Files:** `src/trader/core/risk.py`, `src/trader/config.py`

### 19. Persist signal history across restarts
- **Domain:** Strategy & Risk Logic
- **What:** Write `_signal_history` deque to the portfolio DB and reload on startup.
- **Why:** Signal history is in-memory only. Every restart bypasses the persistence check for 2+ cycles, causing the bot to fire unconfirmed buys immediately after any restart.
- **Files:** `src/trader/strategies/base.py`, `src/trader/portfolio/db.py`

### 20. Fix backtest to include ML and sentiment
- **Domain:** ML & Portfolio
- **What:** Extend `Backtester` to accept an optional `MLPredictor` and use the same scoring logic as the live engine. Replace hardcoded `sentiment=0.0` with either historical sentiment replay or a configurable offset.
- **Why:** Sweep optimizes on zero-sentiment pure-technical signals, but live trading uses 30–40% sentiment weight. Optimized thresholds from sweep are calibrated on a different signal distribution than live.
- **Files:** `src/trader/core/backtest.py`, `src/trader/core/sweep.py`

---

## TIER 4 — New Data Sources

### 21. Replace Reddit with StockTwits
- **Domain:** Signal & Data Sources
- **What:** Reddit API keys are unconfigured — zero social signal for every cycle. StockTwits has a free public API (no auth required for public streams), covers both crypto and stocks.
- **Why:** Reddit subreddit map also only covers 4 of 15 crypto pairs even when configured. StockTwits is higher signal-to-noise for short-term momentum.
- **File:** `src/trader/collectors/reddit.py` (replace or add alongside)

### 22. Add VIX and macro regime signals
- **Domain:** Signal & Data Sources
- **What:** Add a new collector fetching DXY (dollar index), 10Y yield, and VIX via Yahoo Finance (free, no auth).
- **Why:** Crypto has strong inverse DXY correlation. Stocks need VIX for risk-regime detection. Both are freely available and address a major signal gap. Currently zero macro context in either bot.
- **File:** new `src/trader/collectors/macro.py`

### 23. Add earnings calendar for stocks
- **Domain:** Signal & Data Sources
- **What:** Fetch upcoming earnings dates via Alpaca or Polygon free tier. Avoid opening new positions 3–5 days before earnings (IV crush risk), or implement an intentional earnings-volatility strategy.
- **Why:** Entirely absent from signal stack. Buying before earnings is one of the most common retail trading mistakes.
- **File:** new `src/trader/collectors/earnings.py`

### 24. Add on-chain data for BTC and ETH
- **Domain:** Signal & Data Sources
- **What:** Add collector for exchange net flows, active addresses, TVL trends via DeFiLlama and Blockchain.info (both free, no auth).
- **Why:** On-chain metrics are leading indicators that precede price movement by 1–3 days. Large exchange inflows signal sell pressure; outflows signal accumulation.
- **File:** new `src/trader/collectors/onchain.py`

### 25. Add CoinGlass liquidation heatmap
- **Domain:** Signal & Data Sources
- **What:** Add collector for the CoinGlass free public API — cumulative liquidation levels by price.
- **Why:** Liquidation levels show where leveraged positions will be force-closed, predicting price acceleration zones. Strong intraday signal for crypto momentum.
- **File:** new `src/trader/collectors/liquidations.py`

### 26. Use BTC dominance signal (already fetched, discarded)
- **Domain:** Signal & Data Sources
- **What:** CoinGecko collector already fetches BTC dominance but the value is discarded. Wire it as a separate numeric signal.
- **Why:** Rising BTC dominance = capital rotating out of altcoins = bearish signal for non-BTC pairs. Zero-cost improvement — data is already in the payload.
- **File:** `src/trader/collectors/coingecko.py`

### 27. Add Deribit options IV and skew for BTC/ETH
- **Domain:** Signal & Data Sources
- **What:** Add collector for Deribit free public API — put/call ratio and implied volatility skew for BTC and ETH.
- **Why:** Options market leads spot price. IV skew and P/C ratio provide institutional sentiment that retail Reddit/RSS sources cannot.
- **File:** new `src/trader/collectors/deribit.py`

---

## TIER 5 — Quality and Calibration

### 28. Fix RSI formula
- **Domain:** Signal & Data Sources
- **What:** Both oversold and overbought branches use the same formula `(70 - rsi) / 40`. Works accidentally but magnitude is wrong at extremes (RSI=0 gives +1.75, RSI=100 gives −0.75 before clamp). Write explicit separate formulas for each branch.
- **File:** `src/trader/core/signals.py`

### 29. Increase volume signal weight and add MFI
- **Domain:** Signal & Data Sources
- **What:** Raise volume weight from 0.05 to 0.10–0.15 (reduce one oscillator weight to compensate). Add Money Flow Index (MFI = price × volume momentum).
- **Why:** Volume is the most reliable confirmation signal in technical analysis. At 0.05 it is nearly decorative.
- **File:** `src/trader/core/signals.py`

### 30. Add multi-timeframe technical analysis
- **Domain:** Signal & Data Sources
- **What:** Add 4h and daily EMA trend context alongside the existing 1h signals. Engine already calls `get_candles()` — just needs additional timeframe calls.
- **Why:** All signals currently computed on 1h candles only. Daily trend context reduces false signals in choppy 1h periods.
- **Files:** `src/trader/core/signals.py`, `src/trader/core/engine.py`

### 31. Add EMA crossover signal (9/21 EMA)
- **Domain:** Signal & Data Sources
- **What:** Add 9/21 EMA crossover as a new indicator alongside the existing 50 EMA trend check.
- **Why:** Current EMA signal only checks price-vs-50 EMA for broad trend direction. 9/21 crossover is a classic entry-timing signal with much higher precision.
- **File:** `src/trader/core/signals.py`

### 32. Improve LLM sentiment granularity
- **Domain:** ML & LLM Integration
- **What:** Replace 3-class (bullish/bearish/neutral) prompt with numeric −1.0 to +1.0 scale. Reduce `CHUNK_SIZE` from 10 to 5. Add confidence threshold: if LLM self-reported confidence < 0.5, treat as neutral.
- **Why:** 3-class classification loses continuous signal information. Processing 10 headlines per call loses per-headline granularity.
- **File:** `src/trader/llm/sentiment.py`

### 33. Per-symbol funding rate signals
- **Domain:** Signal & Data Sources
- **What:** Current `FundingRateCollector` averages funding rates across all requested symbols. Score per-symbol and return individual values.
- **Why:** Averaging across symbols cancels out strong directional signals — e.g., BTC positive funding masked by ETH negative funding.
- **File:** `src/trader/collectors/funding_rates.py`

### 34. Add market regime detection
- **Domain:** Strategy & Risk Logic
- **What:** Add ADX-based or VIX-based regime classifier (trending / ranging / volatile) that adjusts entry thresholds and stop distances dynamically per regime.
- **Why:** Same parameters applied in bull markets, bear markets, and sideways chop produces inconsistent results. In a trending market trailing stops should be wider; in ranging markets entry thresholds should be higher.
- **Files:** `src/trader/core/engine.py`, `src/trader/core/risk.py`

### 35. Implement rebalancing and drift correction
- **Domain:** Portfolio & Universe Selection
- **What:** When a position's portfolio weight exceeds `max_position_pct × 1.5` due to price appreciation, trim back toward the target weight.
- **Why:** No rebalancing means a position that doubles in value takes up 2× its intended weight with no mechanism to correct it.
- **File:** `src/trader/core/engine.py`

### 36. Add ML model staleness detector
- **Domain:** ML & LLM Integration
- **What:** Compute rolling 30-day correlation between `ml_score` and realized next-period returns. If correlation drops below 0.0, auto-disable ML and alert via Telegram.
- **Why:** Models trained on historical regimes degrade when market conditions change. No staleness detection means the bot may trade on a worse-than-random signal for weeks.
- **Files:** `src/trader/ml/predictor.py`, `src/trader/notifications/telegram.py`

### 37. Persist sweep results and implement rolling walk-forward
- **Domain:** Portfolio & Universe Selection
- **What:** Save `parameter_sweep()` results to the portfolio DB (`sweeps` table). Replace the single 70/30 train/test split in `walk_forward()` with rolling windows and an embargo period (5% of window size) between train and test.
- **Why:** Sweep results are lost on crash. Single split is susceptible to regime dependency. No embargo risks lookahead bias at the boundary.
- **Files:** `src/trader/core/sweep.py`, `src/trader/portfolio/db.py`

---

## Open Questions

1. **MIMO API key** — is it configured in Vultr's `.env`? If `api_key: ''` is truly empty, item #7 (LLM sentiment) is blocked.
2. **Reddit vs StockTwits** — keep Reddit (requires setting up API keys + expanding subreddit map) or switch to StockTwits entirely?
3. **ATR stop multipliers (item #9)** — recommend starting with 2.5× stop / 4.0× trailing for crypto and measuring for one week before tuning.
4. **Rotation score delta (item #14)** — suggest 0.20 as the threshold to trigger position rotation. Needs tuning based on observed score distributions.
