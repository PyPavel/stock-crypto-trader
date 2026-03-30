# Adaptive Strategy Switching Design

**Date:** 2026-03-30
**Feature:** Automatic aggressive/moderate/conservative strategy switching based on market regime
**Status:** Approved — ready for implementation

---

## Overview

The trading engine currently uses a fixed strategy set in `config.yaml`. This feature adds a `RegimeDetector` that continuously reads market state and hot-swaps the active strategy, notifying via Telegram on every change.

---

## Architecture

### New component: `RegimeDetector`

**Location:** `src/trader/core/regime.py`

```python
class RegimeDetector:
    def __init__(self, numeric_collectors, sentiment_analyzer, persistence=3):
        ...
    def detect(self, symbols) -> str | None:
        # Returns "aggressive" | "moderate" | "conservative" | None (no change yet)
```

**Blended score:**
- 70% numeric collectors (Fear & Greed, CoinGecko, Polymarket, FundingRates, GoogleTrends)
- 30% LLM regime score (separate prompt: "Is the current market bullish, neutral, or bearish?")
- Result: float in `[-1.0, +1.0]`

**Thresholds:**
- `score >= +0.35` → aggressive
- `score <= -0.35` → conservative
- else → moderate

**Persistence (anti-thrash):**
- Must signal same regime for `N=3` consecutive cycles before switching
- Counter resets if signal changes mid-transition

### Changes to `TradingEngine`

- Accepts `regime_detector: RegimeDetector | None` in `__init__`
- `run_cycle()` calls `regime_detector.detect()` at the top of each cycle
- On regime change: hot-swap `self._strategy`, migrate `_signal_history`, log + send Telegram message
- Config `strategy` field becomes the *initial* strategy (still respected at startup)

### Strategy hot-swap

When strategy changes:
1. Swap `self._strategy` instance
2. Migrate `_signal_history` — keep existing symbol history, new strategy parameters apply going forward
3. On downgrade (aggressive→moderate or moderate→conservative): tighten stop-losses on the *next* sell signal (no forced liquidation)

---

## Data Flow

```
Each cycle:
  1. numeric_collectors.score(symbols) → [float | None, ...]
  2. llm.analyze(["Is market bullish/neutral/bearish?"]) → float
  3. blend(0.7 * numeric_avg + 0.3 * llm_score) → regime_score
  4. evaluate thresholds → target_regime
  5. increment persistence counter for target_regime
  6. if counter >= 3 and target_regime != current_strategy:
       → hot-swap strategy
       → send Telegram: "[CRYPTO] Strategy switched: moderate → aggressive\nRegime score: +0.52"
       → reset counter
```

---

## Error Handling

- **All numeric collectors return `None`**: `detect()` returns `None` — engine keeps current strategy, logs warning `"Regime detection failed — keeping current strategy"`
- **LLM call fails**: skip the 30% LLM weight, use numeric-only score normalized to `[-1,+1]`
- **Persistence counter mid-transition + collector recovers**: counter resets — avoids phantom switches from partial data
- **Downgrade (aggressive→conservative)**: engine reviews open positions against new risk rules; may tighten stop-losses or reduce sizes on next sell signal — no forced immediate liquidation

---

## Testing

**Unit tests for `RegimeDetector.detect()`:**
- Pure numeric score above/below thresholds with no LLM
- LLM blend shifts a borderline case across threshold
- Persistence requires 3 consecutive matching signals before switching
- `None` collectors → current strategy unchanged

**Integration tests in engine:**
- Mock collectors return bullish scores for 3 cycles → strategy switches to aggressive → Telegram called with correct message
- Mock collectors flip bearish → stays moderate until 3rd cycle
- LLM failure → falls back to numeric-only scoring, switch still happens correctly

---

## Configuration

No new config keys required. Initial strategy from `config.yaml` `strategy:` field is used at startup; `RegimeDetector` may override after 3 cycles.

Optional future: expose `persistence`, `thresholds`, and `llm_weight` in `config.yaml` under a `regime:` block — out of scope for this implementation.

---

## Files Changed

| File | Change |
|------|--------|
| `src/trader/core/regime.py` | New — `RegimeDetector` class |
| `src/trader/core/engine.py` | Accept `regime_detector`, call in `run_cycle()`, hot-swap logic |
| `src/trader/__main__.py` | Instantiate `RegimeDetector`, pass to engine |
| `tests/core/test_regime.py` | New — unit + integration tests |
