# Overnight Catalyst Collector + Time Window Gate

**Date:** 2026-04-29  
**Status:** Approved

## Problem

The stocks trader currently buys on intraday signals at any time during market hours. This causes:
- Intraday noise trades with no edge
- PDT violations (buy and sell same day)
- No exploitation of the overnight drift premium

## Solution

Two independent additions wired into the existing engine:

1. `OvernightCatalystCollector` — identifies stocks with a specific overnight catalyst
2. `TimeWindowGate` — restricts buys to 3–4pm ET and signal sells to 9:30–10:30am ET

## Component 1: OvernightCatalystCollector

**File:** `src/trader/collectors/overnight_catalyst.py`

**Interface:** `score(symbols: list[str]) -> float | None` — same as all numeric collectors.

**Data sources (both free, no API key):**

### Source A: yfinance earnings calendar

For each symbol, check if it reports earnings **tonight after market close** (AMC).

| Condition | Signal |
|-----------|--------|
| AMC tonight + historical beat rate ≥ 65% | `+0.55` to `+0.75` (scaled by beat rate) |
| AMC tonight + historical beat rate < 40% | `-0.25` (likely to disappoint) |
| AMC tonight + insufficient history | `+0.20` (mild positive, uncertain) |
| No AMC tonight | `None` (do not influence score) |

Beat rate computed from last 8 quarters via `yfinance.Ticker.earnings_dates` (reuses logic already in `EarningsCollector`).

### Source B: SEC EDGAR 8-K filings

Query `https://efts.sec.gov/LATEST/search-index?q="TICKER"&dateRange=custom&startdt=TODAY&enddt=TODAY&forms=8-K` for each symbol. Parse filing text for keyword tiers:

| Tier | Keywords | Signal |
|------|----------|--------|
| 1 | merger, acquisition, "FDA approval", "breakthrough designation", "going private" | `+0.45` |
| 2 | "major contract", "licensing agreement", partnership, "strategic alliance" | `+0.30` |
| No filing today | `None` |

### Blending

If a symbol has both an earnings signal and an 8-K signal:
- Average the two non-None scores
- Cap at `+0.80`

**Cache TTL:** 60 minutes per symbol (filings and earnings dates don't change intraday).

**Return value:** Average of all per-symbol scores that are non-None. Returns `None` if no symbols have a catalyst today (neutral — no influence on combined score).

## Component 2: TimeWindowGate

**File:** `src/trader/core/time_gate.py`

**Interface:**
```python
class TimeWindowGate:
    def can_buy(self) -> bool   # True between buy_start and buy_end ET
    def can_sell(self) -> bool  # True between sell_start and sell_end ET
```

All times resolved to US/Eastern automatically (handles DST). Windows are inclusive on both ends.

**Engine integration (`_execute_decisions`):**

After the strategy decision, before the PDT check:
```
if decision == "buy" and not time_gate.can_buy():
    → convert to HOLD, log "time gate: outside buy window"
if decision == "sell" and not time_gate.can_sell():
    → convert to HOLD, log "time gate: outside sell window"
```

**Protective exits are unaffected** — stop-loss, trailing stop, and take-profit checks happen earlier in `_execute_decisions`, above the strategy decision point. They fire at any time unconditionally.

## Config

### New dataclass in `src/trader/config.py`

```python
@dataclass
class TimeGateConfig:
    enabled: bool = False
    buy_start: str = "15:00"   # ET
    buy_end: str = "16:00"     # ET
    sell_start: str = "09:30"  # ET
    sell_end: str = "10:30"    # ET
```

Added as `time_gate: TimeGateConfig` field on the top-level `Config` dataclass.

### `config-stocks.yaml` addition

```yaml
time_gate:
  enabled: true
  buy_start: "15:00"
  buy_end: "16:00"
  sell_start: "09:30"
  sell_end: "10:30"
```

### Crypto unaffected

`config.yaml` gets no `time_gate` block — defaults to `enabled: false`.

## Wiring in `__main__.py` (stocks branch)

1. Instantiate `OvernightCatalystCollector`, append to `numeric_collectors` list
2. Read `config.time_gate`; if `enabled`, instantiate `TimeWindowGate` and pass to `TradingEngine`
3. `TradingEngine.__init__` accepts optional `time_gate: TimeWindowGate | None = None`

## Files Changed

| File | Change |
|------|--------|
| `src/trader/collectors/overnight_catalyst.py` | New file |
| `src/trader/core/time_gate.py` | New file |
| `src/trader/config.py` | Add `TimeGateConfig`, add `time_gate` field to `Config` |
| `src/trader/core/engine.py` | Accept `time_gate` in `__init__`, add 2 guards in `_execute_decisions` |
| `src/trader/__main__.py` | Wire collector + gate in stocks branch |
| `config-stocks.yaml` | Add `time_gate` block |
| `tests/collectors/test_overnight_catalyst.py` | New test file |
| `tests/core/test_time_gate.py` | New test file |

## Testing

**`test_overnight_catalyst.py`** — mock yfinance and EDGAR responses:
- AMC earnings + high beat rate → positive score
- AMC earnings + low beat rate → negative score
- Tier-1 8-K keyword → `+0.45`
- No catalyst → `None`
- Both sources → averaged, capped at `+0.80`

**`test_time_gate.py`** — mock `datetime.now`:
- Inside buy window → `can_buy()` True
- Outside buy window → `can_buy()` False
- Inside sell window → `can_sell()` True
- Outside sell window → `can_sell()` False
- `enabled: false` → both always True

## Out of Scope

- Paid analyst upgrade feeds (Benzinga, Seeking Alpha) — future work
- Pre-market price action signal — future work
- Short interest signal — future work
