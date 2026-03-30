# Dynamic Symbol Universe — Design Spec

**Date:** 2026-03-30
**Status:** Approved

## Problem

Both the crypto and stocks bots trade a hardcoded list of symbols (`pairs` in config). This means the bot misses opportunities in symbols not on the list and cannot react to market-wide momentum shifts.

## Goal

Replace the static `pairs` list with a two-stage dynamic funnel that discovers symbols with active market signals, while preserving backward compatibility via a config flag.

---

## Architecture

### Two-Stage Funnel

```
Universe (top 200 crypto / top 1000 stocks)  — refreshed every 24h
         ↓  momentum filter (price_change × volume_ratio)
Candidates (top 50)                           — scored every cycle
         ↓  full signal scoring (technical + sentiment + numeric collectors)
Active pairs (top 30)                         — traded this cycle
```

### New Class: `SymbolUniverse`

**Location:** `src/trader/core/universe.py`

Owns the full funnel. Three responsibilities:

1. **Universe refresh** (every 24h): fetch the broad symbol pool from exchange APIs
2. **Momentum scoring** (every cycle): cheap single-metric filter to narrow 1000 → 50
3. **Candidate exposure**: `get_candidates() → list[str]` — called by `TradingEngine`

#### Data sources

| Exchange | Universe source | Momentum source |
|----------|----------------|-----------------|
| Coinbase (crypto) | CoinGecko `/coins/markets` — top 200 by market cap | Same response (24h change % + volume ratio already included, no extra call) |
| Alpaca (stocks) | Alpaca `/v2/assets?status=active` filtered by Alpaca `/v2/screener/stocks/most_actives` | Alpaca `most_actives` + `top_movers` endpoints — no per-symbol candle fetch |

#### Momentum score formula

```
momentum_score = price_change_24h_pct × volume_ratio
volume_ratio   = current_24h_volume / avg_24h_volume (from universe data)
```

Top 50 by `abs(momentum_score)` become candidates.

#### Interface

```python
class SymbolUniverse:
    def __init__(self, adapter: ExchangeAdapter, exchange: str, seed_pairs: list[str])
    def get_candidates(self) -> list[str]    # top 50 after momentum filter
    def refresh_universe(self) -> None       # called by scheduler every 24h
```

`seed_pairs` (the existing `config.pairs`) are always included in the candidate pool regardless of momentum rank — preserves behavior for open positions.

---

## Changes to `TradingEngine`

### `run_cycle()` — before

```python
for symbol in self.config.pairs:
    self._process_symbol(symbol, prices)
```

### `run_cycle()` — after

```python
candidates = self._universe.get_candidates()   # up to 50 symbols
scored = []
for symbol in candidates:
    result = self._score_symbol(symbol, prices)
    if result is not None:
        scored.append((symbol, result))

scored.sort(key=lambda x: abs(x[1]["score"]), reverse=True)
for symbol, result in scored[:30]:
    self._execute_decisions(symbol, result, prices)
```

### `_process_symbol` split

Current `_process_symbol` is split into two focused methods:

- `_score_symbol(symbol, prices) → dict | None` — fetches candles + price, runs technical/sentiment/ML scoring, returns score dict. No side effects.
- `_execute_decisions(symbol, result, prices)` — runs strategy decision + risk checks + order execution. Existing logic unchanged.

This split also makes unit testing cleaner: score logic can be tested without triggering orders.

---

## Config Changes

New optional `universe` block in `config.yaml` / `config-stocks.yaml`:

```yaml
universe:
  enabled: true      # false = use pairs list as-is (current behavior, default)
  size: 200          # universe pool size (crypto: 200, stocks: 1000)
  candidates: 50     # symbols surviving momentum filter
  active_pairs: 30   # symbols actually traded per cycle (top by signal strength)
```

`pairs` remains valid. If `universe.enabled: false` (or key absent), behavior is identical to today — `pairs` list is used directly.

---

## Error Handling

| Failure | Behavior |
|---------|----------|
| Universe API down at refresh | Keep previous universe list, log warning, do not block trading |
| Momentum scoring fails for a symbol | Skip symbol silently, exclude from candidates |
| Candidate list empty after filter | Fall back to `config.pairs` as candidate list |
| All full-signal scores are zero | Trade top 30 by momentum score instead |

---

## `__main__.py` Changes

- On startup: instantiate `SymbolUniverse`, pass to `TradingEngine`
- Add a 24h scheduler job: `universe.refresh_universe()`
- Universe refresh runs immediately on startup (no 24h wait on first boot)

---

## Testing

- `SymbolUniverse` tested in isolation with mocked adapter calls
- `_score_symbol` / `_execute_decisions` split enables engine tests without order execution
- All existing tests pass unchanged (`universe.enabled` defaults to `false` in test fixtures)
- New tests cover: universe refresh, momentum scoring, candidate selection, seed_pairs always included, fallback to config.pairs when universe is empty

---

## Out of Scope

- Per-symbol position sizing based on universe rank (future)
- Crypto-specific universe from Coinbase's own asset listing (CoinGecko is sufficient)
- Backtesting the funnel (separate initiative)
