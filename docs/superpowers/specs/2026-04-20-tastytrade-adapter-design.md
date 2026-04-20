# TastyTrade Adapter Design

**Date:** 2026-04-20  
**Status:** Approved

## Problem

Alpaca's PDT (Pattern Day Trader) rule blocks accounts under $25k from making more than 3 day-trades in a rolling 5-day window. This can leave positions stuck. TastyTrade cash accounts are exempt from PDT, making it a viable parallel broker for stocks.

## Goals

- Add `TastyTradeAdapter` as a first-class stock broker alongside Alpaca
- Support TastyTrade paper (sandbox) mode for initial evaluation
- Allow running Alpaca and TastyTrade simultaneously as separate processes
- No changes to existing Alpaca or Coinbase adapters

## Architecture

### Multi-instance model

Each broker runs as an independent process with its own config file:

```
python -m trader --config config-alpaca.yaml    --port 8000 --db alpaca.db
python -m trader --config config-tastytrade.yaml --port 8001 --db tastytrade.db
```

`exchange` remains a single string per config. This matches the existing crypto/stocks split pattern.

### New file: `src/trader/adapters/tastytrade.py`

Implements `ExchangeAdapter` fully. Internally holds two clients:

- `tastytrade.Session` + `tastytrade.Account` — order execution, balances, positions
- Alpaca `StockHistoricalDataClient` — `get_candles` and `get_price` (free IEX feed)

The Alpaca data dependency is an implementation detail hidden inside the adapter. The `alpaca` config section is required alongside `tastytrade` for data credentials.

Paper mode uses TastyTrade's official sandbox (`is_test_env=True`, hits `api.cert.tastyworks.com`).

`is_market_open()` duplicates the NYSE hours guard from `AlpacaAdapter`.

### Methods

| Method | Implementation |
|---|---|
| `get_candles` | Alpaca `StockHistoricalDataClient` (identical to AlpacaAdapter) |
| `get_price` | Alpaca `StockLatestQuoteRequest` (identical to AlpacaAdapter) |
| `get_balance` | `account.get_balances(session)` → returns `USD`, `equity`, `buying_power` |
| `place_order` | `NewOrder` with `NewOrderLeg` for equity market orders |
| `get_open_orders` | `account.get_live_orders(session)` filtered by symbol |
| `cancel_order` | `Order.cancel(session)` |

### Order placement notes

- TastyTrade uses fractional shares for equities
- Buy orders: convert USD notional → shares using current price (same pattern as Alpaca)
- Sell orders: fetch actual position qty from TastyTrade before placing (same defensive pattern as Alpaca)
- Poll for fill after submit (TastyTrade orders may not fill instantly)

## Config changes

### New dataclass in `config.py`

```python
@dataclass
class TastyTradeConfig:
    username: str = ""
    password: str = ""
    account_number: str = ""   # picks first account if empty
    paper: bool = True
```

Added to `Config` dataclass alongside existing `alpaca`.

### Environment variable overrides

| Env var | Field |
|---|---|
| `TASTYTRADE_USERNAME` | `cfg.tastytrade.username` |
| `TASTYTRADE_PASSWORD` | `cfg.tastytrade.password` |
| `TASTYTRADE_ACCOUNT_NUMBER` | `cfg.tastytrade.account_number` |
| `TASTYTRADE_PAPER` | `cfg.tastytrade.paper` |

### Example `config-tastytrade.yaml`

```yaml
exchange: tastytrade
mode: paper
strategy: moderate
capital: 5000
pairs: [AAPL, TSLA, NVDA, MSFT, GOOGL]

tastytrade:
  paper: true  # overridden by TASTYTRADE_PAPER env var

alpaca:        # required for market data (free IEX feed)
  paper: true  # irrelevant for data, but field is required
```

## `__main__.py` wiring

Add `elif cfg.exchange == "tastytrade":` branch with:
- Same collectors as `alpaca` branch (StockNewsCollector, StockTwitsCollector, etc.)
- `TastyTradeAdapter(tastytrade_cfg=cfg.tastytrade, alpaca_data_key=cfg.alpaca.api_key, alpaca_data_secret=cfg.alpaca.api_secret)`
- Same `is_market_open` guard via `hasattr(adapter, "is_market_open")`

## Dependencies

Add `tastytrade` to `requirements.txt`. No other new dependencies.

## Testing approach

1. Run with `exchange: tastytrade`, `tastytrade.paper: true` pointing at TastyTrade sandbox
2. Verify candles/prices come back correctly (from Alpaca data)
3. Verify paper orders submit and fill in TastyTrade sandbox
4. Run both `config-alpaca.yaml` and `config-tastytrade.yaml` simultaneously and confirm independent operation

## Out of scope

- Automatic PDT failover (switching brokers mid-session)
- Shared capital/position tracking across brokers
- TastyTrade options or futures trading
