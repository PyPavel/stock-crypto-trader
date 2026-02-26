# Trading Bot Design

**Date:** 2026-02-26
**Status:** Approved

## Overview

A modular monolith trading bot that starts with crypto (Coinbase Advanced) and is architected to support stocks later. Uses a local LLM (Ollama/Mistral) for real-time sentiment analysis and an optional big LLM (Claude/GPT) as a manual strategy advisor. Supports paper trading and live trading via a config toggle.

---

## Architecture

Single Python app with clean module boundaries, Docker-deployed:

```
trader/
├── adapters/          # Exchange adapters (Coinbase via CCXT, future: Alpaca, Kraken)
├── strategies/        # Strategy plugins (conservative, moderate, aggressive)
├── llm/               # LLM module (Ollama local + optional Claude/GPT advisor)
├── collectors/        # Data collectors (Reddit PRAW, CryptoPanic API)
├── core/              # Engine, scheduler, signal generator, risk manager, order router
├── portfolio/         # Portfolio state, SQLite persistence
├── dashboard/         # FastAPI REST API + simple web UI
├── config.yaml        # Single config file
└── docker-compose.yml
```

### Exchange Adapter Interface

All exchanges implement this interface, making it trivial to add new ones:

```python
class ExchangeAdapter:
    def get_candles(symbol, interval) -> List[Candle]
    def get_balance() -> Dict
    def place_order(side, symbol, amount) -> Order
    def get_open_orders() -> List[Order]
```

- **Now:** Coinbase Advanced via CCXT
- **Later:** Alpaca (stocks), Kraken, Binance — each is a new adapter file, zero core changes

---

## Data Flow

Every N seconds (configurable, default 60s):

```
1. Exchange Adapter   → fetch OHLCV candles + current price
2. Data Collectors    → fetch Reddit posts (PRAW) + CryptoPanic news
                        Ollama/Mistral scores each item → aggregate sentiment score
3. Signal Generator   → run technical indicators: RSI, MACD, Bollinger Bands
4. Strategy Plugin    → combine signals + sentiment → trade decision (buy/sell/hold + size)
5. Risk Manager       → validate against position limits, max drawdown, stop-loss rules
6. Order Router       → paper mode: log simulated trade / live mode: call exchange API
7. Portfolio State    → update balances, positions, P&L in SQLite
8. Dashboard          → serve current state via FastAPI REST + web UI
```

**LLM Advisor** (Claude/GPT) is outside the cycle — called manually via dashboard or on a schedule (e.g. weekly) to review performance and suggest strategy adjustments.

---

## LLM Usage

| Task | Model | Frequency |
|------|-------|-----------|
| News/Reddit sentiment scoring | Ollama (Mistral 7B) | Every cycle (real-time) |
| Trade narrative logging | Ollama (Mistral 7B) | Every trade |
| Strategy performance review | Claude / GPT | Manual or weekly |
| Strategy adjustment suggestions | Claude / GPT | Manual or weekly |

**Principle:** anything running on every tick uses local LLM. High-stakes, infrequent decisions use big LLM.

---

## Sentiment Sources

- **CryptoPanic API** (free tier) — aggregates crypto news + social signals
- **Reddit via PRAW** (free) — r/cryptocurrency + coin-specific subreddits (r/bitcoin, r/ethtrader, etc.)
- **Twitter/X** — excluded (API costs $100+/month)

Sentiment score (bullish/bearish/neutral per item) is aggregated into a single weighted score per trading pair, contributing ~30% weight in the moderate strategy.

---

## Strategies

Three built-in strategy plugins, selectable via config:

| Strategy | Position Size | Stop-Loss | Sentiment Weight | Trade Frequency |
|----------|--------------|-----------|-----------------|-----------------|
| Conservative | Small | Tight | Low | Low |
| **Moderate (default)** | **Medium** | **Balanced** | **~30%** | **Medium** |
| Aggressive | Large | Wide | High | High |

Strategies are plugins — new ones can be added without touching core.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Exchange integration | CCXT (unified crypto library) |
| Scheduler | APScheduler |
| Technical indicators | pandas-ta |
| Local LLM | Ollama Python SDK (Mistral 7B) |
| Reddit | PRAW |
| News | CryptoPanic REST API |
| Persistence | SQLite |
| Dashboard API | FastAPI |
| Containerization | Docker Compose |

---

## Configuration

Single `config.yaml` controls everything:

```yaml
exchange: coinbase
mode: paper           # paper | live
strategy: moderate    # conservative | moderate | aggressive
capital: 100          # starting USD
pairs:
  - BTC-USD
  - ETH-USD
cycle_interval: 60    # seconds
ollama_model: mistral
llm_advisor:
  enabled: false      # enable for Claude/GPT advisor
  provider: claude    # claude | openai
```

---

## Error Handling

| Failure | Behavior |
|---------|---------|
| Exchange API failure | Retry with exponential backoff; skip cycle if unresolvable |
| Ollama unavailable | Trade on indicators only (no sentiment); log warning |
| Unexpected position state | Halt trading; alert via dashboard; require manual review |
| General errors | Log to SQLite for post-mortem analysis |

---

## Testing Strategy

- **Unit tests** — strategy plugins, risk manager, indicator calculations, adapter interface
- **Adapter mocks** — deterministic tests without hitting real APIs
- **Backtesting module** — feed historical OHLCV data through engine, measure P&L / win rate / max drawdown
- **Paper trading** — primary integration test against real market data with simulated orders

### Go-Live Workflow

```
1. Backtest strategy on historical data
2. Run paper trading for 1-2 weeks minimum
3. Review results in dashboard (optionally run LLM Advisor analysis)
4. Switch config: mode: paper → mode: live
```
