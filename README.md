# Stock & Crypto Trader

An automated trading bot for cryptocurrency (Coinbase Advanced Trade) and US stocks (Alpaca), featuring multi-source sentiment analysis, ML-based signal scoring, and a built-in web dashboard.

> **Disclaimer:** This software is for educational and experimental purposes. Trading carries significant financial risk. Paper trading is enabled by default. Never risk money you cannot afford to lose.

---

## Features

- **Dual-market support** — Coinbase Advanced Trade (crypto) and Alpaca (stocks) via a unified adapter interface
- **Dynamic symbol universe** — Scans top 200 assets, narrows to 50 candidates, trades the best 30
- **Multi-source sentiment** — Aggregates signals from Reddit, RSS feeds, Fear & Greed Index, CoinGecko, CryptoPanic, Polymarket, Unusual Whales options flow, and Google Trends
- **LLM sentiment scoring** — Optional LLM-based analysis via Claude or Xiaomi MIMO
- **ML signal scoring** — LightGBM classifiers trained on 2 years of historical OHLCV data with technical indicators
- **Three strategies** — Conservative, Moderate, Aggressive (auto-switchable based on market conditions)
- **Risk management** — Stop-loss, trailing stops, max drawdown circuit breaker, position limits, per-symbol cooldowns
- **Paper & live modes** — Paper trading by default; flip one config flag to go live
- **Telegram notifications** — Trade alerts and daily summaries
- **Web dashboard** — FastAPI dashboard with health endpoint at `http://localhost:8000`
- **Docker-ready** — `docker-compose up` runs both crypto and stocks services

---

## Architecture

```
src/trader/
├── adapters/        # Exchange adapters (Coinbase, Alpaca)
├── collectors/      # Sentiment signal collectors
├── core/            # Trading engine, signals, risk, universe, backtesting
├── strategies/      # Conservative / Moderate / Aggressive strategies
├── llm/             # LLM-based sentiment analysis
├── ml/              # LightGBM scoring
├── notifications/   # Telegram notifier
├── portfolio/       # Portfolio state & SQLite database
└── dashboard/       # FastAPI web UI
```

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/PyPavel/stock-crypto-trader.git
cd stock-crypto-trader

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API credentials
```

Edit `config.yaml` (crypto) or `config-stocks.yaml` (stocks) to adjust trading pairs, risk parameters, and strategy.

### 3. Run in paper mode (default)

```bash
# Crypto bot
python -m trader --config config.yaml

# Stocks bot
python -m trader --config config-stocks.yaml
```

The dashboard is available at `http://localhost:8000`.

---

## Docker

```bash
# Run both services
docker-compose up -d

# Crypto dashboard: http://localhost:8000
# Stocks dashboard: http://localhost:8001
```

---

## Configuration

All credentials are read from environment variables (`.env`). YAML config files hold non-secret parameters.

### `.env` variables

| Variable | Required | Description |
|----------|----------|-------------|
| `COINBASE_API_KEY` | Crypto only | Coinbase Advanced Trade API key path |
| `COINBASE_API_SECRET` | Crypto only | EC private key (PEM, `\n`-escaped) |
| `ALPACA_API_KEY` | Stocks only | Alpaca API key |
| `ALPACA_API_SECRET` | Stocks only | Alpaca API secret |
| `MIMO_API_KEY` | Optional | Xiaomi MIMO API key for LLM sentiment |
| `REDDIT_CLIENT_ID` | Optional | Reddit app client ID |
| `REDDIT_CLIENT_SECRET` | Optional | Reddit app client secret |
| `TELEGRAM_BOT_TOKEN` | Optional | Telegram bot token (from @BotFather) |
| `TELEGRAM_CHAT_ID` | Optional | Telegram chat ID (from @userinfobot) |

### Key config options (`config.yaml`)

```yaml
mode: paper          # paper | live
capital: 800.0       # starting capital in USD
strategy: aggressive # conservative | moderate | aggressive
cycle_interval: 300  # seconds between trading cycles

risk:
  stop_loss_pct: 0.05
  trailing_stop_pct: 0.08
  max_drawdown_pct: 0.15
  max_open_positions: 5
  max_position_pct: 0.20

ml:
  enabled: false
  model_path: models/crypto.lgbm

universe:
  enabled: true
  size: 200        # scan top N assets
  candidates: 50
  active_pairs: 30
```

---

## ML Models

Pre-trained LightGBM models are included (`models/crypto.lgbm`, `models/stocks.lgbm`).

To retrain on fresh data:

```bash
# Fetch 2 years of historical OHLCV data
python scripts/fetch_training_data.py --exchange coinbase

# Train the model
python scripts/train_model.py --data data/training/coinbase --output models/crypto.lgbm
```

---

## Testing

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
pytest

# With coverage
pytest --cov
```

---

## Going Live

1. Set `mode: live` in your config YAML
2. For Alpaca stocks, also set `alpaca.paper: false`
3. Ensure real API credentials are in `.env`
4. Start with small `capital` and monitor closely

---

## License

MIT License — see [LICENSE](LICENSE).
