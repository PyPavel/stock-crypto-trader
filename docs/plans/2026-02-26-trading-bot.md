# Trading Bot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a modular monolith crypto trading bot with Coinbase Advanced, Ollama-powered sentiment analysis, configurable strategies, and paper/live trading support.

**Architecture:** Single Python app with clean module boundaries — abstract ExchangeAdapter interface (CCXT-backed), pluggable Strategy classes, Ollama-scored sentiment from Reddit + CryptoPanic, SQLite state, FastAPI dashboard.

**Tech Stack:** Python 3.12, CCXT, pandas-ta, Ollama, PRAW, FastAPI, APScheduler, SQLite, Docker Compose

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: `src/trader/__init__.py` (and all package `__init__.py` files)
- Create: `tests/__init__.py` (and all test package `__init__.py` files)

**Step 1: Create directory structure**

```bash
mkdir -p src/trader/{adapters,strategies,llm,collectors,core,portfolio,dashboard/templates}
mkdir -p tests/{adapters,strategies,llm,collectors,core,portfolio,dashboard}
touch src/trader/__init__.py
touch src/trader/{adapters,strategies,llm,collectors,core,portfolio,dashboard}/__init__.py
touch tests/__init__.py
touch tests/{adapters,strategies,llm,collectors,core,portfolio,dashboard}/__init__.py
```

**Step 2: Create `pyproject.toml`**

```toml
[project]
name = "trader"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "ccxt>=4.0",
    "pandas-ta>=0.3",
    "pandas>=2.0",
    "numpy>=1.26",
    "praw>=7.7",
    "requests>=2.31",
    "ollama>=0.1",
    "anthropic>=0.20",
    "fastapi>=0.110",
    "uvicorn>=0.27",
    "apscheduler>=3.10",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "jinja2>=3.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
    "pytest-cov>=4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 3: Create `config.yaml`**

```yaml
exchange: coinbase
mode: paper        # paper | live
strategy: moderate # conservative | moderate | aggressive
capital: 100.0
pairs:
  - BTC/USDT
  - ETH/USDT
cycle_interval: 60  # seconds

ollama:
  model: mistral
  base_url: http://localhost:11434

coinbase:
  api_key: ""
  api_secret: ""

reddit:
  client_id: ""
  client_secret: ""
  user_agent: "trader-bot/1.0"

cryptopanic:
  api_key: ""  # free tier works without key for basic endpoints

llm_advisor:
  enabled: false
  provider: claude  # claude | openai
  api_key: ""

risk:
  max_position_pct: 0.20   # max 20% of capital per position
  stop_loss_pct: 0.05      # 5% stop loss
  max_drawdown_pct: 0.15   # halt trading at 15% drawdown
```

**Step 4: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: all packages install without error.

**Step 5: Commit**

```bash
git add .
git commit -m "chore: project scaffolding and dependencies"
```

---

### Task 2: Config Loader

**Files:**
- Create: `src/trader/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from trader.config import Config, load_config


def test_load_config_basic(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "exchange: coinbase\nmode: paper\nstrategy: moderate\n"
        "capital: 100.0\npairs:\n  - BTC/USDT\ncycle_interval: 60\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg.exchange == "coinbase"
    assert cfg.mode == "paper"
    assert cfg.strategy == "moderate"
    assert cfg.capital == 100.0
    assert cfg.pairs == ["BTC/USDT"]
    assert cfg.cycle_interval == 60


def test_invalid_mode_raises(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "exchange: coinbase\nmode: turbo\nstrategy: moderate\n"
        "capital: 100.0\npairs:\n  - BTC/USDT\ncycle_interval: 60\n"
    )
    with pytest.raises(ValueError, match="mode"):
        load_config(str(cfg_file))


def test_invalid_strategy_raises(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "exchange: coinbase\nmode: paper\nstrategy: yolo\n"
        "capital: 100.0\npairs:\n  - BTC/USDT\ncycle_interval: 60\n"
    )
    with pytest.raises(ValueError, match="strategy"):
        load_config(str(cfg_file))
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trader.config'`

**Step 3: Write implementation**

```python
# src/trader/config.py
from dataclasses import dataclass, field
from typing import Literal
import yaml


@dataclass
class OllamaConfig:
    model: str = "mistral"
    base_url: str = "http://localhost:11434"


@dataclass
class CoinbaseConfig:
    api_key: str = ""
    api_secret: str = ""


@dataclass
class RedditConfig:
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "trader-bot/1.0"


@dataclass
class CryptoPanicConfig:
    api_key: str = ""


@dataclass
class LLMAdvisorConfig:
    enabled: bool = False
    provider: str = "claude"
    api_key: str = ""


@dataclass
class RiskConfig:
    max_position_pct: float = 0.20
    stop_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15


@dataclass
class Config:
    exchange: str
    mode: Literal["paper", "live"]
    strategy: Literal["conservative", "moderate", "aggressive"]
    capital: float
    pairs: list[str]
    cycle_interval: int = 60
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    coinbase: CoinbaseConfig = field(default_factory=CoinbaseConfig)
    reddit: RedditConfig = field(default_factory=RedditConfig)
    cryptopanic: CryptoPanicConfig = field(default_factory=CryptoPanicConfig)
    llm_advisor: LLMAdvisorConfig = field(default_factory=LLMAdvisorConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)

    mode = data.get("mode", "paper")
    if mode not in ("paper", "live"):
        raise ValueError(f"mode must be 'paper' or 'live', got '{mode}'")

    strategy = data.get("strategy", "moderate")
    if strategy not in ("conservative", "moderate", "aggressive"):
        raise ValueError(f"strategy must be conservative/moderate/aggressive, got '{strategy}'")

    cfg = Config(
        exchange=data["exchange"],
        mode=mode,
        strategy=strategy,
        capital=float(data["capital"]),
        pairs=data["pairs"],
        cycle_interval=data.get("cycle_interval", 60),
    )

    for key, cls in [
        ("ollama", OllamaConfig),
        ("coinbase", CoinbaseConfig),
        ("reddit", RedditConfig),
        ("cryptopanic", CryptoPanicConfig),
        ("llm_advisor", LLMAdvisorConfig),
        ("risk", RiskConfig),
    ]:
        if key in data:
            setattr(cfg, key, cls(**data[key]))

    return cfg
```

**Step 4: Run tests**

```bash
pytest tests/test_config.py -v
```
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/trader/config.py tests/test_config.py
git commit -m "feat: config loader with validation"
```

---

### Task 3: Data Models

**Files:**
- Create: `src/trader/models.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_models.py
from datetime import datetime, timezone
from trader.models import Candle, Order, Trade, SentimentScore, Signal


def test_candle_creation():
    c = Candle(
        symbol="BTC/USDT",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=40000.0, high=41000.0, low=39000.0, close=40500.0, volume=100.0,
    )
    assert c.symbol == "BTC/USDT"
    assert c.close == 40500.0


def test_signal_score_bounded():
    s = Signal(symbol="BTC/USDT", score=0.75, reason="RSI oversold")
    assert -1.0 <= s.score <= 1.0


def test_sentiment_score():
    ss = SentimentScore(symbol="BTC/USDT", score=0.4, source="reddit", items_analyzed=10)
    assert ss.source == "reddit"


def test_order_sides():
    o = Order(symbol="BTC/USDT", side="buy", amount=0.001, price=40000.0, mode="paper")
    assert o.side in ("buy", "sell")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
import uuid


@dataclass
class Candle:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    symbol: str
    score: float        # -1.0 (strong sell) to +1.0 (strong buy)
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SentimentScore:
    symbol: str
    score: float        # -1.0 to +1.0
    source: str         # "reddit" | "cryptopanic" | "combined"
    items_analyzed: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Order:
    symbol: str
    side: Literal["buy", "sell"]
    amount: float
    price: float
    mode: Literal["paper", "live"]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending | filled | cancelled


@dataclass
class Trade:
    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    amount: float
    price: float
    fee: float
    mode: Literal["paper", "live"]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    pnl: float = 0.0
    narrative: str = ""
```

**Step 4: Run tests**

```bash
pytest tests/test_models.py -v
```
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/trader/models.py tests/test_models.py
git commit -m "feat: core data models"
```

---

### Task 4: Exchange Adapter Interface

**Files:**
- Create: `src/trader/adapters/base.py`
- Create: `tests/adapters/test_base.py`

**Step 1: Write the failing test**

```python
# tests/adapters/test_base.py
import pytest
from trader.adapters.base import ExchangeAdapter
from trader.models import Candle, Order


class ConcreteAdapter(ExchangeAdapter):
    def get_candles(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        return []

    def get_price(self, symbol: str) -> float:
        return 42000.0

    def get_balance(self) -> dict[str, float]:
        return {"USDT": 100.0}

    def place_order(self, side: str, symbol: str, amount: float) -> Order:
        return Order(symbol=symbol, side=side, amount=amount, price=42000.0, mode="paper")

    def get_open_orders(self, symbol: str) -> list[Order]:
        return []

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        return True


def test_adapter_interface():
    adapter = ConcreteAdapter()
    assert adapter.get_price("BTC/USDT") == 42000.0
    assert adapter.get_balance() == {"USDT": 100.0}


def test_abstract_adapter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ExchangeAdapter()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/adapters/test_base.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/adapters/base.py
from abc import ABC, abstractmethod
from trader.models import Candle, Order


class ExchangeAdapter(ABC):
    """Implement this interface to add any exchange (crypto or stocks)."""

    @abstractmethod
    def get_candles(self, symbol: str, interval: str, limit: int = 100) -> list[Candle]:
        """Return OHLCV candles. interval: '1m', '5m', '1h', '1d'"""

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Return current market price."""

    @abstractmethod
    def get_balance(self) -> dict[str, float]:
        """Return {currency: amount} balances."""

    @abstractmethod
    def place_order(self, side: str, symbol: str, amount: float) -> Order:
        """Place a market order. side: 'buy' | 'sell'"""

    @abstractmethod
    def get_open_orders(self, symbol: str) -> list[Order]:
        """Return all open orders for symbol."""

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order. Returns True if cancelled."""
```

**Step 4: Run tests**

```bash
pytest tests/adapters/test_base.py -v
```
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/trader/adapters/base.py tests/adapters/test_base.py
git commit -m "feat: exchange adapter abstract interface"
```

---

### Task 5: Coinbase Adapter (CCXT)

**Files:**
- Create: `src/trader/adapters/coinbase.py`
- Create: `tests/adapters/test_coinbase.py`

**Step 1: Write the failing test**

```python
# tests/adapters/test_coinbase.py
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from trader.adapters.coinbase import CoinbaseAdapter
from trader.models import Candle, Order


def make_adapter():
    with patch("trader.adapters.coinbase.ccxt.coinbaseadvanced") as mock_cls:
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange
        adapter = CoinbaseAdapter(api_key="k", api_secret="s")
        adapter._exchange = mock_exchange
        return adapter, mock_exchange


def test_get_price():
    adapter, exchange = make_adapter()
    exchange.fetch_ticker.return_value = {"last": 42000.0}
    assert adapter.get_price("BTC/USDT") == 42000.0
    exchange.fetch_ticker.assert_called_once_with("BTC/USDT")


def test_get_candles_returns_candle_objects():
    adapter, exchange = make_adapter()
    ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    exchange.fetch_ohlcv.return_value = [
        [ts, 40000.0, 41000.0, 39000.0, 40500.0, 100.0],
    ]
    candles = adapter.get_candles("BTC/USDT", "1h", limit=1)
    assert len(candles) == 1
    assert isinstance(candles[0], Candle)
    assert candles[0].close == 40500.0


def test_get_balance():
    adapter, exchange = make_adapter()
    exchange.fetch_balance.return_value = {"USDT": {"free": 100.0}, "BTC": {"free": 0.001}}
    bal = adapter.get_balance()
    assert bal["USDT"] == 100.0
    assert bal["BTC"] == 0.001


def test_place_order_returns_order():
    adapter, exchange = make_adapter()
    exchange.create_market_order.return_value = {
        "id": "order-123", "price": 42000.0, "amount": 0.001, "status": "filled"
    }
    order = adapter.place_order("buy", "BTC/USDT", 0.001)
    assert isinstance(order, Order)
    assert order.side == "buy"
    assert order.amount == 0.001
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/adapters/test_coinbase.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/adapters/coinbase.py
from datetime import datetime, timezone
import ccxt
from trader.adapters.base import ExchangeAdapter
from trader.models import Candle, Order

INTERVAL_MAP = {"1m": "1m", "5m": "5m", "1h": "1h", "1d": "1d"}


class CoinbaseAdapter(ExchangeAdapter):
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self._exchange = ccxt.coinbaseadvanced({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })

    def get_candles(self, symbol: str, interval: str = "1h", limit: int = 100) -> list[Candle]:
        raw = self._exchange.fetch_ohlcv(symbol, INTERVAL_MAP.get(interval, "1h"), limit=limit)
        return [
            Candle(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
                open=row[1], high=row[2], low=row[3], close=row[4], volume=row[5],
            )
            for row in raw
        ]

    def get_price(self, symbol: str) -> float:
        ticker = self._exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    def get_balance(self) -> dict[str, float]:
        raw = self._exchange.fetch_balance()
        return {k: v["free"] for k, v in raw.items() if isinstance(v, dict) and "free" in v}

    def place_order(self, side: str, symbol: str, amount: float) -> Order:
        raw = self._exchange.create_market_order(symbol, side, amount)
        return Order(
            id=raw.get("id", ""),
            symbol=symbol,
            side=side,
            amount=float(raw.get("amount", amount)),
            price=float(raw.get("price", 0.0)),
            mode="live",
            status=raw.get("status", "filled"),
        )

    def get_open_orders(self, symbol: str) -> list[Order]:
        raw = self._exchange.fetch_open_orders(symbol)
        return [
            Order(id=o["id"], symbol=symbol, side=o["side"],
                  amount=o["amount"], price=o.get("price", 0.0), mode="live")
            for o in raw
        ]

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception:
            return False
```

**Step 4: Run tests**

```bash
pytest tests/adapters/test_coinbase.py -v
```
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/trader/adapters/coinbase.py tests/adapters/test_coinbase.py
git commit -m "feat: Coinbase Advanced adapter via CCXT"
```

---

### Task 6: Technical Signal Generator

**Files:**
- Create: `src/trader/core/signals.py`
- Create: `tests/core/test_signals.py`

**Step 1: Write the failing test**

```python
# tests/core/test_signals.py
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from trader.core.signals import SignalGenerator
from trader.models import Candle


def make_candles(closes: list[float]) -> list[Candle]:
    return [
        Candle(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            open=c, high=c * 1.01, low=c * 0.99, close=c, volume=100.0,
        )
        for i, c in enumerate(closes)
    ]


def test_score_is_bounded():
    # Steadily rising prices → bullish score
    closes = list(range(35000, 35100))  # 100 candles
    gen = SignalGenerator()
    score = gen.score(make_candles(closes))
    assert -1.0 <= score <= 1.0


def test_oversold_gives_positive_score():
    # Falling prices → RSI oversold → positive (buy) signal
    closes = [50000 - i * 200 for i in range(60)]  # strong downtrend
    gen = SignalGenerator()
    score = gen.score(make_candles(closes))
    assert score > 0  # oversold → buy signal


def test_overbought_gives_negative_score():
    # Rising prices → RSI overbought → negative (sell) signal
    closes = [30000 + i * 300 for i in range(60)]  # strong uptrend
    gen = SignalGenerator()
    score = gen.score(make_candles(closes))
    assert score < 0  # overbought → sell signal


def test_needs_minimum_candles():
    gen = SignalGenerator()
    score = gen.score(make_candles([40000] * 5))  # too few
    assert score == 0.0  # neutral when insufficient data
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_signals.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/core/signals.py
import pandas as pd
import pandas_ta as ta
from trader.models import Candle

MIN_CANDLES = 30


class SignalGenerator:
    """Combines RSI, MACD, and Bollinger Band signals into a single score [-1, +1]."""

    def score(self, candles: list[Candle]) -> float:
        """Returns -1.0 (strong sell) to +1.0 (strong buy). 0.0 if insufficient data."""
        if len(candles) < MIN_CANDLES:
            return 0.0

        df = pd.DataFrame([
            {"close": c.close, "high": c.high, "low": c.low, "volume": c.volume}
            for c in candles
        ])

        scores = []

        # RSI signal (weight: 0.4)
        rsi_series = ta.rsi(df["close"], length=14)
        if rsi_series is not None and not rsi_series.empty:
            rsi = rsi_series.iloc[-1]
            if pd.notna(rsi):
                if rsi < 30:
                    scores.append((70 - rsi) / 40 * 0.4)   # oversold → buy
                elif rsi > 70:
                    scores.append((70 - rsi) / 40 * 0.4)   # overbought → sell (negative)
                else:
                    scores.append(0.0)

        # MACD signal (weight: 0.35)
        macd_df = ta.macd(df["close"])
        if macd_df is not None and "MACDh_12_26_9" in macd_df.columns:
            hist = macd_df["MACDh_12_26_9"].iloc[-1]
            if pd.notna(hist):
                normalized = max(-1.0, min(1.0, hist / (df["close"].mean() * 0.002)))
                scores.append(normalized * 0.35)

        # Bollinger Bands signal (weight: 0.25)
        bb = ta.bbands(df["close"], length=20)
        if bb is not None:
            lower_col = [c for c in bb.columns if "BBL" in c]
            upper_col = [c for c in bb.columns if "BBU" in c]
            mid_col = [c for c in bb.columns if "BBM" in c]
            if lower_col and upper_col and mid_col:
                price = df["close"].iloc[-1]
                lower = bb[lower_col[0]].iloc[-1]
                upper = bb[upper_col[0]].iloc[-1]
                mid = bb[mid_col[0]].iloc[-1]
                if pd.notna(lower) and pd.notna(upper) and upper != lower:
                    bb_pct = (price - lower) / (upper - lower)  # 0=at lower, 1=at upper
                    bb_score = (0.5 - bb_pct) * 2 * 0.25       # below mid → buy, above → sell
                    scores.append(bb_score)

        if not scores:
            return 0.0

        total = sum(scores)
        return max(-1.0, min(1.0, total))
```

**Step 4: Run tests**

```bash
pytest tests/core/test_signals.py -v
```
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/trader/core/signals.py tests/core/test_signals.py
git commit -m "feat: technical signal generator (RSI, MACD, Bollinger Bands)"
```

---

### Task 7: Strategy Interface + Moderate Strategy

**Files:**
- Create: `src/trader/strategies/base.py`
- Create: `src/trader/strategies/moderate.py`
- Create: `tests/strategies/test_moderate.py`

**Step 1: Write the failing test**

```python
# tests/strategies/test_moderate.py
import pytest
from trader.strategies.moderate import ModerateStrategy
from trader.models import SentimentScore, Signal
from trader.config import RiskConfig


def make_decision(tech_score: float, sentiment_score: float):
    strategy = ModerateStrategy(risk=RiskConfig())
    tech_signal = Signal(symbol="BTC/USDT", score=tech_score, reason="test")
    sentiment = SentimentScore(symbol="BTC/USDT", score=sentiment_score,
                               source="combined", items_analyzed=5)
    return strategy.decide("BTC/USDT", tech_signal, sentiment, capital=100.0, position=0.0)


def test_strong_buy_signal_returns_buy():
    decision = make_decision(tech_score=0.8, sentiment_score=0.6)
    assert decision["action"] == "buy"
    assert decision["amount"] > 0


def test_strong_sell_signal_returns_sell():
    decision = make_decision(tech_score=-0.8, sentiment_score=-0.6)
    assert decision["action"] == "sell"


def test_neutral_signal_returns_hold():
    decision = make_decision(tech_score=0.1, sentiment_score=0.0)
    assert decision["action"] == "hold"


def test_buy_amount_respects_max_position_pct():
    strategy = ModerateStrategy(risk=RiskConfig(max_position_pct=0.20))
    tech_signal = Signal(symbol="BTC/USDT", score=0.9, reason="test")
    sentiment = SentimentScore(symbol="BTC/USDT", score=0.9, source="combined", items_analyzed=5)
    decision = strategy.decide("BTC/USDT", tech_signal, sentiment, capital=100.0, position=0.0)
    # amount in USD should not exceed 20% of capital
    assert decision.get("usd_amount", 0) <= 20.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/strategies/test_moderate.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/strategies/base.py
from abc import ABC, abstractmethod
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig


class Strategy(ABC):
    """Implement this to add a new trading strategy."""

    def __init__(self, risk: RiskConfig):
        self.risk = risk

    @abstractmethod
    def decide(
        self,
        symbol: str,
        technical: Signal,
        sentiment: SentimentScore,
        capital: float,
        position: float,   # current USD value held in this symbol
    ) -> dict:
        """
        Returns dict with keys:
          action: 'buy' | 'sell' | 'hold'
          usd_amount: float  (USD to spend/receive)
          reason: str
        """
```

```python
# src/trader/strategies/moderate.py
from trader.strategies.base import Strategy
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig

BUY_THRESHOLD = 0.35
SELL_THRESHOLD = -0.35
TECH_WEIGHT = 0.70
SENTIMENT_WEIGHT = 0.30


class ModerateStrategy(Strategy):
    """Balanced risk/reward. Weights technicals 70%, sentiment 30%."""

    def decide(
        self,
        symbol: str,
        technical: Signal,
        sentiment: SentimentScore,
        capital: float,
        position: float,
    ) -> dict:
        combined = technical.score * TECH_WEIGHT + sentiment.score * SENTIMENT_WEIGHT

        if combined >= BUY_THRESHOLD and position == 0.0:
            usd_amount = min(capital * self.risk.max_position_pct, capital)
            return {
                "action": "buy",
                "usd_amount": usd_amount,
                "reason": f"combined score {combined:.2f} (tech={technical.score:.2f}, sentiment={sentiment.score:.2f})",
            }

        if combined <= SELL_THRESHOLD and position > 0.0:
            return {
                "action": "sell",
                "usd_amount": position,
                "reason": f"combined score {combined:.2f} (tech={technical.score:.2f}, sentiment={sentiment.score:.2f})",
            }

        return {"action": "hold", "usd_amount": 0.0, "reason": f"combined score {combined:.2f} within hold range"}
```

**Step 4: Run tests**

```bash
pytest tests/strategies/test_moderate.py -v
```
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/trader/strategies/base.py src/trader/strategies/moderate.py tests/strategies/test_moderate.py
git commit -m "feat: strategy interface and moderate strategy"
```

---

### Task 8: Conservative and Aggressive Strategies

**Files:**
- Create: `src/trader/strategies/conservative.py`
- Create: `src/trader/strategies/aggressive.py`
- Create: `tests/strategies/test_conservative.py`
- Create: `tests/strategies/test_aggressive.py`

**Step 1: Write the failing tests**

```python
# tests/strategies/test_conservative.py
from trader.strategies.conservative import ConservativeStrategy
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig


def decide(tech, sent):
    s = ConservativeStrategy(risk=RiskConfig())
    return s.decide("BTC/USDT",
                    Signal(symbol="BTC/USDT", score=tech, reason=""),
                    SentimentScore(symbol="BTC/USDT", score=sent, source="combined", items_analyzed=5),
                    capital=100.0, position=0.0)


def test_requires_strong_signal_to_buy():
    # moderate signal → hold
    assert decide(0.5, 0.3)["action"] == "hold"


def test_strong_signal_buys():
    assert decide(0.9, 0.8)["action"] == "buy"


def test_buy_uses_small_position():
    result = decide(0.9, 0.9)
    assert result["usd_amount"] <= 10.0  # conservative: max 10% of 100
```

```python
# tests/strategies/test_aggressive.py
from trader.strategies.aggressive import AggressiveStrategy
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig


def decide(tech, sent, position=0.0):
    s = AggressiveStrategy(risk=RiskConfig())
    return s.decide("BTC/USDT",
                    Signal(symbol="BTC/USDT", score=tech, reason=""),
                    SentimentScore(symbol="BTC/USDT", score=sent, source="combined", items_analyzed=5),
                    capital=100.0, position=position)


def test_low_threshold_triggers_buy():
    # aggressive buys on weaker signals
    assert decide(0.2, 0.1)["action"] == "buy"


def test_uses_large_position():
    result = decide(0.9, 0.9)
    assert result["usd_amount"] >= 30.0  # aggressive: up to 40% of 100
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/strategies/test_conservative.py tests/strategies/test_aggressive.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementations**

```python
# src/trader/strategies/conservative.py
from trader.strategies.base import Strategy
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig

BUY_THRESHOLD = 0.65
SELL_THRESHOLD = -0.50
TECH_WEIGHT = 0.80
SENTIMENT_WEIGHT = 0.20
MAX_POSITION_PCT = 0.10  # override to 10%


class ConservativeStrategy(Strategy):
    """Tight entry threshold, small positions, sentiment has minimal weight."""

    def decide(self, symbol, technical, sentiment, capital, position):
        combined = technical.score * TECH_WEIGHT + sentiment.score * SENTIMENT_WEIGHT

        if combined >= BUY_THRESHOLD and position == 0.0:
            usd_amount = capital * MAX_POSITION_PCT
            return {"action": "buy", "usd_amount": usd_amount,
                    "reason": f"conservative buy at {combined:.2f}"}

        if combined <= SELL_THRESHOLD and position > 0.0:
            return {"action": "sell", "usd_amount": position,
                    "reason": f"conservative sell at {combined:.2f}"}

        return {"action": "hold", "usd_amount": 0.0, "reason": f"score {combined:.2f}"}
```

```python
# src/trader/strategies/aggressive.py
from trader.strategies.base import Strategy
from trader.models import Signal, SentimentScore
from trader.config import RiskConfig

BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15
TECH_WEIGHT = 0.55
SENTIMENT_WEIGHT = 0.45
MAX_POSITION_PCT = 0.40  # up to 40% per position


class AggressiveStrategy(Strategy):
    """Low entry threshold, large positions, sentiment has high weight."""

    def decide(self, symbol, technical, sentiment, capital, position):
        combined = technical.score * TECH_WEIGHT + sentiment.score * SENTIMENT_WEIGHT

        if combined >= BUY_THRESHOLD and position == 0.0:
            usd_amount = capital * MAX_POSITION_PCT
            return {"action": "buy", "usd_amount": usd_amount,
                    "reason": f"aggressive buy at {combined:.2f}"}

        if combined <= SELL_THRESHOLD and position > 0.0:
            return {"action": "sell", "usd_amount": position,
                    "reason": f"aggressive sell at {combined:.2f}"}

        return {"action": "hold", "usd_amount": 0.0, "reason": f"score {combined:.2f}"}
```

**Step 4: Run tests**

```bash
pytest tests/strategies/ -v
```
Expected: PASS (all strategy tests)

**Step 5: Commit**

```bash
git add src/trader/strategies/conservative.py src/trader/strategies/aggressive.py \
        tests/strategies/test_conservative.py tests/strategies/test_aggressive.py
git commit -m "feat: conservative and aggressive strategy plugins"
```

---

### Task 9: Strategy Registry

**Files:**
- Create: `src/trader/strategies/registry.py`
- Create: `tests/strategies/test_registry.py`

**Step 1: Write the failing test**

```python
# tests/strategies/test_registry.py
import pytest
from trader.strategies.registry import get_strategy
from trader.strategies.moderate import ModerateStrategy
from trader.strategies.conservative import ConservativeStrategy
from trader.strategies.aggressive import AggressiveStrategy
from trader.config import RiskConfig


def test_get_moderate():
    assert isinstance(get_strategy("moderate", RiskConfig()), ModerateStrategy)


def test_get_conservative():
    assert isinstance(get_strategy("conservative", RiskConfig()), ConservativeStrategy)


def test_get_aggressive():
    assert isinstance(get_strategy("aggressive", RiskConfig()), AggressiveStrategy)


def test_unknown_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("yolo", RiskConfig())
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/strategies/test_registry.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/strategies/registry.py
from trader.strategies.base import Strategy
from trader.strategies.moderate import ModerateStrategy
from trader.strategies.conservative import ConservativeStrategy
from trader.strategies.aggressive import AggressiveStrategy
from trader.config import RiskConfig

_REGISTRY = {
    "moderate": ModerateStrategy,
    "conservative": ConservativeStrategy,
    "aggressive": AggressiveStrategy,
}


def get_strategy(name: str, risk: RiskConfig) -> Strategy:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(_REGISTRY)}")
    return _REGISTRY[name](risk=risk)
```

**Step 4: Run tests**

```bash
pytest tests/strategies/test_registry.py -v
```
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/trader/strategies/registry.py tests/strategies/test_registry.py
git commit -m "feat: strategy registry"
```

---

### Task 10: Risk Manager

**Files:**
- Create: `src/trader/core/risk.py`
- Create: `tests/core/test_risk.py`

**Step 1: Write the failing test**

```python
# tests/core/test_risk.py
import pytest
from trader.core.risk import RiskManager
from trader.config import RiskConfig


def make_rm():
    return RiskManager(RiskConfig(max_position_pct=0.20, stop_loss_pct=0.05, max_drawdown_pct=0.15))


def test_allows_valid_buy():
    rm = make_rm()
    result = rm.validate_buy(symbol="BTC/USDT", usd_amount=20.0, capital=100.0, positions={})
    assert result["allowed"] is True


def test_blocks_oversized_buy():
    rm = make_rm()
    result = rm.validate_buy(symbol="BTC/USDT", usd_amount=50.0, capital=100.0, positions={})
    assert result["allowed"] is False
    assert "position" in result["reason"].lower()


def test_blocks_buy_when_max_drawdown_hit():
    rm = make_rm()
    # Started at 100, now at 84 → 16% drawdown > 15% limit
    result = rm.validate_buy(symbol="BTC/USDT", usd_amount=10.0, capital=84.0,
                              positions={}, starting_capital=100.0)
    assert result["allowed"] is False
    assert "drawdown" in result["reason"].lower()


def test_stop_loss_triggered():
    rm = make_rm()
    # bought at 40000, now at 37000 → 7.5% loss > 5% stop-loss
    triggered = rm.check_stop_loss(symbol="BTC/USDT", entry_price=40000.0, current_price=37000.0)
    assert triggered is True


def test_stop_loss_not_triggered():
    rm = make_rm()
    # bought at 40000, now at 39000 → 2.5% loss < 5%
    triggered = rm.check_stop_loss(symbol="BTC/USDT", entry_price=40000.0, current_price=39000.0)
    assert triggered is False
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_risk.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/core/risk.py
from trader.config import RiskConfig


class RiskManager:
    def __init__(self, risk: RiskConfig):
        self.risk = risk

    def validate_buy(
        self,
        symbol: str,
        usd_amount: float,
        capital: float,
        positions: dict,
        starting_capital: float | None = None,
    ) -> dict:
        # Check drawdown
        if starting_capital and starting_capital > 0:
            drawdown = (starting_capital - capital) / starting_capital
            if drawdown >= self.risk.max_drawdown_pct:
                return {"allowed": False,
                        "reason": f"max drawdown {drawdown:.1%} reached, halting trades"}

        # Check position size
        max_usd = capital * self.risk.max_position_pct
        if usd_amount > max_usd:
            return {"allowed": False,
                    "reason": f"position size ${usd_amount:.2f} exceeds max ${max_usd:.2f} ({self.risk.max_position_pct:.0%} of capital)"}

        return {"allowed": True, "reason": "ok"}

    def check_stop_loss(self, symbol: str, entry_price: float, current_price: float) -> bool:
        loss_pct = (entry_price - current_price) / entry_price
        return loss_pct >= self.risk.stop_loss_pct
```

**Step 4: Run tests**

```bash
pytest tests/core/test_risk.py -v
```
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/trader/core/risk.py tests/core/test_risk.py
git commit -m "feat: risk manager with drawdown and stop-loss checks"
```

---

### Task 11: Portfolio State + SQLite Persistence

**Files:**
- Create: `src/trader/portfolio/db.py`
- Create: `src/trader/portfolio/state.py`
- Create: `tests/portfolio/test_state.py`

**Step 1: Write the failing test**

```python
# tests/portfolio/test_state.py
import pytest
from trader.portfolio.state import Portfolio
from trader.models import Trade


@pytest.fixture
def portfolio(tmp_path):
    return Portfolio(db_path=str(tmp_path / "test.db"), starting_capital=100.0)


def test_initial_balance(portfolio):
    assert portfolio.cash == 100.0
    assert portfolio.positions == {}


def test_record_buy(portfolio):
    trade = Trade(order_id="1", symbol="BTC/USDT", side="buy",
                  amount=0.001, price=40000.0, fee=0.5, mode="paper")
    portfolio.record_trade(trade)
    assert "BTC/USDT" in portfolio.positions
    assert portfolio.cash < 100.0


def test_record_sell(portfolio):
    buy = Trade(order_id="1", symbol="BTC/USDT", side="buy",
                amount=0.001, price=40000.0, fee=0.5, mode="paper")
    portfolio.record_trade(buy)
    sell = Trade(order_id="2", symbol="BTC/USDT", side="sell",
                 amount=0.001, price=42000.0, fee=0.5, mode="paper")
    portfolio.record_trade(sell)
    assert portfolio.positions.get("BTC/USDT", {}).get("amount", 0.0) == 0.0


def test_total_value(portfolio):
    assert portfolio.total_value(prices={"BTC/USDT": 40000.0}) == 100.0


def test_trades_persisted(tmp_path):
    p = Portfolio(db_path=str(tmp_path / "test.db"), starting_capital=100.0)
    trade = Trade(order_id="1", symbol="BTC/USDT", side="buy",
                  amount=0.001, price=40000.0, fee=0.5, mode="paper")
    p.record_trade(trade)
    # Reload from disk
    p2 = Portfolio(db_path=str(tmp_path / "test.db"), starting_capital=100.0)
    assert len(p2.get_trades()) == 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/portfolio/test_state.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/portfolio/db.py
import sqlite3
from datetime import datetime
from trader.models import Trade


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            order_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            amount REAL NOT NULL,
            price REAL NOT NULL,
            fee REAL NOT NULL,
            mode TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            pnl REAL DEFAULT 0.0,
            narrative TEXT DEFAULT ''
        )
    """)
    conn.commit()


def save_trade(conn: sqlite3.Connection, trade: Trade) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)",
        (trade.order_id, trade.symbol, trade.side, trade.amount, trade.price,
         trade.fee, trade.mode, trade.timestamp.isoformat(), trade.pnl, trade.narrative)
    )
    conn.commit()


def load_trades(conn: sqlite3.Connection) -> list[Trade]:
    rows = conn.execute("SELECT * FROM trades ORDER BY timestamp").fetchall()
    return [
        Trade(order_id=r[0], symbol=r[1], side=r[2], amount=r[3], price=r[4],
              fee=r[5], mode=r[6], timestamp=datetime.fromisoformat(r[7]),
              pnl=r[8], narrative=r[9])
        for r in rows
    ]
```

```python
# src/trader/portfolio/state.py
import sqlite3
from trader.models import Trade
from trader.portfolio.db import init_db, save_trade, load_trades


class Portfolio:
    def __init__(self, db_path: str, starting_capital: float):
        self.starting_capital = starting_capital
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        init_db(self._conn)

        # Rebuild in-memory state from persisted trades
        self.cash = starting_capital
        self.positions: dict[str, dict] = {}  # symbol → {amount, entry_price}
        for trade in load_trades(self._conn):
            self._apply(trade)

    def record_trade(self, trade: Trade) -> None:
        save_trade(self._conn, trade)
        self._apply(trade)

    def _apply(self, trade: Trade) -> None:
        cost = trade.amount * trade.price + trade.fee
        if trade.side == "buy":
            self.cash -= cost
            pos = self.positions.setdefault(trade.symbol, {"amount": 0.0, "entry_price": 0.0})
            pos["amount"] += trade.amount
            pos["entry_price"] = trade.price
        else:
            self.cash += trade.amount * trade.price - trade.fee
            if trade.symbol in self.positions:
                self.positions[trade.symbol]["amount"] -= trade.amount
                if self.positions[trade.symbol]["amount"] <= 0:
                    del self.positions[trade.symbol]

    def total_value(self, prices: dict[str, float]) -> float:
        holdings = sum(
            pos["amount"] * prices.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        )
        return self.cash + holdings

    def get_trades(self) -> list[Trade]:
        return load_trades(self._conn)
```

**Step 4: Run tests**

```bash
pytest tests/portfolio/test_state.py -v
```
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/trader/portfolio/ tests/portfolio/test_state.py
git commit -m "feat: portfolio state with SQLite persistence"
```

---

### Task 12: Order Router (Paper + Live)

**Files:**
- Create: `src/trader/core/router.py`
- Create: `tests/core/test_router.py`

**Step 1: Write the failing test**

```python
# tests/core/test_router.py
from unittest.mock import MagicMock
from trader.core.router import OrderRouter
from trader.models import Order
from trader.adapters.base import ExchangeAdapter


def make_router(mode: str):
    adapter = MagicMock(spec=ExchangeAdapter)
    adapter.get_price.return_value = 42000.0
    adapter.place_order.return_value = Order(
        symbol="BTC/USDT", side="buy", amount=0.001, price=42000.0, mode="live"
    )
    return OrderRouter(adapter=adapter, mode=mode), adapter


def test_paper_buy_does_not_call_adapter():
    router, adapter = make_router("paper")
    order = router.execute("buy", "BTC/USDT", usd_amount=10.0)
    assert order.mode == "paper"
    adapter.place_order.assert_not_called()


def test_paper_order_calculates_amount_from_price():
    router, adapter = make_router("paper")
    order = router.execute("buy", "BTC/USDT", usd_amount=42.0)
    assert abs(order.amount - 0.001) < 1e-6  # 42 / 42000


def test_live_buy_calls_adapter():
    router, adapter = make_router("live")
    router.execute("buy", "BTC/USDT", usd_amount=42.0)
    adapter.place_order.assert_called_once()


def test_paper_sell_returns_sell_order():
    router, adapter = make_router("paper")
    order = router.execute("sell", "BTC/USDT", usd_amount=42.0)
    assert order.side == "sell"
    assert order.mode == "paper"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_router.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/core/router.py
from trader.adapters.base import ExchangeAdapter
from trader.models import Order


class OrderRouter:
    def __init__(self, adapter: ExchangeAdapter, mode: str):
        self._adapter = adapter
        self._mode = mode  # "paper" | "live"

    def execute(self, side: str, symbol: str, usd_amount: float) -> Order:
        price = self._adapter.get_price(symbol)
        amount = usd_amount / price

        if self._mode == "paper":
            return Order(
                symbol=symbol, side=side, amount=amount,
                price=price, mode="paper", status="filled",
            )

        # Live: delegate to exchange
        return self._adapter.place_order(side, symbol, amount)
```

**Step 4: Run tests**

```bash
pytest tests/core/test_router.py -v
```
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/trader/core/router.py tests/core/test_router.py
git commit -m "feat: order router with paper and live modes"
```

---

### Task 13: CryptoPanic Collector

**Files:**
- Create: `src/trader/collectors/cryptopanic.py`
- Create: `tests/collectors/test_cryptopanic.py`

**Step 1: Write the failing test**

```python
# tests/collectors/test_cryptopanic.py
from unittest.mock import patch, MagicMock
from trader.collectors.cryptopanic import CryptoPanicCollector


def test_fetch_returns_headlines():
    mock_response = {
        "results": [
            {"title": "Bitcoin hits new high", "currencies": [{"code": "BTC"}]},
            {"title": "ETH upgrade successful", "currencies": [{"code": "ETH"}]},
        ]
    }
    with patch("trader.collectors.cryptopanic.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: mock_response
        )
        collector = CryptoPanicCollector(api_key="")
        headlines = collector.fetch(symbols=["BTC/USDT"])
    assert len(headlines) >= 1
    assert any("Bitcoin" in h for h in headlines)


def test_fetch_empty_on_error():
    with patch("trader.collectors.cryptopanic.requests.get") as mock_get:
        mock_get.side_effect = Exception("network error")
        collector = CryptoPanicCollector(api_key="")
        headlines = collector.fetch(symbols=["BTC/USDT"])
    assert headlines == []
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/collectors/test_cryptopanic.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/collectors/cryptopanic.py
import requests
import logging

logger = logging.getLogger(__name__)

CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"


class CryptoPanicCollector:
    def __init__(self, api_key: str = ""):
        self._api_key = api_key

    def fetch(self, symbols: list[str], limit: int = 20) -> list[str]:
        """Return list of news headlines relevant to the given symbols."""
        # Extract base currencies from pairs like "BTC/USDT" → "BTC"
        currencies = {s.split("/")[0] for s in symbols}

        params = {"public": "true", "kind": "news", "filter": "hot"}
        if self._api_key:
            params["auth_token"] = self._api_key

        try:
            response = requests.get(CRYPTOPANIC_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"CryptoPanic fetch failed: {e}")
            return []

        headlines = []
        for item in data.get("results", [])[:limit]:
            item_currencies = {c["code"] for c in item.get("currencies", [])}
            if not currencies or item_currencies & currencies:
                headlines.append(item["title"])

        return headlines
```

**Step 4: Run tests**

```bash
pytest tests/collectors/test_cryptopanic.py -v
```
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/trader/collectors/cryptopanic.py tests/collectors/test_cryptopanic.py
git commit -m "feat: CryptoPanic news collector"
```

---

### Task 14: Reddit Collector

**Files:**
- Create: `src/trader/collectors/reddit.py`
- Create: `tests/collectors/test_reddit.py`

**Step 1: Write the failing test**

```python
# tests/collectors/test_reddit.py
from unittest.mock import MagicMock, patch
from trader.collectors.reddit import RedditCollector

SUBREDDITS = {"BTC": ["bitcoin"], "ETH": ["ethtrader"]}


def test_fetch_returns_post_titles():
    mock_post = MagicMock()
    mock_post.title = "Bitcoin looking bullish today"
    mock_subreddit = MagicMock()
    mock_subreddit.hot.return_value = [mock_post]

    with patch("trader.collectors.reddit.praw.Reddit") as mock_reddit_cls:
        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_reddit_cls.return_value = mock_reddit

        collector = RedditCollector(client_id="id", client_secret="secret",
                                    user_agent="test", subreddit_map=SUBREDDITS)
        posts = collector.fetch(symbols=["BTC/USDT"])

    assert len(posts) >= 1
    assert "Bitcoin" in posts[0]


def test_fetch_empty_on_error():
    with patch("trader.collectors.reddit.praw.Reddit") as mock_reddit_cls:
        mock_reddit_cls.side_effect = Exception("auth failed")
        collector = RedditCollector(client_id="", client_secret="",
                                    user_agent="test", subreddit_map=SUBREDDITS)
        posts = collector.fetch(symbols=["BTC/USDT"])
    assert posts == []
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/collectors/test_reddit.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/collectors/reddit.py
import logging
import praw

logger = logging.getLogger(__name__)

DEFAULT_SUBREDDIT_MAP = {
    "BTC": ["bitcoin", "cryptocurrency"],
    "ETH": ["ethtrader", "ethereum"],
    "SOL": ["solana"],
    "ADA": ["cardano"],
}


class RedditCollector:
    def __init__(self, client_id: str, client_secret: str, user_agent: str,
                 subreddit_map: dict | None = None):
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._subreddit_map = subreddit_map or DEFAULT_SUBREDDIT_MAP

    def fetch(self, symbols: list[str], limit: int = 15) -> list[str]:
        """Return list of post titles relevant to the given symbols."""
        currencies = {s.split("/")[0] for s in symbols}

        try:
            reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
        except Exception as e:
            logger.warning(f"Reddit init failed: {e}")
            return []

        titles = []
        for currency in currencies:
            subs = self._subreddit_map.get(currency, [])
            for sub_name in subs:
                try:
                    sub = reddit.subreddit(sub_name)
                    for post in sub.hot(limit=limit):
                        titles.append(post.title)
                except Exception as e:
                    logger.warning(f"Reddit fetch from r/{sub_name} failed: {e}")

        return titles
```

**Step 4: Run tests**

```bash
pytest tests/collectors/test_reddit.py -v
```
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/trader/collectors/reddit.py tests/collectors/test_reddit.py
git commit -m "feat: Reddit collector via PRAW"
```

---

### Task 15: Ollama Sentiment Module

**Files:**
- Create: `src/trader/llm/sentiment.py`
- Create: `tests/llm/test_sentiment.py`

**Step 1: Write the failing test**

```python
# tests/llm/test_sentiment.py
from unittest.mock import patch, MagicMock
from trader.llm.sentiment import SentimentAnalyzer


def test_bullish_text_positive_score():
    with patch("trader.llm.sentiment.ollama.chat") as mock_chat:
        mock_chat.return_value = MagicMock(
            message=MagicMock(content='{"sentiment": "bullish", "confidence": 0.85}')
        )
        analyzer = SentimentAnalyzer(model="mistral", base_url="http://localhost:11434")
        score = analyzer.score_texts(["Bitcoin breaks all-time high", "Strong buying pressure"])
    assert score > 0


def test_bearish_text_negative_score():
    with patch("trader.llm.sentiment.ollama.chat") as mock_chat:
        mock_chat.return_value = MagicMock(
            message=MagicMock(content='{"sentiment": "bearish", "confidence": 0.90}')
        )
        analyzer = SentimentAnalyzer(model="mistral", base_url="http://localhost:11434")
        score = analyzer.score_texts(["Crypto market crashes", "Massive sell-off"])
    assert score < 0


def test_returns_zero_on_empty_input():
    analyzer = SentimentAnalyzer(model="mistral", base_url="http://localhost:11434")
    assert analyzer.score_texts([]) == 0.0


def test_returns_zero_on_ollama_failure():
    with patch("trader.llm.sentiment.ollama.chat") as mock_chat:
        mock_chat.side_effect = Exception("connection refused")
        analyzer = SentimentAnalyzer(model="mistral", base_url="http://localhost:11434")
        score = analyzer.score_texts(["some text"])
    assert score == 0.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/llm/test_sentiment.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/llm/sentiment.py
import json
import logging
import ollama

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Analyze the sentiment of the following crypto news/posts.
Respond ONLY with valid JSON in this exact format:
{{"sentiment": "bullish" | "bearish" | "neutral", "confidence": 0.0-1.0}}

Texts:
{texts}"""

SENTIMENT_SCORES = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}


class SentimentAnalyzer:
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url

    def score_texts(self, texts: list[str]) -> float:
        """Score a batch of texts. Returns -1.0 to +1.0. Returns 0.0 on error."""
        if not texts:
            return 0.0

        # Batch into chunks of 5 to keep prompt size manageable
        chunk_scores = []
        for i in range(0, len(texts), 5):
            chunk = texts[i:i + 5]
            score = self._score_chunk(chunk)
            if score is not None:
                chunk_scores.append(score)

        if not chunk_scores:
            return 0.0
        return sum(chunk_scores) / len(chunk_scores)

    def _score_chunk(self, texts: list[str]) -> float | None:
        prompt = PROMPT_TEMPLATE.format(texts="\n".join(f"- {t}" for t in texts))
        try:
            response = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            content = response.message.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            sentiment = data.get("sentiment", "neutral")
            confidence = float(data.get("confidence", 0.5))
            return SENTIMENT_SCORES.get(sentiment, 0.0) * confidence
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return None
```

**Step 4: Run tests**

```bash
pytest tests/llm/test_sentiment.py -v
```
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/trader/llm/sentiment.py tests/llm/test_sentiment.py
git commit -m "feat: Ollama sentiment analyzer"
```

---

### Task 16: Core Trading Engine

**Files:**
- Create: `src/trader/core/engine.py`
- Create: `tests/core/test_engine.py`

**Step 1: Write the failing test**

```python
# tests/core/test_engine.py
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from trader.core.engine import TradingEngine
from trader.config import Config, RiskConfig, OllamaConfig
from trader.models import Candle, Trade


def make_candles(n=50):
    return [
        Candle("BTC/USDT", datetime(2024, 1, i + 1, tzinfo=timezone.utc),
               40000.0, 41000.0, 39000.0, 40000.0 + i * 10, 100.0)
        for i in range(n)
    ]


def make_engine(mode="paper"):
    cfg = Config(exchange="coinbase", mode=mode, strategy="moderate",
                 capital=100.0, pairs=["BTC/USDT"], cycle_interval=60)

    adapter = MagicMock()
    adapter.get_candles.return_value = make_candles()
    adapter.get_price.return_value = 40500.0
    adapter.get_balance.return_value = {"USDT": 100.0}

    sentiment_analyzer = MagicMock()
    sentiment_analyzer.score_texts.return_value = 0.3

    news_collector = MagicMock()
    news_collector.fetch.return_value = ["BTC is bullish"]

    reddit_collector = MagicMock()
    reddit_collector.fetch.return_value = ["Crypto market up"]

    engine = TradingEngine(
        config=cfg,
        adapter=adapter,
        sentiment_analyzer=sentiment_analyzer,
        collectors=[news_collector, reddit_collector],
    )
    return engine


def test_run_cycle_completes_without_error():
    engine = make_engine()
    engine.run_cycle()  # should not raise


def test_run_cycle_updates_portfolio():
    engine = make_engine()
    initial_trades = len(engine.portfolio.get_trades())
    engine.run_cycle()
    # cycle may or may not trade depending on signals — just ensure no crash


def test_engine_respects_paper_mode():
    engine = make_engine(mode="paper")
    engine.run_cycle()
    trades = engine.portfolio.get_trades()
    for t in trades:
        assert t.mode == "paper"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_engine.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/core/engine.py
import logging
from trader.config import Config
from trader.adapters.base import ExchangeAdapter
from trader.llm.sentiment import SentimentAnalyzer
from trader.core.signals import SignalGenerator
from trader.core.risk import RiskManager
from trader.core.router import OrderRouter
from trader.portfolio.state import Portfolio
from trader.strategies.registry import get_strategy
from trader.models import SentimentScore, Trade

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(
        self,
        config: Config,
        adapter: ExchangeAdapter,
        sentiment_analyzer: SentimentAnalyzer,
        collectors: list,
        db_path: str = "trader.db",
    ):
        self.config = config
        self._adapter = adapter
        self._sentiment = sentiment_analyzer
        self._collectors = collectors
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
        sentiment = SentimentScore(symbol=symbol, score=raw_sentiment,
                                   source="combined", items_analyzed=len(texts))

        # 3. Technical signals
        tech_signal = type("Signal", (), {
            "symbol": symbol,
            "score": self._signals.score(candles),
            "reason": "technical indicators",
        })()

        # 4. Strategy decision
        position_usd = 0.0
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            position_usd = pos["amount"] * price

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

            order = self._router.execute("buy", symbol, decision["usd_amount"])
            trade = Trade(order_id=order.id, symbol=symbol, side="buy",
                          amount=order.amount, price=order.price, fee=0.0,
                          mode=self.config.mode, narrative=decision["reason"])
            self.portfolio.record_trade(trade)
            logger.info(f"BUY {symbol}: {order.amount:.6f} @ {order.price:.2f} — {decision['reason']}")

        elif decision["action"] == "sell" and position_usd > 0:
            # Stop-loss check
            entry = self.portfolio.positions[symbol]["entry_price"]
            if self._risk.check_stop_loss(symbol, entry, price):
                logger.info(f"Stop-loss triggered for {symbol}")

            order = self._router.execute("sell", symbol, position_usd)
            trade = Trade(order_id=order.id, symbol=symbol, side="sell",
                          amount=order.amount, price=order.price, fee=0.0,
                          mode=self.config.mode, narrative=decision["reason"])
            self.portfolio.record_trade(trade)
            logger.info(f"SELL {symbol}: {order.amount:.6f} @ {order.price:.2f} — {decision['reason']}")

        else:
            logger.debug(f"HOLD {symbol}: {decision['reason']}")
```

**Step 4: Run tests**

```bash
pytest tests/core/test_engine.py -v
```
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/trader/core/engine.py tests/core/test_engine.py
git commit -m "feat: core trading engine"
```

---

### Task 17: Backtesting Module

**Files:**
- Create: `src/trader/core/backtest.py`
- Create: `tests/core/test_backtest.py`

**Step 1: Write the failing test**

```python
# tests/core/test_backtest.py
from datetime import datetime, timezone
from trader.core.backtest import Backtester
from trader.models import Candle
from trader.config import RiskConfig
from trader.strategies.moderate import ModerateStrategy


def make_candles(closes: list[float]) -> list[Candle]:
    return [
        Candle("BTC/USDT", datetime(2024, 1, i + 1, tzinfo=timezone.utc),
               c, c * 1.01, c * 0.99, c, 100.0)
        for i, c in enumerate(closes)
    ]


def test_backtest_returns_results():
    strategy = ModerateStrategy(risk=RiskConfig())
    bt = Backtester(strategy=strategy, starting_capital=100.0)

    # Simulated price series: downtrend then uptrend
    closes = [40000 - i * 100 for i in range(50)] + [36000 + i * 150 for i in range(50)]
    results = bt.run(symbol="BTC/USDT", candles=make_candles(closes))

    assert "final_value" in results
    assert "num_trades" in results
    assert "win_rate" in results
    assert "max_drawdown" in results
    assert results["final_value"] > 0


def test_backtest_win_rate_between_0_and_1():
    strategy = ModerateStrategy(risk=RiskConfig())
    bt = Backtester(strategy=strategy, starting_capital=100.0)
    closes = [40000 + (i % 10) * 100 for i in range(100)]
    results = bt.run(symbol="BTC/USDT", candles=make_candles(closes))
    assert 0.0 <= results["win_rate"] <= 1.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_backtest.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/core/backtest.py
from trader.core.signals import SignalGenerator
from trader.core.risk import RiskManager
from trader.models import Candle, SentimentScore, Signal
from trader.strategies.base import Strategy
from trader.config import RiskConfig

WINDOW = 50  # candles fed to signal generator per step


class Backtester:
    def __init__(self, strategy: Strategy, starting_capital: float = 100.0):
        self._strategy = strategy
        self._starting_capital = starting_capital
        self._signals = SignalGenerator()
        self._risk = RiskManager(strategy.risk)

    def run(self, symbol: str, candles: list[Candle]) -> dict:
        cash = self._starting_capital
        position = 0.0      # units held
        entry_price = 0.0
        trades = []
        portfolio_values = [cash]

        neutral_sentiment = SentimentScore(symbol=symbol, score=0.0,
                                           source="backtest", items_analyzed=0)

        for i in range(WINDOW, len(candles)):
            window = candles[i - WINDOW:i]
            current = candles[i]
            price = current.close

            tech_score = self._signals.score(window)
            tech_signal = Signal(symbol=symbol, score=tech_score, reason="backtest")
            position_usd = position * price

            decision = self._strategy.decide(symbol, tech_signal, neutral_sentiment,
                                              capital=cash, position=position_usd)

            if decision["action"] == "buy" and position == 0.0:
                check = self._risk.validate_buy(symbol, decision["usd_amount"], cash, {},
                                                self._starting_capital)
                if check["allowed"]:
                    amount = decision["usd_amount"] / price
                    cash -= amount * price
                    position = amount
                    entry_price = price
                    trades.append({"side": "buy", "price": price})

            elif decision["action"] == "sell" and position > 0.0:
                proceeds = position * price
                pnl = (price - entry_price) / entry_price
                trades.append({"side": "sell", "price": price, "pnl": pnl})
                cash += proceeds
                position = 0.0
                entry_price = 0.0

            portfolio_values.append(cash + position * price)

        # Close any open position at end
        if position > 0.0:
            final_price = candles[-1].close
            pnl = (final_price - entry_price) / entry_price
            trades.append({"side": "sell", "price": final_price, "pnl": pnl})
            cash += position * final_price

        sells = [t for t in trades if t["side"] == "sell"]
        win_rate = (sum(1 for t in sells if t.get("pnl", 0) > 0) / len(sells)) if sells else 0.0

        peak = self._starting_capital
        max_drawdown = 0.0
        for v in portfolio_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, dd)

        return {
            "final_value": cash,
            "num_trades": len(sells),
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "pnl_pct": (cash - self._starting_capital) / self._starting_capital,
        }
```

**Step 4: Run tests**

```bash
pytest tests/core/test_backtest.py -v
```
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/trader/core/backtest.py tests/core/test_backtest.py
git commit -m "feat: backtesting module"
```

---

### Task 18: FastAPI Dashboard

**Files:**
- Create: `src/trader/dashboard/api.py`
- Create: `src/trader/dashboard/templates/index.html`
- Create: `tests/dashboard/test_api.py`

**Step 1: Write the failing test**

```python
# tests/dashboard/test_api.py
import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
from trader.dashboard.api import create_app
from trader.portfolio.state import Portfolio


@pytest.fixture
def mock_engine(tmp_path):
    engine = MagicMock()
    engine.config.mode = "paper"
    engine.config.strategy = "moderate"
    engine.config.pairs = ["BTC/USDT"]
    engine.portfolio = Portfolio(db_path=str(tmp_path / "test.db"), starting_capital=100.0)
    engine._adapter.get_price.return_value = 42000.0
    return engine


@pytest.mark.asyncio
async def test_status_endpoint(mock_engine):
    app = create_app(mock_engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "mode" in data
    assert "strategy" in data
    assert "cash" in data


@pytest.mark.asyncio
async def test_trades_endpoint(mock_engine):
    app = create_app(mock_engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/trades")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_set_mode_endpoint(mock_engine):
    app = create_app(mock_engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/mode", json={"mode": "live"})
    assert resp.status_code == 200
    assert mock_engine.config.mode == "live"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/dashboard/test_api.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/trader/dashboard/api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"


class ModeRequest(BaseModel):
    mode: str


class StrategyRequest(BaseModel):
    strategy: str


def create_app(engine) -> FastAPI:
    app = FastAPI(title="Trader Dashboard")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html = (TEMPLATES_DIR / "index.html").read_text()
        return HTMLResponse(content=html)

    @app.get("/api/status")
    async def status():
        prices = {}
        for symbol in engine.config.pairs:
            try:
                prices[symbol] = engine._adapter.get_price(symbol)
            except Exception:
                prices[symbol] = 0.0
        return {
            "mode": engine.config.mode,
            "strategy": engine.config.strategy,
            "pairs": engine.config.pairs,
            "cash": engine.portfolio.cash,
            "positions": engine.portfolio.positions,
            "total_value": engine.portfolio.total_value(prices),
            "prices": prices,
        }

    @app.get("/api/trades")
    async def trades():
        return [
            {"order_id": t.order_id, "symbol": t.symbol, "side": t.side,
             "amount": t.amount, "price": t.price, "mode": t.mode,
             "timestamp": t.timestamp.isoformat(), "pnl": t.pnl, "narrative": t.narrative}
            for t in engine.portfolio.get_trades()
        ]

    @app.post("/api/mode")
    async def set_mode(req: ModeRequest):
        if req.mode not in ("paper", "live"):
            raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")
        engine.config.mode = req.mode
        engine._router._mode = req.mode
        return {"mode": req.mode}

    @app.post("/api/strategy")
    async def set_strategy(req: StrategyRequest):
        from trader.strategies.registry import get_strategy
        if req.strategy not in ("conservative", "moderate", "aggressive"):
            raise HTTPException(status_code=400, detail="invalid strategy")
        engine.config.strategy = req.strategy
        engine._strategy = get_strategy(req.strategy, engine.config.risk)
        return {"strategy": req.strategy}

    @app.post("/api/cycle")
    async def trigger_cycle():
        """Manually trigger one trading cycle."""
        engine.run_cycle()
        return {"status": "cycle complete"}

    return app
```

```html
<!-- src/trader/dashboard/templates/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trader Dashboard</title>
  <style>
    body { font-family: monospace; background: #0d1117; color: #e6edf3; padding: 20px; }
    h1 { color: #58a6ff; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 12px 0; }
    .green { color: #3fb950; } .red { color: #f85149; } .yellow { color: #d29922; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }
    button { background: #238636; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; margin: 4px; }
    button.danger { background: #da3633; }
    select { background: #21262d; color: #e6edf3; border: 1px solid #30363d; padding: 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Trader Dashboard</h1>

  <div class="card" id="status-card">
    <h2>Status</h2>
    <p>Mode: <strong id="mode">-</strong> &nbsp;
       Strategy: <strong id="strategy">-</strong></p>
    <p>Cash: <strong id="cash">-</strong> &nbsp;
       Total Value: <strong id="total">-</strong></p>
    <p>
      <select id="mode-select" onchange="setMode(this.value)">
        <option value="paper">Paper</option>
        <option value="live">Live</option>
      </select>
      <select id="strategy-select" onchange="setStrategy(this.value)">
        <option value="conservative">Conservative</option>
        <option value="moderate">Moderate</option>
        <option value="aggressive">Aggressive</option>
      </select>
      <button onclick="triggerCycle()">Run Cycle Now</button>
    </p>
  </div>

  <div class="card">
    <h2>Trades</h2>
    <table id="trades-table">
      <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Amount</th><th>Price</th><th>Mode</th><th>Narrative</th></tr></thead>
      <tbody id="trades-body"></tbody>
    </table>
  </div>

  <script>
    async function loadStatus() {
      const r = await fetch('/api/status');
      const d = await r.json();
      document.getElementById('mode').textContent = d.mode;
      document.getElementById('strategy').textContent = d.strategy;
      document.getElementById('cash').textContent = '$' + d.cash.toFixed(2);
      document.getElementById('total').textContent = '$' + d.total_value.toFixed(2);
      document.getElementById('mode-select').value = d.mode;
      document.getElementById('strategy-select').value = d.strategy;
    }

    async function loadTrades() {
      const r = await fetch('/api/trades');
      const trades = await r.json();
      const tbody = document.getElementById('trades-body');
      tbody.innerHTML = trades.slice(-50).reverse().map(t => `
        <tr>
          <td>${new Date(t.timestamp).toLocaleString()}</td>
          <td>${t.symbol}</td>
          <td class="${t.side === 'buy' ? 'green' : 'red'}">${t.side.toUpperCase()}</td>
          <td>${t.amount.toFixed(6)}</td>
          <td>$${t.price.toFixed(2)}</td>
          <td class="yellow">${t.mode}</td>
          <td>${t.narrative}</td>
        </tr>`).join('');
    }

    async function setMode(mode) {
      await fetch('/api/mode', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({mode}) });
      loadStatus();
    }

    async function setStrategy(strategy) {
      await fetch('/api/strategy', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({strategy}) });
      loadStatus();
    }

    async function triggerCycle() {
      await fetch('/api/cycle', { method: 'POST' });
      loadStatus(); loadTrades();
    }

    loadStatus(); loadTrades();
    setInterval(() => { loadStatus(); loadTrades(); }, 30000);
  </script>
</body>
</html>
```

**Step 4: Run tests**

```bash
pytest tests/dashboard/test_api.py -v
```
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/trader/dashboard/ tests/dashboard/test_api.py
git commit -m "feat: FastAPI dashboard with status, trades, mode and strategy controls"
```

---

### Task 19: Main Entry Point

**Files:**
- Create: `src/trader/__main__.py`

**Step 1: Write implementation** (no unit test — integration tested via paper trading)

```python
# src/trader/__main__.py
import logging
import argparse
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from trader.config import load_config
from trader.adapters.coinbase import CoinbaseAdapter
from trader.llm.sentiment import SentimentAnalyzer
from trader.collectors.cryptopanic import CryptoPanicCollector
from trader.collectors.reddit import RedditCollector
from trader.core.engine import TradingEngine
from trader.dashboard.api import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Trader Bot")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--db", default="trader.db")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Starting in {cfg.mode.upper()} mode, strategy={cfg.strategy}, pairs={cfg.pairs}")

    adapter = CoinbaseAdapter(api_key=cfg.coinbase.api_key, api_secret=cfg.coinbase.api_secret)
    sentiment = SentimentAnalyzer(model=cfg.ollama.model, base_url=cfg.ollama.base_url)
    collectors = [
        CryptoPanicCollector(api_key=cfg.cryptopanic.api_key),
        RedditCollector(client_id=cfg.reddit.client_id, client_secret=cfg.reddit.client_secret,
                        user_agent=cfg.reddit.user_agent),
    ]

    engine = TradingEngine(config=cfg, adapter=adapter, sentiment_analyzer=sentiment,
                           collectors=collectors, db_path=args.db)

    scheduler = BackgroundScheduler()
    scheduler.add_job(engine.run_cycle, "interval", seconds=cfg.cycle_interval, id="trading_cycle")
    scheduler.start()
    logger.info(f"Scheduler started, cycle every {cfg.cycle_interval}s")

    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test**

```bash
python -m trader --help
```
Expected: prints usage without error.

**Step 3: Commit**

```bash
git add src/trader/__main__.py
git commit -m "feat: main entry point with scheduler and dashboard"
```

---

### Task 20: Docker Setup

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`

**Step 1: Create Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY config.yaml .

EXPOSE 8000

CMD ["python", "-m", "trader", "--config", "config.yaml"]
```

**Step 2: Create docker-compose.yml**

```yaml
# docker-compose.yml
services:
  trader:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - trader-data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
    depends_on:
      - ollama
    command: python -m trader --config config.yaml --db /app/data/trader.db
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama
    restart: unless-stopped

volumes:
  trader-data:
  ollama-models:
```

**Step 3: Create .dockerignore**

```
__pycache__
*.pyc
*.egg-info
.git
tests/
*.db
.env
```

**Step 4: Update config.yaml Ollama URL for Docker**

Change `base_url` in config.yaml to use the Docker service name:
```yaml
ollama:
  model: mistral
  base_url: http://ollama:11434
```

**Step 5: Build and verify**

```bash
docker compose build
```
Expected: builds without error.

**Step 6: Pull Ollama model (first time)**

```bash
docker compose run --rm ollama ollama pull mistral
```

**Step 7: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "feat: Docker and docker-compose setup with Ollama"
```

---

### Task 21: Full Test Suite + Coverage

**Step 1: Run all tests**

```bash
pytest tests/ -v --cov=trader --cov-report=term-missing
```
Expected: all tests pass, coverage >70%

**Step 2: Fix any failures**

Address any import errors or test failures before proceeding.

**Step 3: Commit coverage config**

Add to `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["src/trader"]
omit = ["*/dashboard/templates/*"]
```

```bash
git add pyproject.toml
git commit -m "chore: coverage configuration"
```

---

### Task 22: Smoke Test End-to-End (Paper Mode)

**Goal:** Verify the full system works before live trading.

**Step 1: Start the stack**

```bash
docker compose up -d
```

**Step 2: Check dashboard loads**

Open http://localhost:8000 — should show the dashboard UI.

**Step 3: Trigger a manual cycle**

```bash
curl -X POST http://localhost:8000/api/cycle
```
Expected: `{"status": "cycle complete"}`

**Step 4: Check status**

```bash
curl http://localhost:8000/api/status | python -m json.tool
```
Expected: JSON with mode=paper, cash, positions, total_value.

**Step 5: Check logs**

```bash
docker compose logs trader --tail=50
```
Expected: INFO logs showing cycle running, buy/sell/hold decisions.

**Step 6: Once satisfied, switch to live**

Edit `config.yaml`: `mode: live`

Or via dashboard dropdown → "Live" → confirm Coinbase API keys are set first.

---

## Go-Live Checklist

Before switching `mode: live`:
- [ ] Run paper trading for at least 1 week
- [ ] Review trade history in dashboard — win rate >50%, no unexpected behavior
- [ ] Set real Coinbase API keys in `config.yaml` (or environment variables)
- [ ] Set Reddit PRAW credentials
- [ ] Verify stop-loss is working (check risk config)
- [ ] Start with minimum capital ($10-20) before committing full $100
