# ML Signal Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-tuned weighted-sum score in `SignalGenerator` with a LightGBM classifier trained on real historical price data, falling back to the existing scorer when no model is present.

**Architecture:** A standalone `features.py` extracts raw indicator values from candles into a flat dict. `MLPredictor` loads a saved LightGBM model and turns those features into a `[-1, +1]` score (P(buy) − P(sell)). The engine prefers the ML score when a model is loaded; otherwise it uses `SignalGenerator` unchanged. Two scripts handle data fetching and training offline.

**Tech Stack:** LightGBM, scikit-learn (metrics + split), pyarrow (parquet), CCXT (Coinbase history), alpaca-py (Alpaca history), pandas (already present)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/trader/ml/__init__.py` | Create | Package marker |
| `src/trader/ml/features.py` | Create | Extract raw indicator values from candles → flat dict |
| `src/trader/ml/predictor.py` | Create | Load LightGBM model, `score(candles) -> float \| None` |
| `src/trader/config.py` | Modify | Add `MLConfig` dataclass + env wiring |
| `src/trader/core/engine.py` | Modify | Instantiate `MLPredictor`, use ML score when available |
| `scripts/fetch_training_data.py` | Create | Pull 2y of 1h OHLCV per symbol → `data/training/` parquet |
| `scripts/train_model.py` | Create | Features + labels → train LightGBM → `models/*.lgbm` |
| `pyproject.toml` | Modify | Add `lightgbm`, `scikit-learn`, `pyarrow` |
| `Dockerfile` | Modify | Install new deps + `COPY models/ models/` |
| `tests/ml/test_features.py` | Create | Unit tests for feature extraction |
| `tests/ml/test_predictor.py` | Create | Unit tests for predictor (mocked model) |

---

## Task 1: Feature extraction

**Files:**
- Create: `src/trader/ml/__init__.py`
- Create: `src/trader/ml/features.py`
- Create: `tests/ml/__init__.py`
- Create: `tests/ml/test_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ml/__init__.py` (empty).

Create `tests/ml/test_features.py`:

```python
import math
from trader.models import Candle
from trader.ml.features import compute_features, FEATURE_NAMES

_N = 60  # enough candles for all indicators


def _make_candles(n: int = _N, trend: str = "up") -> list[Candle]:
    from datetime import datetime, timedelta, timezone
    candles = []
    base = 100.0
    for i in range(n):
        if trend == "up":
            close = base + i * 0.5
        else:
            close = base - i * 0.3
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        candles.append(Candle(
            timestamp=ts, open=close - 0.1, high=close + 0.5,
            low=close - 0.5, close=close, volume=1000.0 + i * 10,
        ))
    return candles


def test_returns_all_feature_names():
    candles = _make_candles()
    feat = compute_features(candles)
    assert feat is not None
    for name in FEATURE_NAMES:
        assert name in feat, f"missing feature: {name}"


def test_all_features_are_finite():
    candles = _make_candles()
    feat = compute_features(candles)
    assert feat is not None
    for name, val in feat.items():
        assert math.isfinite(val), f"{name}={val} is not finite"


def test_returns_none_on_insufficient_candles():
    candles = _make_candles(n=10)
    assert compute_features(candles) is None


def test_rsi_reasonable_range():
    candles = _make_candles()
    feat = compute_features(candles)
    assert 0 <= feat["rsi_14"] <= 100


def test_stoch_reasonable_range():
    candles = _make_candles()
    feat = compute_features(candles)
    assert 0 <= feat["stoch_k"] <= 100
    assert 0 <= feat["stoch_d"] <= 100
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd /home/pavel/tools/trader
python -m pytest tests/ml/test_features.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'trader.ml'`

- [ ] **Step 3: Create package marker**

Create `src/trader/ml/__init__.py` (empty file).

- [ ] **Step 4: Implement feature extraction**

Create `src/trader/ml/features.py`:

```python
"""
Extract raw technical indicator values from a candle window into a flat dict.
These are the features fed to the LightGBM model at both training and inference time.
"""
import pandas as pd

try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except Exception:
    _HAS_PANDAS_TA = False

from trader.models import Candle

MIN_CANDLES = 35
VOLUME_AVG_PERIOD = 20

FEATURE_NAMES = [
    "rsi_14",
    "macd_hist_norm",
    "bb_pct",
    "ema_dev_pct",
    "volume_ratio",
    "adx",
    "stoch_k",
    "stoch_d",
    "stoch_crossover",
]


def compute_features(candles: list[Candle]) -> dict[str, float] | None:
    """
    Return a dict of feature name → float for the most recent candle in the window.
    Returns None if there are not enough candles to compute all indicators.
    """
    if len(candles) < MIN_CANDLES:
        return None

    df = pd.DataFrame([
        {"close": c.close, "high": c.high, "low": c.low, "volume": c.volume}
        for c in candles
    ])
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    features: dict[str, float] = {}

    # --- RSI ---
    try:
        if _HAS_PANDAS_TA:
            rsi_s = ta.rsi(close, length=14)
        else:
            rsi_s = _rsi(close)
        rsi_val = float(rsi_s.iloc[-1])
        features["rsi_14"] = rsi_val if pd.notna(rsi_val) else 50.0
    except Exception:
        features["rsi_14"] = 50.0

    # --- MACD histogram (normalised by mean price) ---
    try:
        if _HAS_PANDAS_TA:
            macd_df = ta.macd(close)
            hist_col = [c for c in macd_df.columns if "MACDh" in c] if macd_df is not None else []
            hist = float(macd_df[hist_col[0]].iloc[-1]) if hist_col else 0.0
        else:
            hist_s = _macd_hist(close)
            hist = float(hist_s.iloc[-1])
        mean_price = float(close.mean())
        features["macd_hist_norm"] = (hist / (mean_price * 0.002)) if mean_price > 0 else 0.0
        features["macd_hist_norm"] = max(-5.0, min(5.0, features["macd_hist_norm"]))
    except Exception:
        features["macd_hist_norm"] = 0.0

    # --- Bollinger Band %b (position within bands, 0=lower, 1=upper) ---
    try:
        if _HAS_PANDAS_TA:
            bb = ta.bbands(close, length=20)
            lower_col = [c for c in bb.columns if "BBL" in c] if bb is not None else []
            upper_col = [c for c in bb.columns if "BBU" in c] if bb is not None else []
            lower = float(bb[lower_col[0]].iloc[-1]) if lower_col else None
            upper = float(bb[upper_col[0]].iloc[-1]) if upper_col else None
        else:
            lower_s, _, upper_s = _bbands(close)
            lower = float(lower_s.iloc[-1])
            upper = float(upper_s.iloc[-1])
        price = float(close.iloc[-1])
        if lower is not None and upper is not None and pd.notna(lower) and pd.notna(upper) and upper != lower:
            features["bb_pct"] = (price - lower) / (upper - lower)
            features["bb_pct"] = max(0.0, min(1.0, features["bb_pct"]))
        else:
            features["bb_pct"] = 0.5
    except Exception:
        features["bb_pct"] = 0.5

    # --- EMA deviation % (price distance from 50-EMA, as fraction) ---
    try:
        if _HAS_PANDAS_TA:
            ema_s = ta.ema(close, length=50)
        else:
            ema_s = _ema(close, 50)
        ema_val = float(ema_s.iloc[-1])
        price = float(close.iloc[-1])
        features["ema_dev_pct"] = ((price - ema_val) / ema_val) if ema_val > 0 else 0.0
        features["ema_dev_pct"] = max(-0.2, min(0.2, features["ema_dev_pct"]))
    except Exception:
        features["ema_dev_pct"] = 0.0

    # --- Volume ratio (current / 20-bar average) ---
    try:
        if len(volume) >= VOLUME_AVG_PERIOD:
            vol_avg = float(volume.iloc[-VOLUME_AVG_PERIOD:].mean())
            current_vol = float(volume.iloc[-1])
            features["volume_ratio"] = (current_vol / vol_avg) if vol_avg > 0 else 1.0
            features["volume_ratio"] = max(0.0, min(5.0, features["volume_ratio"]))
        else:
            features["volume_ratio"] = 1.0
    except Exception:
        features["volume_ratio"] = 1.0

    # --- ADX ---
    try:
        if _HAS_PANDAS_TA:
            adx_df = ta.adx(high, low, close, length=14)
            adx_col = [c for c in adx_df.columns if c.startswith("ADX")] if adx_df is not None else []
            adx_val = float(adx_df[adx_col[0]].iloc[-1]) if adx_col else 20.0
        else:
            adx_s = _adx(high, low, close)
            adx_val = float(adx_s.iloc[-1])
        features["adx"] = adx_val if pd.notna(adx_val) else 20.0
        features["adx"] = max(0.0, min(100.0, features["adx"]))
    except Exception:
        features["adx"] = 20.0

    # --- Stochastic %K, %D, crossover ---
    try:
        if _HAS_PANDAS_TA:
            stoch_df = ta.stoch(high, low, close, k=14, d=3)
            k_col = [c for c in stoch_df.columns if "STOCHk" in c] if stoch_df is not None else []
            d_col = [c for c in stoch_df.columns if "STOCHd" in c] if stoch_df is not None else []
            if k_col and d_col:
                stoch_k = float(stoch_df[k_col[0]].iloc[-1])
                stoch_d = float(stoch_df[d_col[0]].iloc[-1])
                stoch_k_prev = float(stoch_df[k_col[0]].iloc[-2]) if len(stoch_df) > 1 else stoch_k
                stoch_d_prev = float(stoch_df[d_col[0]].iloc[-2]) if len(stoch_df) > 1 else stoch_d
            else:
                stoch_k = stoch_d = stoch_k_prev = stoch_d_prev = 50.0
        else:
            k_s, d_s = _stochastic(high, low, close)
            stoch_k = float(k_s.iloc[-1])
            stoch_d = float(d_s.iloc[-1])
            stoch_k_prev = float(k_s.iloc[-2]) if len(k_s) > 1 else stoch_k
            stoch_d_prev = float(d_s.iloc[-2]) if len(d_s) > 1 else stoch_d

        features["stoch_k"] = stoch_k if pd.notna(stoch_k) else 50.0
        features["stoch_d"] = stoch_d if pd.notna(stoch_d) else 50.0

        if stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d:
            features["stoch_crossover"] = 1.0   # bullish crossover
        elif stoch_k_prev >= stoch_d_prev and stoch_k < stoch_d:
            features["stoch_crossover"] = -1.0  # bearish crossover
        else:
            features["stoch_crossover"] = 0.0
    except Exception:
        features["stoch_k"] = 50.0
        features["stoch_d"] = 50.0
        features["stoch_crossover"] = 0.0

    # Verify all expected features are present
    for name in FEATURE_NAMES:
        if name not in features:
            return None

    return features


# ── Manual indicator fallbacks (no pandas-ta dependency) ─────────────────────

def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _macd_hist(series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _bbands(series: pd.Series, length: int = 20):
    mid = series.rolling(length).mean()
    std = series.rolling(length).std()
    return mid - 2 * std, mid, mid + 2 * std


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_s = tr.rolling(window=length).mean().replace(0, float("nan"))
    plus_di = 100 * (plus_dm.rolling(window=length).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(window=length).mean() / atr_s)
    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.rolling(window=length).mean()


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, float("nan"))
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k, d
```

- [ ] **Step 5: Run tests — expect pass**

```bash
python -m pytest tests/ml/test_features.py -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add src/trader/ml/__init__.py src/trader/ml/features.py tests/ml/__init__.py tests/ml/test_features.py
git commit -m "feat: ML feature extraction from candles"
```

---

## Task 2: ML Predictor

**Files:**
- Create: `src/trader/ml/predictor.py`
- Create: `tests/ml/test_predictor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ml/test_predictor.py`:

```python
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone
from trader.models import Candle
from trader.ml.predictor import MLPredictor

_N = 60


def _make_candles(n: int = _N) -> list[Candle]:
    candles = []
    base = 100.0
    for i in range(n):
        close = base + i * 0.3
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        candles.append(Candle(
            timestamp=ts, open=close - 0.1, high=close + 0.5,
            low=close - 0.5, close=close, volume=1000.0,
        ))
    return candles


def test_score_returns_none_without_model():
    predictor = MLPredictor(model_path="nonexistent.lgbm")
    assert predictor.score(_make_candles()) is None


def test_score_returns_none_on_insufficient_candles():
    predictor = MLPredictor(model_path="nonexistent.lgbm")
    assert predictor.score(_make_candles(n=10)) is None


def test_score_in_range_with_mock_model():
    mock_model = MagicMock()
    # LightGBM predict_proba returns shape (n_samples, n_classes): [P(buy), P(hold), P(sell)]
    mock_model.predict.return_value = np.array([[0.6, 0.3, 0.1]])

    predictor = MLPredictor(model_path="any.lgbm")
    predictor._model = mock_model

    score = predictor.score(_make_candles())
    assert score is not None
    assert -1.0 <= score <= 1.0


def test_score_bullish_prediction():
    """High P(buy), low P(sell) → positive score."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8, 0.15, 0.05]])

    predictor = MLPredictor(model_path="any.lgbm")
    predictor._model = mock_model

    score = predictor.score(_make_candles())
    assert score > 0


def test_score_bearish_prediction():
    """Low P(buy), high P(sell) → negative score."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.05, 0.15, 0.8]])

    predictor = MLPredictor(model_path="any.lgbm")
    predictor._model = mock_model

    score = predictor.score(_make_candles())
    assert score < 0
```

- [ ] **Step 2: Run tests — expect failure**

```bash
python -m pytest tests/ml/test_predictor.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'MLPredictor'`

- [ ] **Step 3: Implement predictor**

Create `src/trader/ml/predictor.py`:

```python
"""
ML-based signal scorer. Loads a saved LightGBM model and returns a score in [-1, +1].

Score formula: P(buy) - P(sell)
  - Classes are ordered: 0=buy, 1=hold, 2=sell  (matches training label encoding)
  - Positive score → bullish, negative → bearish, near-zero → hold

Falls back gracefully: returns None when no model is loaded or data is insufficient.
The engine treats None as "use SignalGenerator score instead".
"""
import logging
import numpy as np
from trader.models import Candle
from trader.ml.features import compute_features, FEATURE_NAMES

logger = logging.getLogger(__name__)


class MLPredictor:
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=self._model_path)
            logger.info("ML model loaded from %s", self._model_path)
        except Exception as e:
            logger.info("ML model not loaded (%s) — will use technical scorer", e)
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def score(self, candles: list[Candle]) -> float | None:
        """
        Return a score in [-1.0, +1.0] or None if model unavailable / insufficient data.
        Score = P(buy) - P(sell).
        """
        if self._model is None:
            return None

        features = compute_features(candles)
        if features is None:
            return None

        x = np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float32)
        try:
            probs = self._model.predict(x)   # shape (1, 3): [P(buy), P(hold), P(sell)]
            p_buy = float(probs[0][0])
            p_sell = float(probs[0][2])
            return float(np.clip(p_buy - p_sell, -1.0, 1.0))
        except Exception as e:
            logger.warning("ML prediction failed: %s", e)
            return None
```

- [ ] **Step 4: Run tests — expect pass**

```bash
python -m pytest tests/ml/test_predictor.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/trader/ml/predictor.py tests/ml/test_predictor.py
git commit -m "feat: MLPredictor — loads LightGBM model, scores candles"
```

---

## Task 3: Config + engine integration

**Files:**
- Modify: `src/trader/config.py`
- Modify: `src/trader/core/engine.py`
- Modify: `tests/core/test_engine.py` (add one test)

- [ ] **Step 1: Add MLConfig to config.py**

Open `src/trader/config.py`. After the `MimoConfig` dataclass, add:

```python
@dataclass
class MLConfig:
    enabled: bool = False
    model_path: str = "models/crypto.lgbm"
    min_confidence: float = 0.0   # reserved for future threshold filtering
```

In the `Config` dataclass, add after the `mimo` field:
```python
    ml: MLConfig = field(default_factory=MLConfig)
```

In `load_config`, add `("ml", MLConfig)` to the key/class list:
```python
        ("ml", MLConfig),
```

- [ ] **Step 2: Verify config loads cleanly**

```bash
python -c "from trader.config import load_config; c = load_config('config.yaml'); print(c.ml)"
```

Expected: `MLConfig(enabled=False, model_path='models/crypto.lgbm', min_confidence=0.0)`

- [ ] **Step 3: Wire MLPredictor into engine**

Open `src/trader/core/engine.py`. Add import at top:

```python
from trader.ml.predictor import MLPredictor
```

In `TradingEngine.__init__`, after `self._signals = SignalGenerator()`, add:

```python
        self._ml: MLPredictor | None = None
        if config.ml.enabled:
            self._ml = MLPredictor(model_path=config.ml.model_path)
            if not self._ml.is_loaded:
                self._ml = None
```

In `_process_symbol`, find where `tech_signal` is created (the `score_with_trend` call). After that block, add ML score override:

```python
        # ML override: if model loaded, replace technical score with ML score
        if self._ml is not None:
            ml_score = self._ml.score(candles)
            if ml_score is not None:
                tech_signal = Signal(score=ml_score, trend_bullish=tech_signal.trend_bullish)
```

- [ ] **Step 4: Run existing engine tests**

```bash
python -m pytest tests/core/test_engine.py -v
```

Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/trader/config.py src/trader/core/engine.py
git commit -m "feat: wire MLPredictor into engine, falls back to SignalGenerator"
```

---

## Task 4: Data fetching script

**Files:**
- Create: `scripts/fetch_training_data.py`
- Create: `data/training/.gitkeep`

- [ ] **Step 1: Create data directory**

```bash
mkdir -p /home/pavel/tools/trader/data/training
touch /home/pavel/tools/trader/data/training/.gitkeep
```

- [ ] **Step 2: Create fetch script**

Create `scripts/fetch_training_data.py`:

```python
#!/usr/bin/env python3
"""
Fetch 2 years of 1h OHLCV candles for all configured symbols.
Saves one parquet file per symbol to data/training/{exchange}/{symbol}.parquet

Usage:
  python scripts/fetch_training_data.py --exchange coinbase
  python scripts/fetch_training_data.py --exchange alpaca
  python scripts/fetch_training_data.py --exchange all

Requires env vars: COINBASE_API_KEY, COINBASE_API_SECRET (for coinbase)
                   ALPACA_API_KEY, ALPACA_API_SECRET (for alpaca)
"""
import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

CRYPTO_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD",
    "DOT/USD", "LINK/USD", "LTC/USD", "AVAX/USD", "DOGE/USD",
]
STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "MA",
]
LOOKBACK_DAYS = 730   # 2 years
OUTPUT_DIR = Path("data/training")


def fetch_coinbase(symbols: list[str]) -> None:
    import ccxt
    exchange = ccxt.coinbase({
        "apiKey": os.environ.get("COINBASE_API_KEY", ""),
        "secret": os.environ.get("COINBASE_API_SECRET", ""),
    })
    out_dir = OUTPUT_DIR / "coinbase"
    out_dir.mkdir(parents=True, exist_ok=True)

    since_ms = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)

    for symbol in symbols:
        print(f"Fetching {symbol} from Coinbase...")
        all_rows = []
        fetch_since = since_ms
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", since=fetch_since, limit=300)
            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")
                break
            if not ohlcv:
                break
            all_rows.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts <= fetch_since:
                break
            fetch_since = last_ts + 1
            if last_ts >= int(datetime.now(timezone.utc).timestamp() * 1000) - 3_600_000:
                break
            time.sleep(0.3)  # rate limit

        if not all_rows:
            print(f"  No data for {symbol}, skipping.")
            continue

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        safe_name = symbol.replace("/", "_")
        path = out_dir / f"{safe_name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  Saved {len(df)} rows → {path}")


def fetch_alpaca(symbols: list[str]) -> None:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(
        api_key=os.environ.get("ALPACA_API_KEY", ""),
        secret_key=os.environ.get("ALPACA_API_SECRET", ""),
    )
    out_dir = OUTPUT_DIR / "alpaca"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

    for symbol in symbols:
        print(f"Fetching {symbol} from Alpaca...")
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
            )
            bars = client.get_stock_bars(req)
            df = bars.df.reset_index()
            # Alpaca returns MultiIndex (symbol, timestamp) → flatten
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol].drop(columns=["symbol"])
            df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high",
                                    "l": "low", "c": "close", "v": "volume"})
            keep = ["timestamp", "open", "high", "low", "close", "volume"]
            df = df[[c for c in keep if c in df.columns]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

            path = out_dir / f"{symbol}.parquet"
            df.to_parquet(path, index=False)
            print(f"  Saved {len(df)} rows → {path}")
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", choices=["coinbase", "alpaca", "all"], default="all")
    args = parser.parse_args()

    if args.exchange in ("coinbase", "all"):
        fetch_coinbase(CRYPTO_SYMBOLS)
    if args.exchange in ("alpaca", "all"):
        fetch_alpaca(STOCK_SYMBOLS)

    print("\nDone. Files saved to data/training/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Fetch the data (run with your .env loaded)**

```bash
cd /home/pavel/tools/trader
set -a && source .env && set +a
python scripts/fetch_training_data.py --exchange coinbase
```

Expected output: lines like `Saved 14000 rows → data/training/coinbase/BTC_USD.parquet`

Then fetch stocks:
```bash
python scripts/fetch_training_data.py --exchange alpaca
```

- [ ] **Step 4: Verify parquet files**

```bash
python -c "
import pandas as pd
from pathlib import Path
for p in sorted(Path('data/training').rglob('*.parquet')):
    df = pd.read_parquet(p)
    print(f'{p.name}: {len(df)} rows, {df.timestamp.min()} → {df.timestamp.max()}')
"
```

Expected: 10+ files, each with 10,000–18,000 rows spanning ~2 years.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_training_data.py data/training/.gitkeep
git commit -m "feat: script to fetch 2y historical OHLCV for training"
```

---

## Task 5: Training script

**Files:**
- Create: `scripts/train_model.py`
- Create: `models/.gitkeep`

- [ ] **Step 1: Create models directory**

```bash
mkdir -p /home/pavel/tools/trader/models
touch /home/pavel/tools/trader/models/.gitkeep
```

- [ ] **Step 2: Create training script**

Create `scripts/train_model.py`:

```python
#!/usr/bin/env python3
"""
Train a LightGBM classifier on historical OHLCV data.

For each candle window, compute 9 technical indicator features.
Label = price direction 4 hours forward:
  0 = BUY   (return > +1%)
  1 = HOLD  (-1% <= return <= +1%)
  2 = SELL  (return < -1%)

Trains one model per exchange and saves to models/crypto.lgbm and models/stocks.lgbm.

Usage:
  python scripts/train_model.py
  python scripts/train_model.py --exchange coinbase
  python scripts/train_model.py --exchange alpaca
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing trader package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trader.models import Candle
from trader.ml.features import compute_features, FEATURE_NAMES

FORWARD_HOURS = 4      # predict price direction this many candles ahead
BUY_THRESHOLD = 0.01   # +1% → BUY
SELL_THRESHOLD = -0.01  # -1% → SELL
WINDOW = 60            # candle history fed to compute_features
LABEL_BUY = 0
LABEL_HOLD = 1
LABEL_SELL = 2


def load_parquet_as_candles(path: Path) -> list[Candle]:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    candles = []
    for row in df.itertuples():
        candles.append(Candle(
            timestamp=row.timestamp,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        ))
    return candles


def build_dataset(candles: list[Candle]) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over candles and build (X, y) arrays."""
    X_rows = []
    y_rows = []

    for i in range(WINDOW, len(candles) - FORWARD_HOURS):
        window = candles[i - WINDOW: i]
        features = compute_features(window)
        if features is None:
            continue

        entry_price = candles[i].close
        exit_price = candles[i + FORWARD_HOURS].close
        forward_return = (exit_price - entry_price) / entry_price

        if forward_return > BUY_THRESHOLD:
            label = LABEL_BUY
        elif forward_return < SELL_THRESHOLD:
            label = LABEL_SELL
        else:
            label = LABEL_HOLD

        X_rows.append([features[name] for name in FEATURE_NAMES])
        y_rows.append(label)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


def train(exchange: str, parquet_dir: Path, output_path: Path) -> None:
    import lightgbm as lgb
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {parquet_dir}")
        return

    print(f"\n=== Training {exchange} model ===")
    all_X, all_y = [], []

    for pf in parquet_files:
        print(f"  Loading {pf.name}...")
        candles = load_parquet_as_candles(pf)
        X, y = build_dataset(candles)
        if len(X) == 0:
            print(f"    Skipped (insufficient data)")
            continue
        all_X.append(X)
        all_y.append(y)
        label_counts = {0: (y == 0).sum(), 1: (y == 1).sum(), 2: (y == 2).sum()}
        print(f"    {len(X)} samples — BUY={label_counts[0]}, HOLD={label_counts[1]}, SELL={label_counts[2]}")

    if not all_X:
        print("No training data. Aborting.")
        return

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"\nTotal samples: {len(X)}")
    print(f"Label distribution — BUY: {(y==0).sum()}, HOLD: {(y==1).sum()}, SELL: {(y==2).sum()}")

    # Time-ordered split: first 80% train, last 20% test
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Class weights to handle imbalance (HOLD dominates)
    from collections import Counter
    counts = Counter(y_train.tolist())
    total = sum(counts.values())
    class_weight = {k: total / (3 * v) for k, v in counts.items()}
    sample_weight = np.array([class_weight[int(label)] for label in y_train])

    print("\nTraining LightGBM...")
    train_ds = lgb.Dataset(X_train, label=y_train, weight=sample_weight,
                           feature_name=FEATURE_NAMES)
    val_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=300,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    # Evaluate
    probs = model.predict(X_test)           # shape (n, 3)
    preds = np.argmax(probs, axis=1)
    print("\nTest set classification report:")
    print(classification_report(y_test, preds, target_names=["BUY", "HOLD", "SELL"]))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    print(f"Model saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", choices=["coinbase", "alpaca", "all"], default="all")
    args = parser.parse_args()

    base = Path("data/training")
    models = Path("models")

    if args.exchange in ("coinbase", "all"):
        train("coinbase", base / "coinbase", models / "crypto.lgbm")
    if args.exchange in ("alpaca", "all"):
        train("alpaca", base / "alpaca", models / "stocks.lgbm")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Install training dependencies locally**

```bash
pip install lightgbm scikit-learn pyarrow --break-system-packages -q
```

- [ ] **Step 4: Run training**

```bash
cd /home/pavel/tools/trader
python scripts/train_model.py --exchange coinbase
```

Expected output ends with:
```
Test set classification report:
              precision    recall  f1-score   support
         BUY       ...
        HOLD       ...
        SELL       ...
Model saved → models/crypto.lgbm
```

Then train stocks:
```bash
python scripts/train_model.py --exchange alpaca
```

- [ ] **Step 5: Verify model files exist**

```bash
ls -lh models/
```

Expected: `crypto.lgbm` and `stocks.lgbm`, each a few MB.

- [ ] **Step 6: Quick smoke test — load and score**

```bash
python -c "
from trader.ml.predictor import MLPredictor
from datetime import datetime, timedelta, timezone
from trader.models import Candle

p = MLPredictor('models/crypto.lgbm')
print('loaded:', p.is_loaded)

candles = []
for i in range(60):
    c = 100 + i * 0.3
    ts = datetime(2024,1,1,tzinfo=timezone.utc) + timedelta(hours=i)
    candles.append(Candle(timestamp=ts, open=c-0.1, high=c+0.5, low=c-0.5, close=c, volume=1000))

print('score:', p.score(candles))
"
```

Expected: `loaded: True` and a float between -1 and 1.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_model.py models/.gitkeep
git commit -m "feat: LightGBM training script — features + labels + time-split eval"
```

---

## Task 6: Enable ML in configs + update Docker

**Files:**
- Modify: `config.yaml`
- Modify: `config-stocks.yaml`
- Modify: `Dockerfile`
- Modify: `pyproject.toml`

- [ ] **Step 1: Enable ML in config.yaml**

Add to `config.yaml`:
```yaml
ml:
  enabled: true
  model_path: models/crypto.lgbm
  min_confidence: 0.0
```

- [ ] **Step 2: Enable ML in config-stocks.yaml**

Also fix the stale `claude:` section in config-stocks.yaml — replace it with `mimo:`.

Add to `config-stocks.yaml`:
```yaml
mimo:
  model: mimo-v2-flash
  api_key: ''
ml:
  enabled: true
  model_path: models/stocks.lgbm
  min_confidence: 0.0
```

Remove the old `claude:` block from that file.

- [ ] **Step 3: Update Dockerfile**

In `Dockerfile`, update the pip install line to add the three new packages and copy models:

```dockerfile
RUN pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir pandas-ta --no-deps && \
    pip install --no-cache-dir -e . --no-deps && \
    pip install --no-cache-dir ccxt pandas numpy praw requests openai \
        fastapi uvicorn pyyaml pydantic jinja2 apscheduler alpaca-py pytz pytrends \
        lightgbm scikit-learn pyarrow

COPY models/ models/
```

The `COPY models/ models/` line goes after the existing `COPY src/ src/` line.

- [ ] **Step 4: Update pyproject.toml**

Add to the `dependencies` list:
```toml
    "lightgbm>=4.0",
    "scikit-learn>=1.3",
    "pyarrow>=14.0",
```

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v --ignore=tests/adapters 2>&1 | tail -20
```

Expected: all tests pass (no regressions).

- [ ] **Step 6: Commit**

```bash
git add config.yaml config-stocks.yaml Dockerfile pyproject.toml
git commit -m "feat: enable ML scorer in both configs, update Docker deps"
```

---

## Task 7: Deploy to Vultr

- [ ] **Step 1: Rsync to server**

```bash
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  /home/pavel/tools/trader/ root@<vultr-ip>:/opt/trader/
```

- [ ] **Step 2: Rebuild and restart on server**

SSH in and run:
```bash
cd /opt/trader && docker compose up -d --build
```

- [ ] **Step 3: Verify both services started**

```bash
docker compose logs --tail=30 trader
docker compose logs --tail=30 trader-stocks
```

Expected: logs show `ML model loaded from models/crypto.lgbm` / `models/stocks.lgbm`.

---

## .gitignore additions

Add to `.gitignore` so raw training data isn't committed (models are committed since they're small):

```
data/training/coinbase/
data/training/alpaca/
```

```bash
git add .gitignore
git commit -m "chore: ignore raw training parquet files"
```
