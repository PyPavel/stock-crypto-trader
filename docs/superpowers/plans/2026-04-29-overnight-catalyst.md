# Overnight Catalyst Collector + Time Window Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the stocks trader a specific overnight edge by (1) scoring stocks that have a post-close earnings release or material SEC 8-K filing today, and (2) restricting buys to 3–4pm ET and signal sells to 9:30–10:30am ET.

**Architecture:** `OvernightCatalystCollector` slots into the existing `numeric_collectors` list — no engine changes needed for it. `TimeWindowGate` is a tiny class injected into `TradingEngine.__init__` and checked in `_execute_decisions` after the strategy decision. `TimeGateConfig` is a new dataclass added to `config.py` and parsed in `load_config`.

**Tech Stack:** Python 3.10+, yfinance, requests (already in deps), zoneinfo (stdlib), pytest

---

### Task 1: Add TimeGateConfig to config.py

**Files:**
- Modify: `src/trader/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_config.py`:

```python
def test_time_gate_config_defaults():
    from trader.config import TimeGateConfig
    tg = TimeGateConfig()
    assert tg.enabled is False
    assert tg.buy_start == "15:00"
    assert tg.buy_end == "16:00"
    assert tg.sell_start == "09:30"
    assert tg.sell_end == "10:30"


def test_load_config_parses_time_gate(tmp_path):
    from trader.config import load_config
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("""
exchange: alpaca
mode: paper
strategy: moderate
capital: 1000
pairs: [AAPL]
time_gate:
  enabled: true
  buy_start: "14:30"
  buy_end: "15:30"
  sell_start: "09:30"
  sell_end: "10:00"
""")
    cfg = load_config(str(cfg_file))
    assert cfg.time_gate.enabled is True
    assert cfg.time_gate.buy_start == "14:30"
    assert cfg.time_gate.sell_end == "10:00"


def test_load_config_time_gate_defaults_when_absent(tmp_path):
    from trader.config import load_config
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("""
exchange: alpaca
mode: paper
strategy: moderate
capital: 1000
pairs: [AAPL]
""")
    cfg = load_config(str(cfg_file))
    assert cfg.time_gate.enabled is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/pavel/tools/trader
python -m pytest tests/test_config.py::test_time_gate_config_defaults tests/test_config.py::test_load_config_parses_time_gate tests/test_config.py::test_load_config_time_gate_defaults_when_absent -v
```

Expected: FAIL — `ImportError: cannot import name 'TimeGateConfig'`

- [ ] **Step 3: Add TimeGateConfig dataclass and wire into Config + load_config**

In `src/trader/config.py`, add after the `UniverseConfig` dataclass (before `RiskConfig`):

```python
@dataclass
class TimeGateConfig:
    enabled: bool = False
    buy_start: str = "15:00"   # ET, inclusive
    buy_end: str = "16:00"     # ET, inclusive
    sell_start: str = "09:30"  # ET, inclusive
    sell_end: str = "10:30"    # ET, inclusive
```

In the `Config` dataclass, add after `universe`:

```python
    time_gate: TimeGateConfig = field(default_factory=TimeGateConfig)
```

In `load_config`, add `("time_gate", TimeGateConfig)` to the `for key, cls in [...]` loop:

```python
    for key, cls in [
        ("mimo", MimoConfig),
        ("coinbase", CoinbaseConfig),
        ("alpaca", AlpacaConfig),
        ("tastytrade", TastyTradeConfig),
        ("reddit", RedditConfig),
        ("cryptopanic", CryptoPanicConfig),
        ("discord", DiscordConfig),
        ("llm_advisor", LLMAdvisorConfig),
        ("risk", RiskConfig),
        ("ml", MLConfig),
        ("telegram", TelegramConfig),
        ("universe", UniverseConfig),
        ("time_gate", TimeGateConfig),   # ← add this line
    ]:
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/pavel/tools/trader
python -m pytest tests/test_config.py::test_time_gate_config_defaults tests/test_config.py::test_load_config_parses_time_gate tests/test_config.py::test_load_config_time_gate_defaults_when_absent -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd /home/pavel/tools/trader
git add src/trader/config.py tests/test_config.py
git commit -m "feat: add TimeGateConfig to config (buy/sell time windows, disabled by default)"
```

---

### Task 2: Implement TimeWindowGate

**Files:**
- Create: `src/trader/core/time_gate.py`
- Create: `tests/core/test_time_gate.py`

- [ ] **Step 1: Write failing tests**

Create `tests/core/test_time_gate.py`:

```python
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch
from trader.core.time_gate import TimeWindowGate
from trader.config import TimeGateConfig

_ET = ZoneInfo("America/New_York")


def _gate(enabled=True, buy_start="15:00", buy_end="16:00",
          sell_start="09:30", sell_end="10:30"):
    cfg = TimeGateConfig(enabled=enabled, buy_start=buy_start, buy_end=buy_end,
                         sell_start=sell_start, sell_end=sell_end)
    return TimeWindowGate(cfg)


def _mock_et(hour, minute):
    """Return an ET-aware datetime for the given wall-clock time."""
    return datetime(2026, 4, 29, hour, minute, 0, tzinfo=_ET)


def test_can_buy_inside_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(15, 30)
        assert gate.can_buy() is True


def test_can_buy_at_start_of_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(15, 0)
        assert gate.can_buy() is True


def test_can_buy_at_end_of_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(16, 0)
        assert gate.can_buy() is True


def test_cannot_buy_before_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(10, 0)
        assert gate.can_buy() is False


def test_cannot_buy_after_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(16, 1)
        assert gate.can_buy() is False


def test_can_sell_inside_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(10, 0)
        assert gate.can_sell() is True


def test_cannot_sell_outside_window():
    gate = _gate()
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(14, 0)
        assert gate.can_sell() is False


def test_disabled_gate_always_allows_buy():
    gate = _gate(enabled=False)
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(10, 0)   # outside buy window
        assert gate.can_buy() is True


def test_disabled_gate_always_allows_sell():
    gate = _gate(enabled=False)
    with patch("trader.core.time_gate.datetime") as mock_dt:
        mock_dt.now.return_value = _mock_et(14, 0)   # outside sell window
        assert gate.can_sell() is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/pavel/tools/trader
python -m pytest tests/core/test_time_gate.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'trader.core.time_gate'`

- [ ] **Step 3: Implement TimeWindowGate**

Create `src/trader/core/time_gate.py`:

```python
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from trader.config import TimeGateConfig

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


def _parse_hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


class TimeWindowGate:
    """
    Gates buy and sell decisions to specific time windows (US/Eastern).
    When disabled, both can_buy() and can_sell() always return True.
    """

    def __init__(self, config: TimeGateConfig) -> None:
        self._enabled = config.enabled
        self._buy_start = _parse_hhmm(config.buy_start)
        self._buy_end = _parse_hhmm(config.buy_end)
        self._sell_start = _parse_hhmm(config.sell_start)
        self._sell_end = _parse_hhmm(config.sell_end)

    def can_buy(self) -> bool:
        if not self._enabled:
            return True
        return self._in_window(self._buy_start, self._buy_end)

    def can_sell(self) -> bool:
        if not self._enabled:
            return True
        return self._in_window(self._sell_start, self._sell_end)

    def _in_window(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        now = datetime.now(tz=_ET)
        current = (now.hour, now.minute)
        return start <= current <= end
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/pavel/tools/trader
python -m pytest tests/core/test_time_gate.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
cd /home/pavel/tools/trader
git add src/trader/core/time_gate.py tests/core/test_time_gate.py
git commit -m "feat: add TimeWindowGate — restricts buys to 3-4pm ET, sells to 9:30-10:30am ET"
```

---

### Task 3: Wire TimeWindowGate into TradingEngine

**Files:**
- Modify: `src/trader/core/engine.py` (lines ~22–45 for `__init__`, lines ~510–545 for `_execute_decisions`)

- [ ] **Step 1: Write failing test**

Add to `tests/core/test_engine.py` (append at bottom):

```python
def test_time_gate_blocks_buy_outside_window(tmp_path):
    """TimeWindowGate that always blocks buys converts buy decision to hold."""
    from unittest.mock import MagicMock
    engine = make_engine(tmp_path)
    gate = MagicMock()
    gate.can_buy.return_value = False
    gate.can_sell.return_value = True
    engine._time_gate = gate

    engine._strategy.decide = MagicMock(return_value={
        "action": "buy", "usd_amount": 100.0, "reason": "test"
    })
    engine._advisor = None
    engine._pdt = None

    result = {
        "price": 100.0, "candles": [], "tech_signal": MagicMock(score=0.5, trend_bullish=True, rsi=50),
        "sentiment": MagicMock(score=0.5), "tech_score": 0.5, "ml_score": None,
        "trend_bullish": True, "raw_sentiment": 0.5, "combined_score": 0.5,
        "atr": 1.0, "texts": [],
    }
    with patch.object(engine, "_execute_buy"), patch.object(engine, "_execute_sell"):
        engine._execute_decisions("AAPL", result, {}, can_buy=True)
        engine._execute_buy.assert_not_called()


def test_time_gate_blocks_sell_outside_window(tmp_path):
    """TimeWindowGate that always blocks sells converts sell decision to hold."""
    from unittest.mock import MagicMock, patch
    engine = make_engine(tmp_path)
    gate = MagicMock()
    gate.can_buy.return_value = True
    gate.can_sell.return_value = False
    engine._time_gate = gate

    engine._strategy.decide = MagicMock(return_value={
        "action": "sell", "usd_amount": 100.0, "reason": "test"
    })
    engine._advisor = None
    engine._pdt = None

    # Inject a position so the strategy sell path is reached
    engine.portfolio.positions["AAPL"] = {"amount": 1.0, "entry_price": 100.0}
    engine._peak_prices["AAPL"] = 100.0

    result = {
        "price": 100.0, "candles": [], "tech_signal": MagicMock(score=-0.5, trend_bullish=False, rsi=30),
        "sentiment": MagicMock(score=-0.5), "tech_score": -0.5, "ml_score": None,
        "trend_bullish": False, "raw_sentiment": -0.5, "combined_score": -0.5,
        "atr": 1.0, "texts": [],
    }
    with patch.object(engine, "_execute_sell") as mock_sell:
        engine._execute_decisions("AAPL", result, {"AAPL": 100.0}, can_buy=False)
        # stop-loss won't fire (entry == price, no loss), so _execute_sell is only
        # reachable via strategy path — which is blocked by the time gate
        mock_sell.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/pavel/tools/trader
python -m pytest tests/core/test_engine.py::test_time_gate_blocks_buy_outside_window tests/core/test_engine.py::test_time_gate_blocks_sell_outside_window -v
```

Expected: FAIL — `AttributeError: 'TradingEngine' object has no attribute '_time_gate'`

- [ ] **Step 3: Add time_gate param to TradingEngine.__init__**

In `src/trader/core/engine.py`, add to the import block at the top:

```python
from trader.core.time_gate import TimeWindowGate
```

In `TradingEngine.__init__` signature, add after `universe=None`:

```python
        time_gate: TimeWindowGate | None = None,
```

In `TradingEngine.__init__` body, add after `self._pdt = ...` line:

```python
        self._time_gate = time_gate
```

- [ ] **Step 4: Add time gate guards in _execute_decisions**

In `src/trader/core/engine.py`, in `_execute_decisions`, find the comment `# 5b. LLM Advisor` block (around line 519). Insert the time gate checks **after** the LLM Advisor block and **before** the `# Block new buys if symbol didn't rank` check.

Add after the LLM Advisor block ends (after the `elif advise["judgment"] == "buy"` block):

```python
        # 5c. Time window gate — block buys/sells outside configured windows
        if self._time_gate:
            if decision["action"] == "buy" and not self._time_gate.can_buy():
                logger.info("TIME GATE: buy blocked for %s (outside buy window)", symbol)
                return
            if decision["action"] == "sell" and not self._time_gate.can_sell():
                logger.info("TIME GATE: sell blocked for %s (outside sell window)", symbol)
                return
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/pavel/tools/trader
python -m pytest tests/core/test_engine.py -v
```

Expected: all pass (including the 2 new tests)

- [ ] **Step 6: Commit**

```bash
cd /home/pavel/tools/trader
git add src/trader/core/engine.py
git commit -m "feat: wire TimeWindowGate into TradingEngine — gates buy/sell by time window"
```

---

### Task 4: OvernightCatalystCollector — earnings source

**Files:**
- Create: `src/trader/collectors/overnight_catalyst.py`
- Create: `tests/collectors/test_overnight_catalyst.py`

- [ ] **Step 1: Write failing tests (earnings path only)**

Create `tests/collectors/test_overnight_catalyst.py`:

```python
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import date, datetime, timezone
from trader.collectors.overnight_catalyst import OvernightCatalystCollector


def _make_earnings_dates(rows):
    """rows: list of (date, eps_estimate, eps_actual) — future dates have eps_actual=NaN"""
    index = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows], tz="America/New_York")
    df = pd.DataFrame(
        {"EPS Estimate": [r[1] for r in rows], "Reported EPS": [r[2] for r in rows]},
        index=index,
    )
    return df


def _today():
    return date.today()


def _ticker_with_amc(beat_rate_quarters, tonight=True):
    """Return a mock yfinance Ticker that has AMC earnings tonight."""
    ticker = MagicMock()
    today = _today()

    rows = []
    # Future AMC earnings tonight (or tomorrow if tonight=False)
    future_date = today if tonight else date(today.year, today.month + 1, 1)
    rows.append((future_date, 1.00, float("nan")))

    # Historical quarters to build beat_rate_quarters beat history
    from datetime import timedelta
    for i in range(8):
        past = today - timedelta(days=90 * (i + 1))
        if i < round(beat_rate_quarters * 8):
            rows.append((past, 1.00, 1.20))   # beat
        else:
            rows.append((past, 1.00, 0.80))   # miss

    ticker.earnings_dates = _make_earnings_dates(rows)
    return ticker


def test_amc_high_beat_rate_gives_positive_score():
    collector = OvernightCatalystCollector()
    ticker = _ticker_with_amc(beat_rate_quarters=0.75)  # 75% beat rate
    with patch("yfinance.Ticker", return_value=ticker), \
         patch.object(collector, "_score_edgar", return_value=None):
        score = collector.score(["AAPL"])
    assert score is not None
    assert 0.55 <= score <= 0.80


def test_amc_low_beat_rate_gives_negative_score():
    collector = OvernightCatalystCollector()
    ticker = _ticker_with_amc(beat_rate_quarters=0.25)  # 25% beat rate
    with patch("yfinance.Ticker", return_value=ticker), \
         patch.object(collector, "_score_edgar", return_value=None):
        score = collector.score(["AAPL"])
    assert score is not None
    assert score < 0


def test_no_amc_tonight_returns_none():
    collector = OvernightCatalystCollector()
    ticker = _ticker_with_amc(beat_rate_quarters=0.75, tonight=False)
    with patch("yfinance.Ticker", return_value=ticker), \
         patch.object(collector, "_score_edgar", return_value=None):
        score = collector.score(["AAPL"])
    assert score is None


def test_empty_symbols_returns_none():
    collector = OvernightCatalystCollector()
    assert collector.score([]) is None


def test_amc_unknown_history_gives_mild_positive():
    collector = OvernightCatalystCollector()
    ticker = MagicMock()
    today = _today()
    ticker.earnings_dates = _make_earnings_dates([(today, float("nan"), float("nan"))])
    with patch("yfinance.Ticker", return_value=ticker), \
         patch.object(collector, "_score_edgar", return_value=None):
        score = collector.score(["AAPL"])
    assert score is not None
    assert 0.10 <= score <= 0.30
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/pavel/tools/trader
python -m pytest tests/collectors/test_overnight_catalyst.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'trader.collectors.overnight_catalyst'`

- [ ] **Step 3: Implement earnings source in OvernightCatalystCollector**

Create `src/trader/collectors/overnight_catalyst.py`:

```python
"""
OvernightCatalystCollector — scores stocks with a specific overnight catalyst.

Source A: yfinance earnings calendar
  - AMC tonight + historical beat rate >= 65% → +0.55 to +0.75
  - AMC tonight + historical beat rate < 40%  → -0.25
  - AMC tonight + insufficient history         → +0.20
  - No AMC tonight                             → None

Source B: SEC EDGAR 8-K filings (today)
  - Tier-1 keywords (merger, FDA approval, …)  → +0.45
  - Tier-2 keywords (major contract, …)         → +0.30
  - No filing today                             → None

When both sources produce a signal: average them, cap at +0.80.
Returns None if NO symbol has a catalyst today.
Cache TTL: 60 minutes per symbol.
"""
import logging
import time
import math
from datetime import date

import requests

logger = logging.getLogger(__name__)

_CACHE_TTL = 3600   # 1 hour

_EDGAR_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%22{ticker}%22&dateRange=custom&startdt={today}&enddt={today}&forms=8-K"
)

_TIER1_KEYWORDS = [
    "merger", "acquisition", "fda approval", "fda approved",
    "breakthrough designation", "going private", "going-private",
]
_TIER2_KEYWORDS = [
    "major contract", "licensing agreement", "strategic alliance",
    "partnership agreement", "definitive agreement",
]


class OvernightCatalystCollector:
    def __init__(self):
        self._cache: dict[str, tuple[float, float | None]] = {}  # symbol → (ts, score)

    def score(self, symbols: list[str]) -> float | None:
        if not symbols:
            return None
        scores = []
        for sym in symbols:
            s = self._score_symbol(sym)
            if s is not None:
                scores.append(s)
        if not scores:
            return None
        return sum(scores) / len(scores)

    def _score_symbol(self, symbol: str) -> float | None:
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and (now - cached[0]) < _CACHE_TTL:
            return cached[1]

        earnings_score = self._score_earnings(symbol)
        edgar_score = self._score_edgar(symbol)

        combined = self._blend(earnings_score, edgar_score)
        self._cache[symbol] = (now, combined)
        if combined is not None:
            logger.info(
                "OvernightCatalyst [%s]: earnings=%s edgar=%s → %.3f",
                symbol, earnings_score, edgar_score, combined,
            )
        return combined

    def _blend(self, earnings: float | None, edgar: float | None) -> float | None:
        signals = [s for s in (earnings, edgar) if s is not None]
        if not signals:
            return None
        avg = sum(signals) / len(signals)
        return min(0.80, max(-1.0, avg))

    def _score_earnings(self, symbol: str) -> float | None:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — earnings catalyst disabled")
            return None

        try:
            ticker = yf.Ticker(symbol)
            earnings_dates = ticker.earnings_dates
        except Exception as e:
            logger.warning("OvernightCatalyst: yfinance fetch failed for %s: %s", symbol, e)
            return None

        if earnings_dates is None or earnings_dates.empty:
            return None

        today = date.today()
        has_amc_tonight = False

        for idx in earnings_dates.index:
            try:
                d = idx.date() if hasattr(idx, "date") else idx
                if d == today:
                    has_amc_tonight = True
                    break
            except Exception:
                continue

        if not has_amc_tonight:
            return None

        beat_rate = self._compute_beat_rate(earnings_dates)
        if beat_rate is None:
            return 0.20   # tonight earnings but no history → mild positive

        if beat_rate >= 0.65:
            # Scale from +0.55 (at 65%) to +0.75 (at 100%)
            return 0.55 + (beat_rate - 0.65) / 0.35 * 0.20
        elif beat_rate < 0.40:
            return -0.25
        else:
            return 0.10   # mixed history → mild positive

    def _compute_beat_rate(self, earnings_dates) -> float | None:
        if "EPS Estimate" not in earnings_dates.columns or "Reported EPS" not in earnings_dates.columns:
            return None
        today = date.today()
        beats, total = 0, 0
        for idx in sorted(earnings_dates.index, reverse=True):
            if total >= 8:
                break
            try:
                d = idx.date() if hasattr(idx, "date") else idx
                if d >= today:
                    continue
                est = earnings_dates.loc[idx, "EPS Estimate"]
                act = earnings_dates.loc[idx, "Reported EPS"]
                if str(est) == "nan" or str(act) == "nan":
                    continue
                total += 1
                if float(act) > float(est):
                    beats += 1
            except Exception:
                continue
        return beats / total if total >= 2 else None

    def _score_edgar(self, symbol: str) -> float | None:
        today = date.today().isoformat()
        url = _EDGAR_URL.format(ticker=symbol, today=today)
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "trader-bot/1.0 contact@example.com"})
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("OvernightCatalyst: EDGAR fetch failed for %s: %s", symbol, e)
            return None

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return None

        for hit in hits:
            text = (hit.get("_source", {}).get("file_date", "") + " " +
                    " ".join(hit.get("_source", {}).get("period_of_report", ""))).lower()
            # Also check the display names / entity names
            entity = " ".join(
                str(v).lower()
                for v in hit.get("_source", {}).values()
                if isinstance(v, str)
            )
            combined_text = text + " " + entity

            for kw in _TIER1_KEYWORDS:
                if kw in combined_text:
                    logger.info("OvernightCatalyst EDGAR [%s]: tier-1 keyword '%s'", symbol, kw)
                    return 0.45

            for kw in _TIER2_KEYWORDS:
                if kw in combined_text:
                    logger.info("OvernightCatalyst EDGAR [%s]: tier-2 keyword '%s'", symbol, kw)
                    return 0.30

        # Filing found today but no material keywords
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/pavel/tools/trader
python -m pytest tests/collectors/test_overnight_catalyst.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
cd /home/pavel/tools/trader
git add src/trader/collectors/overnight_catalyst.py tests/collectors/test_overnight_catalyst.py
git commit -m "feat: add OvernightCatalystCollector — yfinance AMC earnings signal"
```

---

### Task 5: Add EDGAR 8-K signal tests + validate blending

**Files:**
- Modify: `tests/collectors/test_overnight_catalyst.py`

- [ ] **Step 1: Add EDGAR and blending tests**

Append to `tests/collectors/test_overnight_catalyst.py`:

```python
def test_edgar_tier1_keyword_gives_045():
    collector = OvernightCatalystCollector()
    edgar_response = {
        "hits": {"hits": [{"_source": {"entity_name": "AAPL", "form_type": "8-K",
                                        "file_date": "2026-04-29",
                                        "period_of_report": "merger agreement signed"}}]}
    }
    with patch("trader.collectors.overnight_catalyst.requests.get") as mock_get, \
         patch.object(collector, "_score_earnings", return_value=None):
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = edgar_response
        score = collector.score(["AAPL"])
    assert score == 0.45


def test_edgar_tier2_keyword_gives_030():
    collector = OvernightCatalystCollector()
    edgar_response = {
        "hits": {"hits": [{"_source": {"entity_name": "AAPL",
                                        "description": "major contract awarded"}}]}
    }
    with patch("trader.collectors.overnight_catalyst.requests.get") as mock_get, \
         patch.object(collector, "_score_earnings", return_value=None):
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = edgar_response
        score = collector.score(["AAPL"])
    assert score == 0.30


def test_no_catalyst_returns_none():
    collector = OvernightCatalystCollector()
    with patch.object(collector, "_score_earnings", return_value=None), \
         patch.object(collector, "_score_edgar", return_value=None):
        score = collector.score(["AAPL"])
    assert score is None


def test_both_sources_averaged_and_capped():
    collector = OvernightCatalystCollector()
    # earnings=0.75, edgar=0.45 → avg=0.60 → within cap
    with patch.object(collector, "_score_earnings", return_value=0.75), \
         patch.object(collector, "_score_edgar", return_value=0.45):
        score = collector.score(["AAPL"])
    assert abs(score - 0.60) < 0.01


def test_blend_cap_at_080():
    collector = OvernightCatalystCollector()
    # earnings=1.0, edgar=1.0 → avg=1.0 → capped at 0.80
    with patch.object(collector, "_score_earnings", return_value=1.0), \
         patch.object(collector, "_score_edgar", return_value=1.0):
        score = collector.score(["AAPL"])
    assert score == 0.80
```

- [ ] **Step 2: Run all overnight catalyst tests**

```bash
cd /home/pavel/tools/trader
python -m pytest tests/collectors/test_overnight_catalyst.py -v
```

Expected: 10 passed

- [ ] **Step 3: Commit**

```bash
cd /home/pavel/tools/trader
git add tests/collectors/test_overnight_catalyst.py
git commit -m "test: add EDGAR and blending tests for OvernightCatalystCollector"
```

---

### Task 6: Wire everything in __main__.py and config-stocks.yaml

**Files:**
- Modify: `src/trader/__main__.py`
- Modify: `config-stocks.yaml`

- [ ] **Step 1: Add OvernightCatalystCollector to both stock exchange branches**

In `src/trader/__main__.py`, in the `if cfg.exchange == "alpaca":` block, add to the `numeric_collectors` list:

```python
        from trader.collectors.overnight_catalyst import OvernightCatalystCollector
        numeric_collectors = [
            MarketSentimentCollector(),
            MacroCollector(asset_class="stock"),
            EarningsCollector(),
            PolymarketCollector(),
            UnusualWhalesCollector(),
            GoogleTrendsCollector(asset_class="stock"),
            OvernightCatalystCollector(),              # ← add this
        ]
```

Do the same in the `elif cfg.exchange == "tastytrade":` block (same change, same collector).

- [ ] **Step 2: Wire TimeWindowGate into the engine instantiation**

In `src/trader/__main__.py`, add imports at the top of the file (after existing imports):

```python
from trader.core.time_gate import TimeWindowGate
```

Replace the `engine = TradingEngine(...)` call with:

```python
    time_gate = TimeWindowGate(cfg.time_gate) if cfg.time_gate.enabled else None
    if time_gate:
        logger.info(
            "TimeWindowGate enabled: buy=%s–%s ET, sell=%s–%s ET",
            cfg.time_gate.buy_start, cfg.time_gate.buy_end,
            cfg.time_gate.sell_start, cfg.time_gate.sell_end,
        )
    engine = TradingEngine(config=cfg, adapter=adapter, sentiment_analyzer=sentiment,
                           advisor=advisor,
                           collectors=collectors, numeric_collectors=numeric_collectors,
                           db_path=args.db, notifier=notifier, universe=universe,
                           time_gate=time_gate)
```

- [ ] **Step 3: Add time_gate block to config-stocks.yaml**

In `config-stocks.yaml`, add after the `strategy: moderate` line:

```yaml
time_gate:
  enabled: true
  buy_start: "15:00"
  buy_end: "16:00"
  sell_start: "09:30"
  sell_end: "10:30"
```

- [ ] **Step 4: Run full test suite to check for regressions**

```bash
cd /home/pavel/tools/trader
python -m pytest --tb=short -q
```

Expected: same pre-existing 13 failures, no new failures.

- [ ] **Step 5: Commit**

```bash
cd /home/pavel/tools/trader
git add src/trader/__main__.py config-stocks.yaml
git commit -m "feat: wire OvernightCatalystCollector + TimeWindowGate into stocks trader"
```

---

### Task 7: Deploy to Vultr

- [ ] **Step 1: Push to origin**

```bash
cd /home/pavel/tools/trader
git push origin master
```

- [ ] **Step 2: Pull and rebuild on server**

```bash
sshpass -p '}Fi9d34Pyb+F][o{' ssh -o StrictHostKeyChecking=no root@144.202.60.186 \
  "cd /opt/trader && git pull origin master && docker compose build && docker compose up -d"
```

- [ ] **Step 3: Verify containers healthy**

```bash
sshpass -p '}Fi9d34Pyb+F][o{' ssh -o StrictHostKeyChecking=no root@144.202.60.186 \
  "sleep 10 && cd /opt/trader && docker compose ps"
```

Expected: all 3 containers `Up (healthy)`.

- [ ] **Step 4: Verify time gate is active in logs**

```bash
sshpass -p '}Fi9d34Pyb+F][o{' ssh -o StrictHostKeyChecking=no root@144.202.60.186 \
  "cd /opt/trader && docker compose logs --tail=30 trader-stocks 2>&1 | grep -i 'TimeWindowGate\|time.gate\|OvernightCatalyst'"
```

Expected: log line containing `TimeWindowGate enabled: buy=15:00–16:00 ET`.
