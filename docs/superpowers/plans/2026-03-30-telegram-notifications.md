# Telegram Trade Notifications Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Send a Telegram message to the user on every buy/sell event (including stop-loss, trailing-stop, take-profit triggers) for both the crypto and stocks traders.

**Architecture:** A thin `TelegramNotifier` is constructed in `__main__.py` and injected into `TradingEngine`. The engine calls `notifier.send(...)` after every trade execution. The notifier uses `requests` (already a dependency) to call the Telegram Bot API and logs a warning on failure — never raising, so a Telegram outage cannot block a trade.

**Tech Stack:** Python stdlib + `requests` (already installed), Telegram Bot API (`sendMessage`), env vars for credentials.

---

### Task 1: TelegramConfig in config.py

**Files:**
- Modify: `src/trader/config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_telegram.py
import os
from trader.config import load_config, TelegramConfig

def test_telegram_config_defaults():
    cfg = TelegramConfig()
    assert cfg.bot_token == ""
    assert cfg.chat_id == ""

def test_telegram_config_loaded_from_yaml(tmp_path):
    yaml_content = """
exchange: coinbase
mode: paper
strategy: moderate
capital: 1000.0
pairs: [BTC-USD]
telegram:
  bot_token: abc123
  chat_id: "99999"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    cfg = load_config(str(config_file))
    assert cfg.telegram.bot_token == "abc123"
    assert cfg.telegram.chat_id == "99999"

def test_telegram_config_env_override(tmp_path, monkeypatch):
    yaml_content = """
exchange: coinbase
mode: paper
strategy: moderate
capital: 1000.0
pairs: [BTC-USD]
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "envtoken")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "envid")
    cfg = load_config(str(config_file))
    assert cfg.telegram.bot_token == "envtoken"
    assert cfg.telegram.chat_id == "envid"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/pavel/tools/trader
python -m pytest tests/test_config_telegram.py -v
```
Expected: FAIL — `TelegramConfig` not found.

- [ ] **Step 3: Add TelegramConfig dataclass and wire it into Config + load_config**

In `src/trader/config.py`, add after the `MLConfig` dataclass:

```python
@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""
```

In the `Config` dataclass, add the field after `ml`:

```python
telegram: TelegramConfig = field(default_factory=TelegramConfig)
```

In `load_config`, add `("telegram", TelegramConfig)` to the existing loop list:

```python
for key, cls in [
    ("mimo", MimoConfig),
    ("coinbase", CoinbaseConfig),
    ("alpaca", AlpacaConfig),
    ("reddit", RedditConfig),
    ("cryptopanic", CryptoPanicConfig),
    ("llm_advisor", LLMAdvisorConfig),
    ("risk", RiskConfig),
    ("ml", MLConfig),
    ("telegram", TelegramConfig),   # <-- add this line
]:
```

After the existing env var overrides block at the bottom of `load_config`, add:

```python
if os.environ.get("TELEGRAM_BOT_TOKEN"):
    cfg.telegram.bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
if os.environ.get("TELEGRAM_CHAT_ID"):
    cfg.telegram.chat_id = os.environ["TELEGRAM_CHAT_ID"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_config_telegram.py -v
```
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/trader/config.py tests/test_config_telegram.py
git commit -m "feat: add TelegramConfig dataclass with env var override"
```

---

### Task 2: TelegramNotifier class

**Files:**
- Create: `src/trader/notifications/__init__.py`
- Create: `src/trader/notifications/telegram.py`
- Create: `tests/notifications/__init__.py`
- Create: `tests/notifications/test_telegram.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/notifications/test_telegram.py
from unittest.mock import patch, MagicMock
from trader.notifications.telegram import TelegramNotifier


def test_send_calls_requests_post():
    notifier = TelegramNotifier(bot_token="tok123", chat_id="42")
    with patch("trader.notifications.telegram.requests.post") as mock_post:
        mock_post.return_value.ok = True
        notifier.send("BUY BTC-USD 0.001 @ 50000.00")
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "tok123" in call_args[0][0]  # URL contains token
        assert call_args[1]["json"]["chat_id"] == "42"
        assert "BUY BTC-USD" in call_args[1]["json"]["text"]


def test_send_no_op_when_unconfigured():
    notifier = TelegramNotifier(bot_token="", chat_id="")
    with patch("trader.notifications.telegram.requests.post") as mock_post:
        notifier.send("anything")
        mock_post.assert_not_called()


def test_send_logs_warning_on_http_error():
    notifier = TelegramNotifier(bot_token="tok", chat_id="42")
    with patch("trader.notifications.telegram.requests.post") as mock_post:
        mock_post.return_value.ok = False
        mock_post.return_value.text = "Bad Request"
        import logging
        with patch.object(logging.getLogger("trader.notifications.telegram"), "warning") as mock_warn:
            notifier.send("test")
            mock_warn.assert_called_once()


def test_send_logs_warning_on_exception():
    notifier = TelegramNotifier(bot_token="tok", chat_id="42")
    with patch("trader.notifications.telegram.requests.post", side_effect=Exception("timeout")):
        import logging
        with patch.object(logging.getLogger("trader.notifications.telegram"), "warning") as mock_warn:
            notifier.send("test")
            mock_warn.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/notifications/test_telegram.py -v
```
Expected: FAIL — `trader.notifications.telegram` not found.

- [ ] **Step 3: Create the package and notifier**

Create `src/trader/notifications/__init__.py` (empty):
```python
```

Create `src/trader/notifications/telegram.py`:
```python
import logging
import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self._token = bot_token
        self._chat_id = chat_id

    def send(self, message: str) -> None:
        """Send a Telegram message. Silently logs on failure — never raises."""
        if not self._token or not self._chat_id:
            return
        url = TELEGRAM_API.format(token=self._token)
        try:
            resp = requests.post(url, json={"chat_id": self._chat_id, "text": message}, timeout=5)
            if not resp.ok:
                logger.warning("Telegram send failed: %s", resp.text)
        except Exception as exc:
            logger.warning("Telegram send error: %s", exc)
```

Create `tests/notifications/__init__.py` (empty):
```python
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/notifications/test_telegram.py -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/trader/notifications/ tests/notifications/
git commit -m "feat: add TelegramNotifier with silent-failure send"
```

---

### Task 3: Inject notifier into TradingEngine

**Files:**
- Modify: `src/trader/core/engine.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/core/test_engine_telegram.py` (new file):

```python
# tests/core/test_engine_telegram.py
from unittest.mock import MagicMock, patch
from trader.core.engine import TradingEngine
from trader.notifications.telegram import TelegramNotifier


def _make_engine(notifier=None):
    from trader.config import Config, RiskConfig
    cfg = Config(
        exchange="coinbase", mode="paper", strategy="moderate",
        capital=10000.0, pairs=["BTC-USD"], cycle_interval=3600,
    )
    adapter = MagicMock()
    adapter.get_price.return_value = 50000.0
    adapter.get_candles.return_value = []
    adapter.place_order.return_value = MagicMock(id="o1", amount=0.001, price=50000.0)

    sentiment = MagicMock()
    sentiment.score_texts.return_value = 0.6

    engine = TradingEngine(
        config=cfg,
        adapter=adapter,
        sentiment_analyzer=sentiment,
        collectors=[],
        notifier=notifier,
    )
    return engine


def test_engine_accepts_notifier():
    notifier = TelegramNotifier(bot_token="", chat_id="")
    engine = _make_engine(notifier=notifier)
    assert engine._notifier is notifier


def test_engine_notifier_defaults_to_none():
    engine = _make_engine()
    assert engine._notifier is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/core/test_engine_telegram.py -v
```
Expected: FAIL — `TradingEngine.__init__` does not accept `notifier`.

- [ ] **Step 3: Add notifier parameter to TradingEngine**

In `src/trader/core/engine.py`, update `__init__` signature (after `db_path`):

```python
def __init__(
    self,
    config: Config,
    adapter: ExchangeAdapter,
    sentiment_analyzer: SentimentAnalyzer,
    collectors: list,
    numeric_collectors: list | None = None,
    db_path: str = "trader.db",
    notifier=None,
):
```

Add the import at the top of `engine.py` (after existing imports):

```python
from trader.notifications.telegram import TelegramNotifier
```

Store the notifier in `__init__` body (after existing assignments):

```python
self._notifier: TelegramNotifier | None = notifier
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/core/test_engine_telegram.py -v
```
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/trader/core/engine.py tests/core/test_engine_telegram.py
git commit -m "feat: TradingEngine accepts optional TelegramNotifier"
```

---

### Task 4: Fire notifications on buy/sell events

**Files:**
- Modify: `src/trader/core/engine.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/core/test_engine_telegram.py`:

```python
def test_notifier_called_on_buy():
    notifier = MagicMock()
    engine = _make_engine(notifier=notifier)

    # Patch strategy to always return buy
    engine._strategy.decide = MagicMock(return_value={
        "action": "buy", "usd_amount": 500.0, "reason": "bullish signal"
    })
    engine._signals.score_with_trend = MagicMock(return_value={"score": 0.8, "trend_bullish": True})
    engine._signals.atr = MagicMock(return_value=None)

    prices = {}
    engine._process_symbol("BTC-USD", prices)

    notifier.send.assert_called_once()
    msg = notifier.send.call_args[0][0]
    assert "BUY" in msg
    assert "BTC-USD" in msg


def test_notifier_called_on_sell():
    notifier = MagicMock()
    engine = _make_engine(notifier=notifier)

    # Seed an open position
    engine.portfolio.positions["BTC-USD"] = {
        "amount": 0.01, "entry_price": 50000.0, "side": "buy"
    }
    engine._peak_prices["BTC-USD"] = 50000.0

    # Patch strategy to always return sell
    engine._strategy.decide = MagicMock(return_value={
        "action": "sell", "usd_amount": 500.0, "reason": "bearish signal"
    })
    engine._signals.score_with_trend = MagicMock(return_value={"score": 0.2, "trend_bullish": False})
    engine._signals.atr = MagicMock(return_value=None)

    prices = {}
    engine._process_symbol("BTC-USD", prices)

    notifier.send.assert_called_once()
    msg = notifier.send.call_args[0][0]
    assert "SELL" in msg
    assert "BTC-USD" in msg


def test_notifier_not_called_on_hold():
    notifier = MagicMock()
    engine = _make_engine(notifier=notifier)

    engine._strategy.decide = MagicMock(return_value={
        "action": "hold", "usd_amount": 0.0, "reason": "neutral"
    })
    engine._signals.score_with_trend = MagicMock(return_value={"score": 0.5, "trend_bullish": True})
    engine._signals.atr = MagicMock(return_value=None)

    prices = {}
    engine._process_symbol("BTC-USD", prices)

    notifier.send.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/core/test_engine_telegram.py -v
```
Expected: 3 new tests FAIL — notifier is stored but never called.

- [ ] **Step 3: Add notify calls in engine.py**

Add a helper method to `TradingEngine` (add after `_record_trade_time`):

```python
def _notify(self, side: str, symbol: str, amount: float, price: float, reason: str) -> None:
    if self._notifier is None:
        return
    label = "CRYPTO" if self.config.exchange != "alpaca" else "STOCK"
    msg = (
        f"[{label}] {side.upper()} {symbol}\n"
        f"Amount: {amount:.6f}  Price: ${price:,.2f}\n"
        f"Reason: {reason}"
    )
    self._notifier.send(msg)
```

In `_process_symbol`, replace the buy log line (currently `logger.info("BUY %s: ...")`) with:

```python
logger.info("BUY %s: %.6f @ %.2f", symbol, order.amount, order.price)
self._notify("buy", symbol, order.amount, order.price, decision["reason"])
```

In `_execute_sell`, replace the sell log line (currently `logger.info("SELL %s: ...")`) with:

```python
logger.info("SELL %s: %.6f @ %.2f (%s)", symbol, order.amount, order.price, reason)
self._notify("sell", symbol, order.amount, order.price, reason)
```

- [ ] **Step 4: Run all notification tests**

```bash
python -m pytest tests/core/test_engine_telegram.py tests/notifications/ -v
```
Expected: all PASSED.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
python -m pytest --tb=short -q
```
Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
git add src/trader/core/engine.py tests/core/test_engine_telegram.py
git commit -m "feat: fire Telegram notification on every buy/sell event"
```

---

### Task 5: Wire notifier in __main__.py and update .env.example

**Files:**
- Modify: `src/trader/__main__.py`
- Modify: `.env.example`

- [ ] **Step 1: Update __main__.py to construct and inject the notifier**

In `src/trader/__main__.py`, add the import after existing imports:

```python
from trader.notifications.telegram import TelegramNotifier
```

After `cfg = load_config(args.config)` and the logger.info line, add:

```python
notifier = TelegramNotifier(
    bot_token=cfg.telegram.bot_token,
    chat_id=cfg.telegram.chat_id,
)
if cfg.telegram.bot_token:
    logger.info("Telegram notifications enabled (chat_id=%s)", cfg.telegram.chat_id)
else:
    logger.info("Telegram notifications disabled (no TELEGRAM_BOT_TOKEN set)")
```

In the `TradingEngine(...)` constructor call, add `notifier=notifier`:

```python
engine = TradingEngine(config=cfg, adapter=adapter, sentiment_analyzer=sentiment,
                       collectors=collectors, numeric_collectors=numeric_collectors,
                       db_path=args.db, notifier=notifier)
```

- [ ] **Step 2: Update .env.example**

Append to `.env.example`:

```
# Telegram notifications — get a bot token from @BotFather, chat_id from @userinfobot
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=123456789
```

- [ ] **Step 3: Smoke test — verify the app starts without errors**

```bash
python -m trader --config config.yaml --help
```
Expected: help text printed, no import errors.

- [ ] **Step 4: Commit**

```bash
git add src/trader/__main__.py .env.example
git commit -m "feat: wire TelegramNotifier into trader entrypoint"
```

---

### Task 6: Deploy to Vultr

**Files:** none (deployment only)

- [ ] **Step 1: Push the branch / commits to remote**

```bash
git push origin master
```

- [ ] **Step 2: SSH to Vultr and pull**

```bash
ssh -i ~/.ssh/key root@144.202.60.186 "cd /home/pavel/tools/trader && git pull"
```

- [ ] **Step 3: Set Telegram env vars on server (if not already in .env)**

```bash
ssh -i ~/.ssh/key root@144.202.60.186
# on server:
echo "TELEGRAM_BOT_TOKEN=<your_token>" >> /home/pavel/tools/trader/.env
echo "TELEGRAM_CHAT_ID=<your_chat_id>" >> /home/pavel/tools/trader/.env
```

- [ ] **Step 4: Rebuild and restart containers**

```bash
ssh -i ~/.ssh/key root@144.202.60.186 "cd /home/pavel/tools/trader && docker compose up -d --build"
```

- [ ] **Step 5: Verify both containers are healthy**

```bash
ssh -i ~/.ssh/key root@144.202.60.186 "docker compose ps"
```
Expected: `trader` and `trader-stocks` both show `healthy` or `running`.

- [ ] **Step 6: Tail logs to confirm notification line appears at startup**

```bash
ssh -i ~/.ssh/key root@144.202.60.186 "docker compose logs --tail=30 trader trader-stocks"
```
Expected: `Telegram notifications enabled (chat_id=...)` or `disabled` in each service log.
