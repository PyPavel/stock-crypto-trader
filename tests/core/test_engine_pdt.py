"""Engine integration tests for PDT (Pattern Day Trader) constraints."""
from unittest.mock import MagicMock
from trader.core.engine import TradingEngine


def _make_engine(daytrade_count: int = 0, notifier=None):
    from trader.config import Config
    cfg = Config(
        exchange="alpaca", mode="paper", strategy="moderate",
        capital=10000.0, pairs=["AAPL"], cycle_interval=3600,
    )
    adapter = MagicMock()
    adapter.get_day_trade_count.return_value = daytrade_count
    adapter.get_price.return_value = 100.0
    adapter.get_candles.return_value = []
    adapter.is_market_open.return_value = True

    sentiment = MagicMock()
    sentiment.score_texts.return_value = 0.5

    engine = TradingEngine(
        config=cfg,
        adapter=adapter,
        sentiment_analyzer=sentiment,
        collectors=[],
        notifier=notifier,
        db_path=":memory:",
    )
    # Seed the PDT guard with the adapter's day-trade count (normally done in run_cycle)
    if engine._pdt:
        engine._pdt.refresh()
    return engine, adapter


def _make_result(price: float, score: float = 0.3) -> dict:
    return {
        "price": price,
        "candles": [],
        "tech_signal": MagicMock(),
        "sentiment": MagicMock(),
        "tech_score": score,
        "ml_score": None,
        "trend_bullish": score > 0,
        "raw_sentiment": score,
        "combined_score": score,
        "atr": None,
        "texts": ["headline"],
    }


# ------------------------------------------------------------------ #
#  PDTGuard instantiation
# ------------------------------------------------------------------ #

def test_pdt_guard_created_when_adapter_has_day_trade_count():
    engine, _ = _make_engine()
    assert engine._pdt is not None


def test_pdt_guard_not_created_for_crypto_adapter():
    from trader.config import Config
    cfg = Config(
        exchange="coinbase", mode="paper", strategy="moderate",
        capital=10000.0, pairs=["BTC-USD"], cycle_interval=3600,
    )
    # spec with no get_day_trade_count — simulates a crypto adapter
    adapter = MagicMock(spec=["get_price", "get_candles", "place_order"])
    adapter.get_price.return_value = 50000.0
    adapter.get_candles.return_value = []

    sentiment = MagicMock()
    sentiment.score_texts.return_value = 0.5

    engine = TradingEngine(
        config=cfg,
        adapter=adapter,
        sentiment_analyzer=sentiment,
        collectors=[],
    )
    assert engine._pdt is None


# ------------------------------------------------------------------ #
#  PDT stop-loss blocking
# ------------------------------------------------------------------ #

def test_pdt_blocks_stop_loss_on_same_day_buy_with_no_budget():
    """Stop-loss must NOT execute when the position was opened today and budget = 0."""
    notifier = MagicMock()
    engine, _ = _make_engine(daytrade_count=3, notifier=notifier)  # budget exhausted
    engine._pdt.record_buy("AAPL")  # mark as bought today
    # Mock strategy to hold so the only close path is via stop-loss
    engine._strategy.decide = MagicMock(return_value={"action": "hold", "reason": "n/a"})

    # entry=$150, current=$140 → 6.7% loss > 5% stop-loss threshold
    engine.portfolio.positions["AAPL"] = {
        "amount": 1.0, "entry_price": 150.0, "side": "buy"
    }
    engine._peak_prices["AAPL"] = 150.0

    engine._execute_decisions("AAPL", _make_result(price=140.0, score=-0.5), {})

    assert "AAPL" in engine.portfolio.positions  # position was NOT closed
    notifier.send.assert_called_once()
    msg = notifier.send.call_args[0][0]
    assert "PDT" in msg
    assert "AAPL" in msg


def test_pdt_allows_stop_loss_for_overnight_hold_even_with_no_budget():
    """Overnight positions are free to exit regardless of PDT budget."""
    engine, _ = _make_engine(daytrade_count=3)  # budget exhausted, but AAPL is overnight

    engine.portfolio.positions["AAPL"] = {
        "amount": 1.0, "entry_price": 150.0, "side": "buy"
    }
    engine._peak_prices["AAPL"] = 150.0

    engine._execute_decisions("AAPL", _make_result(price=140.0, score=-0.5), {})

    assert "AAPL" not in engine.portfolio.positions  # position was closed by stop-loss


# ------------------------------------------------------------------ #
#  PDT buy gating
# ------------------------------------------------------------------ #

def test_pdt_blocks_buy_when_budget_exhausted():
    """No buy orders when remaining day-trade count = 0."""
    engine, _ = _make_engine(daytrade_count=3)  # remaining=0

    engine._strategy.decide = MagicMock(return_value={
        "action": "buy", "usd_amount": 500.0, "reason": "signal"
    })

    engine._execute_decisions("AAPL", _make_result(price=100.0, score=0.8), {}, can_buy=True)

    assert "AAPL" not in engine.portfolio.positions


def test_pdt_blocks_buy_below_adjusted_threshold_with_one_remaining():
    """With 1 day-trade left, threshold raises to 0.45 — score=0.35 must be blocked."""
    engine, _ = _make_engine(daytrade_count=2)  # remaining=1, threshold=0.45

    engine._strategy.decide = MagicMock(return_value={
        "action": "buy", "usd_amount": 500.0, "reason": "signal"
    })

    # combined_score=0.35 < 0.45 → blocked by PDT threshold
    engine._execute_decisions("AAPL", _make_result(price=100.0, score=0.35), {}, can_buy=True)

    assert "AAPL" not in engine.portfolio.positions


def test_pdt_allows_buy_above_adjusted_threshold_with_one_remaining():
    """With 1 day-trade left and score above 0.45, buy should proceed."""
    engine, _ = _make_engine(daytrade_count=2)  # remaining=1, threshold=0.45
    engine._risk.validate_buy = MagicMock(return_value={"allowed": True})

    engine._strategy.decide = MagicMock(return_value={
        "action": "buy", "usd_amount": 500.0, "reason": "signal"
    })

    # combined_score=0.5 > 0.45 → passes PDT threshold
    engine._execute_decisions("AAPL", _make_result(price=100.0, score=0.5), {}, can_buy=True)

    assert "AAPL" in engine.portfolio.positions


# ------------------------------------------------------------------ #
#  record_buy after successful buy
# ------------------------------------------------------------------ #

def test_pdt_records_buy_after_successful_buy():
    """Successful buy must register the symbol in PDTGuard so a same-day sell is tracked."""
    engine, _ = _make_engine(daytrade_count=0)  # full budget
    engine._risk.validate_buy = MagicMock(return_value={"allowed": True})

    engine._strategy.decide = MagicMock(return_value={
        "action": "buy", "usd_amount": 500.0, "reason": "signal"
    })

    engine._execute_decisions("AAPL", _make_result(price=100.0, score=0.5), {}, can_buy=True)

    assert engine._pdt.is_same_day_buy("AAPL") is True


# ------------------------------------------------------------------ #
#  pdt_remaining forwarded to LLM advisor
# ------------------------------------------------------------------ #

def test_pdt_remaining_forwarded_to_llm_advisor():
    """LLM advisor must receive pdt_remaining so it can bias toward overnight picks."""
    advisor = MagicMock()
    advisor.advise.return_value = {"judgment": "avoid", "conviction": 0.0, "reason": "veto"}

    engine, _ = _make_engine(daytrade_count=1)  # remaining=2
    engine._advisor = advisor

    engine._strategy.decide = MagicMock(return_value={
        "action": "buy", "usd_amount": 500.0, "reason": "signal"
    })

    engine._execute_decisions("AAPL", _make_result(price=100.0, score=0.5), {}, can_buy=True)

    advisor.advise.assert_called_once()
    _, kwargs = advisor.advise.call_args
    assert kwargs.get("pdt_remaining") == 2
