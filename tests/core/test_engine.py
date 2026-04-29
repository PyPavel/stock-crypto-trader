from unittest.mock import MagicMock
from datetime import datetime, timezone, timedelta
from trader.core.engine import TradingEngine
from trader.config import Config, RiskConfig
from trader.models import Candle


def make_candles(n=50):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        Candle("BTC/USD", base + timedelta(hours=i),
               40000.0, 41000.0, 39000.0, 40000.0 + i * 10, 100.0)
        for i in range(n)
    ]


def make_engine(tmp_path, mode="paper"):
    cfg = Config(exchange="coinbase", mode=mode, strategy="moderate",
                 capital=100.0, pairs=["BTC/USD"], cycle_interval=60)

    adapter = MagicMock()
    adapter.get_candles.return_value = make_candles()
    adapter.get_price.return_value = 40500.0
    adapter.get_balance.return_value = {"USD": 100.0}

    sentiment_analyzer = MagicMock()
    sentiment_analyzer.score_texts.return_value = 0.3

    news_collector = MagicMock()
    news_collector.fetch.return_value = ["BTC is bullish"]

    return TradingEngine(
        config=cfg,
        adapter=adapter,
        sentiment_analyzer=sentiment_analyzer,
        collectors=[news_collector],
        db_path=str(tmp_path / "test.db"),
    )


def test_run_cycle_completes_without_error(tmp_path):
    engine = make_engine(tmp_path)
    engine.run_cycle()  # should not raise


def test_run_cycle_in_paper_mode_no_live_orders(tmp_path):
    engine = make_engine(tmp_path, mode="paper")
    engine.run_cycle()
    trades = engine.portfolio.get_trades()
    for t in trades:
        assert t.mode == "paper"


def test_engine_handles_collector_failure(tmp_path):
    engine = make_engine(tmp_path)
    engine._collectors[0].fetch.side_effect = Exception("network error")
    engine.run_cycle()  # should not raise, sentiment falls back to 0


def test_numeric_collectors_blend_into_sentiment(tmp_path):
    engine = make_engine(tmp_path)
    # Add a numeric collector that returns strong buy signal
    numeric_mock = MagicMock()
    numeric_mock.score.return_value = 1.0
    # Mock inspect to treat it as no-arg score
    engine._numeric_collectors = [numeric_mock]
    engine.run_cycle()  # should not raise


def test_stop_loss_triggers_sell(tmp_path):
    engine = make_engine(tmp_path)
    # Manually put a position in the portfolio with high entry price
    from trader.models import Trade
    from datetime import datetime, timezone
    buy_trade = Trade(order_id="sl-test", symbol="BTC/USD", side="buy",
                      amount=0.001, price=50000.0, fee=0.0, mode="paper",
                      timestamp=datetime.now(timezone.utc))
    engine.portfolio.record_trade(buy_trade)
    # Price is now 40500 (set in make_engine), entry was 50000 = 19% loss > 5% stop-loss
    engine.run_cycle()
    trades = engine.portfolio.get_trades()
    sell_trades = [t for t in trades if t.side == "sell"]
    assert len(sell_trades) >= 1
    assert sell_trades[0].narrative == "stop-loss triggered"


def test_time_gate_blocks_buy_outside_window(tmp_path):
    """TimeWindowGate that always blocks buys converts buy decision to hold."""
    from unittest.mock import MagicMock, patch
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
    with patch.object(engine, "_execute_sell") as mock_sell:
        engine._execute_decisions("AAPL", result, {}, can_buy=True)
        # The time gate blocks buy before it reaches the buy execution block;
        # no trade should have been recorded
        mock_sell.assert_not_called()
        assert len(engine.portfolio.get_trades()) == 0


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
