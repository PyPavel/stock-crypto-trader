from datetime import datetime, timezone, timedelta
from trader.core.signals import SignalGenerator
from trader.models import Candle

_BASE_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_candles(closes):
    return [
        Candle(
            symbol="BTC/USD",
            timestamp=_BASE_TS + timedelta(hours=i),
            open=c, high=c * 1.01, low=c * 0.99, close=c, volume=100.0,
        )
        for i, c in enumerate(closes)
    ]


def test_score_is_bounded():
    closes = list(range(35000, 35100))
    gen = SignalGenerator()
    score = gen.score(make_candles(closes))
    assert -1.0 <= score <= 1.0


def test_oversold_gives_positive_score():
    closes = [50000 - i * 200 for i in range(60)]
    gen = SignalGenerator()
    score = gen.score(make_candles(closes))
    assert score > 0


def test_overbought_gives_negative_score():
    closes = [30000 + i * 300 for i in range(60)]
    gen = SignalGenerator()
    score = gen.score(make_candles(closes))
    assert score < 0


def test_needs_minimum_candles():
    gen = SignalGenerator()
    score = gen.score(make_candles([40000] * 5))
    assert score == 0.0


# ------------------------------------------------------------------ #
#  ATR computation
# ------------------------------------------------------------------ #

def test_atr_returns_value_for_sufficient_candles():
    closes = [100 + i * 0.5 for i in range(30)]
    gen = SignalGenerator()
    atr = gen.atr(make_candles(closes), period=14)
    assert atr is not None
    assert atr > 0


def test_atr_returns_none_for_insufficient_candles():
    gen = SignalGenerator()
    atr = gen.atr(make_candles([100] * 10), period=14)
    assert atr is None


def test_atr_reflects_volatility():
    # High volatility candles
    high_vol = [
        Candle(symbol="X", timestamp=_BASE_TS + timedelta(hours=i),
               open=100, high=110, low=90, close=100, volume=100.0)
        for i in range(20)
    ]
    # Low volatility candles
    low_vol = [
        Candle(symbol="X", timestamp=_BASE_TS + timedelta(hours=i),
               open=100, high=100.5, low=99.5, close=100, volume=100.0)
        for i in range(20)
    ]
    gen = SignalGenerator()
    high_atr = gen.atr(high_vol, period=14)
    low_atr = gen.atr(low_vol, period=14)
    assert high_atr is not None and low_atr is not None
    assert high_atr > low_atr
