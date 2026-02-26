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
