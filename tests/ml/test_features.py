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
            symbol="BTC-USD",
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
