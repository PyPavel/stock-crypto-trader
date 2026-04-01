import pytest
import numpy as np
from unittest.mock import MagicMock
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
            symbol="BTC-USD",
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
