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
