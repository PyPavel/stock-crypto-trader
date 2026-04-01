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
