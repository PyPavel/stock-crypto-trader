import pandas as pd
try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except Exception:
    _HAS_PANDAS_TA = False

from trader.models import Candle

MIN_CANDLES = 30


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Manual RSI fallback."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, float('nan'))
    return 100 - (100 / (1 + rs))


def _macd_hist(series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    """Manual MACD histogram fallback."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _bbands(series: pd.Series, length: int = 20):
    """Manual Bollinger Bands fallback. Returns (lower, middle, upper)."""
    mid = series.rolling(length).mean()
    std = series.rolling(length).std()
    return mid - 2 * std, mid, mid + 2 * std


class SignalGenerator:
    """Combines RSI, MACD, and Bollinger Band signals into a single score [-1, +1]."""

    def score(self, candles: list[Candle]) -> float:
        """Returns -1.0 (strong sell) to +1.0 (strong buy). 0.0 if insufficient data."""
        if len(candles) < MIN_CANDLES:
            return 0.0

        df = pd.DataFrame([
            {"close": c.close, "high": c.high, "low": c.low, "volume": c.volume}
            for c in candles
        ])

        scores = []
        close = df["close"]

        # RSI signal (weight: 0.4)
        try:
            if _HAS_PANDAS_TA:
                rsi_series = ta.rsi(close, length=14)
            else:
                rsi_series = _rsi(close, length=14)

            if rsi_series is not None and not rsi_series.empty:
                rsi = rsi_series.iloc[-1]
                if pd.notna(rsi):
                    if rsi < 30:
                        scores.append((70 - rsi) / 40 * 0.4)
                    elif rsi > 70:
                        scores.append((70 - rsi) / 40 * 0.4)
                    else:
                        scores.append(0.0)
        except Exception:
            pass

        # MACD signal (weight: 0.35)
        try:
            if _HAS_PANDAS_TA:
                macd_df = ta.macd(close)
                hist_col = [c for c in macd_df.columns if "MACDh" in c] if macd_df is not None else []
                hist = macd_df[hist_col[0]].iloc[-1] if hist_col else None
            else:
                hist_series = _macd_hist(close)
                hist = hist_series.iloc[-1]

            if hist is not None and pd.notna(hist):
                normalized = max(-1.0, min(1.0, hist / (close.mean() * 0.002)))
                scores.append(normalized * 0.35)
        except Exception:
            pass

        # Bollinger Bands signal (weight: 0.25)
        try:
            if _HAS_PANDAS_TA:
                bb = ta.bbands(close, length=20)
                lower_col = [c for c in bb.columns if "BBL" in c] if bb is not None else []
                upper_col = [c for c in bb.columns if "BBU" in c] if bb is not None else []
                if lower_col and upper_col:
                    lower = bb[lower_col[0]].iloc[-1]
                    upper = bb[upper_col[0]].iloc[-1]
                else:
                    lower = upper = None
            else:
                lower_s, _, upper_s = _bbands(close)
                lower = lower_s.iloc[-1]
                upper = upper_s.iloc[-1]

            price = close.iloc[-1]
            if lower is not None and upper is not None and pd.notna(lower) and pd.notna(upper) and upper != lower:
                bb_pct = (price - lower) / (upper - lower)
                scores.append((0.5 - bb_pct) * 2 * 0.25)
        except Exception:
            pass

        if not scores:
            return 0.0

        total = sum(scores)
        return max(-1.0, min(1.0, total))
