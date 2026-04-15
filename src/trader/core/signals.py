import logging
import pandas as pd
try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except Exception:
    _HAS_PANDAS_TA = False

from trader.models import Candle, Signal

logger = logging.getLogger(__name__)

MIN_CANDLES = 35
EMA_TREND_PERIOD = 50
VOLUME_AVG_PERIOD = 20


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


def _ema(series: pd.Series, length: int) -> pd.Series:
    """Manual EMA."""
    return series.ewm(span=length, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Manual ADX fallback. Returns ADX series."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_s = tr.rolling(window=length).mean().replace(0, float('nan'))
    plus_di = 100 * (plus_dm.rolling(window=length).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(window=length).mean() / atr_s)
    di_sum = (plus_di + minus_di).replace(0, float('nan'))
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.rolling(window=length).mean()


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Manual Stochastic fallback. Returns (%K, %D)."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, float('nan'))
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k, d


class SignalGenerator:
    """Combines RSI, MACD, Bollinger Bands, EMA trend, volume, ADX, and Stochastic into a score [-1, +1]."""

    @staticmethod
    def atr(candles: list[Candle], period: int = 14) -> float | None:
        """Compute Average True Range from candles. Returns latest ATR value or None."""
        if len(candles) < period + 1:
            return None
        df = pd.DataFrame([
            {"high": c.high, "low": c.low, "close": c.close}
            for c in candles
        ])
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(window=period).mean()
        val = atr_series.iloc[-1]
        return float(val) if pd.notna(val) else None

    def score(self, candles: list[Candle]) -> float:
        """Returns -1.0 (strong sell) to +1.0 (strong buy). 0.0 if insufficient data."""
        result = self.score_with_trend(candles)
        return result["score"]

    def score_with_trend(self, candles: list[Candle]) -> dict:
        """
        Returns dict with:
          score: float [-1, +1]
          trend_bullish: bool (True if price > 50 EMA)
        """
        if len(candles) < MIN_CANDLES:
            return {"score": 0.0, "trend_bullish": True, "rsi": None}

        df = pd.DataFrame([
            {"close": c.close, "high": c.high, "low": c.low, "volume": c.volume}
            for c in candles
        ])

        scores = []
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        latest_rsi: float | None = None

        # --- RSI signal (weight: 0.20) ---
        try:
            if _HAS_PANDAS_TA:
                rsi_series = ta.rsi(close, length=14)
            else:
                rsi_series = _rsi(close, length=14)

            if rsi_series is not None and not rsi_series.empty:
                rsi = rsi_series.iloc[-1]
                if pd.notna(rsi):
                    latest_rsi = float(rsi)
                    if rsi < 30:
                        scores.append((70 - rsi) / 40 * 0.20)
                    elif rsi > 70:
                        scores.append((70 - rsi) / 40 * 0.20)
                    else:
                        scores.append(0.0)
        except Exception:
            logger.debug("RSI calculation failed", exc_info=True)

        # --- MACD signal (weight: 0.20) ---
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
                scores.append(normalized * 0.20)
        except Exception:
            logger.debug("MACD calculation failed", exc_info=True)

        # --- Bollinger Bands signal (weight: 0.15) ---
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
                scores.append((0.5 - bb_pct) * 2 * 0.15)
        except Exception:
            logger.debug("Bollinger Bands calculation failed", exc_info=True)

        # --- EMA trend signal (weight: 0.10) ---
        # Bullish: price above 50-EMA → positive contribution; bearish: below → negative
        trend_bullish = True
        try:
            if _HAS_PANDAS_TA:
                ema_series = ta.ema(close, length=EMA_TREND_PERIOD)
            else:
                ema_series = _ema(close, EMA_TREND_PERIOD)

            if ema_series is not None and not ema_series.empty:
                ema_val = ema_series.iloc[-1]
                price_now = close.iloc[-1]
                if pd.notna(ema_val) and ema_val > 0:
                    trend_bullish = price_now > ema_val
                    # Normalize deviation: cap at ±5% for full signal
                    deviation_pct = (price_now - ema_val) / ema_val
                    normalized = max(-1.0, min(1.0, deviation_pct / 0.05))
                    scores.append(normalized * 0.10)
        except Exception:
            logger.debug("EMA trend calculation failed", exc_info=True)

        # --- Volume confirmation signal (weight: 0.05) ---
        # High volume in the direction of price movement amplifies the signal.
        # Above-average volume on a rising close → bullish; on a falling close → bearish.
        try:
            if len(volume) >= VOLUME_AVG_PERIOD:
                vol_avg = volume.iloc[-VOLUME_AVG_PERIOD:].mean()
                current_vol = volume.iloc[-1]
                if pd.notna(vol_avg) and vol_avg > 0 and pd.notna(current_vol):
                    vol_ratio = current_vol / vol_avg  # >1 = above average
                    # Price direction: positive if close > previous close
                    prev_close = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]
                    price_direction = 1.0 if close.iloc[-1] >= prev_close else -1.0
                    # Scale: vol_ratio=2.0 → full weight; cap at 1.0
                    vol_signal = min(1.0, (vol_ratio - 1.0)) * price_direction
                    scores.append(vol_signal * 0.05)
        except Exception:
            logger.debug("Volume confirmation calculation failed", exc_info=True)

        # --- ADX signal (weight: 0.15) ---
        # ADX > 25 = strong trend → weight the directional signal.
        # +DI > -DI = bullish, -DI > +DI = bearish. Low ADX → neutral.
        try:
            if _HAS_PANDAS_TA:
                adx_df = ta.adx(high, low, close, length=14)
                if adx_df is not None:
                    adx_col = [c for c in adx_df.columns if "ADX" in c]
                    pdi_col = [c for c in adx_df.columns if "DMP" in c]
                    mdi_col = [c for c in adx_df.columns if "DMN" in c]
                    if adx_col and pdi_col and mdi_col:
                        adx_val = adx_df[adx_col[0]].iloc[-1]
                        plus_di = adx_df[pdi_col[0]].iloc[-1]
                        minus_di = adx_df[mdi_col[0]].iloc[-1]
                    else:
                        adx_val = plus_di = minus_di = None
            else:
                adx_series = _adx(high, low, close, length=14)
                adx_val = adx_series.iloc[-1] if not adx_series.empty else None
                # Compute DI values for direction
                plus_dm = high.diff()
                minus_dm = -low.diff()
                plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
                minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
                prev_close = close.shift(1)
                tr = pd.concat([
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ], axis=1).max(axis=1)
                atr_s = tr.rolling(window=14).mean().replace(0, float('nan'))
                plus_di = (100 * plus_dm.rolling(window=14).mean() / atr_s).iloc[-1]
                minus_di = (100 * minus_dm.rolling(window=14).mean() / atr_s).iloc[-1]

            if (adx_val is not None and pd.notna(adx_val)
                    and pd.notna(plus_di) and pd.notna(minus_di)):
                if adx_val >= 25:
                    # Strong trend: score based on direction, scaled by ADX strength (capped at 50)
                    direction = 1.0 if plus_di > minus_di else -1.0
                    strength = min(1.0, (adx_val - 25) / 25)  # 25→0, 50→1.0
                    scores.append(direction * strength * 0.15)
                else:
                    # Weak trend: slightly negative (avoid choppy markets)
                    scores.append(-0.05)
        except Exception:
            logger.debug("ADX calculation failed", exc_info=True)

        # --- Stochastic signal (weight: 0.15) ---
        # %K < 20 = oversold → bullish; %K > 80 = overbought → bearish.
        # %K crossing above %D = bullish crossover; crossing below = bearish crossover.
        try:
            if _HAS_PANDAS_TA:
                stoch_df = ta.stoch(high, low, close, k=14, d=3)
                if stoch_df is not None:
                    k_col = [c for c in stoch_df.columns if "STOCHk" in c]
                    d_col = [c for c in stoch_df.columns if "STOCHd" in c]
                    if k_col and d_col:
                        stoch_k = stoch_df[k_col[0]].iloc[-1]
                        stoch_d = stoch_df[d_col[0]].iloc[-1]
                        stoch_k_prev = stoch_df[k_col[0]].iloc[-2] if len(stoch_df) > 1 else stoch_k
                        stoch_d_prev = stoch_df[d_col[0]].iloc[-2] if len(stoch_df) > 1 else stoch_d
                    else:
                        stoch_k = stoch_d = stoch_k_prev = stoch_d_prev = None
            else:
                k_series, d_series = _stochastic(high, low, close, k_period=14, d_period=3)
                stoch_k = k_series.iloc[-1] if not k_series.empty else None
                stoch_d = d_series.iloc[-1] if not d_series.empty else None
                stoch_k_prev = k_series.iloc[-2] if len(k_series) > 1 else stoch_k
                stoch_d_prev = d_series.iloc[-2] if len(d_series) > 1 else stoch_d

            if (stoch_k is not None and pd.notna(stoch_k)
                    and stoch_d is not None and pd.notna(stoch_d)):
                stoch_score = 0.0
                # Oversold/overbought signal
                if stoch_k < 20:
                    stoch_score += (20 - stoch_k) / 20 * 0.5  # up to 0.5 at 0
                elif stoch_k > 80:
                    stoch_score -= (stoch_k - 80) / 20 * 0.5  # up to -0.5 at 100
                # Crossover signal
                if (pd.notna(stoch_k_prev) and pd.notna(stoch_d_prev)):
                    if stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d:
                        stoch_score += 0.5  # bullish crossover
                    elif stoch_k_prev >= stoch_d_prev and stoch_k < stoch_d:
                        stoch_score -= 0.5  # bearish crossover
                scores.append(max(-1.0, min(1.0, stoch_score)) * 0.15)
        except Exception:
            logger.debug("Stochastic calculation failed", exc_info=True)

        if not scores:
            return {"score": 0.0, "trend_bullish": True, "rsi": latest_rsi}

        total = sum(scores)
        return {
            "score": max(-1.0, min(1.0, total)),
            "trend_bullish": trend_bullish,
            "rsi": latest_rsi,
        }
