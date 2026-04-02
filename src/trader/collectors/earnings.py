"""
EarningsCollector — earnings-aware signal for stocks.

Two strategies:
  1. Pre-earnings anticipation: buy before if beat probability is high
  2. Post-earnings dip buying: stock drops hard but beat estimates → overreaction → buy

Signal logic:
  PRE-EARNINGS (earnings in next 1–5 days):
    - Compute historical EPS beat rate from last 8 quarters
    - High beat rate (>65%) + positive analyst revisions → mild positive signal (+0.15 to +0.25)
    - Low beat rate (<40%) or declining estimates → mild negative signal (caution)
    - Day before / day of earnings → signal dampened (uncertainty)

  POST-EARNINGS DIP (earnings in last 1–3 days, stock down >4%):
    - If actual EPS beat estimate → market overreacted → strong positive signal (+0.35 to +0.5)
    - If missed but stock down >10% and other signals bullish → mild positive (oversold)
    - If beat AND guidance raised → maximum positive signal (+0.5)

  NO EARNINGS EVENT: returns 0.0 (neutral, don't affect other signals)

Data source: yfinance (free, no API key)
Cache TTL: 4 hours per symbol (earnings data changes slowly)
"""

import logging
import time
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

_CACHE_TTL = 14400   # 4 hours
_PRE_EARNINGS_DAYS = 5   # days before earnings to start signal
_POST_EARNINGS_DAYS = 3  # days after earnings to look for dip opportunity
_DIP_THRESHOLD_PCT = 4.0  # stock must drop this much post-earnings for dip signal


class EarningsCollector:
    """
    Earnings-aware signal generator for individual stocks.
    Returns a numeric signal [-1, +1] per symbol.
    """

    def __init__(self):
        # {symbol: (fetched_at, signal, metadata)}
        self._cache: dict[str, tuple[float, float, dict]] = {}

    def score(self, symbols: list[str]) -> float | None:
        """
        Return average earnings signal across the given symbols.
        Returns None if no earnings events detected (neutral, don't influence).
        """
        signals = []
        for sym in symbols:
            sig = self._score_symbol(sym)
            if sig is not None:
                signals.append(sig)

        if not signals:
            return None
        return sum(signals) / len(signals)

    def _score_symbol(self, symbol: str) -> float | None:
        """Score a single stock symbol. Returns None if no earnings event nearby."""
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and (now - cached[0]) < _CACHE_TTL:
            score, meta = cached[1], cached[2]
            if meta:
                logger.debug("Earnings [%s] cached: signal=%.3f reason=%s", symbol, score, meta.get("reason", ""))
            return score if meta else None  # None means "no event", 0.0 means "event but neutral"

        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — EarningsCollector disabled")
            return None

        try:
            ticker = yf.Ticker(symbol)
            signal, meta = self._compute_signal(ticker, symbol)

            self._cache[symbol] = (now, signal if signal is not None else 0.0, meta or {})
            if meta:
                logger.info(
                    "Earnings [%s]: signal=%.3f reason=%s",
                    symbol, signal or 0.0, meta.get("reason", ""),
                )
            return signal

        except Exception as e:
            logger.warning("EarningsCollector failed for %s: %s", symbol, e)
            return None

    def _compute_signal(self, ticker, symbol: str) -> tuple[float | None, dict | None]:
        """
        Core earnings signal computation. Returns (signal, metadata) or (None, None).
        """
        today = datetime.now(timezone.utc).date()

        # --- Get earnings dates ---
        try:
            earnings_dates = ticker.earnings_dates
        except Exception:
            return None, None

        if earnings_dates is None or earnings_dates.empty:
            return None, None

        # Find next (upcoming) and most recent (past) earnings
        next_earnings = None
        last_earnings = None
        last_eps_actual = None
        last_eps_estimate = None

        for idx in earnings_dates.index:
            try:
                date = idx.date() if hasattr(idx, 'date') else idx
                eps_est = earnings_dates.loc[idx, "EPS Estimate"] if "EPS Estimate" in earnings_dates.columns else None
                eps_act = earnings_dates.loc[idx, "Reported EPS"] if "Reported EPS" in earnings_dates.columns else None

                if date >= today:
                    # Future earnings
                    if next_earnings is None or date < next_earnings:
                        next_earnings = date
                else:
                    # Past earnings
                    if last_earnings is None or date > last_earnings:
                        last_earnings = date
                        try:
                            last_eps_actual = float(eps_act) if eps_act and str(eps_act) != 'nan' else None
                            last_eps_estimate = float(eps_est) if eps_est and str(eps_est) != 'nan' else None
                        except (ValueError, TypeError):
                            pass
            except Exception:
                continue

        days_to_next = (next_earnings - today).days if next_earnings else None
        days_since_last = (today - last_earnings).days if last_earnings else None

        # --- Historical beat rate from last 8 quarters ---
        beat_rate = self._compute_beat_rate(earnings_dates, n_quarters=8)

        # --- Get recent price change for post-earnings dip detection ---
        price_drop_pct = self._get_recent_price_drop(ticker, days=3)

        # ================================================================
        # POST-EARNINGS DIP BUYING (highest priority)
        # Earnings just happened (1–3 days ago) AND stock dropped hard
        # ================================================================
        if days_since_last is not None and 0 < days_since_last <= _POST_EARNINGS_DAYS:
            if price_drop_pct is not None and price_drop_pct <= -_DIP_THRESHOLD_PCT:
                # Did they beat estimates?
                beat = (
                    last_eps_actual is not None
                    and last_eps_estimate is not None
                    and last_eps_actual > last_eps_estimate
                )
                miss = (
                    last_eps_actual is not None
                    and last_eps_estimate is not None
                    and last_eps_actual < last_eps_estimate
                )

                if beat:
                    # Beat estimates but stock dropped — classic sell-the-news overreaction
                    surprise_pct = (
                        (last_eps_actual - last_eps_estimate) / abs(last_eps_estimate) * 100
                        if last_eps_estimate and last_eps_estimate != 0 else 0
                    )
                    # Stronger signal for bigger drop (more oversold) + bigger beat
                    drop_magnitude = min(abs(price_drop_pct) / 10.0, 0.5)  # caps at 0.5
                    beat_bonus = min(surprise_pct / 20.0, 0.2)             # caps at 0.2
                    signal = min(0.5, 0.25 + drop_magnitude + beat_bonus)
                    return signal, {
                        "reason": f"post-earnings dip BUY: dropped {price_drop_pct:.1f}% but beat EPS by {surprise_pct:.1f}%",
                        "days_since": days_since_last,
                        "beat": True,
                        "surprise_pct": surprise_pct,
                        "price_drop_pct": price_drop_pct,
                    }

                elif miss and price_drop_pct <= -10.0:
                    # Missed AND dropped >10% — potentially oversold but risky
                    # Weak positive signal, let other signals confirm
                    return 0.1, {
                        "reason": f"post-earnings miss but extreme drop {price_drop_pct:.1f}% — possible oversold",
                        "days_since": days_since_last,
                        "beat": False,
                        "price_drop_pct": price_drop_pct,
                    }

                else:
                    # Dropped but unknown/insufficient EPS data
                    return 0.0, {
                        "reason": f"post-earnings drop {price_drop_pct:.1f}% — no EPS data to confirm",
                        "days_since": days_since_last,
                    }

        # ================================================================
        # PRE-EARNINGS ANTICIPATION (earnings in next 1–5 days)
        # ================================================================
        if days_to_next is not None and 0 < days_to_next <= _PRE_EARNINGS_DAYS:
            if days_to_next == 1:
                # Day before: uncertainty is high — dampen any signal
                if beat_rate is not None and beat_rate >= 0.70:
                    return 0.10, {
                        "reason": f"earnings tomorrow, strong beat history ({beat_rate:.0%})",
                        "days_to": days_to_next,
                        "beat_rate": beat_rate,
                    }
                return 0.0, {
                    "reason": f"earnings tomorrow — holding neutral (uncertainty)",
                    "days_to": days_to_next,
                }

            if beat_rate is not None:
                if beat_rate >= 0.70:
                    # Strong historical beat rate → anticipation signal
                    signal = 0.15 + (beat_rate - 0.70) * 0.5  # 0.15 to 0.25
                    return min(signal, 0.25), {
                        "reason": f"pre-earnings anticipation: beat rate {beat_rate:.0%}, earnings in {days_to_next}d",
                        "days_to": days_to_next,
                        "beat_rate": beat_rate,
                    }
                elif beat_rate <= 0.40:
                    # Weak beat history → caution
                    return -0.10, {
                        "reason": f"pre-earnings caution: low beat rate {beat_rate:.0%}, earnings in {days_to_next}d",
                        "days_to": days_to_next,
                        "beat_rate": beat_rate,
                    }
                else:
                    # Mixed history — neutral
                    return 0.0, {
                        "reason": f"pre-earnings neutral: beat rate {beat_rate:.0%}, earnings in {days_to_next}d",
                        "days_to": days_to_next,
                        "beat_rate": beat_rate,
                    }

        # No earnings event in range — no signal
        return None, None

    def _compute_beat_rate(self, earnings_dates, n_quarters: int = 8) -> float | None:
        """
        Compute fraction of last N quarters where EPS beat the estimate.
        Returns None if insufficient data.
        """
        if earnings_dates is None or earnings_dates.empty:
            return None
        if "EPS Estimate" not in earnings_dates.columns or "Reported EPS" not in earnings_dates.columns:
            return None

        today = datetime.now(timezone.utc).date()
        beats = 0
        total = 0

        for idx in sorted(earnings_dates.index, reverse=True):
            if total >= n_quarters:
                break
            try:
                date = idx.date() if hasattr(idx, 'date') else idx
                if date >= today:
                    continue
                eps_est = earnings_dates.loc[idx, "EPS Estimate"]
                eps_act = earnings_dates.loc[idx, "Reported EPS"]
                if str(eps_est) == 'nan' or str(eps_act) == 'nan':
                    continue
                est = float(eps_est)
                act = float(eps_act)
                total += 1
                if act > est:
                    beats += 1
            except Exception:
                continue

        if total < 2:
            return None
        return beats / total

    def _get_recent_price_drop(self, ticker, days: int = 3) -> float | None:
        """
        Get price change % over the last N days.
        Negative = drop. Returns None if data unavailable.
        """
        try:
            hist = ticker.history(period=f"{days + 2}d")
            if hist is None or len(hist) < 2:
                return None
            start_price = float(hist["Close"].iloc[0])
            end_price = float(hist["Close"].iloc[-1])
            if start_price <= 0:
                return None
            return (end_price - start_price) / start_price * 100
        except Exception:
            return None
