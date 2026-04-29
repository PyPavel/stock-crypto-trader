"""
OvernightCatalystCollector — scores stocks with a specific overnight catalyst.

Source A: yfinance earnings calendar
  - AMC tonight + historical beat rate >= 65% → +0.55 to +0.75
  - AMC tonight + historical beat rate < 40%  → -0.25
  - AMC tonight + insufficient history         → +0.20
  - No AMC tonight                             → None

Source B: SEC EDGAR 8-K filings (today)
  - Tier-1 keywords (merger, FDA approval, …)  → +0.45
  - Tier-2 keywords (major contract, …)         → +0.30
  - No filing today                             → None

When both sources produce a signal: average them, cap at +0.80.
Returns None if NO symbol has a catalyst today.
Cache TTL: 60 minutes per symbol.
"""
import logging
import time
from datetime import date

import requests

logger = logging.getLogger(__name__)

_CACHE_TTL = 3600   # 1 hour

_EDGAR_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%22{ticker}%22&dateRange=custom&startdt={today}&enddt={today}&forms=8-K"
)

_TIER1_KEYWORDS = [
    "merger", "acquisition", "fda approval", "fda approved",
    "breakthrough designation", "going private", "going-private",
]
_TIER2_KEYWORDS = [
    "major contract", "licensing agreement", "strategic alliance",
    "partnership agreement", "definitive agreement",
]


class OvernightCatalystCollector:
    def __init__(self):
        self._cache: dict[str, tuple[float, float | None]] = {}  # symbol → (ts, score)

    def score(self, symbols: list[str]) -> float | None:
        if not symbols:
            return None
        scores = []
        for sym in symbols:
            s = self._score_symbol(sym)
            if s is not None:
                scores.append(s)
        if not scores:
            return None
        return sum(scores) / len(scores)

    def _score_symbol(self, symbol: str) -> float | None:
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and (now - cached[0]) < _CACHE_TTL:
            return cached[1]

        earnings_score = self._score_earnings(symbol)
        edgar_score = self._score_edgar(symbol)

        combined = self._blend(earnings_score, edgar_score)
        self._cache[symbol] = (now, combined)
        if combined is not None:
            logger.info(
                "OvernightCatalyst [%s]: earnings=%s edgar=%s → %.3f",
                symbol, earnings_score, edgar_score, combined,
            )
        return combined

    def _blend(self, earnings: float | None, edgar: float | None) -> float | None:
        signals = [s for s in (earnings, edgar) if s is not None]
        if not signals:
            return None
        avg = sum(signals) / len(signals)
        return min(0.80, max(-1.0, avg))

    def _score_earnings(self, symbol: str) -> float | None:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — earnings catalyst disabled")
            return None

        try:
            ticker = yf.Ticker(symbol)
            earnings_dates = ticker.earnings_dates
        except Exception as e:
            logger.warning("OvernightCatalyst: yfinance fetch failed for %s: %s", symbol, e)
            return None

        if earnings_dates is None or earnings_dates.empty:
            return None

        today = date.today()
        has_amc_tonight = False

        for idx in earnings_dates.index:
            try:
                d = idx.date() if hasattr(idx, "date") else idx
                if d == today:
                    has_amc_tonight = True
                    break
            except Exception:
                continue

        if not has_amc_tonight:
            return None

        beat_rate = self._compute_beat_rate(earnings_dates)
        if beat_rate is None:
            return 0.20   # tonight earnings but no history → mild positive

        if beat_rate >= 0.65:
            # Scale from +0.55 (at 65%) to +0.75 (at 100%)
            return 0.55 + (beat_rate - 0.65) / 0.35 * 0.20
        elif beat_rate < 0.40:
            return -0.25
        else:
            return 0.10   # mixed history → mild positive

    def _compute_beat_rate(self, earnings_dates) -> float | None:
        if "EPS Estimate" not in earnings_dates.columns or "Reported EPS" not in earnings_dates.columns:
            return None
        today = date.today()
        beats, total = 0, 0
        for idx in sorted(earnings_dates.index, reverse=True):
            if total >= 8:
                break
            try:
                d = idx.date() if hasattr(idx, "date") else idx
                if d >= today:
                    continue
                est = earnings_dates.loc[idx, "EPS Estimate"]
                act = earnings_dates.loc[idx, "Reported EPS"]
                if str(est) == "nan" or str(act) == "nan":
                    continue
                total += 1
                if float(act) > float(est):
                    beats += 1
            except Exception:
                continue
        return beats / total if total >= 2 else None

    def _score_edgar(self, symbol: str) -> float | None:
        today = date.today().isoformat()
        url = _EDGAR_URL.format(ticker=symbol, today=today)
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "trader-bot/1.0 contact@example.com"})
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("OvernightCatalyst: EDGAR fetch failed for %s: %s", symbol, e)
            return None

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return None

        for hit in hits:
            text = (hit.get("_source", {}).get("file_date", "") + " " +
                    " ".join(hit.get("_source", {}).get("period_of_report", ""))).lower()
            entity = " ".join(
                str(v).lower()
                for v in hit.get("_source", {}).values()
                if isinstance(v, str)
            )
            combined_text = text + " " + entity

            for kw in _TIER1_KEYWORDS:
                if kw in combined_text:
                    logger.info("OvernightCatalyst EDGAR [%s]: tier-1 keyword '%s'", symbol, kw)
                    return 0.45

            for kw in _TIER2_KEYWORDS:
                if kw in combined_text:
                    logger.info("OvernightCatalyst EDGAR [%s]: tier-2 keyword '%s'", symbol, kw)
                    return 0.30

        return None
