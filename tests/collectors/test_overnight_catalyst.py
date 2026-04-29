from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import date, datetime, timezone
from trader.collectors.overnight_catalyst import OvernightCatalystCollector


def _make_earnings_dates(rows):
    """rows: list of (date, eps_estimate, eps_actual) — future dates have eps_actual=NaN"""
    index = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows], tz="America/New_York")
    df = pd.DataFrame(
        {"EPS Estimate": [r[1] for r in rows], "Reported EPS": [r[2] for r in rows]},
        index=index,
    )
    return df


def _today():
    return date.today()


def _ticker_with_amc(beat_rate_quarters, tonight=True):
    """Return a mock yfinance Ticker that has AMC earnings tonight."""
    ticker = MagicMock()
    today = _today()

    rows = []
    # Future AMC earnings tonight (or tomorrow if tonight=False)
    future_date = today if tonight else date(today.year, today.month + 1, 1)
    rows.append((future_date, 1.00, float("nan")))

    # Historical quarters to build beat history
    from datetime import timedelta
    for i in range(8):
        past = today - timedelta(days=90 * (i + 1))
        if i < round(beat_rate_quarters * 8):
            rows.append((past, 1.00, 1.20))   # beat
        else:
            rows.append((past, 1.00, 0.80))   # miss

    ticker.earnings_dates = _make_earnings_dates(rows)
    return ticker


def test_amc_high_beat_rate_gives_positive_score():
    collector = OvernightCatalystCollector()
    ticker = _ticker_with_amc(beat_rate_quarters=0.75)  # 75% beat rate
    with patch("yfinance.Ticker", return_value=ticker), \
         patch.object(collector, "_score_edgar", return_value=None):
        score = collector.score(["AAPL"])
    assert score is not None
    assert 0.55 <= score <= 0.80


def test_amc_low_beat_rate_gives_negative_score():
    collector = OvernightCatalystCollector()
    ticker = _ticker_with_amc(beat_rate_quarters=0.25)  # 25% beat rate
    with patch("yfinance.Ticker", return_value=ticker), \
         patch.object(collector, "_score_edgar", return_value=None):
        score = collector.score(["AAPL"])
    assert score is not None
    assert score < 0


def test_no_amc_tonight_returns_none():
    collector = OvernightCatalystCollector()
    ticker = _ticker_with_amc(beat_rate_quarters=0.75, tonight=False)
    with patch("yfinance.Ticker", return_value=ticker), \
         patch.object(collector, "_score_edgar", return_value=None):
        score = collector.score(["AAPL"])
    assert score is None


def test_empty_symbols_returns_none():
    collector = OvernightCatalystCollector()
    assert collector.score([]) is None


def test_amc_unknown_history_gives_mild_positive():
    collector = OvernightCatalystCollector()
    ticker = MagicMock()
    today = _today()
    ticker.earnings_dates = _make_earnings_dates([(today, float("nan"), float("nan"))])
    with patch("yfinance.Ticker", return_value=ticker), \
         patch.object(collector, "_score_edgar", return_value=None):
        score = collector.score(["AAPL"])
    assert score is not None
    assert 0.10 <= score <= 0.30
