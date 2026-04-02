"""
MacroCollector — free macro regime signals via Yahoo Finance.

Fetches VIX, DXY (dollar index), and 10Y yield.
Returns a composite macro signal for both crypto and stock bots.

No API key required — uses yfinance which scrapes Yahoo Finance.

Signal logic:
  - VIX:  rising VIX → fear/risk-off → bearish for both crypto and stocks
  - DXY:  rising dollar → bearish for crypto (inverse correlation), mixed for stocks
  - 10Y:  rising yields → bearish for growth stocks, mixed for crypto

Returns score in [-1, +1]. Cached 30 minutes (macro data changes slowly).
"""

import logging
import time

logger = logging.getLogger(__name__)

_CACHE_TTL = 1800  # 30 minutes

# Yahoo Finance tickers
_VIX_TICKER = "^VIX"
_DXY_TICKER = "DX-Y.NYB"
_YIELD_10Y_TICKER = "^TNX"


class MacroCollector:
    """
    Macro regime signal from VIX, DXY, and 10Y yield.

    asset_class: "crypto" or "stock" — affects how DXY and yield signals are weighted.
    """

    def __init__(self, asset_class: str = "crypto"):
        self._asset_class = asset_class
        self._cached_score: float | None = None
        self._cache_ts: float = 0.0

    def score(self) -> float | None:
        if self._cached_score is not None and (time.time() - self._cache_ts) < _CACHE_TTL:
            return self._cached_score

        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — MacroCollector disabled")
            return self._cached_score

        try:
            tickers = yf.download(
                [_VIX_TICKER, _DXY_TICKER, _YIELD_10Y_TICKER],
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )

            close = tickers["Close"] if "Close" in tickers.columns else tickers.xs("Close", axis=1, level=0)

            def pct_change_2d(col_name: str) -> float | None:
                """2-day % change for a ticker column."""
                if col_name not in close.columns:
                    return None
                series = close[col_name].dropna()
                if len(series) < 2:
                    return None
                return float((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2] * 100)

            vix_chg = pct_change_2d(_VIX_TICKER)
            dxy_chg = pct_change_2d(_DXY_TICKER)
            yield_chg = pct_change_2d(_YIELD_10Y_TICKER)

            # Also get current VIX level for context
            vix_level = float(close[_VIX_TICKER].dropna().iloc[-1]) if _VIX_TICKER in close.columns else None

            components: list[float] = []

            # VIX signal: rising VIX = fear = bearish for everything
            # VIX > 30 is high fear, < 15 is complacency
            if vix_chg is not None:
                vix_signal = -max(-1.0, min(1.0, vix_chg / 10.0))
                components.append(vix_signal)

            # VIX level: sustained high VIX is itself bearish
            if vix_level is not None:
                if vix_level > 30:
                    components.append(-0.4)   # panic territory
                elif vix_level > 20:
                    components.append(-0.15)  # elevated uncertainty
                elif vix_level < 13:
                    components.append(0.1)    # complacency (mild contrarian negative)

            # DXY signal
            if dxy_chg is not None:
                if self._asset_class == "crypto":
                    # Rising dollar → bearish for crypto (strong inverse correlation)
                    dxy_signal = -max(-1.0, min(1.0, dxy_chg / 2.0))
                    components.append(dxy_signal * 0.8)
                else:
                    # For stocks: rising dollar hurts multinationals, mild negative
                    dxy_signal = -max(-1.0, min(1.0, dxy_chg / 3.0))
                    components.append(dxy_signal * 0.4)

            # 10Y yield signal
            if yield_chg is not None:
                if self._asset_class == "stock":
                    # Rising yields hurt growth stocks (higher discount rate)
                    yield_signal = -max(-1.0, min(1.0, yield_chg / 5.0))
                    components.append(yield_signal * 0.5)
                else:
                    # For crypto: rising yields = risk-off = mild bearish
                    yield_signal = -max(-1.0, min(1.0, yield_chg / 8.0))
                    components.append(yield_signal * 0.3)

            if not components:
                return self._cached_score

            macro_score = sum(components) / len(components)
            macro_score = max(-1.0, min(1.0, macro_score))

            logger.info(
                "Macro [%s]: VIX=%.1f(chg%+.1f%%) DXY=%+.1f%% 10Y=%+.1f%% → signal=%.3f",
                self._asset_class,
                vix_level or 0, vix_chg or 0,
                dxy_chg or 0, yield_chg or 0,
                macro_score,
            )

            self._cached_score = macro_score
            self._cache_ts = time.time()
            return macro_score

        except Exception as e:
            logger.warning("MacroCollector failed: %s", e)
            return self._cached_score
