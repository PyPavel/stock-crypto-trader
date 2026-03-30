"""
SymbolUniverse — dynamic symbol pool with two-stage momentum funnel.

Stage 1: Universe refresh every 24h
  - Crypto  → CoinGecko /coins/markets (top N by market cap)
  - Stocks  → Alpaca /v1beta1/screener/stocks/most_actives

Stage 2: Momentum filter every cycle
  - Score   = price_change_24h_pct × volume_ratio
  - Top 50 by abs(score) become candidates
  - seed_pairs are always included regardless of rank
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

# CoinGecko free API — no key required
_COINGECKO_MARKETS_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&sparkline=false"
    "&price_change_percentage=24h"
)

# Alpaca screener endpoint (data API)
_ALPACA_MOVERS_URL = (
    "https://data.alpaca.markets/v1beta1/screener/stocks/movers"
    "?top={top}"
)

_24H_SECONDS = 86_400


@dataclass
class _SymbolData:
    """Lightweight holder for per-symbol universe data."""
    symbol: str                   # exchange-ready symbol, e.g. "BTC/USD" or "AAPL"
    price_change_24h_pct: float   # raw 24h price change in percent
    volume_24h: float             # 24h traded volume in native units or USD


class SymbolUniverse:
    """
    Manages a dynamic universe of tradeable symbols.

    Parameters
    ----------
    exchange:
        Exchange identifier string, e.g. "coinbase" or "alpaca".
    seed_pairs:
        Symbols from config.pairs that are always included in candidates,
        regardless of momentum rank.
    universe_config:
        UniverseConfig dataclass (enabled, size, candidates, active_pairs).
        When enabled=False (default), get_candidates() returns seed_pairs unchanged.
    alpaca_cfg:
        AlpacaConfig — required for the Alpaca stocks universe refresh.
    """

    def __init__(
        self,
        exchange: str,
        seed_pairs: list[str],
        universe_config=None,
        alpaca_cfg=None,
    ) -> None:
        self._exchange = exchange.lower()
        self._seed_pairs: list[str] = list(seed_pairs)
        self._universe_cfg = universe_config
        self._alpaca_api_key: str = alpaca_cfg.api_key if alpaca_cfg else ""
        self._alpaca_api_secret: str = alpaca_cfg.api_secret if alpaca_cfg else ""

        # Derived settings — fall back to defaults when no config supplied
        self._universe_size: int = universe_config.size if universe_config else 200
        self._candidates_count: int = universe_config.candidates if universe_config else 50

        # Internal state
        self._universe: list[_SymbolData] = []
        self._last_refresh_ts: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self._universe_cfg and self._universe_cfg.enabled)

    @property
    def active_pairs(self) -> int:
        if self._universe_cfg and self._universe_cfg.enabled:
            return self._universe_cfg.active_pairs
        return len(self._seed_pairs)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_candidates(self) -> list[str]:
        """
        Return up to `candidates_count` symbols after the momentum filter.

        Always includes seed_pairs. Falls back to seed_pairs if the universe
        is disabled, empty, or scoring fails entirely.
        """
        if not self.enabled:
            return list(self._seed_pairs)

        if not self._universe:
            logger.warning("Universe is empty — falling back to seed_pairs")
            return list(self._seed_pairs)

        scored = self._score_momentum()

        if not scored:
            logger.warning("Momentum scoring produced no results — falling back to seed_pairs")
            return list(self._seed_pairs)

        # Sort by abs(momentum_score) descending
        scored.sort(key=lambda x: abs(x[1]), reverse=True)

        # Take top N
        top_symbols = [sym for sym, _ in scored[: self._candidates_count]]

        # Merge seed_pairs: add any not already in the list
        seed_set = set(self._seed_pairs)
        top_set = set(top_symbols)
        extras = [s for s in self._seed_pairs if s not in top_set]
        result = top_symbols + extras

        logger.info(
            "Candidates: %d momentum + %d seed extras = %d total",
            len(top_symbols),
            len(extras),
            len(result),
        )
        return result

    def refresh_universe(self) -> None:
        """
        Fetch the broad symbol pool from the exchange data source.

        Safe to call at any time — on failure keeps the previous universe
        and logs a warning rather than raising.
        """
        try:
            if self._exchange == "coinbase":
                data = self._fetch_coingecko_universe()
            elif self._exchange == "alpaca":
                data = self._fetch_alpaca_universe()
            else:
                logger.warning("Unknown exchange '%s' — cannot refresh universe", self._exchange)
                return

            if data:
                self._universe = data
                self._last_refresh_ts = time.time()
                logger.info(
                    "Universe refreshed: %d symbols (exchange=%s)",
                    len(self._universe),
                    self._exchange,
                )
            else:
                logger.warning(
                    "Universe refresh returned 0 symbols — keeping previous list (size=%d)",
                    len(self._universe),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Universe refresh failed: %s — keeping previous list", exc)

    def needs_refresh(self) -> bool:
        """Return True if 24 h have elapsed since last successful refresh."""
        return (time.time() - self._last_refresh_ts) >= _24H_SECONDS

    # ------------------------------------------------------------------
    # Momentum scoring
    # ------------------------------------------------------------------

    def _score_momentum(self) -> list[tuple[str, float]]:
        """
        Compute momentum_score = price_change_24h_pct × volume_ratio for each
        symbol in the universe.

        volume_ratio = symbol_volume / mean_volume_of_universe

        Returns list of (symbol, score) tuples.  Symbols that fail scoring are
        silently skipped.
        """
        volumes = [s.volume_24h for s in self._universe if s.volume_24h > 0]
        if not volumes:
            return []

        mean_vol = sum(volumes) / len(volumes)
        if mean_vol == 0:
            return []

        results: list[tuple[str, float]] = []
        for entry in self._universe:
            try:
                volume_ratio = entry.volume_24h / mean_vol
                score = entry.price_change_24h_pct * volume_ratio
                results.append((entry.symbol, score))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping %s in momentum scoring: %s", entry.symbol, exc)

        return results

    # ------------------------------------------------------------------
    # Data fetchers
    # ------------------------------------------------------------------

    def _fetch_coingecko_universe(self) -> list[_SymbolData]:
        """Fetch top-N crypto from CoinGecko /coins/markets."""
        per_page = min(self._universe_size, 250)  # CoinGecko max per_page = 250
        pages_needed = max(1, -(-self._universe_size // per_page))  # ceiling division

        all_entries: list[_SymbolData] = []

        for page in range(1, pages_needed + 1):
            url = (
                f"{_COINGECKO_MARKETS_URL}"
                f"&per_page={per_page}&page={page}"
            )
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "trader-bot/1.0"},
            )
            resp.raise_for_status()
            coins: list[dict] = resp.json()

            if not coins:
                break

            for coin in coins:
                raw_symbol: str = coin.get("symbol", "").upper()
                if not raw_symbol:
                    continue

                # Convert CoinGecko symbol (e.g. "BTC") → Coinbase format "BTC/USD"
                symbol = f"{raw_symbol}/USD"

                price_change = float(coin.get("price_change_percentage_24h") or 0.0)
                volume = float(coin.get("total_volume") or 0.0)

                all_entries.append(
                    _SymbolData(
                        symbol=symbol,
                        price_change_24h_pct=price_change,
                        volume_24h=volume,
                    )
                )

            if len(all_entries) >= self._universe_size:
                break

            # Respect CoinGecko free-tier rate limit (be polite)
            if page < pages_needed:
                time.sleep(1.0)

        return all_entries[: self._universe_size]

    def _fetch_alpaca_universe(self) -> list[_SymbolData]:
        """
        Fetch top movers (gainers + losers) from Alpaca screener.

        Endpoint: GET /v1beta1/screener/stocks/movers?top=N
        Returns gainers and losers with percent_change and price.
        Volume is not available; we use abs(percent_change) as proxy so
        momentum_score = percent_change * abs(percent_change) = signed(change²).
        """
        if not self._alpaca_api_key or not self._alpaca_api_secret:
            logger.warning("Alpaca API credentials missing — cannot fetch stock universe")
            return []

        # Alpaca movers endpoint max is 50 per side (gainers + losers)
        top = min(self._universe_size // 2, 50)
        url = _ALPACA_MOVERS_URL.format(top=top)

        resp = requests.get(
            url,
            timeout=15,
            headers={
                "APCA-API-KEY-ID": self._alpaca_api_key,
                "APCA-API-SECRET-KEY": self._alpaca_api_secret,
                "User-Agent": "trader-bot/1.0",
            },
        )
        resp.raise_for_status()
        payload = resp.json()

        # Response: {"gainers": [...], "losers": [...]}
        # Each entry: {"symbol": "NVDA", "percent_change": 5.2, "change": 4.1, "price": 890.0}
        gainers: list[dict] = payload.get("gainers", [])
        losers: list[dict] = payload.get("losers", [])

        results: list[_SymbolData] = []
        for item in gainers + losers:
            symbol: str = item.get("symbol", "").upper()
            if not symbol:
                continue
            pct = float(item.get("percent_change") or 0.0)
            # Use abs(pct) as volume proxy so momentum = pct * abs(pct) = signed(pct²)
            results.append(
                _SymbolData(
                    symbol=symbol,
                    price_change_24h_pct=pct,
                    volume_24h=abs(pct),
                )
            )

        return results
