"""Tests for SymbolUniverse — dynamic symbol funnel."""
from unittest.mock import patch, MagicMock
from trader.core.universe import SymbolUniverse, _SymbolData
from trader.config import UniverseConfig


def _cfg(enabled=True, size=10, candidates=5, active_pairs=3) -> UniverseConfig:
    return UniverseConfig(enabled=enabled, size=size, candidates=candidates, active_pairs=active_pairs)


def _make_universe_data(n: int) -> list[_SymbolData]:
    """Generate n fake universe entries with varying momentum."""
    return [
        _SymbolData(
            symbol=f"COIN{i}/USD",
            price_change_24h_pct=float(i - n // 2),   # mix of positive and negative
            volume_24h=float(1_000_000 * (n - i + 1)),  # first entries have higher volume
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# get_candidates with universe disabled
# ---------------------------------------------------------------------------

def test_disabled_universe_returns_seed_pairs():
    u = SymbolUniverse(exchange="coinbase", seed_pairs=["BTC/USD", "ETH/USD"],
                       universe_config=_cfg(enabled=False))
    assert u.get_candidates() == ["BTC/USD", "ETH/USD"]


def test_no_config_returns_seed_pairs():
    u = SymbolUniverse(exchange="coinbase", seed_pairs=["BTC/USD"], universe_config=None)
    assert u.get_candidates() == ["BTC/USD"]


# ---------------------------------------------------------------------------
# get_candidates fallback when universe is empty
# ---------------------------------------------------------------------------

def test_empty_universe_falls_back_to_seed_pairs():
    u = SymbolUniverse(exchange="coinbase", seed_pairs=["BTC/USD", "ETH/USD"],
                       universe_config=_cfg())
    # No refresh called — universe is empty
    assert u.get_candidates() == ["BTC/USD", "ETH/USD"]


# ---------------------------------------------------------------------------
# Momentum scoring and candidate selection
# ---------------------------------------------------------------------------

def test_candidates_capped_at_configured_count():
    u = SymbolUniverse(exchange="coinbase", seed_pairs=[],
                       universe_config=_cfg(candidates=3))
    u._universe = _make_universe_data(10)
    candidates = u.get_candidates()
    # Should be at most `candidates` count (no seed extras here)
    assert len(candidates) == 3


def test_seed_pairs_always_included_even_if_low_momentum():
    low_momentum_seed = "LOWVOL/USD"
    u = SymbolUniverse(exchange="coinbase", seed_pairs=[low_momentum_seed],
                       universe_config=_cfg(candidates=3))
    # Universe has 10 symbols, none of which is the seed
    u._universe = _make_universe_data(10)
    candidates = u.get_candidates()
    assert low_momentum_seed in candidates


def test_top_symbols_sorted_by_abs_momentum():
    u = SymbolUniverse(exchange="coinbase", seed_pairs=[],
                       universe_config=_cfg(candidates=2))
    # Two symbols: one with large positive change, one large negative, one small
    u._universe = [
        _SymbolData("SMALL/USD", price_change_24h_pct=0.1, volume_24h=1_000_000),
        _SymbolData("BIG_UP/USD", price_change_24h_pct=10.0, volume_24h=1_000_000),
        _SymbolData("BIG_DOWN/USD", price_change_24h_pct=-9.0, volume_24h=1_000_000),
    ]
    candidates = u.get_candidates()
    assert candidates[0] == "BIG_UP/USD"
    assert candidates[1] == "BIG_DOWN/USD"
    assert "SMALL/USD" not in candidates


# ---------------------------------------------------------------------------
# Crypto symbol format
# ---------------------------------------------------------------------------

def test_crypto_symbols_converted_to_slash_usd_format():
    """CoinGecko returns 'btc' — universe must store 'BTC/USD'."""
    coingecko_response = [
        {"symbol": "btc", "price_change_percentage_24h": 2.5, "total_volume": 50_000_000_000},
        {"symbol": "eth", "price_change_percentage_24h": 1.0, "total_volume": 20_000_000_000},
    ]
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = coingecko_response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        u = SymbolUniverse(exchange="coinbase", seed_pairs=[],
                           universe_config=_cfg(size=2, candidates=5))
        u.refresh_universe()

    symbols = [d.symbol for d in u._universe]
    assert "BTC/USD" in symbols
    assert "ETH/USD" in symbols
    assert "btc" not in symbols


# ---------------------------------------------------------------------------
# refresh_universe — failure handling
# ---------------------------------------------------------------------------

def test_refresh_failure_preserves_previous_universe():
    u = SymbolUniverse(exchange="coinbase", seed_pairs=["BTC/USD"],
                       universe_config=_cfg())
    # Pre-populate universe
    previous = _make_universe_data(5)
    u._universe = previous

    with patch("requests.get", side_effect=Exception("network error")):
        u.refresh_universe()

    # Previous universe must be preserved
    assert u._universe == previous


def test_refresh_failure_does_not_raise():
    u = SymbolUniverse(exchange="coinbase", seed_pairs=["BTC/USD"],
                       universe_config=_cfg())
    with patch("requests.get", side_effect=Exception("timeout")):
        u.refresh_universe()  # must not raise


# ---------------------------------------------------------------------------
# Stock universe (Alpaca)
# ---------------------------------------------------------------------------

def test_stock_universe_missing_credentials_returns_empty():
    u = SymbolUniverse(exchange="alpaca", seed_pairs=["AAPL"],
                       universe_config=_cfg(), alpaca_cfg=None)
    # Should not raise, just log a warning and return []
    u.refresh_universe()
    assert u._universe == []


def test_stock_universe_parses_alpaca_response():
    alpaca_cfg = MagicMock()
    alpaca_cfg.api_key = "test-key"
    alpaca_cfg.api_secret = "test-secret"

    alpaca_response = {
        "most_actives": [
            {"symbol": "NVDA", "volume": 80_000_000, "change": 3.2},
            {"symbol": "AAPL", "volume": 60_000_000, "change": -1.5},
        ]
    }
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = alpaca_response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        u = SymbolUniverse(exchange="alpaca", seed_pairs=[],
                           universe_config=_cfg(size=10, candidates=5),
                           alpaca_cfg=alpaca_cfg)
        u.refresh_universe()

    symbols = [d.symbol for d in u._universe]
    assert "NVDA" in symbols
    assert "AAPL" in symbols


# ---------------------------------------------------------------------------
# needs_refresh
# ---------------------------------------------------------------------------

def test_needs_refresh_true_when_never_refreshed():
    u = SymbolUniverse(exchange="coinbase", seed_pairs=[], universe_config=_cfg())
    assert u.needs_refresh() is True


def test_needs_refresh_false_immediately_after_refresh():
    coingecko_response = [
        {"symbol": "btc", "price_change_percentage_24h": 1.0, "total_volume": 1_000_000},
    ]
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = coingecko_response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        u = SymbolUniverse(exchange="coinbase", seed_pairs=[],
                           universe_config=_cfg())
        u.refresh_universe()

    assert u.needs_refresh() is False
