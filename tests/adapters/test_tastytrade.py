from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from trader.models import Candle, Order


def make_adapter(paper=True):
    """Build a TastyTradeAdapter with all external clients mocked."""
    # Python 3.14 patch() uses pkgutil._resolve_name which traverses attributes —
    # the module must be in sys.modules before the with-patch context is entered.
    import trader.adapters.tastytrade  # noqa: F401
    with patch("trader.adapters.tastytrade.Session") as mock_session_cls, \
         patch("trader.adapters.tastytrade.Account") as mock_account_cls, \
         patch("trader.adapters.tastytrade.StockHistoricalDataClient") as mock_data_cls:

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_account = MagicMock()
        mock_account_cls.get_accounts.return_value = [mock_account]
        mock_account.account_number = "ABC123"

        mock_data = MagicMock()
        mock_data_cls.return_value = mock_data

        from trader.adapters.tastytrade import TastyTradeAdapter
        from trader.config import TastyTradeConfig

        tt_cfg = TastyTradeConfig(username="u", password="p", account_number="ABC123", paper=paper)
        adapter = TastyTradeAdapter(
            tastytrade_cfg=tt_cfg,
            alpaca_data_key="ak",
            alpaca_data_secret="as",
        )
        adapter._session = mock_session
        adapter._account = mock_account
        adapter._data = mock_data
        return adapter, mock_session, mock_account, mock_data


def test_get_price_returns_midprice():
    adapter, _, _, mock_data = make_adapter()
    mock_quote = MagicMock()
    mock_quote.ask_price = 101.0
    mock_quote.bid_price = 99.0
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

    price = adapter.get_price("AAPL")
    assert price == 100.0


def test_get_price_ask_only():
    adapter, _, _, mock_data = make_adapter()
    mock_quote = MagicMock()
    mock_quote.ask_price = 105.0
    mock_quote.bid_price = 0.0
    mock_data.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

    price = adapter.get_price("AAPL")
    assert price == 105.0


def test_get_candles_returns_candle_objects():
    adapter, _, _, mock_data = make_adapter()
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    mock_bar = MagicMock()
    mock_bar.timestamp = ts
    mock_bar.open = 100.0
    mock_bar.high = 110.0
    mock_bar.low = 90.0
    mock_bar.close = 105.0
    mock_bar.volume = 1000.0
    mock_bars = MagicMock()
    mock_bars.data = {"AAPL": [mock_bar]}
    mock_data.get_stock_bars.return_value = mock_bars

    candles = adapter.get_candles("AAPL", "1h", limit=1)
    assert len(candles) == 1
    assert isinstance(candles[0], Candle)
    assert candles[0].symbol == "AAPL"
    assert candles[0].close == 105.0
    assert candles[0].volume == 1000.0
