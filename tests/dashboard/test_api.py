import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
from trader.dashboard.api import create_app
from trader.portfolio.state import Portfolio
from trader.config import Config, RiskConfig
from trader.core.router import OrderRouter


@pytest.fixture
def mock_engine(tmp_path):
    engine = MagicMock()
    engine.config.mode = "paper"
    engine.config.strategy = "moderate"
    engine.config.pairs = ["BTC/USD"]
    engine.config.risk = RiskConfig()
    engine.portfolio = Portfolio(db_path=str(tmp_path / "test.db"), starting_capital=100.0)
    engine._adapter.get_price.return_value = 42000.0
    # Give engine a real router so set_mode works
    engine._router = MagicMock()
    engine._router._mode = "paper"
    return engine


@pytest.mark.asyncio
async def test_status_endpoint(mock_engine):
    app = create_app(mock_engine, config_path=str(mock_engine.config))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "mode" in data
    assert "strategy" in data
    assert "cash" in data


@pytest.mark.asyncio
async def test_trades_endpoint(mock_engine):
    app = create_app(mock_engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/trades")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_set_mode_endpoint(mock_engine):
    app = create_app(mock_engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/mode", json={"mode": "live"})
    assert resp.status_code == 200
    assert mock_engine.config.mode == "live"


@pytest.mark.asyncio
async def test_invalid_mode_returns_400(mock_engine):
    app = create_app(mock_engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/mode", json={"mode": "turbo"})
    assert resp.status_code == 400
