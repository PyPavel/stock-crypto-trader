from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"


class ModeRequest(BaseModel):
    mode: str


class StrategyRequest(BaseModel):
    strategy: str


def create_app(engine) -> FastAPI:
    app = FastAPI(title="Trader Dashboard")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html = (TEMPLATES_DIR / "index.html").read_text()
        return HTMLResponse(content=html)

    @app.get("/api/status")
    async def status():
        prices = {}
        for symbol in engine.config.pairs:
            try:
                prices[symbol] = engine._adapter.get_price(symbol)
            except Exception:
                prices[symbol] = 0.0
        return {
            "mode": engine.config.mode,
            "strategy": engine.config.strategy,
            "pairs": engine.config.pairs,
            "cash": engine.portfolio.cash,
            "positions": engine.portfolio.positions,
            "total_value": engine.portfolio.total_value(prices),
            "prices": prices,
        }

    @app.get("/api/trades")
    async def trades():
        return [
            {"order_id": t.order_id, "symbol": t.symbol, "side": t.side,
             "amount": t.amount, "price": t.price, "mode": t.mode,
             "timestamp": t.timestamp.isoformat(), "pnl": t.pnl, "narrative": t.narrative}
            for t in engine.portfolio.get_trades()
        ]

    @app.post("/api/mode")
    async def set_mode(req: ModeRequest):
        if req.mode not in ("paper", "live"):
            raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")
        engine.config.mode = req.mode
        engine._router._mode = req.mode
        return {"mode": req.mode}

    @app.post("/api/strategy")
    async def set_strategy(req: StrategyRequest):
        from trader.strategies.registry import get_strategy
        if req.strategy not in ("conservative", "moderate", "aggressive"):
            raise HTTPException(status_code=400, detail="invalid strategy")
        engine.config.strategy = req.strategy
        engine._strategy = get_strategy(req.strategy, engine.config.risk)
        return {"strategy": req.strategy}

    @app.post("/api/cycle")
    async def trigger_cycle():
        engine.run_cycle()
        return {"status": "cycle complete"}

    return app
