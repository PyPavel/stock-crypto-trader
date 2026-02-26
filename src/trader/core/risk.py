from trader.config import RiskConfig


class RiskManager:
    def __init__(self, risk: RiskConfig):
        self.risk = risk

    def validate_buy(
        self,
        symbol: str,
        usd_amount: float,
        capital: float,
        positions: dict,
        starting_capital: float | None = None,
    ) -> dict:
        # Check drawdown
        if starting_capital and starting_capital > 0:
            drawdown = (starting_capital - capital) / starting_capital
            if drawdown >= self.risk.max_drawdown_pct:
                return {"allowed": False,
                        "reason": f"max drawdown {drawdown:.1%} reached, halting trades"}

        # Check position size
        max_usd = capital * self.risk.max_position_pct
        if usd_amount > max_usd:
            return {"allowed": False,
                    "reason": f"position size ${usd_amount:.2f} exceeds max ${max_usd:.2f}"}

        return {"allowed": True, "reason": "ok"}

    def check_stop_loss(self, symbol: str, entry_price: float, current_price: float) -> bool:
        loss_pct = (entry_price - current_price) / entry_price
        return loss_pct >= self.risk.stop_loss_pct
