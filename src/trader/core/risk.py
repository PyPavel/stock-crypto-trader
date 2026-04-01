import logging
from datetime import datetime, timezone
from trader.config import RiskConfig

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, risk: RiskConfig):
        self.risk = risk
        # Daily loss tracking
        self._daily_start_capital: float | None = None
        self._daily_start_date: str | None = None

    # ------------------------------------------------------------------ #
    #  Position sizing
    # ------------------------------------------------------------------ #

    def calc_position_size(
        self,
        capital: float,
        price: float,
        atr: float | None = None,
    ) -> float:
        """Return the USD amount to allocate for a new position.

        If use_atr_sizing is enabled and ATR is available, size inversely to
        volatility so that each trade risks a comparable dollar amount.
        Otherwise falls back to fixed max_position_pct of capital.
        """
        if self.risk.use_atr_sizing and atr and atr > 0 and price > 0:
            # Risk a fixed fraction of capital; stop-loss distance = atr_stop_multiplier * ATR
            risk_per_trade = capital * self.risk.max_position_pct * 0.5  # risk 50% of max alloc
            stop_distance = atr * self.risk.atr_stop_multiplier
            # Shares we can buy so that stop-hit loss ≈ risk_per_trade
            shares = risk_per_trade / stop_distance
            usd = shares * price
            return min(usd, capital * self.risk.max_position_pct)
        return capital * self.risk.max_position_pct

    # ------------------------------------------------------------------ #
    #  Buy validation
    # ------------------------------------------------------------------ #

    def validate_buy(
        self,
        symbol: str,
        usd_amount: float,
        capital: float,
        positions: dict,
        starting_capital: float | None = None,
        prices: dict[str, float] | None = None,
        daily_pnl: float | None = None,
    ) -> dict:
        # Check drawdown based on total portfolio value (cash + open positions)
        if starting_capital and starting_capital > 0:
            holdings_value = 0.0
            if prices:
                for sym, pos in positions.items():
                    holdings_value += pos.get("amount", 0.0) * prices.get(sym, 0.0)
            total_value = capital + holdings_value
            drawdown = (starting_capital - total_value) / starting_capital
            if drawdown >= self.risk.max_drawdown_pct:
                return {"allowed": False,
                        "reason": f"max drawdown {drawdown:.1%} reached, halting trades"}

        # Check daily loss limit
        if self.risk.max_daily_loss_pct > 0 and daily_pnl is not None:
            if starting_capital and starting_capital > 0:
                daily_loss_pct = -daily_pnl / starting_capital
                if daily_pnl < 0 and daily_loss_pct >= self.risk.max_daily_loss_pct:
                    return {"allowed": False,
                            "reason": f"daily loss limit {daily_loss_pct:.1%} reached, halting trades"}

        # Check max open positions
        open_count = len(positions)
        if open_count >= self.risk.max_open_positions:
            return {"allowed": False,
                    "reason": f"max open positions {self.risk.max_open_positions} reached ({open_count} active)"}

        # Check correlation limits
        corr_result = self._check_correlation(symbol, positions)
        if not corr_result["allowed"]:
            return corr_result

        # Check position size
        max_usd = capital * self.risk.max_position_pct
        if usd_amount > max_usd:
            return {"allowed": False,
                    "reason": f"position size ${usd_amount:.2f} exceeds max ${max_usd:.2f}"}

        return {"allowed": True, "reason": "ok"}

    # ------------------------------------------------------------------ #
    #  Correlation awareness
    # ------------------------------------------------------------------ #

    def _check_correlation(self, symbol: str, positions: dict) -> dict:
        """Block buy if already holding max_correlated_positions in the same group."""
        if not self.risk.correlation_groups:
            return {"allowed": True, "reason": "ok"}

        base = symbol.split("/")[0]  # e.g. "BTC" from "BTC/USD"
        # Find which group this symbol belongs to
        group_name = None
        group_members = set()
        for gname, members in self.risk.correlation_groups.items():
            members_upper = [m.upper() for m in members]
            if base.upper() in members_upper:
                group_name = gname
                group_members = set(members_upper)
                break

        if group_name is None:
            return {"allowed": True, "reason": "ok"}

        # Count open positions in this group
        count = 0
        for pos_sym in positions:
            pos_base = pos_sym.split("/")[0].upper()
            if pos_base in group_members:
                count += 1

        if count >= self.risk.max_correlated_positions:
            return {"allowed": False,
                    "reason": (f"correlation limit: {count} positions in "
                               f"'{group_name}' group (max {self.risk.max_correlated_positions})")}

        return {"allowed": True, "reason": "ok"}

    # ------------------------------------------------------------------ #
    #  Stop-loss (fixed or ATR-adaptive)
    # ------------------------------------------------------------------ #

    def check_stop_loss(self, symbol: str, entry_price: float, current_price: float,
                        atr: float | None = None) -> bool:
        if self.risk.use_atr_stops and atr and atr > 0:
            stop_price = entry_price - atr * self.risk.atr_stop_multiplier
            return current_price <= stop_price
        loss_pct = (entry_price - current_price) / entry_price
        return loss_pct >= self.risk.stop_loss_pct

    # ------------------------------------------------------------------ #
    #  Trailing stop (fixed or ATR-adaptive)
    # ------------------------------------------------------------------ #

    def check_trailing_stop(self, peak_price: float, current_price: float,
                            atr: float | None = None) -> bool:
        """Returns True if current price has dropped enough from its peak."""
        if peak_price <= 0:
            return False
        if self.risk.use_atr_stops and atr and atr > 0:
            trail_price = peak_price - atr * self.risk.atr_trail_multiplier
            return current_price <= trail_price
        drop_pct = (peak_price - current_price) / peak_price
        return drop_pct >= self.risk.trailing_stop_pct

    # ------------------------------------------------------------------ #
    #  Take-profit
    # ------------------------------------------------------------------ #

    def check_take_profit(self, entry_price: float, current_price: float) -> bool:
        """Returns True if full take-profit target is hit."""
        if self.risk.take_profit_pct <= 0 or entry_price <= 0:
            return False
        profit_pct = (current_price - entry_price) / entry_price
        return profit_pct >= self.risk.take_profit_pct

    def check_partial_take_profit(self, entry_price: float, current_price: float) -> bool:
        """Returns True if partial take-profit trigger is hit."""
        if self.risk.partial_tp_trigger_pct <= 0 or entry_price <= 0:
            return False
        profit_pct = (current_price - entry_price) / entry_price
        return profit_pct >= self.risk.partial_tp_trigger_pct

    def partial_sell_amount(self, position_value: float) -> float:
        """Return the USD amount to sell for a partial take-profit."""
        return position_value * self.risk.partial_take_profit_pct

    # ------------------------------------------------------------------ #
    #  Per-trade loss limit
    # ------------------------------------------------------------------ #

    def check_trade_loss_limit(self, entry_price: float, current_price: float,
                               starting_capital: float) -> bool:
        """Returns True if per-trade loss limit is exceeded."""
        if self.risk.max_trade_loss_pct <= 0 or entry_price <= 0:
            return False
        loss_pct = (entry_price - current_price) / entry_price
        return loss_pct >= self.risk.max_trade_loss_pct

    # ------------------------------------------------------------------ #
    #  Daily loss tracking helpers
    # ------------------------------------------------------------------ #

    def reset_daily_tracking(self, current_capital: float) -> None:
        """Call at the start of each trading day to reset daily P&L baseline."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_start_date != today:
            self._daily_start_capital = current_capital
            self._daily_start_date = today

    def get_daily_pnl(self, current_capital: float) -> float:
        """Return P&L since start of today (negative = loss)."""
        if self._daily_start_capital is None:
            return 0.0
        return current_capital - self._daily_start_capital
