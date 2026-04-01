from trader.core.risk import RiskManager
from trader.config import RiskConfig


def make_rm():
    return RiskManager(RiskConfig(max_position_pct=0.20, stop_loss_pct=0.05, max_drawdown_pct=0.15))


# ------------------------------------------------------------------ #
#  Original tests
# ------------------------------------------------------------------ #

def test_allows_valid_buy():
    rm = make_rm()
    result = rm.validate_buy("BTC/USD", 20.0, 100.0, {})
    assert result["allowed"] is True


def test_blocks_oversized_buy():
    rm = make_rm()
    result = rm.validate_buy("BTC/USD", 50.0, 100.0, {})
    assert result["allowed"] is False
    assert "position" in result["reason"].lower()


def test_blocks_buy_when_max_drawdown_hit():
    rm = make_rm()
    result = rm.validate_buy("BTC/USD", 10.0, 84.0, {}, starting_capital=100.0)
    assert result["allowed"] is False
    assert "drawdown" in result["reason"].lower()


def test_blocks_buy_when_total_drawdown_hit():
    rm = make_rm()
    positions = {"BTC/USD": {"amount": 0.001, "entry_price": 60000.0}}
    prices = {"BTC/USD": 60000.0}
    result = rm.validate_buy("BTC/USD", 5.0, 90.0, positions, starting_capital=200.0, prices=prices)
    assert result["allowed"] is False
    assert "drawdown" in result["reason"].lower()


def test_stop_loss_triggered():
    rm = make_rm()
    assert rm.check_stop_loss("BTC/USD", 40000.0, 37000.0) is True


def test_stop_loss_not_triggered():
    rm = make_rm()
    assert rm.check_stop_loss("BTC/USD", 40000.0, 39000.0) is False


# ------------------------------------------------------------------ #
#  ATR-based position sizing
# ------------------------------------------------------------------ #

def test_calc_position_size_fixed():
    rm = make_rm()
    # No ATR sizing enabled → fixed percentage
    size = rm.calc_position_size(1000.0, 50000.0, atr=500.0)
    assert size == 200.0  # 20% of 1000


def test_calc_position_size_atr():
    rm = RiskManager(RiskConfig(
        max_position_pct=0.20,
        use_atr_sizing=True,
        atr_stop_multiplier=2.0,
    ))
    # risk_per_trade = 1000 * 0.20 * 0.5 = 100
    # stop_distance = 500 * 2.0 = 1000
    # shares = 100 / 1000 = 0.1
    # usd = 0.1 * 50000 = 5000 → capped at 1000 * 0.20 = 200
    size = rm.calc_position_size(1000.0, 50000.0, atr=500.0)
    assert size == 200.0  # capped by max_position_pct


def test_calc_position_size_atr_no_cap():
    rm = RiskManager(RiskConfig(
        max_position_pct=0.50,
        use_atr_sizing=True,
        atr_stop_multiplier=2.0,
    ))
    # risk_per_trade = 10000 * 0.50 * 0.5 = 2500
    # stop_distance = 50 * 2.0 = 100
    # shares = 2500 / 100 = 25
    # usd = 25 * 500 = 12500 → capped at 10000 * 0.50 = 5000
    size = rm.calc_position_size(10000.0, 500.0, atr=50.0)
    assert size == 5000.0


def test_calc_position_size_atr_fallback_when_none():
    rm = RiskManager(RiskConfig(
        max_position_pct=0.20,
        use_atr_sizing=True,
    ))
    # ATR is None → falls back to fixed
    size = rm.calc_position_size(1000.0, 50000.0, atr=None)
    assert size == 200.0


# ------------------------------------------------------------------ #
#  Take-profit
# ------------------------------------------------------------------ #

def test_take_profit_triggered():
    rm = RiskManager(RiskConfig(take_profit_pct=0.10))
    assert rm.check_take_profit(100.0, 111.0) is True


def test_take_profit_not_triggered():
    rm = RiskManager(RiskConfig(take_profit_pct=0.10))
    assert rm.check_take_profit(100.0, 109.0) is False


def test_take_profit_disabled():
    rm = RiskManager(RiskConfig(take_profit_pct=0.0))
    assert rm.check_take_profit(100.0, 200.0) is False


def test_partial_take_profit_triggered():
    rm = RiskManager(RiskConfig(
        partial_take_profit_pct=0.50,
        partial_tp_trigger_pct=0.05,
    ))
    assert rm.check_partial_take_profit(100.0, 106.0) is True


def test_partial_take_profit_not_triggered():
    rm = RiskManager(RiskConfig(
        partial_take_profit_pct=0.50,
        partial_tp_trigger_pct=0.05,
    ))
    assert rm.check_partial_take_profit(100.0, 104.0) is False


def test_partial_tp_sell_amount():
    rm = RiskManager(RiskConfig(partial_take_profit_pct=0.50))
    assert rm.partial_sell_amount(1000.0) == 500.0


# ------------------------------------------------------------------ #
#  Correlation awareness
# ------------------------------------------------------------------ #

def test_correlation_blocks_correlated_positions():
    rm = RiskManager(RiskConfig(
        max_correlated_positions=2,
        correlation_groups={
            "layer1": ["BTC", "ETH", "SOL"],
            "memecoins": ["DOGE", "SHIB"],
        },
    ))
    positions = {
        "BTC/USD": {"amount": 0.01, "entry_price": 50000.0},
        "ETH/USD": {"amount": 0.1, "entry_price": 3000.0},
    }
    result = rm.validate_buy("SOL/USD", 100.0, 1000.0, positions)
    assert result["allowed"] is False
    assert "correlation" in result["reason"].lower()


def test_correlation_allows_within_limit():
    rm = RiskManager(RiskConfig(
        max_correlated_positions=2,
        correlation_groups={
            "layer1": ["BTC", "ETH", "SOL"],
        },
    ))
    positions = {
        "BTC/USD": {"amount": 0.01, "entry_price": 50000.0},
    }
    result = rm.validate_buy("SOL/USD", 100.0, 1000.0, positions)
    assert result["allowed"] is True


def test_correlation_ignores_ungrouped():
    rm = RiskManager(RiskConfig(
        max_correlated_positions=1,
        correlation_groups={
            "layer1": ["BTC", "ETH"],
        },
    ))
    positions = {
        "BTC/USD": {"amount": 0.01, "entry_price": 50000.0},
    }
    # SOL is not in any group → allowed
    result = rm.validate_buy("SOL/USD", 100.0, 1000.0, positions)
    assert result["allowed"] is True


def test_correlation_disabled_without_groups():
    rm = RiskManager(RiskConfig())  # no correlation_groups
    positions = {"BTC/USD": {"amount": 0.01, "entry_price": 50000.0},
                 "ETH/USD": {"amount": 0.1, "entry_price": 3000.0},
                 "SOL/USD": {"amount": 1.0, "entry_price": 100.0}}
    result = rm.validate_buy("ADA/USD", 100.0, 1000.0, positions)
    assert result["allowed"] is True


# ------------------------------------------------------------------ #
#  ATR-adaptive stops
# ------------------------------------------------------------------ #

def test_atr_stop_loss_triggered():
    rm = RiskManager(RiskConfig(
        use_atr_stops=True,
        stop_loss_pct=0.05,  # ignored when ATR mode
        atr_stop_multiplier=2.0,
    ))
    # entry=100, ATR=5, stop = 100 - 10 = 90
    assert rm.check_stop_loss("BTC/USD", 100.0, 89.0, atr=5.0) is True
    assert rm.check_stop_loss("BTC/USD", 100.0, 91.0, atr=5.0) is False


def test_atr_trailing_stop_triggered():
    rm = RiskManager(RiskConfig(
        use_atr_stops=True,
        trailing_stop_pct=0.08,  # ignored when ATR mode
        atr_trail_multiplier=3.0,
    ))
    # peak=200, ATR=10, trail = 200 - 30 = 170
    assert rm.check_trailing_stop(200.0, 169.0, atr=10.0) is True
    assert rm.check_trailing_stop(200.0, 171.0, atr=10.0) is False


def test_atr_stop_fallback_to_fixed():
    rm = RiskManager(RiskConfig(
        use_atr_stops=True,
        stop_loss_pct=0.05,
        atr_stop_multiplier=2.0,
    ))
    # No ATR provided → falls back to fixed percentage
    assert rm.check_stop_loss("BTC/USD", 40000.0, 37000.0) is True


# ------------------------------------------------------------------ #
#  Per-trade loss limit
# ------------------------------------------------------------------ #

def test_trade_loss_limit_triggered():
    rm = RiskManager(RiskConfig(max_trade_loss_pct=0.03))
    assert rm.check_trade_loss_limit(100.0, 96.0, 1000.0) is True


def test_trade_loss_limit_not_triggered():
    rm = RiskManager(RiskConfig(max_trade_loss_pct=0.03))
    assert rm.check_trade_loss_limit(100.0, 98.0, 1000.0) is False


def test_trade_loss_limit_disabled():
    rm = RiskManager(RiskConfig(max_trade_loss_pct=0.0))
    assert rm.check_trade_loss_limit(100.0, 50.0, 1000.0) is False


# ------------------------------------------------------------------ #
#  Daily loss limit
# ------------------------------------------------------------------ #

def test_daily_loss_limit_blocks_buy():
    rm = RiskManager(RiskConfig(max_daily_loss_pct=0.05))
    rm.reset_daily_tracking(1000.0)
    # Simulate losing 6% of starting capital
    daily_pnl = -60.0
    result = rm.validate_buy("BTC/USD", 100.0, 940.0, {}, starting_capital=1000.0, daily_pnl=daily_pnl)
    assert result["allowed"] is False
    assert "daily loss" in result["reason"].lower()


def test_daily_loss_limit_allows_small_loss():
    rm = RiskManager(RiskConfig(max_daily_loss_pct=0.05))
    rm.reset_daily_tracking(1000.0)
    # 3% loss → under limit
    daily_pnl = -30.0
    result = rm.validate_buy("BTC/USD", 100.0, 970.0, {}, starting_capital=1000.0, daily_pnl=daily_pnl)
    assert result["allowed"] is True


def test_daily_loss_limit_disabled():
    rm = RiskManager(RiskConfig(max_daily_loss_pct=0.0, max_drawdown_pct=0.50))
    result = rm.validate_buy("BTC/USD", 100.0, 900.0, {}, starting_capital=1000.0, daily_pnl=-100.0)
    assert result["allowed"] is True


def test_daily_pnl_tracking():
    rm = RiskManager(RiskConfig())
    rm.reset_daily_tracking(1000.0)
    assert rm.get_daily_pnl(950.0) == -50.0
    assert rm.get_daily_pnl(1100.0) == 100.0


# ------------------------------------------------------------------ #
#  Max positions check still works
# ------------------------------------------------------------------ #

def test_blocks_buy_when_max_positions():
    rm = RiskManager(RiskConfig(max_open_positions=2))
    positions = {
        "BTC/USD": {"amount": 0.01, "entry_price": 50000.0},
        "ETH/USD": {"amount": 0.1, "entry_price": 3000.0},
    }
    result = rm.validate_buy("SOL/USD", 100.0, 1000.0, positions)
    assert result["allowed"] is False
    assert "max open positions" in result["reason"].lower()
