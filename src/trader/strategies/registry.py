from trader.strategies.base import Strategy
from trader.strategies.moderate import ModerateStrategy
from trader.strategies.conservative import ConservativeStrategy
from trader.strategies.aggressive import AggressiveStrategy
from trader.config import RiskConfig

_REGISTRY = {
    "moderate": ModerateStrategy,
    "conservative": ConservativeStrategy,
    "aggressive": AggressiveStrategy,
}


def get_strategy(name: str, risk: RiskConfig) -> Strategy:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(_REGISTRY)}")
    return _REGISTRY[name](risk=risk)
