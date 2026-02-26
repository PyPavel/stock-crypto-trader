import pytest
from trader.strategies.registry import get_strategy
from trader.strategies.moderate import ModerateStrategy
from trader.strategies.conservative import ConservativeStrategy
from trader.strategies.aggressive import AggressiveStrategy
from trader.config import RiskConfig


def test_get_moderate():
    assert isinstance(get_strategy("moderate", RiskConfig()), ModerateStrategy)


def test_get_conservative():
    assert isinstance(get_strategy("conservative", RiskConfig()), ConservativeStrategy)


def test_get_aggressive():
    assert isinstance(get_strategy("aggressive", RiskConfig()), AggressiveStrategy)


def test_unknown_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("yolo", RiskConfig())
