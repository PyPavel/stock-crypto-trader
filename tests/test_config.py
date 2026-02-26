# tests/test_config.py
import pytest
from trader.config import Config, load_config


def test_load_config_basic(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "exchange: coinbase\nmode: paper\nstrategy: moderate\n"
        "capital: 100.0\npairs:\n  - BTC/USD\ncycle_interval: 60\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg.exchange == "coinbase"
    assert cfg.mode == "paper"
    assert cfg.strategy == "moderate"
    assert cfg.capital == 100.0
    assert cfg.pairs == ["BTC/USD"]
    assert cfg.cycle_interval == 60


def test_invalid_mode_raises(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "exchange: coinbase\nmode: turbo\nstrategy: moderate\n"
        "capital: 100.0\npairs:\n  - BTC/USD\ncycle_interval: 60\n"
    )
    with pytest.raises(ValueError, match="mode"):
        load_config(str(cfg_file))


def test_invalid_strategy_raises(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "exchange: coinbase\nmode: paper\nstrategy: yolo\n"
        "capital: 100.0\npairs:\n  - BTC/USD\ncycle_interval: 60\n"
    )
    with pytest.raises(ValueError, match="strategy"):
        load_config(str(cfg_file))
