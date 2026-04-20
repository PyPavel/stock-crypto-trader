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


def test_tastytrade_config_defaults():
    from trader.config import TastyTradeConfig
    cfg = TastyTradeConfig()
    assert cfg.username == ""
    assert cfg.password == ""
    assert cfg.account_number == ""
    assert cfg.paper is True


def test_tastytrade_config_in_main_config(tmp_path):
    from trader.config import load_config
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "exchange: tastytrade\n"
        "mode: paper\n"
        "strategy: moderate\n"
        "capital: 5000\n"
        "pairs: [AAPL]\n"
        "tastytrade:\n"
        "  username: user\n"
        "  password: pass\n"
        "  account_number: ABC123\n"
        "  paper: true\n"
    )
    cfg = load_config(str(config_file))
    assert cfg.tastytrade.username == "user"
    assert cfg.tastytrade.password == "pass"
    assert cfg.tastytrade.account_number == "ABC123"
    assert cfg.tastytrade.paper is True


def test_tastytrade_config_env_overrides(tmp_path, monkeypatch):
    from trader.config import load_config
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "exchange: tastytrade\n"
        "mode: paper\n"
        "strategy: moderate\n"
        "capital: 5000\n"
        "pairs: [AAPL]\n"
    )
    monkeypatch.setenv("TASTYTRADE_USERNAME", "envuser")
    monkeypatch.setenv("TASTYTRADE_PASSWORD", "envpass")
    monkeypatch.setenv("TASTYTRADE_ACCOUNT_NUMBER", "ENV999")
    monkeypatch.setenv("TASTYTRADE_PAPER", "false")

    cfg = load_config(str(config_file))
    assert cfg.tastytrade.username == "envuser"
    assert cfg.tastytrade.password == "envpass"
    assert cfg.tastytrade.account_number == "ENV999"
    assert cfg.tastytrade.paper is False
