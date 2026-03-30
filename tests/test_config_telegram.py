# tests/test_config_telegram.py
import os
from trader.config import load_config, TelegramConfig

def test_telegram_config_defaults():
    cfg = TelegramConfig()
    assert cfg.bot_token == ""
    assert cfg.chat_id == ""

def test_telegram_config_loaded_from_yaml(tmp_path):
    yaml_content = """
exchange: coinbase
mode: paper
strategy: moderate
capital: 1000.0
pairs: [BTC-USD]
telegram:
  bot_token: abc123
  chat_id: "99999"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    cfg = load_config(str(config_file))
    assert cfg.telegram.bot_token == "abc123"
    assert cfg.telegram.chat_id == "99999"

def test_telegram_config_env_override(tmp_path, monkeypatch):
    yaml_content = """
exchange: coinbase
mode: paper
strategy: moderate
capital: 1000.0
pairs: [BTC-USD]
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "envtoken")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "envid")
    cfg = load_config(str(config_file))
    assert cfg.telegram.bot_token == "envtoken"
    assert cfg.telegram.chat_id == "envid"
