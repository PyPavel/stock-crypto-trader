# src/trader/config.py
from dataclasses import dataclass, field
from typing import Literal
import yaml


@dataclass
class OllamaConfig:
    model: str = "mistral"
    base_url: str = "http://localhost:11434"


@dataclass
class CoinbaseConfig:
    api_key: str = ""
    api_secret: str = ""


@dataclass
class RedditConfig:
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "trader-bot/1.0"


@dataclass
class CryptoPanicConfig:
    api_key: str = ""


@dataclass
class LLMAdvisorConfig:
    enabled: bool = False
    provider: str = "claude"
    api_key: str = ""


@dataclass
class RiskConfig:
    max_position_pct: float = 0.20
    stop_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15


@dataclass
class Config:
    exchange: str
    mode: Literal["paper", "live"]
    strategy: Literal["conservative", "moderate", "aggressive"]
    capital: float
    pairs: list[str]
    cycle_interval: int = 60
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    coinbase: CoinbaseConfig = field(default_factory=CoinbaseConfig)
    reddit: RedditConfig = field(default_factory=RedditConfig)
    cryptopanic: CryptoPanicConfig = field(default_factory=CryptoPanicConfig)
    llm_advisor: LLMAdvisorConfig = field(default_factory=LLMAdvisorConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)

    mode = data.get("mode", "paper")
    if mode not in ("paper", "live"):
        raise ValueError(f"mode must be 'paper' or 'live', got '{mode}'")

    strategy = data.get("strategy", "moderate")
    if strategy not in ("conservative", "moderate", "aggressive"):
        raise ValueError(f"strategy must be conservative/moderate/aggressive, got '{strategy}'")

    cfg = Config(
        exchange=data["exchange"],
        mode=mode,
        strategy=strategy,
        capital=float(data["capital"]),
        pairs=data["pairs"],
        cycle_interval=data.get("cycle_interval", 60),
    )

    for key, cls in [
        ("ollama", OllamaConfig),
        ("coinbase", CoinbaseConfig),
        ("reddit", RedditConfig),
        ("cryptopanic", CryptoPanicConfig),
        ("llm_advisor", LLMAdvisorConfig),
        ("risk", RiskConfig),
    ]:
        if key in data:
            setattr(cfg, key, cls(**data[key]))

    return cfg
