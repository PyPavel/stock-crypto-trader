# src/trader/config.py
from dataclasses import dataclass, field
from typing import Literal
import os
import yaml


@dataclass
class MimoConfig:
    model: str = "mimo-v2-flash"
    api_key: str = ""


@dataclass
class CoinbaseConfig:
    api_key: str = ""
    api_secret: str = ""


@dataclass
class AlpacaConfig:
    api_key: str = ""
    api_secret: str = ""
    paper: bool = True


@dataclass
class RedditConfig:
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "trader-bot/1.0"


@dataclass
class CryptoPanicConfig:
    api_key: str = ""


@dataclass
class MLConfig:
    enabled: bool = False
    model_path: str = "models/signal_scorer.lgb"


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""


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
    max_open_positions: int = 5
    trailing_stop_pct: float = 0.08
    cooldown_minutes: int = 30
    # Take-profit
    take_profit_pct: float = 0.00        # 0 = disabled; e.g. 0.10 = 10% profit target
    partial_take_profit_pct: float = 0.0  # 0 = disabled; fraction to sell at take_profit_pct
    partial_tp_trigger_pct: float = 0.0   # profit % at which partial TP fires
    # Volatility-adjusted sizing
    use_atr_sizing: bool = False          # scale position size by 1/ATR
    atr_period: int = 14
    atr_size_factor: float = 1.0          # multiplier for ATR-based sizing target
    # ATR-adaptive stops
    use_atr_stops: bool = False           # if True, stop_loss and trailing_stop are ATR multiples
    atr_stop_multiplier: float = 2.0      # stop-loss = entry - ATR * multiplier
    atr_trail_multiplier: float = 3.0     # trailing stop = peak - ATR * multiplier
    # Correlation awareness
    max_correlated_positions: int = 2     # max positions in same correlation group
    correlation_groups: dict[str, list[str]] = field(default_factory=dict)
    # Loss limits
    max_daily_loss_pct: float = 0.0       # 0 = disabled; halt trading if daily loss exceeds this
    max_trade_loss_pct: float = 0.0       # 0 = disabled; max loss on a single trade as % of capital


@dataclass
class Config:
    exchange: str
    mode: Literal["paper", "live"]
    strategy: Literal["conservative", "moderate", "aggressive"]
    capital: float
    pairs: list[str]
    cycle_interval: int = 60
    mimo: MimoConfig = field(default_factory=MimoConfig)
    coinbase: CoinbaseConfig = field(default_factory=CoinbaseConfig)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    reddit: RedditConfig = field(default_factory=RedditConfig)
    cryptopanic: CryptoPanicConfig = field(default_factory=CryptoPanicConfig)
    llm_advisor: LLMAdvisorConfig = field(default_factory=LLMAdvisorConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)

    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


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
        ("mimo", MimoConfig),
        ("coinbase", CoinbaseConfig),
        ("alpaca", AlpacaConfig),
        ("reddit", RedditConfig),
        ("cryptopanic", CryptoPanicConfig),
        ("llm_advisor", LLMAdvisorConfig),
        ("risk", RiskConfig),
        ("ml", MLConfig),
        ("telegram", TelegramConfig),
    ]:
        if key in data:
            setattr(cfg, key, cls(**data[key]))

    # Environment variable overrides — env vars take precedence over config.yaml values.
    if os.environ.get("COINBASE_API_KEY"):
        cfg.coinbase.api_key = os.environ["COINBASE_API_KEY"]
    if os.environ.get("COINBASE_API_SECRET"):
        # .env files store PEM keys with literal \n — convert to real newlines
        cfg.coinbase.api_secret = os.environ["COINBASE_API_SECRET"].replace("\\n", "\n")
    if os.environ.get("CRYPTOPANIC_API_KEY"):
        cfg.cryptopanic.api_key = os.environ["CRYPTOPANIC_API_KEY"]
    if os.environ.get("MIMO_API_KEY"):
        cfg.mimo.api_key = os.environ["MIMO_API_KEY"]
    if os.environ.get("LLM_ADVISOR_API_KEY"):
        cfg.llm_advisor.api_key = os.environ["LLM_ADVISOR_API_KEY"]
    if os.environ.get("REDDIT_CLIENT_ID"):
        cfg.reddit.client_id = os.environ["REDDIT_CLIENT_ID"]
    if os.environ.get("REDDIT_CLIENT_SECRET"):
        cfg.reddit.client_secret = os.environ["REDDIT_CLIENT_SECRET"]
    if os.environ.get("ALPACA_API_KEY"):
        cfg.alpaca.api_key = os.environ["ALPACA_API_KEY"]
    if os.environ.get("ALPACA_API_SECRET"):
        cfg.alpaca.api_secret = os.environ["ALPACA_API_SECRET"]
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        cfg.telegram.bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
    if os.environ.get("TELEGRAM_CHAT_ID"):
        cfg.telegram.chat_id = os.environ["TELEGRAM_CHAT_ID"]

    return cfg
