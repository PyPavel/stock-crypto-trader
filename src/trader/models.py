from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
import uuid


@dataclass
class Candle:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    symbol: str
    score: float        # -1.0 (strong sell) to +1.0 (strong buy)
    reason: str
    trend_bullish: bool = True   # True if price is above 50-period EMA (bullish trend)
    rsi: float | None = None     # Latest RSI value (0-100), used for oversold reversal filter
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SentimentScore:
    symbol: str
    score: float        # -1.0 to +1.0
    source: str         # "reddit" | "cryptopanic" | "combined"
    items_analyzed: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Order:
    symbol: str
    side: Literal["buy", "sell"]
    amount: float
    price: float
    mode: Literal["paper", "live"]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"


@dataclass
class Trade:
    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    amount: float
    price: float
    fee: float
    mode: Literal["paper", "live"]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pnl: float = 0.0
    narrative: str = ""
