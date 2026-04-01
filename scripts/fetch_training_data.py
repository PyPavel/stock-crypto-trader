#!/usr/bin/env python3
"""
Fetch 2 years of 1h OHLCV candles for all configured symbols.
Saves one parquet file per symbol to data/training/{exchange}/{symbol}.parquet

Usage:
  python scripts/fetch_training_data.py --exchange coinbase
  python scripts/fetch_training_data.py --exchange alpaca
  python scripts/fetch_training_data.py --exchange all

Requires env vars: COINBASE_API_KEY, COINBASE_API_SECRET (for coinbase)
                   ALPACA_API_KEY, ALPACA_API_SECRET (for alpaca)
"""
import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

CRYPTO_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD",
    "DOT/USD", "LINK/USD", "LTC/USD", "AVAX/USD", "DOGE/USD",
]
STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "MA",
]
LOOKBACK_DAYS = 730   # 2 years
OUTPUT_DIR = Path("data/training")


def fetch_coinbase(symbols: list[str]) -> None:
    import ccxt
    exchange = ccxt.coinbase({
        "apiKey": os.environ.get("COINBASE_API_KEY", ""),
        "secret": os.environ.get("COINBASE_API_SECRET", ""),
    })
    out_dir = OUTPUT_DIR / "coinbase"
    out_dir.mkdir(parents=True, exist_ok=True)

    since_ms = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)

    for symbol in symbols:
        print(f"Fetching {symbol} from Coinbase...")
        all_rows = []
        fetch_since = since_ms
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", since=fetch_since, limit=300)
            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")
                break
            if not ohlcv:
                break
            all_rows.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts <= fetch_since:
                break
            fetch_since = last_ts + 1
            if last_ts >= int(datetime.now(timezone.utc).timestamp() * 1000) - 3_600_000:
                break
            time.sleep(0.3)  # rate limit

        if not all_rows:
            print(f"  No data for {symbol}, skipping.")
            continue

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        safe_name = symbol.replace("/", "_")
        path = out_dir / f"{safe_name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  Saved {len(df)} rows → {path}")


def fetch_alpaca(symbols: list[str]) -> None:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(
        api_key=os.environ.get("ALPACA_API_KEY", ""),
        secret_key=os.environ.get("ALPACA_API_SECRET", ""),
    )
    out_dir = OUTPUT_DIR / "alpaca"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

    for symbol in symbols:
        print(f"Fetching {symbol} from Alpaca...")
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
            )
            bars = client.get_stock_bars(req)
            df = bars.df.reset_index()
            # Alpaca returns MultiIndex (symbol, timestamp) → flatten
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol].drop(columns=["symbol"])
            df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high",
                                    "l": "low", "c": "close", "v": "volume"})
            keep = ["timestamp", "open", "high", "low", "close", "volume"]
            df = df[[c for c in keep if c in df.columns]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

            path = out_dir / f"{symbol}.parquet"
            df.to_parquet(path, index=False)
            print(f"  Saved {len(df)} rows → {path}")
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV data for ML training")
    parser.add_argument("--exchange", choices=["coinbase", "alpaca", "all"], default="all")
    args = parser.parse_args()

    if args.exchange in ("coinbase", "all"):
        fetch_coinbase(CRYPTO_SYMBOLS)
    if args.exchange in ("alpaca", "all"):
        fetch_alpaca(STOCK_SYMBOLS)

    print("\nDone. Files saved to data/training/")


if __name__ == "__main__":
    main()
