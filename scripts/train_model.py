#!/usr/bin/env python3
"""
Train a LightGBM classifier on historical OHLCV data.

For each candle window, compute 9 technical indicator features.
Label = price direction 4 hours forward:
  0 = BUY   (return > +1%)
  1 = HOLD  (-1% <= return <= +1%)
  2 = SELL  (return < -1%)

Trains one model per exchange and saves to models/crypto.lgbm and models/stocks.lgbm.

Usage:
  python scripts/train_model.py
  python scripts/train_model.py --exchange coinbase
  python scripts/train_model.py --exchange alpaca
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing trader package from scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trader.models import Candle
from trader.ml.features import compute_features, FEATURE_NAMES

FORWARD_HOURS = 4       # predict price direction this many candles ahead
BUY_THRESHOLD = 0.01    # +1% → BUY
SELL_THRESHOLD = -0.01  # -1% → SELL
WINDOW = 60             # candle history fed to compute_features
LABEL_BUY = 0
LABEL_HOLD = 1
LABEL_SELL = 2


def load_parquet_as_candles(path: Path) -> list[Candle]:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    candles = []
    for row in df.itertuples():
        candles.append(Candle(
            symbol=path.stem,
            timestamp=row.timestamp,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        ))
    return candles


def build_dataset(candles: list[Candle]) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over candles and build (X, y) arrays."""
    X_rows = []
    y_rows = []

    for i in range(WINDOW, len(candles) - FORWARD_HOURS):
        window = candles[i - WINDOW: i]
        features = compute_features(window)
        if features is None:
            continue

        entry_price = candles[i].close
        exit_price = candles[i + FORWARD_HOURS].close
        forward_return = (exit_price - entry_price) / entry_price

        if forward_return > BUY_THRESHOLD:
            label = LABEL_BUY
        elif forward_return < SELL_THRESHOLD:
            label = LABEL_SELL
        else:
            label = LABEL_HOLD

        X_rows.append([features[name] for name in FEATURE_NAMES])
        y_rows.append(label)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


def train(exchange: str, parquet_dir: Path, output_path: Path) -> None:
    import lightgbm as lgb
    from sklearn.metrics import classification_report

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {parquet_dir}")
        return

    print(f"\n=== Training {exchange} model ===")
    all_X, all_y = [], []

    for pf in parquet_files:
        print(f"  Loading {pf.name}...")
        candles = load_parquet_as_candles(pf)
        X, y = build_dataset(candles)
        if len(X) == 0:
            print(f"    Skipped (insufficient data)")
            continue
        all_X.append(X)
        all_y.append(y)
        label_counts = {0: (y == 0).sum(), 1: (y == 1).sum(), 2: (y == 2).sum()}
        print(f"    {len(X)} samples — BUY={label_counts[0]}, HOLD={label_counts[1]}, SELL={label_counts[2]}")

    if not all_X:
        print("No training data. Aborting.")
        return

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"\nTotal samples: {len(X)}")
    print(f"Label distribution — BUY: {(y==0).sum()}, HOLD: {(y==1).sum()}, SELL: {(y==2).sum()}")

    # Time-ordered split: first 80% train, last 20% test
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Class weights to handle imbalance (HOLD dominates)
    from collections import Counter
    counts = Counter(y_train.tolist())
    total = sum(counts.values())
    class_weight = {k: total / (3 * v) for k, v in counts.items()}
    sample_weight = np.array([class_weight[int(label)] for label in y_train])

    print("\nTraining LightGBM...")
    train_ds = lgb.Dataset(X_train, label=y_train, weight=sample_weight,
                           feature_name=FEATURE_NAMES)
    val_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=300,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    # Evaluate
    probs = model.predict(X_test)           # shape (n, 3)
    preds = np.argmax(probs, axis=1)
    print("\nTest set classification report:")
    print(classification_report(y_test, preds, target_names=["BUY", "HOLD", "SELL"]))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    print(f"Model saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM ML signal scorer")
    parser.add_argument("--exchange", choices=["coinbase", "alpaca", "all"], default="all")
    args = parser.parse_args()

    base = Path("data/training")
    models = Path("models")

    if args.exchange in ("coinbase", "all"):
        train("coinbase", base / "coinbase", models / "crypto.lgbm")
    if args.exchange in ("alpaca", "all"):
        train("alpaca", base / "alpaca", models / "stocks.lgbm")


if __name__ == "__main__":
    main()
