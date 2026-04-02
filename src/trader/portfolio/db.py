import json
import sqlite3
from datetime import datetime
from trader.models import Trade


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            order_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            amount REAL NOT NULL,
            price REAL NOT NULL,
            fee REAL NOT NULL,
            mode TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            pnl REAL DEFAULT 0.0,
            narrative TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_history (
            symbol TEXT PRIMARY KEY,
            scores TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()


def save_signal_history(conn: sqlite3.Connection, symbol: str, scores) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO signal_history VALUES (?, ?, ?)",
        (symbol, json.dumps(list(scores)), datetime.utcnow().isoformat()),
    )
    conn.commit()


def load_signal_history(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT symbol, scores FROM signal_history").fetchall()
    return {row[0]: json.loads(row[1]) for row in rows}


def save_trade(conn: sqlite3.Connection, trade: Trade) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)",
        (trade.order_id, trade.symbol, trade.side, trade.amount, trade.price,
         trade.fee, trade.mode, trade.timestamp.isoformat(), trade.pnl, trade.narrative)
    )
    conn.commit()


def load_trades(conn: sqlite3.Connection) -> list[Trade]:
    rows = conn.execute("SELECT * FROM trades ORDER BY timestamp").fetchall()
    return [
        Trade(order_id=r[0], symbol=r[1], side=r[2], amount=r[3], price=r[4],
              fee=r[5], mode=r[6], timestamp=datetime.fromisoformat(r[7]),
              pnl=r[8], narrative=r[9])
        for r in rows
    ]
