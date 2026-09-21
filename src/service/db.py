"""SQLite database schema and persistence layer for paper trading service.

Ensures restart-safety and idempotency:
- Unique constraints on (symbol, bar_timestamp) prevent duplicate order execution.
- Persists signals, orders, fills, positions, and equity curve snapshots.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger


class Database:
    """SQLite manager for paper-trading state persistence."""

    def __init__(self, db_path: str = "data/paper_trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Create tables and indexes with unique idempotency constraints."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Signals table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    bar_timestamp TEXT NOT NULL,
                    prob_long REAL NOT NULL,
                    confidence REAL NOT NULL,
                    regime_id INTEGER NOT NULL,
                    sentiment_score REAL NOT NULL,
                    raw_data JSON,
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, bar_timestamp)
                );
                """
            )

            # 2. Orders table (idempotent: 1 order per symbol per bar_timestamp)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    bar_timestamp TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    fill_price REAL NOT NULL,
                    fee REAL NOT NULL,
                    target_weight REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, bar_timestamp)
                );
                """
            )

            # 3. Portfolio snapshots table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    drawdown REAL NOT NULL,
                    positions_json JSON
                );
                """
            )

            # 4. Model registry table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    metrics_json JSON,
                    created_at TEXT NOT NULL
                );
                """
            )

            # 5. Processed bars table (guarantees post-processing idempotency)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_bars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    bar_timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    decision_reason TEXT NOT NULL,
                    target_weight REAL NOT NULL,
                    processed_at TEXT NOT NULL,
                    UNIQUE(symbol, bar_timestamp)
                );
                """
            )
            conn.commit()
            logger.info(f"Initialized paper trading SQLite DB at {self.db_path}")

    def has_order_for_bar(self, symbol: str, bar_timestamp: str) -> bool:
        """Check if an order was already processed for this bar (restart safety)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM orders WHERE symbol = ? AND bar_timestamp = ?",
                (symbol, bar_timestamp),
            )
            return cursor.fetchone() is not None

    def record_signal(
        self,
        symbol: str,
        bar_timestamp: str,
        prob_long: float,
        confidence: float,
        regime_id: int,
        sentiment_score: float,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record generated signal; ignores if already exists."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO signals 
                    (symbol, bar_timestamp, prob_long, confidence, regime_id, sentiment_score, raw_data, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        bar_timestamp,
                        prob_long,
                        confidence,
                        regime_id,
                        sentiment_score,
                        json.dumps(raw_data or {}),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to record signal: {e}")
            return False

    def record_order(
        self,
        symbol: str,
        bar_timestamp: str,
        side: str,
        qty: float,
        fill_price: float,
        fee: float,
        target_weight: float,
        status: str = "FILLED",
    ) -> bool:
        """Record executed trade; guarantees idempotency."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO orders 
                    (symbol, bar_timestamp, side, qty, fill_price, fee, target_weight, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        bar_timestamp,
                        side,
                        qty,
                        fill_price,
                        fee,
                        target_weight,
                        status,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to record order: {e}")
            return False

    def record_snapshot(
        self,
        equity: float,
        cash: float,
        positions_value: float,
        drawdown: float,
        positions: Dict[str, float],
    ):
        """Record portfolio equity curve snapshot."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO portfolio_snapshots (timestamp, equity, cash, positions_value, drawdown, positions_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        equity,
                        cash,
                        positions_value,
                        drawdown,
                        json.dumps(positions),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record snapshot: {e}")

    def get_latest_signals(self, limit: int = 50) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def get_recent_orders(self, limit: int = 50) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def get_portfolio_history(self, limit: int = 500) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM portfolio_snapshots ORDER BY id ASC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def is_bar_processed(self, symbol: str, bar_timestamp: str) -> bool:
        """Check if a bar has already completed evaluation and execution."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM processed_bars WHERE symbol = ? AND bar_timestamp = ?",
                (symbol, bar_timestamp),
            )
            return cursor.fetchone() is not None

    def record_processed_bar(
        self,
        symbol: str,
        bar_timestamp: str,
        action: str,
        decision_reason: str,
        target_weight: float,
    ) -> bool:
        """Record bar post-processing completion idempotency key."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO processed_bars
                    (symbol, bar_timestamp, action, decision_reason, target_weight, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        bar_timestamp,
                        action,
                        decision_reason,
                        target_weight,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to record processed bar: {e}")
            return False

    def clear_all(self):
        """Clear all paper trading records to allow fresh simulation/testing."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM orders;")
            cursor.execute("DELETE FROM signals;")
            cursor.execute("DELETE FROM portfolio_snapshots;")
            cursor.execute("DELETE FROM processed_bars;")
            conn.commit()
            logger.info("Database paper trading history wiped clean.")
