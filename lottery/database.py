from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS draws (
    issue TEXT PRIMARY KEY,
    draw_date TEXT NOT NULL,
    red1 INTEGER NOT NULL,
    red2 INTEGER NOT NULL,
    red3 INTEGER NOT NULL,
    red4 INTEGER NOT NULL,
    red5 INTEGER NOT NULL,
    red6 INTEGER NOT NULL,
    blue INTEGER NOT NULL,
    sales REAL,
    pool REAL,
    first_prize_count INTEGER,
    first_prize_amount REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_draws_date ON draws(draw_date);
"""


@dataclass
class Draw:
    issue: str
    draw_date: str  # ISO 日期字符串 (YYYY-MM-DD)
    reds: List[int]
    blue: int
    sales: Optional[float] = None
    pool: Optional[float] = None
    first_prize_count: Optional[int] = None
    first_prize_amount: Optional[float] = None


def get_conn(db_path: Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


def _validate_draw(d: Draw) -> None:
    """基础校验，防止脏数据入库。"""
    if len(d.reds) != 6:
        raise ValueError(f"期号 {d.issue} 红球数量不是6个")
    reds = [int(r) for r in d.reds]
    if len(set(reds)) != 6:
        raise ValueError(f"期号 {d.issue} 红球存在重复: {reds}")
    for r in reds:
        if r < 1 or r > 33:
            raise ValueError(f"期号 {d.issue} 红球超出范围1-33: {r}")
    blue = int(d.blue)
    if blue < 1 or blue > 16:
        raise ValueError(f"期号 {d.issue} 蓝球超出范围1-16: {blue}")
    d.reds = reds
    d.blue = blue


def latest_issue(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("SELECT issue FROM draws ORDER BY issue DESC LIMIT 1").fetchone()
    return row["issue"] if row else None


def upsert_draws(conn: sqlite3.Connection, draws: Iterable[Draw]) -> int:
    now = datetime.utcnow().isoformat()
    payload = []
    for d in draws:
        _validate_draw(d)
        payload.append(
            (
                d.issue,
                d.draw_date,
                d.reds[0],
                d.reds[1],
                d.reds[2],
                d.reds[3],
                d.reds[4],
                d.reds[5],
                d.blue,
                d.sales,
                d.pool,
                d.first_prize_count,
                d.first_prize_amount,
                now,
                now,
            )
        )
    if not payload:
        return 0

    sql = """
    INSERT INTO draws (
        issue, draw_date,
        red1, red2, red3, red4, red5, red6,
        blue, sales, pool, first_prize_count, first_prize_amount,
        created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(issue) DO UPDATE SET
        draw_date=excluded.draw_date,
        red1=excluded.red1,
        red2=excluded.red2,
        red3=excluded.red3,
        red4=excluded.red4,
        red5=excluded.red5,
        red6=excluded.red6,
        blue=excluded.blue,
        sales=excluded.sales,
        pool=excluded.pool,
        first_prize_count=excluded.first_prize_count,
        first_prize_amount=excluded.first_prize_amount,
        updated_at=excluded.updated_at;
    """
    cur = conn.executemany(sql, payload)
    conn.commit()
    return cur.rowcount


def fetch_all(conn: sqlite3.Connection):
    return conn.execute(
        """
        SELECT
            issue, draw_date,
            red1, red2, red3, red4, red5, red6,
            blue, sales, pool, first_prize_count, first_prize_amount
        FROM draws
        ORDER BY draw_date
        """
    ).fetchall()

