"""Database layer.

Demo mode: loads all CSVs into a local SQLite database (data/demo.db).
Production: the same tables are created in MySQL using sql/01_schema_mysql.sql
and loaded with PyMySQL/SQLAlchemy (see load_mysql()).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cardholders (
    cardholder_id INTEGER PRIMARY KEY,
    company_name TEXT, industry TEXT, region TEXT, segment TEXT,
    employees REAL, tenure_months REAL, annual_spend_baseline REAL
);
CREATE TABLE IF NOT EXISTS engagement_monthly (
    cardholder_id INTEGER, month TEXT, month_index INTEGER,
    logins INTEGER, app_sessions INTEGER, email_clicks INTEGER,
    spend REAL, txn_count INTEGER, engagement_score REAL,
    score_fwd3 REAL, engagement_decline INTEGER,
    PRIMARY KEY (cardholder_id, month_index)
);
CREATE TABLE IF NOT EXISTS transactions (
    txn_id TEXT PRIMARY KEY, cardholder_id INTEGER, txn_date TEXT,
    merchant_category TEXT, amount REAL, channel TEXT,
    rewards_rate REAL, rewards_earned REAL
);
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY, campaign_name TEXT, channel TEXT,
    target_segment TEXT, months_active INTEGER, budget REAL,
    impressions INTEGER, clicks INTEGER, conversions INTEGER,
    attributed_revenue REAL, ctr REAL, cvr REAL, roas REAL, cpa REAL
);
CREATE TABLE IF NOT EXISTS ab_test_daily (
    date TEXT, arm TEXT, impressions INTEGER, conversions INTEGER, cvr REAL,
    PRIMARY KEY (date, arm)
);
CREATE INDEX IF NOT EXISTS ix_txn_cardholder ON transactions(cardholder_id);
CREATE INDEX IF NOT EXISTS ix_txn_date ON transactions(txn_date);
CREATE INDEX IF NOT EXISTS ix_eng_cardholder ON engagement_monthly(cardholder_id);
"""

TABLES = ["cardholders", "engagement_monthly", "transactions", "campaigns", "ab_test_daily"]


def load_sqlite(data_dir, db_path) -> str:
    """(Re)create the SQLite demo database and load every CSV."""
    data_dir, db_path = Path(data_dir), Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(SQLITE_SCHEMA)
    for table in TABLES:
        csv = data_dir / f"{table}.csv"
        df = pd.read_csv(csv)
        df.to_sql(table, conn, if_exists="append", index=False)
    conn.commit()
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    conn.close()
    return counts


def load_mysql(data_dir, mysql_cfg: dict) -> dict:
    """Production loader — requires PyMySQL + SQLAlchemy (see requirements.txt)
    and a running MySQL server with sql/01_schema_mysql.sql applied."""
    try:
        import pymysql  # noqa: F401
        from sqlalchemy import create_engine
    except ImportError as e:
        raise RuntimeError(
            "MySQL mode needs: pip install PyMySQL SQLAlchemy "
            "(and a running MySQL server with sql/01_schema_mysql.sql applied)"
        ) from e
    eng = create_engine(
        f"mysql+pymysql://{mysql_cfg['user']}:{mysql_cfg['password']}"
        f"@{mysql_cfg['host']}:{mysql_cfg['port']}/{mysql_cfg['database']}"
    )
    counts = {}
    for table in TABLES:
        df = pd.read_csv(Path(data_dir) / f"{table}.csv")
        df.to_sql(table, eng, if_exists="append", index=False)
        counts[table] = len(df)
    return counts


def run_query(db_path, sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()
