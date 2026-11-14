"""Build data/app.db — the single SQLite database handed to students.

Three tables:
  houses — the Ames housing data (pre-loaded)
  users  — empty, schema only
  usage  — empty, schema only

Run once:  .venv/bin/python data/make_db.py
"""
import os
import sqlite3

import pandas as pd
from sklearn.datasets import fetch_openml

# Id = identifier. The rest are ~80-99% missing ("NA" mostly means "doesn't have it").
DROP = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature"]

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.db")

USERS_SCHEMA = """
CREATE TABLE users (
    user_id     TEXT PRIMARY KEY,              -- the user's email
    key_hash    TEXT NOT NULL,                 -- SHA-256 of the API key, never the key
    credits     INTEGER NOT NULL DEFAULT 100,  -- prepaid balance (top-up)
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

USAGE_SCHEMA = """
CREATE TABLE usage (
    user_id TEXT NOT NULL,
    action  TEXT NOT NULL,   -- 'chat' | 'predict_price' | 'search_knowledge'
    credits INTEGER NOT NULL,
    ts      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def main():
    bunch = fetch_openml("house_prices", as_frame=True, parser="auto")
    X = bunch.data
    y = bunch.target
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    df = X.copy()
    df["SalePrice"] = y
    df = df.drop(columns=[c for c in DROP if c in df.columns])

    conn = sqlite3.connect(DB)
    df.to_sql("houses", conn, if_exists="replace", index=False)
    conn.executescript("DROP TABLE IF EXISTS usage; DROP TABLE IF EXISTS users;")
    conn.executescript(USERS_SCHEMA + USAGE_SCHEMA)
    conn.commit()
    conn.close()

    print(f"Wrote {DB}")
    print(f"  houses: {df.shape[0]} rows x {df.shape[1]} columns")
    print("  users:  empty (schema only)")
    print("  usage:  empty (schema only)")


if __name__ == "__main__":
    main()
