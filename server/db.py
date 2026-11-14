"""SQLite database for the app: users, API keys, credits, and the usage ledger.

Run once before starting the server:  python -m server.db
"""
import hashlib
import os
import secrets
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "app.db")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def new_api_key() -> str:
    return "sk-" + secrets.token_hex(24)


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


SCHEMA = """
DROP TABLE IF EXISTS usage;
DROP TABLE IF EXISTS users;

-- who can call: identified by email, authenticated by a hashed API key
CREATE TABLE users (
    user_id        TEXT PRIMARY KEY,       -- the user's email
    key_hash       TEXT NOT NULL,          -- SHA-256 of the API key, never the key
    monthly_tokens INTEGER NOT NULL DEFAULT 5000,  -- monthly allowance
    credits        INTEGER NOT NULL DEFAULT 100,    -- prepaid balance (top-up)
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- a ledger: every billed action records how many credits it cost
CREATE TABLE usage (
    user_id TEXT NOT NULL,
    action  TEXT NOT NULL,   -- 'chat' | 'predict_price' | 'search_knowledge'
    credits INTEGER NOT NULL,
    ts      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def init_db() -> None:
    conn = connect()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Created {DB}")
