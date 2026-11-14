"""Access the provided SQLite database (data/app.db).

The database (houses, users, usage) is given to students ready-made;
the server only connects to it and never creates the schema.
"""
import hashlib
import os
import secrets
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "data", "app.db")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def new_api_key() -> str:
    return "sk-" + secrets.token_hex(24)


def connect() -> sqlite3.Connection:
    if not os.path.exists(DB):
        raise RuntimeError(f"Database not found at {DB} — it should be provided.")
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn
