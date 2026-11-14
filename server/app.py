"""FastAPI service: admission control + the agent.

Gates a request passes through (cheapest and most fundamental first):

    POST /v1/chat:   401 auth -> 429 rate limit -> 402 credits -> agent loop
    GET  /v1/usage:  401 auth (your own usage only)
    POST /v1/signup: no key needed -> returns a one-time API key

Run (from the repo root):

    .venv/bin/python -m server.db          # once, to create the database
    .venv/bin/uvicorn server.app:app --host 127.0.0.1 --port 8000

Requires the MCP server to be running separately on port 8001.
"""
import sqlite3
import time
from collections import defaultdict

from fastapi import Body, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from server.agent import CACHE_HIT_COST, COSTS, run_agent
from server.db import connect, new_api_key, sha256

REQUESTS_PER_MINUTE = 10
CHAT_COST = 1

rate_hits = defaultdict(list)

app = FastAPI(title="House Hunt Agent")


class SignupRequest(BaseModel):
    email: str


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class TopUpRequest(BaseModel):
    amount: int


# ---------------------------------------------------------------------------
# GATE 1 — authentication (401)
# ---------------------------------------------------------------------------
def require_key(x_api_key: str = Header(default=None)) -> dict:
    if not x_api_key:
        raise HTTPException(401, "Missing API key")
    conn = connect()
    row = conn.execute(
        "SELECT user_id FROM users WHERE key_hash = ?", (sha256(x_api_key),)
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(401, "Unknown API key")
    return {"user_id": row["user_id"]}


# ---------------------------------------------------------------------------
# GATE 2 — rate limit (429), sliding 60-second window, per caller
# ---------------------------------------------------------------------------
def rate_limit(caller: dict = Depends(require_key)) -> dict:
    now = time.time()
    recent = [t for t in rate_hits[caller["user_id"]] if now - t < 60]
    if len(recent) >= REQUESTS_PER_MINUTE:
        raise HTTPException(429, f"Rate limit: {REQUESTS_PER_MINUTE} requests/min")
    recent.append(now)
    rate_hits[caller["user_id"]] = recent
    return caller


# ---------------------------------------------------------------------------
# Billing: monthly quota (402) then prepaid credits (402), atomic
# ---------------------------------------------------------------------------
def spend(conn, user_id: str, action: str, amount: int) -> None:
    used = conn.execute(
        "SELECT COALESCE(SUM(credits), 0) AS s FROM usage "
        "WHERE user_id = ? AND ts >= datetime('now', 'start of month')",
        (user_id,),
    ).fetchone()["s"]
    allowance = conn.execute(
        "SELECT monthly_tokens AS t FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()["t"]
    if used + amount > allowance:
        raise HTTPException(402, f"Monthly quota exceeded: {used}/{allowance} this month")

    cur = conn.execute(
        "UPDATE users SET credits = credits - ? WHERE user_id = ? AND credits >= ?",
        (amount, user_id, amount),
    )
    if cur.rowcount == 0:
        raise HTTPException(402, "Insufficient credits — please top up")
    conn.execute("INSERT INTO usage(user_id, action, credits) VALUES (?, ?, ?)",
                 (user_id, action, amount))
    conn.commit()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/signup")
def signup(body: SignupRequest):
    email = body.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(422, "A valid email is required")
    key = new_api_key()
    conn = connect()
    try:
        conn.execute("INSERT INTO users(user_id, key_hash) VALUES (?, ?)",
                     (email, sha256(key)))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(409, "Email already registered")
    conn.close()
    return {"email": email, "api_key": key,
            "note": "Store this key now — it is shown only once."}


@app.post("/v1/chat")
async def chat(caller: dict = Depends(rate_limit), body: ChatRequest = Body(...)):
    user_id = caller["user_id"]

    # charge the chat message first (402 if there is nothing left)
    conn = connect()
    try:
        spend(conn, user_id, "chat", CHAT_COST)
    finally:
        conn.close()

    tool_costs = []

    def charge(name: str, result_dict):
        cost = COSTS.get(name, 1)
        if name == "search_knowledge" and result_dict and result_dict.get("cache_hit"):
            cost = CACHE_HIT_COST
        tool_costs.append(cost)
        c = connect()
        try:
            spend(c, user_id, name, cost)
        finally:
            c.close()

    reply, _ = await run_agent(body.message, body.history, charge)

    conn = connect()
    credits = conn.execute(
        "SELECT credits AS b FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()["b"]
    conn.close()

    return {
        "reply": reply,
        "credits_spent": CHAT_COST + sum(tool_costs),
        "credits_remaining": credits,
    }


@app.get("/v1/usage")
def usage(caller: dict = Depends(require_key)):
    conn = connect()
    uid = caller["user_id"]
    used = conn.execute(
        "SELECT COALESCE(SUM(credits), 0) AS s FROM usage "
        "WHERE user_id = ? AND ts >= datetime('now', 'start of month')",
        (uid,),
    ).fetchone()["s"]
    allowance = conn.execute(
        "SELECT monthly_tokens AS t FROM users WHERE user_id = ?", (uid,)
    ).fetchone()["t"]
    credits = conn.execute(
        "SELECT credits AS b FROM users WHERE user_id = ?", (uid,)
    ).fetchone()["b"]
    conn.close()
    return {"user_id": uid, "monthly_used": used,
            "monthly_allowance": allowance, "credits": credits}


@app.post("/v1/credits/topup")
def topup(body: TopUpRequest, caller: dict = Depends(require_key)):
    conn = connect()
    conn.execute("UPDATE users SET credits = credits + ? WHERE user_id = ?",
                 (body.amount, caller["user_id"]))
    conn.commit()
    balance = conn.execute(
        "SELECT credits AS b FROM users WHERE user_id = ?", (caller["user_id"],)
    ).fetchone()["b"]
    conn.close()
    return {"user_id": caller["user_id"], "credits": balance}


# teaching helper: reset the demo state between runs
@app.post("/_reset")
def reset():
    rate_hits.clear()
    conn = connect()
    conn.execute("DELETE FROM usage")
    conn.execute("UPDATE users SET credits = 100")
    conn.commit()
    conn.close()
    return {"reset": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
