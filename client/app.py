"""Streamlit client for the House Hunt Agent.

Run (from the repo root):

    .venv/bin/streamlit run client/app.py

Talks to the FastAPI server (default http://127.0.0.1:8000).
"""
import os

import httpx
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="House Hunt Agent", layout="wide")

for key, default in [("api_key", None), ("email", None), ("history", [])]:
    if key not in st.session_state:
        st.session_state[key] = default


def headers():
    return {"x-api-key": st.session_state.api_key}


# ---------------------------------------------------------------------------
# Sidebar — account + credits
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("House Hunt")

    if st.session_state.api_key is None:
        mode = st.radio("Account", ["Sign up", "Log in"])
        if mode == "Sign up":
            email = st.text_input("Email")
            if st.button("Sign up"):
                if email:
                    r = httpx.post(f"{API}/v1/signup", json={"email": email}, timeout=30)
                    if r.status_code == 200:
                        d = r.json()
                        st.session_state.email = email
                        st.session_state.api_key = d["api_key"]
                        st.success("Copy your API key now — it is shown only once.")
                        st.code(d["api_key"])
                    else:
                        st.error(r.json().get("detail", r.text))
                else:
                    st.warning("Enter your email.")
        else:
            key = st.text_input("API key", type="password")
            if st.button("Log in"):
                r = httpx.get(f"{API}/v1/usage", headers={"x-api-key": key}, timeout=30)
                if r.status_code == 200:
                    st.session_state.api_key = key
                    st.session_state.email = r.json().get("user_id")
                    st.rerun()
                else:
                    st.error("Invalid API key.")
    else:
        st.success(f"Logged in: {st.session_state.email}")
        try:
            u = httpx.get(f"{API}/v1/usage", headers=headers(), timeout=30).json()
            st.metric("Credits", u.get("credits"))
            st.metric("Used this month", u.get("monthly_used"))
            st.progress(min(1.0, u.get("monthly_used", 0) / max(1, u.get("monthly_allowance", 1))))
        except Exception as e:
            st.warning(f"Server offline: {e}")
        with st.form("topup"):
            amount = st.number_input("Top up credits", min_value=1, value=50)
            if st.form_submit_button("Top up"):
                httpx.post(f"{API}/v1/credits/topup", json={"amount": int(amount)},
                           headers=headers(), timeout=30)
                st.rerun()
        if st.button("Log out"):
            st.session_state.api_key = None
            st.session_state.email = None
            st.session_state.history = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main — chat
# ---------------------------------------------------------------------------
st.title("House Hunting Assistant")
st.caption("Ask whether a house is fairly priced, or for homebuying advice.")

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if prompt := st.chat_input("Ask about a house..."):
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.api_key is None:
        st.warning("Sign up or log in first (sidebar).")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                r = httpx.post(
                    f"{API}/v1/chat",
                    json={"message": prompt, "history": st.session_state.history},
                    headers=headers(),
                    timeout=180,
                )
            if r.status_code == 200:
                d = r.json()
                reply = d["reply"]
                st.write(reply)
                st.caption(f"spent {d['credits_spent']} credit(s) — "
                           f"{d['credits_remaining']} remaining")
                st.session_state.history.append({"role": "user", "content": prompt})
                st.session_state.history.append({"role": "assistant", "content": reply})
            else:
                try:
                    err = r.json().get("detail", r.text)
                except Exception:
                    err = r.text
                st.error(f"{r.status_code}: {err}")
