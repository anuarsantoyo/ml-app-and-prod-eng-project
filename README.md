# House Hunt Agent

A full-stack AI-agent project: a FastAPI service, a standalone MCP server, a
self-trained ML model, and RAG — fronted by a Streamlit client. Every piece is
built from scratch; the repo doubles as a cheatsheet for the techniques.

A user signs up with just an email, receives an API key (shown once), and chats
with a homebuying agent. The agent decides on its own whether to ask for more
info, call the price model, or search the knowledge base.

## Architecture

    client/   Streamlit UI — signup/login, chat, credits display + top-up
    server/   FastAPI (auth, rate limit, credits) + agent loop + MCP server
    ml/       model training + serving (joblib)
    rag/      document ingestion, Chroma index, semantic search + cache
    data/     the Ames housing dataset as SQLite

The agent loop (`server/agent.py`) is an MCP client: it connects to the
standalone MCP server (`server/mcp_server.py`) over streamable HTTP, lists its
tools, and lets the LLM call them — reason -> act -> observe, repeated.

## Setup

    cd ml-app-and-prod-eng-project
    /usr/bin/python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    cp .env.example .env        # then put your OpenRouter key in .env

`.env` is gitignored — never commit the key.

## Data preparation (run once, in order)

    .venv/bin/python data/make_db.py          # build data/ames.db (SQLite)
    .venv/bin/python ml/train_model.py        # train + save house_price_model.joblib
    .venv/bin/python -m rag.download_docs     # fetch the homebuying PDFs
    .venv/bin/python -m rag.build_index       # chunk + embed + index in Chroma

## Run (three processes, from the repo root)

    .venv/bin/python -m server.db                          # create the app DB
    .venv/bin/uvicorn server.mcp_server:app --port 8001    # MCP server (tools)
    .venv/bin/uvicorn server.app:app --port 8000           # FastAPI (auth + agent)
    .venv/bin/streamlit run client/app.py                  # the UI

Open http://127.0.0.1:8000/docs for the API, http://localhost:8501 for the app.

## API

    POST /v1/signup          {"email": "..."}              -> {"api_key": "sk-..."} (once)
    POST /v1/chat            {"message": "...", "history": [...]} -> {"reply", ...}
    GET  /v1/usage                                          -> credits + monthly usage
    POST /v1/credits/topup   {"amount": 50}                 -> new balance

All endpoints except `/v1/signup` require the `x-api-key` header. Gates applied
in order: 401 (auth) -> 429 (rate limit) -> 402 (monthly quota / credits).

## Credits

    chat             1 credit  (charged up front)
    predict_price    3 credits
    search_knowledge 2 credits (1 if served from the RAG cache)

New users get 100 credits. Top up from the sidebar.

## The model

`ml/train_model.py` loads the Ames data from SQLite, preprocesses (impute /
encode / scale, log target), and compares three models with hyperparameter
tuning; the best is saved with its metrics and feature list. At serving time,
`ml/predict.py` takes ~12 features a buyer would know and leaves the rest NaN —
the pipeline's own imputers fill them with the training median/mode.
