# House Hunt Agent — Project Instructions

## 1. What you're building

A full-stack AI agent that helps a homebuyer decide whether a house is fairly priced. A user signs up with just an email, receives an API key, and chats with an agent. The agent decides on its own whether it needs more information, should predict a price, or should look something up.

One user turn looks like this:

    user:  "I found a 3-bed, 2-bath, 1,820 sqft house built in 1995, listed at $250k."
    agent: (missing neighborhood, condition, lot size...) -> asks for them in one question
    user:  "OldTown neighborhood, condition 7/10, 7,000 sqft lot, 2-car garage,
            800 sqft basement, central air, 2-story, 1 fireplace."
    agent: predict_price(...) -> $170,490 -> "At $250k it's overpriced; the model
            values it around $170k."

## 2. What you'll practice

Each piece is a skill from the course, assembled into one product:

- Admission control — API-key auth, per-caller rate limiting, credits + top-up
- Secrets — keep the OpenRouter key in .env, never commit it
- The ML pipeline — load data from SQLite, preprocess, train, save with joblib
- Model serving — load the joblib once, predict from a handful of user inputs
- MCP — expose your tools through a standalone MCP server
- RAG — chunk documents, embed, index, retrieve, cache
- The agent loop — an LLM that reasons, calls tools, and observes
- Serving — FastAPI with the security gates applied via Depends
- The client — a Streamlit UI

## 3. The requirements

### 3.1 The ML model (predict_price)

Train a house-price model on the Ames dataset you were given as SQLite.

- Load the SQLite file into a pandas DataFrame — practise getting the data with
  `pd.read_sql_query`; don't hardcode a CSV.
- Build one sklearn Pipeline: impute missing values, one-hot encode the categorical
  columns, scale the numeric columns, and log-transform the target (prices are
  right-skewed).
- Train a linear regression on the log of the price. This is the model to use.
  (Comparing other models or doing hyperparameter tuning is optional — not the focus.)
- Save the whole pipeline (preprocessing + model) to `house_price_model.joblib`.

Serving notes (you'll need these for the tool later):

- The model expects all ~75 columns, but a buyer only gives you ~12: bedrooms,
  bathrooms, sqft, year built, lot size, neighborhood, condition (1-10), garage
  spaces, basement sqft, central air, house style, fireplaces.
- For the columns the buyer didn't give, leave them as NaN — the pipeline's own
  imputers fill them with the training median/mode.
- The neighborhood must be one of the 25 Ames neighborhoods; reject anything else
  with a clear message listing the valid ones.

### 3.2 RAG — the knowledge base (search_knowledge)

Build a retrieval system over the homeownership/tax documents you were given (three IRS publications: Pub 530, Pub 936, Pub 523).

- Extract the text from the PDFs.
- Chunk the text into 600-word chunks with 100-word overlap.
- Embed the chunks with a local embedding model: fastembed's `BAAI/bge-small-en-v1.5`
  (384-dim, no API key, works offline). Don't use a paid embedding API.
- Store the embeddings in a vector store (Chroma, as in class).
- `search_knowledge(query)`: embed the query, retrieve the top few most similar chunks
  (cosine similarity), and return them together with their source document.
- Add a cache: if a very similar query was answered before, return the cached result
  instead of recomputing — a semantic cache (compare the query embedding to cached ones
  with cosine similarity and a threshold).

(Why local embeddings: free, no key, and identical for every student.)

### 3.3 The MCP server (your tools)

Build a standalone MCP server that exposes your two tools over the streamable-HTTP transport (same pattern as the MCP sprint in class):

- `predict_price(...)` — loads your saved model and returns the predicted price (also
  return the model's RMSE so the agent can be honest about uncertainty).
- `search_knowledge(query)` — calls your RAG search and returns the relevant passages.

Requirements:

- Use the mcp SDK's `MCPServer` and the `@mcp.tool` decorator.
- Give each tool a clear docstring — the LLM reads it to decide when and how to call the
  tool, and the parameter types define the required inputs.
- `predict_price` must declare all 12 buyer inputs as required parameters — that's what
  makes the agent ask for the ones that are missing.
- Serve it on its own port with uvicorn (`streamable_http_app`).

### 3.4 The agent loop

This is the heart of the project: an LLM that reasons, calls a tool, observes the result, and repeats until it can answer.

- The agent has exactly two tools: `predict_price` (calls your trained model) and
  `search_knowledge` (calls your RAG). No other tools.
- Use OpenRouter for the LLM (chat model set in .env, e.g. `google/gemini-2.5-flash`).
- Act as an MCP client: connect to your MCP server over streamable HTTP, initialize,
  list its tools, and convert them into the tool schemas you pass to the LLM.
- Give the LLM a system prompt that tells it how to behave:
  - price question -> gather every missing `predict_price` input (ask for all of them in
    ONE question, never guess a value) -> call `predict_price` -> compare to any listing
    price the user gave -> give a short verdict (fair / overpriced / underpriced).
  - general question -> call `search_knowledge` -> answer from what it returns, naming
    the source document.
  - keep answers short and in plain, friendly English; never invent prices or facts.
- The loop: send the conversation + tools to the LLM. If it returns a tool call, run it
  through the MCP client, put the result back into the conversation, and call the LLM
  again. When it returns plain text, that's your answer.
- Cap the number of tool-call rounds (e.g. 4) so it can never loop forever.

### 3.5 The FastAPI service (admission control + chat)

Build the FastAPI backend that fronts the agent.

Endpoints:

- `POST /v1/signup` — `{email}` -> creates a user, generates an API key, stores its
  hash, returns the key ONCE. Email only, no password.
- `POST /v1/chat` — `{message, history}` -> runs the agent, behind the gates below.
- `GET /v1/usage` — the caller's credits and what they've used.
- `POST /v1/credits/topup` — `{amount}` -> adds credits.
- `GET /health`

Security gates, applied in order with FastAPI `Depends` (as in Class 1):

- 401 authentication — hash the presented `x-api-key` and look it up; store only the
  SHA-256 hash, never the key itself.
- 429 rate limit — a sliding 60-second window, max N requests/minute per caller, kept
  in memory.
- 402 billing — deduct credits per action; if the balance is empty, refuse and say
  "top up".

Credit model:

- a new user starts with 100 credits
- chat message = 1, `predict_price` = 3, `search_knowledge` = 2
- keep a usage ledger: who, what action, how many credits, when

Wire the agent loop into `/v1/chat`: charge the chat credit up front, run the loop, and charge credits for each tool call the agent makes. Return the reply + credits spent + credits remaining.

Data: SQLite (a `users` table + a `usage` table), like Class 1.

### 3.6 The client (Streamlit)

A simple Streamlit app is enough — the point is that you can actually use the product, not that it looks fancy.

- Sign up (email -> get your API key) or log in with an existing key.
- A chat box to talk to the agent.
- Show your remaining credits and a way to top up.

That's it. Keep it plain.

## 4. What you're given

One SQLite database and three PDFs. Everything else you build.

- `data/app.db` — a single SQLite file with three tables:
  - `houses` — the Ames dataset, pre-loaded: 1,460 rows, ~75 feature columns plus the
    `SalePrice` target (identifier and mostly-empty columns already dropped).
  - `users` — empty, schema only: `user_id` (email, primary key), `key_hash`,
    `credits` (default 100), `created_at`. You insert a row on signup.
  - `usage` — empty, schema only: `user_id`, `action`, `credits`, `ts`. You insert a
    row for every billed action.
- `rag/docs/` — three IRS publications (Pub 530, Pub 936, Pub 523) as PDFs.

## 5. Architecture

    Streamlit (browser)
        |  HTTP + x-api-key
        v
    FastAPI  :8000     auth, rate limit, credits, /v1/chat
        |                 (the agent loop runs here)
        |  MCP client (streamable HTTP)
        v
    MCP server  :8001  predict_price, search_knowledge
        |                    |
        v                    v
    house_price_model.joblib   Chroma index (RAG)

One SQLite database holds everything relational: `houses` (training), `users` + `usage`
(auth/credits). The RAG vectors live in Chroma.

## 6. Suggested build order

Build in this order — each step is testable on its own before you move on:

1. The ML model (3.1) — first, because it fixes the 12 features everything else depends on.
2. The RAG search (3.2) — independent of the model; do it second (or in parallel).
3. The MCP server (3.3) — wrap 1 and 2 into your two tools.
4. The FastAPI service (3.5) — auth/rate-limit/credits first (the Class 1 pattern), then
   the agent loop (3.4) that connects to your MCP server.
5. The Streamlit client (3.6) — last, once the API works.

Dependency chain: ML + RAG -> MCP -> FastAPI (agent) -> Streamlit. The model is the
keystone — build it first and the rest falls into place.

Verify as you go: predict a known house, query a tax question, call each tool, curl the
endpoints, then click the UI.

## 7. Setup and run

Prerequisites: Python 3.11+, and an OpenRouter API key.

- Create a virtual environment and write your own `requirements.txt` as you install
  what you need (fastapi, uvicorn, mcp, openai, chromadb, fastembed, scikit-learn,
  pandas, joblib, pypdf, streamlit, ...).
- Put your key in `.env` (gitignored — never commit it):
  `OPENROUTER_API_KEY=...` and `OPENROUTER_MODEL=...` (a model that supports tool
  calling, e.g. `google/gemini-2.5-flash`).

One-time data prep (your own scripts):

    build the ML model   load houses from app.db -> preprocess -> train -> save .joblib
    build the RAG index  extract PDFs -> chunk (600/100) -> embed -> index in Chroma

Run (three terminals, from the project root):

    uvicorn your_mcp_server:app --port 8001        # the MCP server
    uvicorn your_app:app --port 8000               # the FastAPI service
    streamlit run your_client.py                   # the UI

(No database-creation step — the provided `app.db` already has the three tables.)

## 8. Acceptance criteria

Your project is done when every one of these works:

- Signup returns an API key shown once; signing up the same email again fails.
- A missing or wrong key -> 401.
- More than N requests in a minute -> 429.
- Empty credits -> 402; topping up lets you continue.
- Ask about a house with only some details -> the agent asks for the missing inputs
  (in one question).
- Give it all 12 details -> it predicts a price and compares it to the listing price
  with a verdict.
- Ask a tax/homebuying question -> it answers from the RAG docs and names the source.
- Credits decrement correctly: chat = 1, predict_price = 3, search_knowledge = 2.
- The usage endpoint shows your spend; a repeated RAG question hits the cache.
