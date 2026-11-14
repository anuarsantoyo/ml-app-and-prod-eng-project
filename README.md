# House Hunting Agent

A student capstone for the ML App & Product Engineering course: an AI agent that
helps you judge whether a house is fairly priced.

The agent predicts a fair price from a trained ML model, grounds its advice in a
housing/tax knowledge base (RAG), and uses one geocoding API to fill in missing
location data.

## The core loop

    user:  "I found a 3-bed, 2-bath, 1,820 sqft house, built 1995, listed at $550k."
    agent: gathers the features the model needs (asking or geocoding)
           -> predict_price(...) -> ~$495k
           -> "At $550k it's ~11% over what the model values it at. For a 1995
               build, check the roof and HVAC."   (last line from RAG)

The key agent behavior: it knows the model's required inputs and asks when
something is missing, instead of guessing.

## Architecture

    client/   Streamlit app: sign up / log in (API key), chat, top-up credits, see usage
    server/   FastAPI: the agent, the MCP server, auth / rate-limit / credits (SQLite)
    ml/       house-price model: full pipeline (notebook) -> house_price_model.joblib
    rag/      housing/tax knowledge base: chunks + embeddings + hybrid search + re-rank
    data/     SQLite (users, api keys, credits, usage) + seed data

## The agent's tools (via MCP)

    1. predict_price(features)    -> the trained joblib model
    2. search_knowledge(query)    -> RAG over housing/tax docs
    3. geocode(address)           -> neighborhood / coordinates (Nominatim, free)

## Data

    ML training   Ames Housing dataset (house-price regression, ~79 features)
    RAG corpus    IRS Pub 530 + 936, HUD "Buying a Home", CFPB "Know Before You Owe"
    Live API      Nominatim geocoding (free, no key)

## Security (from the classes)

    API-key auth, per-caller rate limiting, token credits + usage tracking —
    all enforced with FastAPI Depends, backed by SQLite.

## Status

    In progress. See the classes repo for the techniques this project exercises:
    admission control, secrets, CI/CD, Docker, deploy, model serving, logging, UI.
