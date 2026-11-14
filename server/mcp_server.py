"""Standalone MCP server exposing the agent's tools.

Run (from the repo root):

    .venv/bin/uvicorn server.mcp_server:app --host 127.0.0.1 --port 8001

The server speaks the streamable-HTTP MCP transport at /mcp — the same
pattern the students used in the basic course.
"""
from mcp.server import MCPServer

from ml.predict import predict_price as _predict_price
from rag.search import search_knowledge as _search_knowledge

mcp = MCPServer("house-hunt")


@mcp.tool()
def predict_price(
    bedrooms: int,
    bathrooms: float,
    sqft: int,
    year_built: int,
    lot_size: int,
    neighborhood: str,
    condition: int,
    garage: int,
    basement_sqft: int,
    central_air: bool,
    house_style: str,
    fireplaces: int,
) -> dict:
    """Predict the fair market price of a house (Ames, Iowa model).

    Use this when the user asks what a house is worth or whether a listing
    price is fair. You MUST have all twelve inputs before calling; if any is
    missing, ask the user for it (do NOT guess). Ask for all missing inputs
    in one combined question.

    bedrooms: number of bedrooms above ground
    bathrooms: total bathrooms, e.g. 2.5 means 2 full + 1 half
    sqft: above-ground living area in square feet
    year_built: year the house was built
    lot_size: lot area in square feet
    neighborhood: an Ames, Iowa neighborhood name; if the tool rejects it,
        read the error and ask the user to pick from the valid list
    condition: overall condition/quality, 1 (poor) to 10 (excellent)
    garage: number of cars the garage fits (0 if none)
    basement_sqft: total basement area in square feet (0 if none)
    central_air: true if the house has central air conditioning
    house_style: e.g. '1-story', '2-story', 'split'
    fireplaces: number of fireplaces
    """
    return _predict_price(
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        sqft=sqft,
        year_built=year_built,
        lot_size=lot_size,
        neighborhood=neighborhood,
        condition=condition,
        garage=garage,
        basement_sqft=basement_sqft,
        central_air=central_air,
        house_style=house_style,
        fireplaces=fireplaces,
    )


@mcp.tool()
def search_knowledge(query: str) -> dict:
    """Search the housing and tax knowledge base for authoritative advice.

    Use this for general questions about homebuying, mortgages, taxes,
    inspections, closing costs, etc. — anything that is NOT a price
    prediction. Returns the most relevant passages with their source
    documents. Caches repeated or very similar queries.
    """
    return _search_knowledge(query)


app = mcp.streamable_http_app(streamable_http_path="/mcp", host="127.0.0.1")
