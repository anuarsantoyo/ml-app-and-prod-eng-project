"""The agent loop: an LLM (OpenRouter) that calls MCP tools.

One turn = repeat "reason -> act -> observe" until the model answers:

    messages -> LLM -> (tool call? -> MCP tool -> result -> messages) -> answer

The MCP server is a separate process; this loop is the MCP client.
"""
import json
import os

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from server.llm import get_client, model_name

load_dotenv()

MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8001/mcp")
MAX_TOOL_ROUNDS = 4   # max LLM->tool round-trips in one turn
MAX_TOOL_CALLS = 6    # absolute cap on total tool invocations

COSTS = {"predict_price": 3, "search_knowledge": 2}
CACHE_HIT_COST = 1    # a cached RAG search is cheaper

SYSTEM_PROMPT = """You are a friendly homebuying assistant. You help a buyer decide whether a house is fairly priced.

You can call two tools:
- predict_price: estimates the fair market price of a house (trained on Ames, Iowa home sales). It needs all 12 inputs; if any are missing, ask the user for ALL of them in ONE combined question. Never guess a value.
- search_knowledge: looks up authoritative guidance on homebuying, mortgages, taxes, inspections, and closing costs.

How to behave:
- Price question: gather every missing predict_price input (ask once, together), call predict_price, then compare the result to any asking price the user gave. Give a short verdict — fair, overpriced, or underpriced — with the numbers.
- General question: call search_knowledge and answer from what it returns, naming the source document.
- Keep answers short and in plain, friendly English. Never invent prices or facts your tools did not give you.
"""


def _tools_schema(mcp_tools):
    schemas = []
    for t in mcp_tools:
        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": (t.description or "").strip(),
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        })
    return schemas


def _assistant_message(msg):
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (msg.tool_calls or [])
        ],
    }


def _result_dict(result):
    sc = getattr(result, "structured_content", None)
    return sc if isinstance(sc, dict) else None


def _result_text(result):
    sc = getattr(result, "structured_content", None)
    if sc is not None:
        return sc if isinstance(sc, str) else json.dumps(sc)
    parts = []
    for block in (getattr(result, "content", None) or []):
        parts.append(block.text if hasattr(block, "text") else str(block))
    return "\n".join(parts)


async def run_agent(user_message, history=None, charge=None):
    """Run one turn. Returns (reply_text, tool_calls_made).

    charge(name, result_dict) is an optional callback to bill credits for a
    tool call; it may raise HTTPException to abort the turn (e.g. 402).
    """
    client = get_client()
    model = model_name()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in (history or []):
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    tool_calls_made = []

    async with streamable_http_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            schemas = _tools_schema(tools)

            for _ in range(MAX_TOOL_ROUNDS):
                resp = client.chat.completions.create(
                    model=model, messages=messages, tools=schemas, tool_choice="auto",
                )
                msg = resp.choices[0].message
                if not msg.tool_calls:
                    return (msg.content or ""), tool_calls_made

                messages.append(_assistant_message(msg))
                for tc in msg.tool_calls:
                    if len(tool_calls_made) >= MAX_TOOL_CALLS:
                        break
                    args = json.loads(tc.function.arguments or "{}")
                    result = await session.call_tool(tc.function.name, args)
                    text = _result_text(result)
                    tool_calls_made.append(tc.function.name)
                    if charge:
                        charge(tc.function.name, _result_dict(result))
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})

            # out of rounds: force a final answer from what we have so far
            resp = client.chat.completions.create(model=model, messages=messages)
            return (resp.choices[0].message.content or ""), tool_calls_made
