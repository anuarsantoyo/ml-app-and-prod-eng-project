"""Semantic search over the housing docs, with a semantic cache.

search_knowledge(query) -> top-k relevant chunks as a single context string.
A semantic cache (SQLite) skips recompute when a very similar query was
already answered: the query embedding is compared to cached ones by cosine
similarity, and a hit returns the stored context directly.
"""
import json
import sqlite3
from pathlib import Path

import chromadb
import numpy as np

from rag.embed import embed

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
CACHE_DB = Path(__file__).resolve().parent.parent / "data" / "cache.db"
COLLECTION = "housing_knowledge"
TOP_K = 4
CACHE_THRESHOLD = 0.92  # cosine similarity needed for a cache hit

_client = None
_collection = None


def _coll():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_collection(COLLECTION)
    return _collection


def _cos(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / denom)


def _cache_lookup(qvec):
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("CREATE TABLE IF NOT EXISTS cache (query TEXT PRIMARY KEY, embedding TEXT, answer TEXT)")
    best, best_sim = None, 0.0
    for q, e, a in conn.execute("SELECT query, embedding, answer FROM cache"):
        sim = _cos(qvec, json.loads(e))
        if sim > best_sim:
            best_sim, best = sim, a
    conn.close()
    if best is not None and best_sim >= CACHE_THRESHOLD:
        return best, best_sim
    return None, best_sim


def _cache_store(query, qvec, context):
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("CREATE TABLE IF NOT EXISTS cache (query TEXT PRIMARY KEY, embedding TEXT, answer TEXT)")
    conn.execute(
        "INSERT OR REPLACE INTO cache (query, embedding, answer) VALUES (?,?,?)",
        (query, json.dumps(qvec), context),
    )
    conn.commit()
    conn.close()


def search_knowledge(query, use_cache=True):
    qvec = embed(query)

    if use_cache:
        cached, sim = _cache_lookup(qvec)
        if cached is not None:
            return {"context": cached, "source": "cache", "cache_hit": True}

    coll = _coll()
    res = coll.query(query_embeddings=[qvec], n_results=TOP_K, include=["documents", "metadatas"])
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    context = "\n\n".join(f"[{m.get('source')}] {d}" for d, m in zip(docs, metas))
    if use_cache:
        _cache_store(query, qvec, context)
    return {"context": context, "source": "chroma", "cache_hit": False, "n_chunks": len(docs)}
