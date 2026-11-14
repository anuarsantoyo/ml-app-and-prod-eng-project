"""Local sentence embeddings via fastembed.

No API key and no torch needed. Uses BAAI/bge-small-en-v1.5 (384-dim);
the model downloads once on first use and is cached on disk.
"""
from fastembed import TextEmbedding

_MODEL = None


def _get():
    global _MODEL
    if _MODEL is None:
        _MODEL = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _MODEL


def embed(texts):
    """str -> list[float]; list[str] -> list[list[float]]."""
    single = isinstance(texts, str)
    inp = [texts] if single else list(texts)
    model = _get()
    vecs = [v.tolist() for v in model.embed(inp)]
    return vecs[0] if single else vecs
