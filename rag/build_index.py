"""Chunk the extracted docs and index them in Chroma.

Run once (after download_docs.py):  .venv/bin/python rag/build_index.py
"""
from pathlib import Path

import chromadb

from rag.embed import embed

CHUNK_SIZE = 600   # words per chunk
CHUNK_OVERLAP = 100
DOCS_TXT = Path(__file__).resolve().parent / "docs" / "txt"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION = "housing_knowledge"
BATCH = 64


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        if len(chunk.strip()) >= 50:
            chunks.append(chunk)
    return chunks


def build():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    docs, metas, ids = [], [], []
    for txt in sorted(DOCS_TXT.glob("*.txt")):
        for j, chunk in enumerate(chunk_text(txt.read_text())):
            docs.append(chunk)
            metas.append({"source": txt.stem})
            ids.append(f"{txt.stem}-{j}")

    print(f"indexing {len(docs)} chunks into '{COLLECTION}'")
    for i in range(0, len(docs), BATCH):
        embs = embed(docs[i:i + BATCH])
        coll.add(
            ids=ids[i:i + BATCH],
            documents=docs[i:i + BATCH],
            embeddings=embs,
            metadatas=metas[i:i + BATCH],
        )
        print(f"  {min(i + BATCH, len(docs))}/{len(docs)}")
    print("done")


if __name__ == "__main__":
    build()
