#!/usr/bin/env python3
"""
Hybrid retrieval index (FAISS + BM25) built from Postgres chunks table.

Reads chunk rows from Postgres (configured via .env), creates index files if
missing, and supports:
  - build: export vectors + metadata to a local FAISS index + BM25 corpus
  - query: run hybrid search locally (vector + lexical)

Why both?
- FAISS handles semantic similarity using embeddings.
- BM25 handles exact keyword matching (useful for part numbers, codes).

Config
All configuration is read from `.env` (or environment variables).
See `.env.example`.

Usage
  # Build index
  python data-pipeline/05-push-to-index.py --build

  # Query index
  python data-pipeline/05-push-to-index.py --query "medium intensity type ab" --top-k 10
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: python-dotenv. Install requirements.txt") from e

    # Prefer `.env` (typical). If it doesn't exist, fall back to `.env.example`
    # so users can run without renaming.
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        return

    example_path = root / ".env.example"
    if example_path.exists():
        load_dotenv(dotenv_path=example_path)
        return

    # Fall back to default behavior (current working directory), if any.
    load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def _required_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _pg_dsn() -> str:
    dsn = _env("DATABASE_URL") or _env("POSTGRES_DSN")
    if dsn:
        return dsn
    host = _required_env("PGHOST")
    port = _env("PGPORT", "5432")
    db = _required_env("PGDATABASE")
    user = _required_env("PGUSER")
    pw = _required_env("PGPASSWORD")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def _stable_int_id(chunk_id: str) -> int:
    # Stable 63-bit integer id for FAISS IndexIDMap.
    import hashlib

    h = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def _tokenize(text: str) -> list[str]:
    # Basic tokenizer: lowercase + keep alnum words.
    return re.findall(r"[a-z0-9]+", (text or "").lower())


@dataclass(frozen=True)
class Paths:
    index_dir: Path
    faiss_index: Path
    docstore: Path
    bm25: Path


def _paths() -> Paths:
    index_dir = Path(_env("INDEX_DIR", "vectorstore") or "vectorstore")
    index_dir.mkdir(parents=True, exist_ok=True)
    return Paths(
        index_dir=index_dir,
        faiss_index=index_dir / (_env("FAISS_INDEX_FILE", "chunks.faiss") or "chunks.faiss"),
        docstore=index_dir / (_env("DOCSTORE_FILE", "chunks_docstore.jsonl") or "chunks_docstore.jsonl"),
        bm25=index_dir / (_env("BM25_FILE", "chunks_bm25.pkl") or "chunks_bm25.pkl"),
    )


def _read_chunks_from_postgres() -> list[dict[str, Any]]:
    import psycopg

    schema = _env("PG_SCHEMA", "public") or "public"
    table = _env("CHUNKS_TABLE", "ocr_markdown_chunks") or "ocr_markdown_chunks"
    dsn = _pg_dsn()

    sql = f"""
    SELECT
      chunk_id,
      pdf_path,
      citation_path,
      page_number,
      title,
      content,
      content_vector,
      title_vector
    FROM "{schema}"."{table}"
    WHERE
      length(trim(content)) > 0
      AND length(trim(title)) > 0
      AND cardinality(content_vector) > 0
      AND cardinality(title_vector) > 0
    ORDER BY created_at ASC
    """

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    out: list[dict[str, Any]] = []
    for (
        chunk_id,
        pdf_path,
        citation_path,
        page_number,
        title,
        content,
        content_vector,
        title_vector,
    ) in rows:
        out.append(
            {
                "chunk_id": chunk_id,
                "faiss_id": _stable_int_id(chunk_id),
                "pdf_path": pdf_path,
                "citationPath": citation_path,
                "page_number": page_number,
                "title": title,
                "content": content,
                "contentVector": list(content_vector or []),
                "titleVector": list(title_vector or []),
            }
        )
    return out


def _dequantize_int_vec(ints: list[int], scale: int) -> np.ndarray:
    v = np.asarray(ints, dtype=np.float32) / float(scale)
    # Re-normalize for cosine/IP search.
    norm = np.linalg.norm(v) or 1.0
    return (v / norm).astype(np.float32)


def build_index() -> None:
    from rank_bm25 import BM25Okapi
    import faiss

    paths = _paths()
    quant_scale = int(_env("VECTOR_QUANT_SCALE", "1000") or "1000")

    rows = _read_chunks_from_postgres()
    if not rows:
        raise RuntimeError(
            "No rows found to index. Ensure you ran chunking + embedding and that "
            "content/title + vectors are non-empty."
        )

    # Build vectors (we’ll use content vectors for semantic search).
    vectors = np.stack([_dequantize_int_vec(r["contentVector"], quant_scale) for r in rows]).astype(np.float32)
    ids = np.asarray([r["faiss_id"] for r in rows], dtype=np.int64)

    dim = vectors.shape[1]
    base = faiss.IndexFlatIP(dim)  # cosine similarity when vectors are normalized
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(vectors, ids)

    faiss.write_index(index, str(paths.faiss_index))

    # Docstore (jsonl) for retrieval
    with paths.docstore.open("w", encoding="utf-8") as f:
        for r in rows:
            doc = {
                "faiss_id": r["faiss_id"],
                "chunk_id": r["chunk_id"],
                "pdf_path": r["pdf_path"],
                "citationPath": r["citationPath"],
                "page_number": r["page_number"],
                "title": r["title"],
                "content": r["content"],
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # BM25 corpus (title + content)
    corpus_texts = [f"{r['title']}\n{r['content']}" for r in rows]
    tokenized = [_tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    with paths.bm25.open("wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "faiss_ids": [r["faiss_id"] for r in rows],
                "texts": corpus_texts,
            },
            f,
        )

    print(f"Built FAISS index: {paths.faiss_index}")
    print(f"Wrote docstore: {paths.docstore}")
    print(f"Wrote BM25: {paths.bm25}")
    print(f"Indexed chunks: {len(rows)} (dim={dim})")


def _load_docstore(docstore_path: Path) -> dict[int, dict[str, Any]]:
    docs: dict[int, dict[str, Any]] = {}
    with docstore_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs[int(obj["faiss_id"])] = obj
    return docs


def query_index(query: str, top_k: int | None = None) -> None:
    from rank_bm25 import BM25Okapi
    import faiss
    from fastembed import TextEmbedding

    paths = _paths()
    if not paths.faiss_index.exists() or not paths.docstore.exists() or not paths.bm25.exists():
        raise RuntimeError("Index files not found. Run with --build first.")

    quant_scale = int(_env("VECTOR_QUANT_SCALE", "1000") or "1000")
    alpha = float(_env("HYBRID_ALPHA", "0.6") or "0.6")
    k = int(top_k or int(_env("TOP_K", "10") or "10"))

    docs = _load_docstore(paths.docstore)
    index = faiss.read_index(str(paths.faiss_index))

    with paths.bm25.open("rb") as f:
        bm25_pack = pickle.load(f)
    bm25: BM25Okapi = bm25_pack["bm25"]
    bm25_ids: list[int] = bm25_pack["faiss_ids"]

    # Embed query using same embedding model (must match embedding step).
    model_name = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5") or "BAAI/bge-small-en-v1.5"
    embedder = TextEmbedding(model_name=model_name)
    q_vec = next(embedder.embed([query])).tolist()
    q_int = [int(round(x * quant_scale)) for x in (np.asarray(q_vec, dtype=np.float32) / (np.linalg.norm(q_vec) or 1.0))]
    q = _dequantize_int_vec(q_int, quant_scale).reshape(1, -1)

    # Vector search
    scores_v, ids_v = index.search(q.astype(np.float32), k)
    vec_hits = {int(i): float(s) for i, s in zip(ids_v[0], scores_v[0]) if int(i) != -1}

    # BM25 search
    q_tokens = _tokenize(query)
    bm_scores = bm25.get_scores(q_tokens)
    # take top k BM25
    top_bm = np.argsort(bm_scores)[::-1][:k]
    bm_hits = {int(bm25_ids[i]): float(bm_scores[i]) for i in top_bm}

    # Normalize scores and combine
    def _norm_map(m: dict[int, float]) -> dict[int, float]:
        if not m:
            return {}
        vals = np.asarray(list(m.values()), dtype=np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-9:
            return {k: 1.0 for k in m}
        return {k: (v - lo) / (hi - lo) for k, v in m.items()}

    vec_n = _norm_map(vec_hits)
    bm_n = _norm_map(bm_hits)

    all_ids = set(vec_n) | set(bm_n)
    combined = []
    for i in all_ids:
        combined_score = alpha * vec_n.get(i, 0.0) + (1.0 - alpha) * bm_n.get(i, 0.0)
        combined.append((combined_score, i, vec_hits.get(i, 0.0), bm_hits.get(i, 0.0)))
    combined.sort(reverse=True)

    print(f"Query: {query}")
    print(f"Hybrid alpha={alpha}, top_k={k}")
    for rank, (s, i, sv, sb) in enumerate(combined[:k], start=1):
        doc = docs.get(i, {})
        title = doc.get("title", "")
        cite = doc.get("citationPath", "")
        print(f"\n[{rank}] score={s:.3f}  vec={sv:.3f}  bm25={sb:.3f}")
        print(f"title: {title}")
        print(f"citationPath: {cite}")
        snippet = (doc.get("content", "") or "").strip().replace("\n", " ")
        print(f"content: {snippet[:240]}...")


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build FAISS+BM25 indexes from Postgres")
    parser.add_argument("--query", type=str, default=None, help="Run a query against local indexes")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    if args.build:
        build_index()
        return

    if args.query:
        query_index(args.query, top_k=args.top_k)
        return

    raise SystemExit("Nothing to do. Use --build or --query")


if __name__ == "__main__":
    main()
