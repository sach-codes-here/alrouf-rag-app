#!/usr/bin/env python3
"""
Embed markdown chunks from Postgres and fill contentVector/titleVector.

This script reads rows from a chunks table where:
  - content is non-empty
  - title is non-empty
  - content_vector/title_vector are still empty

It generates embeddings using an open-source local model.

Default backend is ONNX-based `fastembed` to avoid PyTorch/NumPy ABI issues.
and writes integer vectors back into Postgres.

NOTE: Your chunks table stores vectors as INTEGER[].
Embedding models return float vectors; we quantize to integers by:
  - L2-normalize each vector
  - scale by --quant-scale (default 1000)
  - round to nearest int

Usage:
  python data-pipeline/04-embed-chunks.py \
    --pg-dsn "postgresql://user:pass@localhost:5432/db" \
    --schema rag --table ocr_markdown_chunks
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable


def _ensure_repo_root_on_path() -> None:
    here = Path(__file__).resolve()
    repo_root = str(here.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _l2_normalize(vec: list[float]) -> list[float]:
    denom = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / denom for x in vec]


def _quantize_to_ints(vec: list[float], scale: int) -> list[int]:
    v = _l2_normalize(vec)
    return [int(round(x * scale)) for x in v]


def _chunks(it: list, n: int) -> Iterable[list]:
    for i in range(0, len(it), n):
        yield it[i : i + n]


def main() -> None:
    _ensure_repo_root_on_path()

    parser = argparse.ArgumentParser(description="Embed chunks and update vectors in Postgres.")
    parser.add_argument("--pg-dsn", type=str, default=None)
    parser.add_argument("--schema", type=str, default="public")
    parser.add_argument("--table", type=str, default="ocr_markdown_chunks")
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Embedding model name (fastembed model by default).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N rows.")
    parser.add_argument(
        "--quant-scale",
        type=int,
        default=1000,
        help="Scale factor used when quantizing unit vectors to INTEGER[].",
    )

    args = parser.parse_args()

    from db import connect, get_pg_dsn

    dsn = get_pg_dsn(args.pg_dsn)

    # Lazy import so requirements aren't needed for non-embedding steps.
    # Prefer fastembed (ONNX) to avoid torch/numpy ABI issues.
    from fastembed import TextEmbedding

    embedder = TextEmbedding(model_name=args.model)

    schema_q = f'"{args.schema}"'
    table_q = f'"{args.table}"'

    select_sql = f"""
    SELECT chunk_id, title, content
    FROM {schema_q}.{table_q}
    WHERE
      length(trim(content)) > 0
      AND length(trim(title)) > 0
      AND (content_vector IS NULL OR cardinality(content_vector) = 0)
      AND (title_vector IS NULL OR cardinality(title_vector) = 0)
    ORDER BY created_at ASC
    """
    if args.limit:
        select_sql += f" LIMIT {int(args.limit)}"

    update_sql = f"""
    UPDATE {schema_q}.{table_q}
    SET content_vector = %(content_vector)s,
        title_vector = %(title_vector)s
    WHERE chunk_id = %(chunk_id)s
    """

    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(select_sql)
            rows = cur.fetchall()

        if not rows:
            print("No rows to embed (already embedded or empty).")
            return

        print(f"Embedding {len(rows)} chunk(s) using model: {args.model}")

        for batch in _chunks(rows, max(1, args.batch_size)):
            chunk_ids = [r[0] for r in batch]
            titles = [r[1] for r in batch]
            contents = [r[2] for r in batch]

            title_embs = [v.tolist() for v in embedder.embed(titles)]
            content_embs = [v.tolist() for v in embedder.embed(contents)]

            params = []
            for chunk_id, t_vec, c_vec in zip(chunk_ids, title_embs, content_embs):
                params.append(
                    {
                        "chunk_id": chunk_id,
                        "title_vector": _quantize_to_ints([float(x) for x in t_vec], args.quant_scale),
                        "content_vector": _quantize_to_ints([float(x) for x in c_vec], args.quant_scale),
                    }
                )

            with conn.cursor() as cur:
                cur.executemany(update_sql, params)
            conn.commit()
            print(f"Updated {len(batch)} chunk(s)")


if __name__ == "__main__":
    main()

