#!/usr/bin/env python3
"""
Chunk OCR Markdown (stored in Postgres) into heading-based chunks and write them
into a new chunks table.

Chunking rule:
  - Split by Markdown headings starting with '## '.
  - Each chunk is from a heading until the next '## ' (or end of document).
  - If no '## ' exists, the whole markdown becomes one chunk.

Also builds a citationPath linking to the PDF with an anchor tag for page:
  file:///.../docs/My.pdf#page=3

Usage:
  python data-pipeline/024-chunk-ocr-markdown-to-postgres.py \
    --pg-dsn "postgresql://user:pass@localhost:5432/db" \
    --ocr-schema rag --ocr-table ocr_png_pages \
    --chunks-schema rag --chunks-table ocr_markdown_chunks
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    here = Path(__file__).resolve()
    repo_root = str(here.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


HEADING_RE = re.compile(r"^\s*##\s*(.*)\s*$")


def _split_markdown(md: str) -> list[tuple[str, str]]:
    """
    Returns list of (title, content) chunks.
    Title is heading text if present, else ''.
    Content is the markdown BETWEEN this heading and the next heading.
    (The heading line itself is NOT included in content.)
    """
    md = (md or "").strip()
    if not md:
        return []

    # Find all heading line positions (multiline)
    matches = list(re.finditer(r"(?m)^\s*##\s*.*$", md))
    if not matches:
        return []

    chunks: list[tuple[str, str]] = []

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        block = md[m.end():end].strip()

        heading_line = m.group(0)
        hm = HEADING_RE.match(heading_line)
        title = hm.group(1).strip() if hm else ""
        chunks.append((title, block))

    return chunks


def _derive_pdf_path_from_image_path(image_path: str) -> str:
    """
    image_path like: docs/<stem>_pdf/page_001.png -> docs/<stem>.pdf
    Falls back to returning empty string if it can't infer.
    """
    p = Path(image_path)
    parent = p.parent.name
    if parent.endswith("_pdf"):
        stem = parent[: -len("_pdf")]
        pdf = p.parent.parent / f"{stem}.pdf"
        return str(pdf.resolve()) if pdf.exists() else str(pdf)
    return ""


def _citation_path(pdf_path: str, page_number: int | None) -> str:
    # Use a file URL with page anchor (commonly supported by PDF viewers).
    if not pdf_path:
        return ""
    suffix = f"#page={page_number}" if page_number else ""
    return f"file://{pdf_path}{suffix}"


def _chunk_id(pdf_path: str, page_number: int | None, title: str, content: str, idx: int) -> str:
    base = f"{pdf_path}::{page_number}::{idx}::{title}::{content[:200]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]


def main() -> None:
    _ensure_repo_root_on_path()

    parser = argparse.ArgumentParser(description="Chunk OCR markdown into a chunks table.")
    parser.add_argument("--pg-dsn", type=str, default=None)

    parser.add_argument("--ocr-schema", type=str, default="public")
    parser.add_argument("--ocr-table", type=str, default="ocr_png_pages")

    parser.add_argument("--chunks-schema", type=str, default="public")
    parser.add_argument("--chunks-table", type=str, default="ocr_markdown_chunks")

    parser.add_argument("--limit", type=int, default=None, help="Process only N OCR rows.")

    args = parser.parse_args()

    from db import (
        PostgresTarget,
        connect,
        ensure_chunks_table,
        ensure_ocr_table,
        get_pg_dsn,
        upsert_chunks,
    )

    dsn = get_pg_dsn(args.pg_dsn)
    ocr_target = PostgresTarget(schema=args.ocr_schema, table=args.ocr_table)
    chunks_target = PostgresTarget(schema=args.chunks_schema, table=args.chunks_table)

    with connect(dsn) as conn:
        # Ensure source/target exist (source table may already exist; this is safe).
        ensure_ocr_table(conn, ocr_target)
        ensure_chunks_table(conn, chunks_target)

        sql = f'SELECT id, image_path, page_num, markdown, created_at FROM "{ocr_target.schema}"."{ocr_target.table}" ORDER BY created_at ASC'
        if args.limit:
            sql += f" LIMIT {int(args.limit)}"

        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        out_records: list[dict] = []
        for (_id, image_path, page_num, markdown, _created_at) in rows:
            pdf_path = _derive_pdf_path_from_image_path(image_path)
            cite = _citation_path(pdf_path, page_num)

            chunks = _split_markdown(markdown or "")
            if not chunks:
                continue

            for idx, (title, content) in enumerate(chunks):
                out_records.append(
                    {
                        "chunk_id": _chunk_id(pdf_path, page_num, title, content, idx),
                        "pdf_path": pdf_path,
                        "citationPath": cite,
                        "page_number": page_num,
                        "content": content,
                        "contentVector": [],
                        "title": title or (Path(pdf_path).stem if pdf_path else ""),
                        "titleVector": [],
                    }
                )

        inserted = upsert_chunks(conn, chunks_target, out_records)
        print(f"Chunks upserted: {inserted} into {chunks_target.schema}.{chunks_target.table}")


if __name__ == "__main__":
    main()

