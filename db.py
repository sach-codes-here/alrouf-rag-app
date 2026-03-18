from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PostgresTarget:
    schema: str = "public"
    table: str = "pdf_pages"


def _quote_ident(name: str) -> str:
    # Minimal identifier quoting for schema/table names.
    return '"' + name.replace('"', '""') + '"'


def get_pg_dsn(explicit_dsn: str | None = None) -> str:
    dsn = explicit_dsn or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError(
            "Postgres DSN not provided. Set DATABASE_URL (or POSTGRES_DSN) "
            "or pass --pg-dsn."
        )
    return dsn


def connect(dsn: str):
    # psycopg v3
    import psycopg

    return psycopg.connect(dsn)


def ensure_table(conn, target: PostgresTarget) -> None:
    schema_q = _quote_ident(target.schema)
    table_q = _quote_ident(target.table)

    create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {schema_q};"

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {schema_q}.{table_q} (
      id TEXT PRIMARY KEY,
      file_name TEXT NOT NULL,
      file_extension TEXT NOT NULL,
      content TEXT NOT NULL,
      title TEXT NOT NULL,
      filepath TEXT NOT NULL,
      content_vector REAL[] NOT NULL DEFAULT '{{}}',
      title_vector REAL[] NOT NULL DEFAULT '{{}}',
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """

    with conn.cursor() as cur:
        cur.execute(create_schema_sql)
        cur.execute(create_table_sql)
    conn.commit()


def upsert_pages(
    conn,
    target: PostgresTarget,
    records: Sequence[Mapping],
) -> int:
    """
    Upsert extracted page records into Postgres.

    Expected record keys (Spark schema):
      - id, fileName, fileExtension, content, title, filepath, contentVector, titleVector
    """
    if not records:
        return 0

    schema_q = _quote_ident(target.schema)
    table_q = _quote_ident(target.table)

    sql = f"""
    INSERT INTO {schema_q}.{table_q} (
      id, file_name, file_extension, content, title, filepath, content_vector, title_vector
    )
    VALUES (
      %(id)s, %(file_name)s, %(file_extension)s, %(content)s, %(title)s, %(filepath)s,
      %(content_vector)s, %(title_vector)s
    )
    ON CONFLICT (id) DO UPDATE SET
      file_name = EXCLUDED.file_name,
      file_extension = EXCLUDED.file_extension,
      content = EXCLUDED.content,
      title = EXCLUDED.title,
      filepath = EXCLUDED.filepath,
      content_vector = EXCLUDED.content_vector,
      title_vector = EXCLUDED.title_vector;
    """

    def normalize(r: Mapping) -> dict:
        return {
            "id": r["id"],
            "file_name": r["fileName"],
            "file_extension": r["fileExtension"],
            "content": r["content"],
            "title": r["title"],
            "filepath": r["filepath"],
            "content_vector": list(r.get("contentVector") or []),
            "title_vector": list(r.get("titleVector") or []),
        }

    params = [normalize(r) for r in records]

    with conn.cursor() as cur:
        cur.executemany(sql, params)
        rowcount = cur.rowcount if cur.rowcount is not None else len(records)
    conn.commit()
    return rowcount


# ---------------------------------------------------------------------------
# OCR table helpers
# ---------------------------------------------------------------------------

def ensure_ocr_table(conn, target: PostgresTarget) -> None:
    schema_q = _quote_ident(target.schema)
    table_q = _quote_ident(target.table)

    create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {schema_q};"

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {schema_q}.{table_q} (
      id TEXT PRIMARY KEY,
      image_path TEXT NOT NULL,
      pdf_stem TEXT,
      page_num INT,
      text TEXT NOT NULL,
      markdown TEXT NOT NULL DEFAULT '',
      raw_ocr_json JSONB,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """

    # Best-effort migration: add new columns if table already exists
    # but was created with the old schema.
    alter_sqls = [
        f"ALTER TABLE {schema_q}.{table_q} ADD COLUMN IF NOT EXISTS markdown TEXT NOT NULL DEFAULT '';",
        f"ALTER TABLE {schema_q}.{table_q} ADD COLUMN IF NOT EXISTS raw_ocr_json JSONB;",
    ]

    with conn.cursor() as cur:
        cur.execute(create_schema_sql)
        cur.execute(create_table_sql)
        for sql in alter_sqls:
            cur.execute(sql)
    conn.commit()


def upsert_ocr_pages(
    conn,
    target: PostgresTarget,
    records: Sequence[Mapping],
) -> int:
    """Upsert OCR page records into Postgres.

    Expected record keys:
      id, image_path, pdf_stem, page_num, text
      Optional: markdown, raw_ocr_json
    """
    if not records:
        return 0

    schema_q = _quote_ident(target.schema)
    table_q = _quote_ident(target.table)

    sql = f"""
    INSERT INTO {schema_q}.{table_q} (
      id, image_path, pdf_stem, page_num, text, markdown, raw_ocr_json
    )
    VALUES (
      %(id)s, %(image_path)s, %(pdf_stem)s, %(page_num)s, %(text)s,
      %(markdown)s, %(raw_ocr_json)s
    )
    ON CONFLICT (id) DO UPDATE SET
      image_path = EXCLUDED.image_path,
      pdf_stem = EXCLUDED.pdf_stem,
      page_num = EXCLUDED.page_num,
      text = EXCLUDED.text,
      markdown = EXCLUDED.markdown,
      raw_ocr_json = EXCLUDED.raw_ocr_json;
    """

    import json as _json

    def _normalize(r: Mapping) -> dict:
        ocr_json = r.get("raw_ocr_json")
        if ocr_json is not None and not isinstance(ocr_json, str):
            ocr_json = _json.dumps(ocr_json, ensure_ascii=False)
        return {
            "id": r["id"],
            "image_path": r["image_path"],
            "pdf_stem": r.get("pdf_stem"),
            "page_num": r.get("page_num"),
            "text": r.get("text", ""),
            "markdown": r.get("markdown", ""),
            "raw_ocr_json": ocr_json,
        }

    params = [_normalize(r) for r in records]

    with conn.cursor() as cur:
        cur.executemany(sql, params)
        rowcount = cur.rowcount if cur.rowcount is not None else len(records)
    conn.commit()
    return rowcount


# ---------------------------------------------------------------------------
# Chunk table helpers (Markdown chunks)
# ---------------------------------------------------------------------------


def ensure_chunks_table(conn, target: PostgresTarget) -> None:
    schema_q = _quote_ident(target.schema)
    table_q = _quote_ident(target.table)

    create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {schema_q};"

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {schema_q}.{table_q} (
      chunk_id TEXT PRIMARY KEY,
      pdf_path TEXT NOT NULL,
      citation_path TEXT NOT NULL,
      page_number INT,
      content TEXT NOT NULL,
      content_vector INTEGER[] NOT NULL DEFAULT '{{}}',
      title TEXT NOT NULL DEFAULT '',
      title_vector INTEGER[] NOT NULL DEFAULT '{{}}',
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """

    with conn.cursor() as cur:
        cur.execute(create_schema_sql)
        cur.execute(create_table_sql)
    conn.commit()


def upsert_chunks(
    conn,
    target: PostgresTarget,
    records: Sequence[Mapping],
) -> int:
    if not records:
        return 0

    schema_q = _quote_ident(target.schema)
    table_q = _quote_ident(target.table)

    sql = f"""
    INSERT INTO {schema_q}.{table_q} (
      chunk_id, pdf_path, citation_path, page_number, content,
      content_vector, title, title_vector
    )
    VALUES (
      %(chunk_id)s, %(pdf_path)s, %(citation_path)s, %(page_number)s, %(content)s,
      %(content_vector)s, %(title)s, %(title_vector)s
    )
    ON CONFLICT (chunk_id) DO UPDATE SET
      pdf_path = EXCLUDED.pdf_path,
      citation_path = EXCLUDED.citation_path,
      page_number = EXCLUDED.page_number,
      content = EXCLUDED.content,
      content_vector = EXCLUDED.content_vector,
      title = EXCLUDED.title,
      title_vector = EXCLUDED.title_vector;
    """

    def _normalize(r: Mapping) -> dict:
        return {
            "chunk_id": r["chunk_id"],
            "pdf_path": r["pdf_path"],
            "citation_path": r["citationPath"],
            "page_number": r.get("page_number"),
            "content": r["content"],
            "content_vector": list(r.get("contentVector") or []),
            "title": r.get("title", ""),
            "title_vector": list(r.get("titleVector") or []),
        }

    params = [_normalize(r) for r in records]

    with conn.cursor() as cur:
        cur.executemany(sql, params)
        rowcount = cur.rowcount if cur.rowcount is not None else len(records)
    conn.commit()
    return rowcount

