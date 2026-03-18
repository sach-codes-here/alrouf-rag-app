from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env() -> None:
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        return
    example_path = root / ".env.example"
    if example_path.exists():
        load_dotenv(dotenv_path=example_path)
        return
    load_dotenv()


def env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Paths:
    index_dir: Path
    faiss_index: Path
    docstore: Path
    bm25: Path


def get_paths() -> Paths:
    root = Path(__file__).resolve().parent.parent
    index_dir = root / (env("INDEX_DIR", "vectorstore") or "vectorstore")
    return Paths(
        index_dir=index_dir,
        faiss_index=index_dir / (env("FAISS_INDEX_FILE", "chunks.faiss") or "chunks.faiss"),
        docstore=index_dir / (env("DOCSTORE_FILE", "chunks_docstore.jsonl") or "chunks_docstore.jsonl"),
        bm25=index_dir / (env("BM25_FILE", "chunks_bm25.pkl") or "chunks_bm25.pkl"),
    )

