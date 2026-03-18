from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import env, get_paths
from .schemas import SearchHit


def ms() -> float:
    return time.perf_counter() * 1000.0


def tokenize(text: str) -> list[str]:
    """
    Unicode-aware tokenizer for English + Arabic.
    Keeps letters/digits/underscore as tokens (\\w is Unicode by default in Python 3).
    """
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def load_docstore(docstore_path: Path) -> dict[int, dict[str, Any]]:
    docs: dict[int, dict[str, Any]] = {}
    with docstore_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs[int(obj["faiss_id"])] = obj
    return docs


class HybridIndex:
    def __init__(self) -> None:
        self.loaded = False
        self.docs: dict[int, dict[str, Any]] = {}
        self.faiss_index = None
        self.bm25 = None
        self.bm25_ids: list[int] = []
        self.alpha: float = 0.6
        self.embedder = None

    def load(self) -> None:
        import faiss
        from rank_bm25 import BM25Okapi
        from fastembed import TextEmbedding

        paths = get_paths()
        if not paths.faiss_index.exists() or not paths.docstore.exists() or not paths.bm25.exists():
            raise RuntimeError(
                "Index files not found. Run `python data-pipeline/05-push-to-index.py --build` first."
            )

        self.alpha = float(env("HYBRID_ALPHA", "0.6") or "0.6")
        self.docs = load_docstore(paths.docstore)
        self.faiss_index = faiss.read_index(str(paths.faiss_index))

        with paths.bm25.open("rb") as f:
            pack = pickle.load(f)
        self.bm25 = pack["bm25"]
        self.bm25_ids = list(pack["faiss_ids"])
        if not isinstance(self.bm25, BM25Okapi):
            pass

        model_name = env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5") or "BAAI/bge-small-en-v1.5"
        try:
            self.embedder = TextEmbedding(model_name=model_name)
        except ValueError:
            self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        self.loaded = True

    def search(self, query: str, *, top_k: int, alpha: float | None) -> tuple[list[SearchHit], dict[str, float]]:
        if not self.loaded:
            self.load()
        assert self.faiss_index is not None
        assert self.bm25 is not None
        assert self.embedder is not None

        a = self.alpha if alpha is None else float(alpha)

        t0 = ms()
        q_vec = next(self.embedder.embed([query])).tolist()
        t_embed = ms() - t0
        q_arr = np.asarray(q_vec, dtype=np.float32)
        q_arr = q_arr / (np.linalg.norm(q_arr) or 1.0)
        q = q_arr.reshape(1, -1).astype(np.float32)

        t0 = ms()
        scores_v, ids_v = self.faiss_index.search(q, top_k)
        t_faiss = ms() - t0
        vec_hits = {int(i): float(s) for i, s in zip(ids_v[0], scores_v[0]) if int(i) != -1}

        t0 = ms()
        q_tokens = tokenize(query)
        bm_scores = self.bm25.get_scores(q_tokens)
        t_bm25 = ms() - t0
        top_bm = np.argsort(bm_scores)[::-1][:top_k]
        bm_hits = {int(self.bm25_ids[i]): float(bm_scores[i]) for i in top_bm}

        def norm_map(m: dict[int, float]) -> dict[int, float]:
            if not m:
                return {}
            vals = np.asarray(list(m.values()), dtype=np.float32)
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 1e-9:
                return {k: 1.0 for k in m}
            return {k: (v - lo) / (hi - lo) for k, v in m.items()}

        vec_n = norm_map(vec_hits)
        bm_n = norm_map(bm_hits)

        all_ids = set(vec_n) | set(bm_n)
        combined: list[tuple[float, int, float, float]] = []
        for i in all_ids:
            combined_score = a * vec_n.get(i, 0.0) + (1.0 - a) * bm_n.get(i, 0.0)
            combined.append((combined_score, i, vec_hits.get(i, 0.0), bm_hits.get(i, 0.0)))
        combined.sort(reverse=True)

        hits: list[SearchHit] = []
        t0 = ms()
        for s, i, sv, sb in combined[:top_k]:
            doc = self.docs.get(i)
            if not doc:
                continue
            hits.append(
                SearchHit(
                    score=float(s),
                    vec_score=float(sv),
                    bm25_score=float(sb),
                    chunk_id=str(doc.get("chunk_id", "")),
                    title=str(doc.get("title", "")),
                    content=str(doc.get("content", "")),
                    citationPath=str(doc.get("citationPath", "")),
                    pdf_path=doc.get("pdf_path"),
                    page_number=doc.get("page_number"),
                )
            )
        t_materialize = ms() - t0

        return hits, {
            "embed_ms": t_embed,
            "faiss_ms": t_faiss,
            "bm25_ms": t_bm25,
            "materialize_ms": t_materialize,
        }

