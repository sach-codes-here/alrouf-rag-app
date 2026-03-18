from __future__ import annotations

from .schemas import SearchHit


def merge_hits(hit_lists: list[list[SearchHit]], top_k: int) -> list[SearchHit]:
    best: dict[str, SearchHit] = {}
    for hits in hit_lists:
        for h in hits:
            prev = best.get(h.chunk_id)
            if prev is None or h.score > prev.score:
                best[h.chunk_id] = h
    merged = sorted(best.values(), key=lambda h: h.score, reverse=True)
    return merged[:top_k]

