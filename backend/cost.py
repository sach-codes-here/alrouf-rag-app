from __future__ import annotations

from typing import Any

from .config import env


def cost_rates() -> dict[str, float]:
    return {
        "embedding_usd": float(env("COST_EMBEDDING_USD", "0.5") or "0.5"),
        "search_usd": float(env("COST_SEARCH_USD", "0.2") or "0.2"),
        "translation_usd": float(env("COST_TRANSLATION_USD", "0.05") or "0.05"),
        "merge_usd": float(env("COST_MERGE_USD", "0.01") or "0.01"),
        "overhead_usd": float(env("COST_OVERHEAD_USD", "0.01") or "0.01"),
    }


def compute_cost_usd(
    *,
    embedding_calls: int,
    variants: int,
    did_translate: bool,
    did_merge: bool,
) -> dict[str, Any]:
    rates = cost_rates()
    embedding_cost = rates["embedding_usd"] * float(embedding_calls)
    search_cost = rates["search_usd"] * float(variants)
    translation_cost = rates["translation_usd"] if did_translate else 0.0
    merge_cost = rates["merge_usd"] if did_merge else 0.0
    overhead_cost = rates["overhead_usd"]
    total = embedding_cost + search_cost + translation_cost + merge_cost + overhead_cost

    return {
        "currency": "USD",
        "rates": rates,
        "line_items": {
            "embedding": embedding_cost,
            "search": search_cost,
            "translation": translation_cost,
            "merge": merge_cost,
            "overhead": overhead_cost,
        },
        "total": total,
    }

