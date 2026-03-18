from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException

from .config import env, get_paths, load_env
from .cost import compute_cost_usd
from .query_utils import build_query_variants, is_out_of_scope
from .reports import ReportStore
from .schemas import ReportSummary, SearchRequest, SmartSearchRequest, SmartSearchResponse
from .search import merge_hits
from .search_index import HybridIndex, ms
from .translation import contains_arabic, maybe_translate_ar_to_en


_index = HybridIndex()
_reports = ReportStore()

app = FastAPI(title="Alrouf RAG Search API", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    load_env()
    # Lazy load index; don't crash service if index missing.
    try:
        _index.load()
    except Exception:
        pass


@app.get("/health")
def health() -> dict[str, Any]:
    paths = get_paths()
    return {
        "loaded": _index.loaded,
        "index_dir": str(paths.index_dir),
        "faiss_index_exists": paths.faiss_index.exists(),
        "docstore_exists": paths.docstore.exists(),
        "bm25_exists": paths.bm25.exists(),
        "embedding_model": env("EMBEDDING_MODEL"),
    }


@app.post("/search")
def search(req: SearchRequest) -> Any:
    report_id = uuid.uuid4().hex
    t_start = ms()
    oos, reason = is_out_of_scope(req.query)
    if oos:
        report = {
            "report_id": report_id,
            "route": "/search",
            "query": req.query,
            "detected_language": "ar" if contains_arabic(req.query) else "en",
            "created_at_ms": int(time.time() * 1000),
            "total_ms": ms() - t_start,
            "embedding_calls": 0,
            "variants": 0,
            "notes": [reason],
        }
        report["cost_usd"] = compute_cost_usd(
            embedding_calls=0,
            variants=0,
            did_translate=False,
            did_merge=False,
        )
        _reports.push(report)
        return {
            "message": "Hello, I am happy to answer Alrouf product related questions only.",
            "report": report,
        }
    try:
        hits, timings = _index.search(req.query, top_k=req.top_k, alpha=req.alpha)
        report = {
            "report_id": report_id,
            "route": "/search",
            "query": req.query,
            "detected_language": "ar" if contains_arabic(req.query) else "en",
            "created_at_ms": int(time.time() * 1000),
            "total_ms": ms() - t_start,
            "embedding_calls": 1,
            "variants": 1,
            "timings_ms": timings,
            "notes": [],
        }
        report["cost_usd"] = compute_cost_usd(
            embedding_calls=1,
            variants=1,
            did_translate=False,
            did_merge=False,
        )
        _reports.push(report)
        return {"hits": hits, "report": report}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/smart-search", response_model=SmartSearchResponse)
def smart_search(req: SmartSearchRequest) -> SmartSearchResponse:
    """
    Arabic-friendly search:
    - Detect Arabic
    - Translate Arabic->English (local, optional via Argos)
    - Create query variants (\"rephrased\") and run hybrid retrieval
    - Merge results
    """
    report_id = uuid.uuid4().hex
    t_start = ms()
    original = req.query
    oos, reason = is_out_of_scope(original)
    if oos:
        report = {
            "report_id": report_id,
            "route": "/smart-search",
            "query": original,
            "detected_language": "ar" if contains_arabic(original) else "en",
            "created_at_ms": int(time.time() * 1000),
            "total_ms": ms() - t_start,
            "embedding_calls": 0,
            "variants": 0,
            "notes": [reason],
        }
        report["cost_usd"] = compute_cost_usd(
            embedding_calls=0,
            variants=0,
            did_translate=False,
            did_merge=False,
        )
        _reports.push(report)
        return SmartSearchResponse(
            original_query=original,
            detected_language="ar" if contains_arabic(original) else "en",
            translated_query=None,
            query_variants=[],
            hits=[],
            message="Hello, I am happy to answer Alrouf product related questions only.",
            report=report,
        )

    is_ar = contains_arabic(original)
    detected = "ar" if is_ar else "en"

    t0 = ms()
    translated = maybe_translate_ar_to_en(original) if is_ar else None
    t_translate = ms() - t0
    t0 = ms()
    variants = build_query_variants(original, translated, req.rephrase)
    t_variants = ms() - t0

    try:
        hit_lists = []
        per_variant_timings = []
        for v in variants:
            h, t = _index.search(v, top_k=req.top_k, alpha=req.alpha)
            hit_lists.append(h)
            per_variant_timings.append(t)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    t0 = ms()
    merged = merge_hits(hit_lists, req.top_k)
    t_merge = ms() - t0

    report = {
        "report_id": report_id,
        "route": "/smart-search",
        "query": original,
        "detected_language": detected,
        "created_at_ms": int(time.time() * 1000),
        "total_ms": ms() - t_start,
        "embedding_calls": len(variants),
        "variants": len(variants),
        "timings_ms": {
            "translate_ms": t_translate,
            "variant_build_ms": t_variants,
            "merge_ms": t_merge,
            "per_variant": per_variant_timings,
        },
        # Local compute; no external API billing.
        "cost": {"billing": "local_only", "embedding_calls": len(variants)},
        "notes": [],
    }
    report["cost_usd"] = compute_cost_usd(
        embedding_calls=len(variants),
        variants=len(variants),
        did_translate=bool(translated),
        did_merge=True,
    )
    _reports.push(report)
    return SmartSearchResponse(
        original_query=original,
        detected_language=detected,
        translated_query=translated,
        query_variants=variants,
        hits=merged,
        report=report,
    )


@app.post("/reload")
def reload_index() -> dict[str, Any]:
    _index.loaded = False
    _index.load()
    return {"ok": True, "loaded": True}


@app.get("/reports", response_model=list[ReportSummary])
def list_reports(limit: int = 50) -> list[ReportSummary]:
    return _reports.list(limit=limit)


@app.get("/reports/{report_id}")
def get_report(report_id: str) -> dict[str, Any]:
    r = _reports.get(report_id)
    if r is None:
        raise HTTPException(status_code=404, detail="report_not_found")
    return r

