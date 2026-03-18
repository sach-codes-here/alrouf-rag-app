from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)


class SearchHit(BaseModel):
    score: float
    vec_score: float
    bm25_score: float
    chunk_id: str
    title: str
    content: str
    citationPath: str
    pdf_path: str | None = None
    page_number: int | None = None


class SmartSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    rephrase: bool = True


class SmartSearchResponse(BaseModel):
    original_query: str
    detected_language: str
    translated_query: str | None = None
    query_variants: list[str]
    hits: list[SearchHit]
    message: str | None = None
    report: dict[str, Any] | None = None


class ReportSummary(BaseModel):
    report_id: str
    route: str
    query: str
    detected_language: str | None = None
    created_at_ms: int
    total_ms: float
    embedding_calls: int
    variants: int
    notes: list[str] = []

