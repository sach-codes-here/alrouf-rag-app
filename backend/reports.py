from __future__ import annotations

import os
from collections import deque
from typing import Any

from .schemas import ReportSummary


class ReportStore:
    def __init__(self) -> None:
        self._reports: deque[dict[str, Any]] = deque(
            maxlen=int(os.getenv("REPORT_BUFFER", "200") or "200")
        )

    def push(self, report: dict[str, Any]) -> None:
        self._reports.append(report)

    def list(self, limit: int = 50) -> list[ReportSummary]:
        lim = max(1, min(int(limit), self._reports.maxlen or 200))
        reports = list(self._reports)[-lim:]
        out: list[ReportSummary] = []
        for r in reversed(reports):
            out.append(
                ReportSummary(
                    report_id=r.get("report_id", ""),
                    route=r.get("route", ""),
                    query=r.get("query", ""),
                    detected_language=r.get("detected_language"),
                    created_at_ms=int(r.get("created_at_ms", 0)),
                    total_ms=float(r.get("total_ms", 0.0)),
                    embedding_calls=int(r.get("embedding_calls", 0)),
                    variants=int(r.get("variants", 0)),
                    notes=list(r.get("notes", [])),
                )
            )
        return out

    def get(self, report_id: str) -> dict[str, Any] | None:
        for r in reversed(self._reports):
            if r.get("report_id") == report_id:
                return r
        return None

