#!/usr/bin/env python3
"""
Layout-aware OCR using PaddleOCR: extract text + bounding boxes from PNGs,
reconstruct reading order into Markdown (with best-effort table detection),
and optionally upsert into Postgres.

Usage:
  python data-pipeline/023-ocr-png-to-markdown-paddle.py --input-dir docs --limit 2
  python data-pipeline/023-ocr-png-to-markdown-paddle.py --input-dir docs --write-postgres --pg-schema rag
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from statistics import median


def _ensure_repo_root_on_path() -> None:
    here = Path(__file__).resolve()
    repo_root = str(here.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _get_ocr_engine():
    from paddleocr import PaddleOCR

    return PaddleOCR(use_textline_orientation=True, lang="en")


def _resize_max_side(image_path: Path, max_side: int | None):
    """
    Optionally downscale an image so its longest side == max_side.
    Returns either the original path (as str) or a numpy array (BGR) for PaddleOCR.
    """
    if not max_side or max_side <= 0:
        return str(image_path)

    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return str(image_path)
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def _run_ocr(
    ocr_engine,
    image_input,
    *,
    fast: bool,
) -> list[dict]:
    """Run PaddleOCR on one image and return normalised box records.

    Each record: {"box": [[x,y],...], "text": str, "conf": float,
                  "cx": float, "cy": float, "y_min": float, "y_max": float}
    """
    predict_kwargs = {}
    if fast:
        # Disable optional (slower) pre-processing and orientation stages.
        predict_kwargs.update(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    results = ocr_engine.predict(image_input, **predict_kwargs)
    if not results:
        return []

    boxes: list[dict] = []
    for res in results:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        polys = res.get("rec_polys", [])

        for text, conf, poly in zip(texts, scores, polys):
            coords = poly.tolist() if hasattr(poly, "tolist") else list(poly)
            ys = [pt[1] for pt in coords]
            xs = [pt[0] for pt in coords]
            boxes.append({
                "box": coords,
                "text": text,
                "conf": float(conf),
                "cx": sum(xs) / len(xs),
                "cy": sum(ys) / len(ys),
                "y_min": min(ys),
                "y_max": max(ys),
                "x_min": min(xs),
                "x_max": max(xs),
            })
    return boxes


# ---------------------------------------------------------------------------
# Layout reconstruction → Markdown
# ---------------------------------------------------------------------------

def _group_into_lines(boxes: list[dict], overlap_ratio: float = 0.5) -> list[list[dict]]:
    """Group OCR boxes into visual lines by vertical overlap."""
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: (b["y_min"], b["x_min"]))
    lines: list[list[dict]] = [[sorted_boxes[0]]]

    for box in sorted_boxes[1:]:
        last_line = lines[-1]
        ref = last_line[0]
        ref_height = ref["y_max"] - ref["y_min"]
        if ref_height == 0:
            ref_height = 1

        overlap = min(box["y_max"], ref["y_max"]) - max(box["y_min"], ref["y_min"])
        if overlap / ref_height >= overlap_ratio:
            last_line.append(box)
        else:
            lines.append([box])

    for line in lines:
        line.sort(key=lambda b: b["x_min"])

    return lines


def _detect_table_region(lines: list[list[dict]], min_rows: int = 3, min_cols: int = 2) -> list[tuple[int, int]]:
    """Detect contiguous runs of lines that look like table rows.

    Heuristic: a table region is a sequence of >=min_rows consecutive lines
    where each line has the same number of cells (>=min_cols) and cell X
    centres are roughly aligned across rows.
    """
    if len(lines) < min_rows:
        return []

    regions: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        ncols = len(lines[i])
        if ncols < min_cols:
            i += 1
            continue

        j = i + 1
        while j < len(lines) and len(lines[j]) == ncols:
            j += 1

        if j - i >= min_rows:
            regions.append((i, j))
            i = j
        else:
            i += 1

    return regions


def _line_to_text(line: list[dict]) -> str:
    return "  ".join(b["text"] for b in line)


def _render_table_md(lines: list[list[dict]]) -> str:
    """Best-effort Markdown table from a block of equally-celled lines."""
    if not lines:
        return ""

    ncols = len(lines[0])
    rows = []
    for line in lines:
        cells = [b["text"].strip() for b in line]
        while len(cells) < ncols:
            cells.append("")
        rows.append(cells)

    md_lines = []
    header = rows[0]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)


def boxes_to_markdown(boxes: list[dict], min_conf: float = 0.5) -> str:
    """Convert raw OCR boxes into layout-aware Markdown."""
    filtered = [b for b in boxes if b["conf"] >= min_conf]
    if not filtered:
        return ""

    lines = _group_into_lines(filtered)
    table_regions = _detect_table_region(lines)

    table_line_indices: set[int] = set()
    for start, end in table_regions:
        table_line_indices.update(range(start, end))

    md_parts: list[str] = []
    i = 0
    while i < len(lines):
        if i in table_line_indices:
            region_start = i
            while i in table_line_indices:
                i += 1
            region_end = i
            md_parts.append("")
            md_parts.append(_render_table_md(lines[region_start:region_end]))
            md_parts.append("")
        else:
            text = _line_to_text(lines[i])

            # Heuristic heading detection: short, mostly uppercase or larger font
            line_height = max(
                (b["y_max"] - b["y_min"] for b in lines[i]), default=0
            )
            all_heights = [b["y_max"] - b["y_min"] for b in filtered]
            med_height = median(all_heights) if all_heights else 0

            if len(text) < 80 and line_height > med_height * 1.3:
                md_parts.append(f"\n## {text}\n")
            else:
                md_parts.append(text)
            i += 1

    return "\n".join(md_parts).strip()


# ---------------------------------------------------------------------------
# File discovery + metadata
# ---------------------------------------------------------------------------

def _deterministic_id(image_path: str) -> str:
    return hashlib.sha256(image_path.encode()).hexdigest()[:32]


def _infer_metadata(png_path: Path) -> tuple[str | None, int | None]:
    pdf_stem: str | None = None
    page_num: int | None = None

    folder_name = png_path.parent.name
    if folder_name.endswith("_pdf"):
        pdf_stem = folder_name[: -len("_pdf")]

    match = re.match(r"page_(\d+)", png_path.stem)
    if match:
        page_num = int(match.group(1))

    return pdf_stem, page_num


def discover_pngs(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_repo_root_on_path()

    parser = argparse.ArgumentParser(
        description="Layout-aware OCR (PaddleOCR) → Markdown, with optional Postgres upsert."
    )
    parser.add_argument("--input-dir", type=str, default="docs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-conf", type=float, default=0.5,
                        help="Drop OCR boxes below this confidence.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster OCR: disable doc orientation/unwarp/textline orientation stages.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1600,
        help="Downscale images so longest side <= N for speed (0 disables). Default: 1600",
    )
    parser.add_argument("--write-postgres", action="store_true")
    parser.add_argument("--pg-dsn", type=str, default=None)
    parser.add_argument("--pg-schema", type=str, default="public")
    parser.add_argument("--pg-table", type=str, default="ocr_png_pages")

    args = parser.parse_args()

    # --- Preflight Postgres connection (before OCR) ---
    pg_conn = None
    pg_target = None
    if args.write_postgres:
        from db import PostgresTarget, connect, ensure_ocr_table, get_pg_dsn

        dsn = get_pg_dsn(args.pg_dsn)
        pg_target = PostgresTarget(schema=args.pg_schema, table=args.pg_table)
        try:
            pg_conn = connect(dsn)
            ensure_ocr_table(pg_conn, pg_target)
            print(f"Postgres OK: ready to write into {pg_target.schema}.{pg_target.table}")
        except Exception:
            if pg_conn is not None:
                try:
                    pg_conn.close()
                except Exception:
                    pass
            raise

    # --- Resolve input dir ---
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        input_dir = Path("..") / args.input_dir
    if not input_dir.exists():
        print(f"ERROR: input directory not found: {args.input_dir}", file=sys.stderr)
        raise SystemExit(1)

    pngs = discover_pngs(input_dir)
    if not pngs:
        print(f"No *.png files found under {input_dir.resolve()}")
        return

    if args.limit:
        pngs = pngs[: args.limit]

    print(f"Found {len(pngs)} PNG(s) under {input_dir.resolve()}")

    # --- Init PaddleOCR ---
    ocr_engine = _get_ocr_engine()

    # --- Process each image ---
    records: list[dict] = []
    for png_path in pngs:
        rel_path = str(png_path)
        pdf_stem, page_num = _infer_metadata(png_path)

        print(f"  OCR: {rel_path} ...", end="", flush=True)
        image_input = _resize_max_side(png_path, args.max_side)
        boxes = _run_ocr(ocr_engine, image_input, fast=args.fast)
        md = boxes_to_markdown(boxes, min_conf=args.min_conf)
        plain = "\n".join(b["text"] for b in boxes if b["conf"] >= args.min_conf)
        print(f"  {len(boxes)} boxes, {len(md)} chars MD")

        raw_json = [
            {"text": b["text"], "conf": round(b["conf"], 4),
             "box": [[round(x, 1) for x in pt] for pt in b["box"]]}
            for b in boxes
        ]

        records.append({
            "id": _deterministic_id(rel_path),
            "image_path": rel_path,
            "pdf_stem": pdf_stem,
            "page_num": page_num,
            "text": plain,
            "markdown": md,
            "raw_ocr_json": raw_json,
        })

    # --- Print preview ---
    print(f"\nDone: {len(records)} record(s)\n")
    for r in records[:2]:
        print(f"=== [{r['image_path']}] page={r['page_num']} ===")
        print(r["markdown"][:600])
        print()

    # --- Write to Postgres ---
    if args.write_postgres:
        from db import upsert_ocr_pages

        assert pg_conn is not None and pg_target is not None
        try:
            inserted = upsert_ocr_pages(pg_conn, pg_target, records)
            print(f"Upserted {inserted} record(s) into {pg_target.schema}.{pg_target.table}")
        finally:
            pg_conn.close()


if __name__ == "__main__":
    main()
