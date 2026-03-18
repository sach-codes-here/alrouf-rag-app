## Alrouf RAG App (End-to-End)

This repo implements a local RAG pipeline:

1. Convert PDFs to per-page PNGs (`docs/*_pdf/page_XXX.png`)
2. OCR PNGs into layout-aware Markdown and store in Postgres
3. Chunk Markdown by `##` headings and store chunks in Postgres
4. Embed chunk `title` + `content` and store integer vectors in Postgres
5. Build a local hybrid search index (FAISS + BM25)
6. Serve search via a FastAPI backend (`/search`, `/smart-search`)

## Prerequisites

- Python installed (recommend Python 3.10+)
- Postgres running locally
- Model + index files built locally via the pipeline scripts

## 1) Configure environment

1. Copy `.env.example` to `.env`
2. Update at least `DATABASE_URL` (and `ARGOS_ENABLED` if you want local Arabic->English translation)

The `.env` file controls:
- Postgres connection (`DATABASE_URL` or `PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD`)
- Which tables to read/write (`PG_SCHEMA`, `CHUNKS_TABLE`)
- Vector index output paths (`INDEX_DIR`, `FAISS_INDEX_FILE`, etc.)
- Embedding model and quantization scale

## 2) Put your PDFs in `docs/`

Place PDFs inside `docs/`:
- `docs/your_document.pdf`

After conversion, you should see folders like:
- `docs/your_document_pdf/page_001.png`

## 3) Convert PDFs to PNGs

This step is implemented as a Jupyter notebook:
- `data-pipeline/01-convert-pdf-to-image.ipynb`

Workflow:
1. Open the notebook
2. Run all cells
3. It will render every `docs/*.pdf` into `docs/<stem>_pdf/page_XXX.png`

## 4) OCR PNGs -> Markdown into Postgres

Run layout-aware OCR with PaddleOCR and upsert results into Postgres:

```bash
python data-pipeline/02-ocr-png-to-markdown-paddle.py \
  --input-dir docs \
  --write-postgres \
  --pg-schema rag \
  --pg-table ocr_png_pages \
  --pg-dsn "postgresql://YOUR_USER:YOUR_PASS@localhost:5432/YOUR_DB" \
  --fast --max-side 1200
```

Defaults:
- `--pg-table` defaults to `ocr_png_pages`
- `--pg-schema` defaults to `public` (set it to `rag` to match the rest of the pipeline)

## 5) Chunk Markdown (`##` headings) -> chunks table in Postgres

Chunk the OCR markdown stored in `ocr_png_pages` and write to `ocr_markdown_chunks`:

```bash
python data-pipeline/03-chunk-ocr-markdown-to-postgres.py \
  --pg-dsn "postgresql://YOUR_USER:YOUR_PASS@localhost:5432/YOUR_DB" \
  --ocr-schema rag --ocr-table ocr_png_pages \
  --chunks-schema rag --chunks-table ocr_markdown_chunks
```

Chunking behavior:
- One chunk per Markdown `##` heading section
- The chunk `content` explicitly excludes the heading line itself
- `citationPath` is built as a local `file://.../My.pdf#page=PAGE` link

## 6) Embed chunks (fill `contentVector` + `titleVector`)

```bash
python data-pipeline/04-embed-chunks.py \
  --pg-dsn "postgresql://YOUR_USER:YOUR_PASS@localhost:5432/YOUR_DB" \
  --schema rag \
  --table ocr_markdown_chunks \
  --model BAAI/bge-small-en-v1.5 \
  --quant-scale 1000
```

This script only embeds rows where:
- `content` and `title` are non-empty
- vectors are missing/empty

## 7) Build hybrid search index (FAISS + BM25)

This script reads your Postgres chunks and writes index files to `INDEX_DIR` (from `.env`):

```bash
python data-pipeline/05-push-to-index.py --build
```

Once built, you should have:
- `vectorstore/chunks.faiss`
- `vectorstore/chunks_docstore.jsonl`
- `vectorstore/chunks_bm25.pkl`

## 8) Run the FastAPI backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Backend endpoints:
- `GET /health`
- `POST /search`
- `POST /smart-search`
- `POST /reload`
- `GET /reports`
- `GET /reports/{report_id}`

## 9) Test with curl

### Hybrid search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Warranty information","top_k":10,"alpha":0.6}'
```

### Arabic-friendly smart search

```bash
curl -X POST http://localhost:8000/smart-search \
  -H "Content-Type: application/json" \
  -d '{"query":"معلومات الضمان","top_k":10,"alpha":0.6,"rephrase":true}'
```

Out-of-scope queries (casual greetings / low-signal) are handled with a friendly 200 response (they do not query the index).