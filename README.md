python data-pipeline/024-chunk-ocr-markdown-to-postgres.py \
  --pg-dsn "postgresql://alrouf_user:change_me@localhost:5432/alrouf_rag" \
  --ocr-schema rag --ocr-table ocr_png_pages \
  --chunks-schema rag --chunks-table ocr_markdown_chunks

python data-pipeline/023-ocr-png-to-markdown-paddle.py \
  --input-dir docs --write-postgres --pg-schema rag \
  --pg-dsn "postgresql://alrouf_user:change_me@localhost:5432/alrouf_rag" \
  --fast --max-side 1200