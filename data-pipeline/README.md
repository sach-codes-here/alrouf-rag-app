# Data Pipeline

## Convert PDFs to PNGs

This pipeline step converts each PDF into per-page PNG images using PyMuPDF.

For `docs/filename.pdf` it creates:

- `docs/filename_pdf/page_001.png`
- `docs/filename_pdf/page_002.png`
- ...

Run:

```bash
python data-pipeline/01-convert-pdf-to-image.py --input-dir docs
```

Optional rendering quality:

```bash
python data-pipeline/01-convert-pdf-to-image.py --input-dir docs --zoom 2.5
```

