from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.rag_pipeline import LegalRAGPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-ingest SignSure PDF corpus into ChromaDB + BM25.")
    parser.add_argument("root_dir", type=Path, help="Path to SignSure root folder")
    parser.add_argument("--force", action="store_true", help="Re-index already-indexed PDFs")
    args = parser.parse_args()

    if not args.root_dir.exists() or not args.root_dir.is_dir():
        print(f"Error: folder not found: {args.root_dir}", file=sys.stderr)
        return 1

    pdf_files = sorted(
        p for p in args.root_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"
    )
    total = len(pdf_files)
    if total == 0:
        print(f"No PDF files found under {args.root_dir}")
        return 0

    pipeline = LegalRAGPipeline()
    start_time = time.perf_counter()
    indexed = 0
    skipped = 0
    ocr_pages_total = 0

    print(f"Found {total} PDFs under '{args.root_dir}'. Starting ingestion...\n")

    for idx, pdf_path in enumerate(pdf_files, start=1):
        summary = pipeline.ingest_single_pdf(pdf_path, force=args.force)
        ocr_pages_total += summary.ocr_pages
        if summary.status == "indexed":
            indexed += 1
        else:
            skipped += 1

        bench_label = f" [{summary.bench}]" if summary.bench else ""
        print(
            f"[{idx}/{total}] {summary.status.upper():7s} | {summary.jurisdiction}/{summary.doc_type}{bench_label} | "
            f"{summary.file_name} -> '{summary.doc_title}' ({summary.chunks_indexed} chunks, {summary.ocr_pages} OCR pages)"
        )

    elapsed = time.perf_counter() - start_time
    print("\n--- Ingestion Summary ---")
    print(f"Total PDFs : {total}")
    print(f"Indexed    : {indexed}")
    print(f"Skipped    : {skipped}")
    print(f"OCR Pages  : {ocr_pages_total}")
    print(f"Elapsed    : {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())