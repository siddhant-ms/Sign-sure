from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pdfplumber
import pypdf
import pypdfium2 as pdfium
import pytesseract

from app.config import settings
from app.models import ExtractedDocument, ExtractedPage


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_legal_text(raw: str) -> str:
    # Rejoin hyphenated line breaks across wrapped statutory/judgment lines
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_with_pypdf(reader: pypdf.PdfReader, page_idx: int) -> str:
    try:
        return reader.pages[page_idx].extract_text() or ""
    except Exception:
        return ""


def _extract_with_pdfplumber(plumber_pdf: pdfplumber.PDF, page_idx: int) -> str:
    try:
        page = plumber_pdf.pages[page_idx]
        text = page.extract_text(layout=False) or ""
        tables = page.extract_tables() or []
        table_lines: list[str] = []
        for tbl in tables:
            for row in tbl:
                cleaned = [cell.strip().replace("\n", " ") for cell in row if cell and cell.strip()]
                if cleaned:
                    table_lines.append(" | ".join(cleaned))
        if table_lines:
            text = f"{text}\n" + "\n".join(table_lines)
        return text
    except Exception:
        return ""


def _ocr_single_page(pdfium_doc: pdfium.PdfDocument, page_idx: int, dpi: int) -> str:
    scale = dpi / 72.0
    page = pdfium_doc[page_idx]
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    try:
        return pytesseract.image_to_string(pil_image, lang="eng") or ""
    finally:
        page.close()


def extract_pdf_document(
    file_path: Path,
    min_chars: int = settings.ocr_min_chars_threshold,
    ocr_dpi: int = settings.ocr_dpi,
) -> ExtractedDocument:
    sha256 = compute_sha256(file_path)
    reader = pypdf.PdfReader(str(file_path))
    page_count = len(reader.pages)

    pages: list[ExtractedPage] = []
    ocr_pages_count = 0

    plumber_ctx = None
    pdfium_doc = None

    try:
        for idx in range(page_count):
            page_num = idx + 1
            text = normalize_legal_text(_extract_with_pypdf(reader, idx))
            method: str = "text"

            # Fallback to pdfplumber when pypdf returns very sparse text
            if len(text) < min_chars:
                if plumber_ctx is None:
                    plumber_ctx = pdfplumber.open(str(file_path)).__enter__()
                plumber_text = normalize_legal_text(_extract_with_pdfplumber(plumber_ctx, idx))
                if len(plumber_text) > len(text):
                    text = plumber_text

            # Fallback to per-page OCR for scanned pages
            if len(text) < min_chars:
                if pdfium_doc is None:
                    pdfium_doc = pdfium.PdfDocument(str(file_path))
                ocr_text = normalize_legal_text(_ocr_single_page(pdfium_doc, idx, ocr_dpi))
                if ocr_text:
                    text = ocr_text
                    method = "ocr"
                    ocr_pages_count += 1

            pages.append(
                ExtractedPage(
                    page_number=page_num,
                    text=text,
                    extraction_method=method,  # type: ignore[arg-type]
                )
            )
    finally:
        if plumber_ctx is not None:
            try:
                plumber_ctx.__exit__(None, None, None)
            except Exception:
                pass
        if pdfium_doc is not None:
            try:
                pdfium_doc.close()
            except Exception:
                pass

    return ExtractedDocument(
        file_path=str(file_path.resolve()),
        file_name=file_path.name,
        sha256=sha256,
        page_count=page_count,
        ocr_pages=ocr_pages_count,
        pages=pages,
    )