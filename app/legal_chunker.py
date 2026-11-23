from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.models import DocMetadata, ExtractedDocument, LegalChunk


@dataclass
class _Segment:
    section_ref: str
    text: str
    page_start: int
    page_end: int
    extraction_method: str


SECTION_BOUNDARY_RE = re.compile(
    r"(?m)^(?:CHAPTER\s+[IVXLCDM0-9]+[^\n]*|PART\s+[IVXLCDM0-9]+[^\n]*|ORDER\s+[IVXLCDM0-9]+[^\n]*|"
    r"(?:Section|Article|Rule)\s+\d+[A-Za-z0-9\-().]*[^\n]*)"
)

JUDGMENT_BOUNDARY_RE = re.compile(
    r"(?m)^(?:\[\d+\]|\d{1,3}\.\s+|(?:FACTS|ISSUES?|POINT FOR DETERMINATION|CONTENTIONS|"
    r"ANALYSIS|REASONING|HELD|CONCLUSION|ORDER)\b[^\n]*)"
)


def _split_oversized_segment(
    seg: _Segment,
    max_chars: int,
    overlap_chars: int,
) -> list[_Segment]:
    if len(seg.text) <= max_chars:
        return [seg]

    step = max(1, max_chars - overlap_chars)
    parts: list[_Segment] = []
    start = 0
    idx = 1
    text_len = len(seg.text)

    while start < text_len:
        end = min(text_len, start + max_chars)
        # Try breaking cleanly at paragraph or sentence boundary near end
        if end < text_len:
            window = seg.text[start:end]
            para_break = window.rfind("\n\n")
            sent_break = max(window.rfind(". "), window.rfind(".\n"))
            best_cut = para_break if para_break > max_chars // 2 else sent_break
            if best_cut > max_chars // 3:
                end = start + best_cut + 1

        piece = seg.text[start:end].strip()
        if piece:
            parts.append(
                _Segment(
                    section_ref=f"{seg.section_ref} (part {idx})",
                    text=piece,
                    page_start=seg.page_start,
                    page_end=seg.page_end,
                    extraction_method=seg.extraction_method,
                )
            )
            idx += 1

        if end >= text_len:
            break
        start = max(start + 1, end - overlap_chars)

    return parts


def _build_segments_for_act(doc: ExtractedDocument) -> list[_Segment]:
    segments: list[_Segment] = []
    current_ref = "Preamble / Preliminary"
    buffer_lines: list[str] = []
    buf_page_start: int | None = None
    buf_page_end: int = 1
    buf_method: str = "text"

    def flush() -> None:
        nonlocal buffer_lines, buf_page_start
        content = "\n".join(buffer_lines).strip()
        if content and buf_page_start is not None:
            segments.append(
                _Segment(
                    section_ref=current_ref,
                    text=content,
                    page_start=buf_page_start,
                    page_end=buf_page_end,
                    extraction_method=buf_method,
                )
            )
        buffer_lines = []
        buf_page_start = None

    for page in doc.pages:
        if not page.text.strip():
            continue
        for line in page.text.splitlines():
            stripped = line.strip()
            if not stripped:
                if buffer_lines and buffer_lines[-1] != "":
                    buffer_lines.append("")
                continue

            if SECTION_BOUNDARY_RE.match(stripped) and len("\n".join(buffer_lines)) > 150:
                flush()
                current_ref = stripped[:100]
                buf_page_start = page.page_number
                buf_page_end = page.page_number
                buf_method = page.extraction_method
                buffer_lines.append(stripped)
            else:
                if buf_page_start is None:
                    buf_page_start = page.page_number
                    buf_method = page.extraction_method
                buf_page_end = page.page_number
                if page.extraction_method == "ocr":
                    buf_method = "ocr"
                buffer_lines.append(stripped)

    flush()
    return segments


def _build_segments_for_judgment(doc: ExtractedDocument) -> list[_Segment]:
    segments: list[_Segment] = []
    current_ref = "Header & Facts"
    buffer_lines: list[str] = []
    buf_page_start: int | None = None
    buf_page_end: int = 1
    buf_method: str = "text"

    def flush() -> None:
        nonlocal buffer_lines, buf_page_start
        content = "\n".join(buffer_lines).strip()
        if content and buf_page_start is not None:
            segments.append(
                _Segment(
                    section_ref=current_ref,
                    text=content,
                    page_start=buf_page_start,
                    page_end=buf_page_end,
                    extraction_method=buf_method,
                )
            )
        buffer_lines = []
        buf_page_start = None

    for page in doc.pages:
        if not page.text.strip():
            continue
        for line in page.text.splitlines():
            stripped = line.strip()
            if not stripped:
                if buffer_lines and buffer_lines[-1] != "":
                    buffer_lines.append("")
                continue

            m = JUDGMENT_BOUNDARY_RE.match(stripped)
            if m and len("\n".join(buffer_lines)) > 150:
                flush()
                para_match = re.match(r"^\[(\d+)\]|^(\d{1,3})\.", stripped)
                if para_match:
                    p_num = para_match.group(1) or para_match.group(2)
                    current_ref = f"Para {p_num}"
                else:
                    current_ref = stripped[:80]
                buf_page_start = page.page_number
                buf_page_end = page.page_number
                buf_method = page.extraction_method
                buffer_lines.append(stripped)
            else:
                if buf_page_start is None:
                    buf_page_start = page.page_number
                    buf_method = page.extraction_method
                buf_page_end = page.page_number
                if page.extraction_method == "ocr":
                    buf_method = "ocr"
                buffer_lines.append(stripped)

    flush()
    return segments


def chunk_legal_document(
    doc: ExtractedDocument,
    meta: DocMetadata,
    max_chars: int = settings.chunk_max_chars,
    overlap_chars: int = settings.chunk_overlap_chars,
) -> list[LegalChunk]:
    raw_segments = (
        _build_segments_for_judgment(doc)
        if meta.doc_type == "judgment"
        else _build_segments_for_act(doc)
    )

    final_segments: list[_Segment] = []
    for seg in raw_segments:
        final_segments.extend(_split_oversized_segment(seg, max_chars=max_chars, overlap_chars=overlap_chars))

    chunks: list[LegalChunk] = []
    for idx, seg in enumerate(final_segments):
        chunk_id = f"{meta.doc_id}_{idx:04d}"
        bench_part = f" | Bench: {meta.bench}" if meta.bench else ""
        prefix = (
            f"[Document: {meta.doc_title} | Jurisdiction: {meta.jurisdiction}"
            f"{bench_part} | Section: {seg.section_ref} | Pages: {seg.page_start}-{seg.page_end}]"
        )
        contextual_text = f"{prefix}\n{seg.text}"

        chunks.append(
            LegalChunk(
                chunk_id=chunk_id,
                doc_id=meta.doc_id,
                doc_title=meta.doc_title,
                jurisdiction=meta.jurisdiction,
                doc_type=meta.doc_type,
                bench=meta.bench,
                domain=meta.domain,
                year=meta.year,
                section_ref=seg.section_ref,
                page_start=seg.page_start,
                page_end=seg.page_end,
                extraction_method=seg.extraction_method,  # type: ignore[arg-type]
                text=seg.text,
                contextual_text=contextual_text,
            )
        )

    return chunks