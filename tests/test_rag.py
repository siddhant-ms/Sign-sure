from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.legal_chunker import chunk_legal_document
from app.main import app
from app.metadata_extractor import build_doc_metadata, parse_folder_hierarchy
from app.models import ExtractedDocument, ExtractedPage, QueryRequest, SearchFilter, SearchRequest
from app.ollama_client import parse_reasoning_output
from app.query_expander import expand_legal_query
from app.rag_pipeline import LegalRAGPipeline


def test_folder_metadata_parsing() -> None:
    p1 = Path("SignSure/Category 1 — Central Acts & Legislation/a1908-05.pdf")
    assert parse_folder_hierarchy(p1) == ("central", "act", None, None)

    p2 = Path("SignSure/Category 2 — Karnataka State Acts & Legislation/karnataka_land.pdf")
    assert parse_folder_hierarchy(p2) == ("karnataka", "act", None, None)

    p3 = Path("SignSure/Category 3 — Karnataka High Court Judgments/Kalaburagi Bench/wp_100.pdf")
    assert parse_folder_hierarchy(p3) == ("karnataka", "judgment", "kalaburagi", "High Court of Karnataka")


def test_act_title_and_year_extraction(tmp_path: Path) -> None:
    folder = tmp_path / "Category 1 — Central Acts & Legislation"
    folder.mkdir(parents=True)
    pdf_file = folder / "a1908-05.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    doc = ExtractedDocument(
        file_path=str(pdf_file),
        file_name="a1908-05.pdf",
        sha256="a" * 64,
        page_count=1,
        ocr_pages=0,
        pages=[
            ExtractedPage(
                page_number=1,
                text="THE CODE OF CIVIL PROCEDURE, 1908\nACT NO. 5 OF 1908\nSection 9. Courts to try all civil suits unless barred.",
                extraction_method="text",
            )
        ],
    )
    meta = build_doc_metadata(doc)
    assert meta.doc_title == "THE CODE OF CIVIL PROCEDURE, 1908"
    assert meta.year == 1908
    assert meta.domain in {"procedural", "civil"}


def test_chunker_adds_contextual_prefix() -> None:
    doc = ExtractedDocument(
        file_path="/tmp/test.pdf",
        file_name="test.pdf",
        sha256="b" * 64,
        page_count=1,
        ocr_pages=0,
        pages=[
            ExtractedPage(
                page_number=1,
                text="Section 138. Dishonour of cheque for insufficiency of funds.\nWhere any cheque drawn by a person is returned unpaid.",
                extraction_method="text",
            )
        ],
    )
    meta = build_doc_metadata(doc)
    chunks = chunk_legal_document(doc, meta)
    assert len(chunks) >= 1
    assert "[Document:" in chunks[0].contextual_text
    assert "Jurisdiction: central" in chunks[0].contextual_text


def test_reasoning_tag_stripping() -> None:
    raw = "<think>\nLet me analyze section 138 carefully.\n</think>\nDishonour of cheque requires notice within 30 days."
    cleaned, reasoning = parse_reasoning_output(raw)
    assert cleaned == "Dishonour of cheque requires notice within 30 days."
    assert reasoning is not None and "section 138" in reasoning


def test_query_expansion() -> None:
    expanded = expand_legal_query("how to file fir and anticipatory bail")
    assert "BNSS" in expanded
    assert "CrPC" in expanded


def test_hybrid_search_and_query_with_mock_ollama(tmp_path: Path) -> None:
    with (
        patch("app.config.settings.data_dir", tmp_path),
        patch("app.config.settings.chroma_dir", tmp_path / "chroma"),
        patch("app.config.settings.bm25_path", tmp_path / "bm25.pkl"),
        patch("app.config.settings.registry_path", tmp_path / "registry.json"),
    ):
        from app.hybrid_store import HybridLegalStore

        store = HybridLegalStore()
        mock_ollama = MagicMock()
        mock_ollama.embed_texts.side_effect = lambda texts, task_prefix, model, batch_size=32: [
            [1.0, 0.0, 0.0] if "bail" in t.lower() else [0.0, 1.0, 0.0] for t in texts
        ]
        mock_ollama.generate_answer.return_value = (
            "Anticipatory bail provisions appear in the retrieved passage [1].",
            "Internal reasoning trace",
        )

        rag = LegalRAGPipeline(store=store, ollama=mock_ollama)

        dharwad_dir = tmp_path / "SignSure" / "Category 3 — Karnataka High Court Judgments" / "dharwad bench"
        dharwad_dir.mkdir(parents=True)
        pdf_path = dharwad_dir / "wp_55.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("app.rag_pipeline.extract_pdf_document") as mock_extract:
            mock_extract.return_value = ExtractedDocument(
                file_path=str(pdf_path),
                file_name="wp_55.pdf",
                sha256="c" * 64,
                page_count=1,
                ocr_pages=0,
                pages=[
                    ExtractedPage(
                        page_number=1,
                        text="[1] Consideration of anticipatory bail petition under Section 482 BNSS.",
                        extraction_method="text",
                    )
                ],
            )
            summary = rag.ingest_single_pdf(pdf_path)
            assert summary.status == "indexed"
            assert summary.bench == "dharwad"

        search_res = rag.search(
            SearchRequest(
                query="anticipatory bail",
                filters=SearchFilter(jurisdiction="karnataka", bench="dharwad"),
                top_k=3,
            )
        )
        assert len(search_res.results) >= 1
        assert search_res.results[0].bench == "dharwad"

        query_res = rag.answer_query(
            QueryRequest(
                query="What is noted about anticipatory bail?",
                filters=SearchFilter(jurisdiction="karnataka", bench="dharwad"),
                debug=True,
            )
        )
        assert "[1]" in query_res.answer
        assert len(query_res.citations) == 1
        assert query_res.reasoning == "Internal reasoning trace"
        assert query_res.retrieved_chunks is not None


def test_health_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "ollama_reachable" in body
    assert "chroma_chunks_count" in body