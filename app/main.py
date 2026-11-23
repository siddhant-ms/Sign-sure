from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.config import settings
from app.models import (
    Bench,
    DocType,
    IngestDocSummary,
    IngestFolderRequest,
    IngestResponse,
    Jurisdiction,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from app.rag_pipeline import LegalRAGPipeline

app = FastAPI(
    title="SignSure Local Legal RAG API",
    version="0.1.0",
    description="Local RAG backend for Indian Central Acts, Karnataka Acts, and Karnataka High Court Judgments.",
)

pipeline = LegalRAGPipeline()


@app.get("/health")
def health() -> dict:
    ollama_ok = False
    models: list[str] = []
    try:
        models = pipeline.ollama.list_models()
        ollama_ok = True
    except Exception:
        ollama_ok = False

    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama_reachable": ollama_ok,
        "ollama_models": models,
        "embed_model": settings.embed_model,
        "default_llm_model": settings.default_llm_model,
        **pipeline.store.stats(),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_folder(req: IngestFolderRequest) -> IngestResponse:
    root = Path(req.root_dir)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.root_dir}")
    return pipeline.ingest_folder(root, force=req.force)


@app.post("/ingest/upload", response_model=IngestDocSummary)
def ingest_upload(
    file: UploadFile = File(...),
    jurisdiction: Jurisdiction = Form("central"),
    doc_type: DocType = Form("act"),
    bench: Bench | None = Form(None),
    force: bool = Form(False),
) -> IngestDocSummary:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    sub_path = Path("Category 1 — Central Acts & Legislation")
    if doc_type == "judgment":
        bench_folder = f"{bench} bench" if bench else "bengaluru bench"
        sub_path = Path("Category 3 — Karnataka High Court Judgments") / bench_folder
    elif jurisdiction == "karnataka":
        sub_path = Path("Category 2 — Karnataka State Acts & Legislation")

    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = Path(tmp_dir) / sub_path
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_file = target_dir / Path(file.filename).name
        with dest_file.open("wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
        return pipeline.ingest_single_pdf(dest_file, force=force)


@app.post("/search", response_model=SearchResponse)
def search_chunks(req: SearchRequest) -> SearchResponse:
    return pipeline.search(req)


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest) -> QueryResponse:
    return pipeline.answer_query(req)


@app.get("/documents")
def list_documents() -> dict:
    docs = list(pipeline.store.doc_registry.values())
    return {"count": len(docs), "documents": docs}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    deleted = pipeline.store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"deleted": True, "doc_id": doc_id}