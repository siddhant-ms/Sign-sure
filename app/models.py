from typing import Literal
from pydantic import BaseModel, Field


Jurisdiction = Literal["central", "karnataka"]
DocType = Literal["act", "judgment"]
Bench = Literal["bengaluru", "dharwad", "kalaburagi"]
LegalDomain = Literal[
    "criminal",
    "civil",
    "property",
    "procedural",
    "constitutional",
    "consumer",
    "administrative",
    "general",
]


class ExtractedPage(BaseModel):
    page_number: int
    text: str
    extraction_method: Literal["text", "ocr"]


class ExtractedDocument(BaseModel):
    file_path: str
    file_name: str
    sha256: str
    page_count: int
    ocr_pages: int
    pages: list[ExtractedPage]


class DocMetadata(BaseModel):
    doc_id: str
    file_name: str
    sha256: str
    jurisdiction: Jurisdiction
    doc_type: DocType
    bench: Bench | None = None
    court: str | None = None
    doc_title: str
    year: int | None = None
    case_number: str | None = None
    judgment_date: str | None = None
    domain: LegalDomain = "general"


class LegalChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    jurisdiction: Jurisdiction
    doc_type: DocType
    bench: Bench | None = None
    domain: LegalDomain
    year: int | None = None
    section_ref: str
    page_start: int
    page_end: int
    extraction_method: Literal["text", "ocr"]
    text: str
    contextual_text: str


class SearchFilter(BaseModel):
    jurisdiction: Jurisdiction | None = None
    doc_type: DocType | None = None
    bench: Bench | None = None
    domain: LegalDomain | None = None
    year: int | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    filters: SearchFilter | None = None
    top_k: int = Field(default=6, ge=1, le=25)
    expand_query: bool = True


class RankedChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    jurisdiction: str
    doc_type: str
    bench: str | None = None
    domain: str
    year: int | None = None
    section_ref: str
    page_start: int
    page_end: int
    extraction_method: str
    text: str
    rrf_score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    expanded_query: str
    results: list[RankedChunkResult]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    filters: SearchFilter | None = None
    model: str | None = None
    top_k: int = Field(default=6, ge=1, le=12)
    expand_query: bool = True
    debug: bool = False


class CitationItem(BaseModel):
    index: int
    doc_id: str
    doc_title: str
    jurisdiction: str
    doc_type: str
    bench: str | None = None
    section_ref: str
    page_start: int
    page_end: int


class QueryResponse(BaseModel):
    query: str
    expanded_query: str
    model_used: str
    answer: str
    citations: list[CitationItem]
    retrieved_chunks: list[RankedChunkResult] | None = None
    reasoning: str | None = None


class IngestFolderRequest(BaseModel):
    root_dir: str
    force: bool = False


class IngestDocSummary(BaseModel):
    doc_id: str
    file_name: str
    doc_title: str
    jurisdiction: str
    doc_type: str
    bench: str | None = None
    chunks_indexed: int
    ocr_pages: int
    status: Literal["indexed", "skipped"]


class IngestResponse(BaseModel):
    total_files_found: int
    indexed_count: int
    skipped_count: int
    total_ocr_pages: int
    documents: list[IngestDocSummary]