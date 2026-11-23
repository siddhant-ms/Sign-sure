from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from app.config import settings
from app.models import DocMetadata, LegalChunk, RankedChunkResult, SearchFilter


def tokenize_legal(text: str) -> list[str]:
    # Preserve alphanumeric tokens, section numbers, case numbers, and years
    return re.findall(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", text.lower())


class HybridLegalStore:
    def __init__(self) -> None:
        settings.ensure_dirs()
        self.chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.doc_registry: dict[str, dict[str, Any]] = self._load_registry()
        self.bm25_records: list[dict[str, Any]] = []
        self.bm25_index: BM25Okapi | None = None
        self._load_bm25()

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        if settings.registry_path.exists():
            return json.loads(settings.registry_path.read_text(encoding="utf-8"))
        return {}

    def _save_registry(self) -> None:
        settings.registry_path.write_text(
            json.dumps(self.doc_registry, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_bm25(self) -> None:
        if settings.bm25_path.exists():
            with settings.bm25_path.open("rb") as f:
                payload = pickle.load(f)
            self.bm25_records = payload.get("records", [])
            corpus = payload.get("corpus", [])
            if corpus:
                self.bm25_index = BM25Okapi(corpus)
            else:
                self.bm25_index = None
        else:
            self.bm25_records = []
            self.bm25_index = None

    def _rebuild_and_save_bm25(self) -> None:
        corpus = [
            tokenize_legal(f"{rec['doc_title']} {rec['section_ref']} {rec['text']}")
            for rec in self.bm25_records
        ]
        self.bm25_index = BM25Okapi(corpus) if corpus else None
        with settings.bm25_path.open("wb") as f:
            pickle.dump({"records": self.bm25_records, "corpus": corpus}, f)

    def is_already_indexed(self, sha256: str) -> bool:
        doc_id = sha256[:16]
        return doc_id in self.doc_registry and self.doc_registry[doc_id].get("sha256") == sha256

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.doc_registry:
            return False
        self.collection.delete(where={"doc_id": doc_id})
        self.bm25_records = [r for r in self.bm25_records if r["doc_id"] != doc_id]
        self._rebuild_and_save_bm25()
        del self.doc_registry[doc_id]
        self._save_registry()
        return True

    def add_document_chunks(
        self,
        meta: DocMetadata,
        chunks: list[LegalChunk],
        embeddings: list[list[float]],
        ocr_pages: int,
        force: bool = False,
    ) -> int:
        if not chunks:
            return 0

        if force and meta.doc_id in self.doc_registry:
            self.delete_document(meta.doc_id)

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas: list[dict[str, Any]] = []
        new_bm25_rows: list[dict[str, Any]] = []

        for c in chunks:
            flat_meta: dict[str, Any] = {
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "jurisdiction": c.jurisdiction,
                "doc_type": c.doc_type,
                "bench": c.bench or "",
                "domain": c.domain,
                "year": c.year if c.year is not None else -1,
                "section_ref": c.section_ref,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "extraction_method": c.extraction_method,
            }
            metadatas.append(flat_meta)
            new_bm25_rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "doc_title": c.doc_title,
                    "jurisdiction": c.jurisdiction,
                    "doc_type": c.doc_type,
                    "bench": c.bench,
                    "domain": c.domain,
                    "year": c.year,
                    "section_ref": c.section_ref,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "extraction_method": c.extraction_method,
                    "text": c.text,
                }
            )

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        self.bm25_records.extend(new_bm25_rows)
        self._rebuild_and_save_bm25()

        self.doc_registry[meta.doc_id] = {
            "doc_id": meta.doc_id,
            "file_name": meta.file_name,
            "sha256": meta.sha256,
            "jurisdiction": meta.jurisdiction,
            "doc_type": meta.doc_type,
            "bench": meta.bench,
            "court": meta.court,
            "doc_title": meta.doc_title,
            "year": meta.year,
            "case_number": meta.case_number,
            "judgment_date": meta.judgment_date,
            "domain": meta.domain,
            "chunks_count": len(chunks),
            "ocr_pages": ocr_pages,
        }
        self._save_registry()
        return len(chunks)

    @staticmethod
    def _build_chroma_where(filters: SearchFilter | None) -> dict[str, Any] | None:
        if not filters:
            return None
        clauses: list[dict[str, Any]] = []
        if filters.jurisdiction:
            clauses.append({"jurisdiction": filters.jurisdiction})
        if filters.doc_type:
            clauses.append({"doc_type": filters.doc_type})
        if filters.bench:
            clauses.append({"bench": filters.bench})
        if filters.domain:
            clauses.append({"domain": filters.domain})
        if filters.year is not None:
            clauses.append({"year": filters.year})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _matches_filter(rec: dict[str, Any], filters: SearchFilter | None) -> bool:
        if not filters:
            return True
        if filters.jurisdiction and rec.get("jurisdiction") != filters.jurisdiction:
            return False
        if filters.doc_type and rec.get("doc_type") != filters.doc_type:
            return False
        if filters.bench and rec.get("bench") != filters.bench:
            return False
        if filters.domain and rec.get("domain") != filters.domain:
            return False
        if filters.year is not None and rec.get("year") != filters.year:
            return False
        return True

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        filters: SearchFilter | None = None,
        top_k_per_retriever: int = settings.retrieval_top_k,
        final_k: int = settings.final_context_k,
        rrf_k: int = settings.rrf_k,
    ) -> list[RankedChunkResult]:
        where_clause = self._build_chroma_where(filters)
        total_collection = self.collection.count()
        n_results = min(top_k_per_retriever, total_collection) if total_collection > 0 else 0

        vector_ranks: dict[str, int] = {}
        record_map: dict[str, dict[str, Any]] = {}

        if n_results > 0:
            vec_res = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas"],
            )
            ids = (vec_res.get("ids") or [[]])[0]
            docs = (vec_res.get("documents") or [[]])[0]
            metas = (vec_res.get("metadatas") or [[]])[0]

            for rank_idx, (cid, doc_txt, meta) in enumerate(zip(ids, docs, metas), start=1):
                vector_ranks[cid] = rank_idx
                record_map[cid] = {
                    "chunk_id": cid,
                    "doc_id": meta.get("doc_id", ""),
                    "doc_title": meta.get("doc_title", ""),
                    "jurisdiction": meta.get("jurisdiction", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "bench": meta.get("bench") or None,
                    "domain": meta.get("domain", "general"),
                    "year": None if meta.get("year", -1) == -1 else meta.get("year"),
                    "section_ref": meta.get("section_ref", ""),
                    "page_start": meta.get("page_start", 1),
                    "page_end": meta.get("page_end", 1),
                    "extraction_method": meta.get("extraction_method", "text"),
                    "text": doc_txt or "",
                }

        bm25_ranks: dict[str, int] = {}
        if self.bm25_index is not None and self.bm25_records:
            tokens = tokenize_legal(query_text)
            scores = self.bm25_index.get_scores(tokens)
            scored_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )
            bm25_rank_counter = 0
            for idx in scored_indices:
                if scores[idx] <= 0:
                    break
                rec = self.bm25_records[idx]
                if not self._matches_filter(rec, filters):
                    continue
                bm25_rank_counter += 1
                cid = rec["chunk_id"]
                bm25_ranks[cid] = bm25_rank_counter
                if cid not in record_map:
                    record_map[cid] = dict(rec)
                if bm25_rank_counter >= top_k_per_retriever:
                    break

        candidate_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        fused: list[RankedChunkResult] = []

        for cid in candidate_ids:
            v_rank = vector_ranks.get(cid)
            b_rank = bm25_ranks.get(cid)
            score = 0.0
            if v_rank is not None:
                score += 1.0 / (rrf_k + v_rank)
            if b_rank is not None:
                score += 1.0 / (rrf_k + b_rank)

            rec = record_map[cid]
            fused.append(
                RankedChunkResult(
                    chunk_id=cid,
                    doc_id=rec["doc_id"],
                    doc_title=rec["doc_title"],
                    jurisdiction=rec["jurisdiction"],
                    doc_type=rec["doc_type"],
                    bench=rec.get("bench"),
                    domain=rec.get("domain", "general"),
                    year=rec.get("year"),
                    section_ref=rec["section_ref"],
                    page_start=rec["page_start"],
                    page_end=rec["page_end"],
                    extraction_method=rec.get("extraction_method", "text"),
                    text=rec["text"],
                    rrf_score=round(score, 6),
                    vector_rank=v_rank,
                    bm25_rank=b_rank,
                )
            )

        fused.sort(key=lambda r: r.rrf_score, reverse=True)
        return fused[:final_k]

    def stats(self) -> dict[str, int]:
        return {
            "documents_count": len(self.doc_registry),
            "chroma_chunks_count": self.collection.count(),
            "bm25_chunks_count": len(self.bm25_records),
        }