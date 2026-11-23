from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SIGNSURE_", env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text:latest"
    default_llm_model: str = "deepseek-r1:7b"
    fast_llm_model: str = "llama3.2:3b"
    llm_num_ctx: int = 8192
    llm_temperature: float = 0.1
    embed_batch_size: int = 32
    ollama_timeout_seconds: float = 180.0

    chunk_max_chars: int = 1400
    chunk_overlap_chars: int = 200
    ocr_min_chars_threshold: int = 50
    ocr_dpi: int = 200

    retrieval_top_k: int = 25
    final_context_k: int = 6
    rrf_k: int = 60

    data_dir: Path = Path("./data_store")
    chroma_dir: Path = Path("./data_store/chroma")
    bm25_path: Path = Path("./data_store/bm25_index.pkl")
    registry_path: Path = Path("./data_store/doc_registry.json")
    chroma_collection_name: str = "signsure_legal_chunks"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()