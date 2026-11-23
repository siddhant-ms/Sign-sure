from __future__ import annotations

import re
from typing import Any

import httpx

from app.config import settings


class OllamaClient:
    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        timeout: float = settings.ollama_timeout_seconds,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> list[str]:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    def embed_texts(
        self,
        texts: list[str],
        task_prefix: str,
        model: str = settings.embed_model,
        batch_size: int = settings.embed_batch_size,
    ) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), batch_size):
                batch = [f"{task_prefix}: {t}" for t in texts[start : start + batch_size]]
                resp = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": model, "input": batch},
                )
                resp.raise_for_status()
                payload = resp.json()
                embeddings = payload.get("embeddings")
                if not embeddings or len(embeddings) != len(batch):
                    raise RuntimeError(
                        f"Ollama /api/embed returned {len(embeddings) if embeddings else 0} embeddings for {len(batch)} inputs."
                    )
                all_embeddings.extend(embeddings)

        return all_embeddings

    def generate_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = settings.default_llm_model,
        num_ctx: int = settings.llm_num_ctx,
        temperature: float = settings.llm_temperature,
    ) -> tuple[str, str | None]:
        payload: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "num_ctx": num_ctx,
                "temperature": temperature,
            },
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            raw_text = resp.json().get("response", "")

        return parse_reasoning_output(raw_text)


def parse_reasoning_output(raw_text: str) -> tuple[str, str | None]:
    think_blocks = re.findall(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL | re.IGNORECASE)
    reasoning = "\n\n".join(b.strip() for b in think_blocks if b.strip()) or None
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
    return cleaned, reasoning