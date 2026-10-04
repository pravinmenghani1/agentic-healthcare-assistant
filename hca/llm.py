"""Local Ollama model access (chat LLM + embeddings)."""
from functools import lru_cache

import requests
from langchain_ollama import ChatOllama, OllamaEmbeddings

from . import config


@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.0, model: str | None = None, num_ctx: int = 8192) -> ChatOllama:
    return ChatOllama(
        model=model or config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=num_ctx,
    )


class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text expects task prefixes; other models get the text unchanged."""

    def _is_nomic(self) -> bool:
        return "nomic" in self.model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._is_nomic():
            texts = [f"search_document: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"search_query: {text}" if self._is_nomic() else text)


@lru_cache(maxsize=1)
def get_embeddings() -> NomicEmbeddings:
    return NomicEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)


def ollama_status() -> dict:
    """Report whether Ollama is reachable and the configured models are pulled."""
    try:
        tags = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3).json()
    except requests.RequestException as exc:
        return {"ok": False, "error": f"Ollama not reachable at {config.OLLAMA_BASE_URL}: {exc}"}
    names = {m["name"] for m in tags.get("models", [])}
    names |= {n.split(":")[0] for n in names if n.endswith(":latest")}
    missing = [m for m in (config.LLM_MODEL, config.EMBED_MODEL, config.EVAL_MODEL) if m not in names]
    return {"ok": not missing, "missing": missing, "models": sorted(names)}
