from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings


@dataclass
class LLMConfig:
    model: str = "llama3.1:8b"
    temperature: float = 0.2
    max_tokens: int = 260
    seed: Optional[int] = None  # Ollama may ignore seed; keep for interface compatibility.


def build_chat_llm(cfg: LLMConfig) -> ChatOllama:
    return ChatOllama(
        model=cfg.model,
        temperature=cfg.temperature,
        num_predict=cfg.max_tokens,  # max output tokens [web:204]
        num_ctx=2048,                # keep context small for speed
        timeout=120,                 # request stream timeout [web:204]
        keep_alive="10m",            # keep model loaded to avoid reload delays [web:204]
        )



def build_embeddings(model: str = "mxbai-embed-large") -> OllamaEmbeddings:
    # Ollama provides local embedding models, usable via LangChain embeddings wrappers. [web:169]
    return OllamaEmbeddings(model=model)

