from collections.abc import Sequence
from typing import Protocol

from rag_anatomy.domain import Embedding


class Embedder(Protocol):
    """Embeds documents and queries into one space, in input order; raises EmbeddingError."""

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]: ...

    async def embed_query(self, text: str) -> Embedding: ...
