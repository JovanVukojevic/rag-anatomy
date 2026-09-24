from collections.abc import Sequence
from typing import Protocol

from rag_anatomy.domain import Embedding


class Embedder(Protocol):
    """Embeds documents and queries into one space; the two may be encoded differently."""

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]: ...

    async def embed_query(self, text: str) -> Embedding: ...
