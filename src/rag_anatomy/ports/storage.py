from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from rag_anatomy.domain import Chunk, Document, Embedding, RetrievedChunk


class DocumentRepository(Protocol):
    """Saves or replaces documents atomically; rejects duplicate content and filenames (any case)."""

    async def find_by_hash(self, content_hash: str) -> Document | None: ...

    async def find_by_filename(self, filename: str) -> Document | None: ...

    async def save(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None: ...

    async def replace(
        self,
        existing_id: UUID,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None: ...


class VectorSearch(Protocol):
    """Returns the k chunks nearest by cosine distance (fewer only if fewer exist), best first, ties by chunk id."""

    async def vector_search(
        self, embedding: Embedding, k: int
    ) -> list[RetrievedChunk]: ...


class KeywordSearch(Protocol):
    """Returns up to k chunks matching the query terms, best first, ties by chunk id."""

    async def keyword_search(self, query: str, k: int) -> list[RetrievedChunk]: ...
