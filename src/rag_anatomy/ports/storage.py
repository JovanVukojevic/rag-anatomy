from collections.abc import Sequence
from typing import Protocol

from rag_anatomy.domain import Chunk, Document, Embedding, RetrievedChunk


class DocumentRepository(Protocol):
    """Saves documents atomically; rejects duplicate content and filenames (any case)."""

    async def find_by_hash(self, content_hash: str) -> Document | None: ...

    async def find_by_filename(self, filename: str) -> Document | None: ...

    async def save(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None: ...


class VectorSearch(Protocol):
    """Returns up to k chunks nearest to the embedding, best first."""

    async def vector_search(
        self, embedding: Embedding, k: int
    ) -> list[RetrievedChunk]: ...


class KeywordSearch(Protocol):
    """Returns up to k chunks matching the query terms, best first."""

    async def keyword_search(self, query: str, k: int) -> list[RetrievedChunk]: ...
