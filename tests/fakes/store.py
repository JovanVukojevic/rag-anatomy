import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from rag_anatomy.domain import (
    Chunk,
    Document,
    DuplicateContentError,
    Embedding,
    FilenameConflictError,
    RetrievedChunk,
    StageRank,
)
from rag_anatomy.ports import DocumentRepository, KeywordSearch, VectorSearch
from tests.fakes._text import overlap


@dataclass(frozen=True, slots=True)
class _Entry:
    chunk: Chunk
    embedding: Embedding
    document: Document


class InMemoryStore:
    def __init__(self) -> None:
        self._documents: dict[UUID, Document] = {}
        self._entries: list[_Entry] = []

    async def find_by_hash(self, content_hash: str) -> Document | None:
        return next(
            (d for d in self._documents.values() if d.content_hash == content_hash),
            None,
        )

    async def find_by_filename(self, filename: str) -> Document | None:
        key = filename.lower()
        return next(
            (d for d in self._documents.values() if d.filename.lower() == key),
            None,
        )

    async def save(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        self._check(document, chunks, embeddings, others=list(self._documents.values()))
        self._insert(document, chunks, embeddings)

    async def replace(
        self,
        existing_id: UUID,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        if existing_id not in self._documents:
            raise LookupError(f"no document with id {existing_id}")
        others = [d for d in self._documents.values() if d.id != existing_id]
        self._check(document, chunks, embeddings, others=others)
        del self._documents[existing_id]
        self._entries = [e for e in self._entries if e.chunk.document_id != existing_id]
        self._insert(document, chunks, embeddings)

    def _check(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
        *,
        others: list[Document],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if any(chunk.document_id != document.id for chunk in chunks):
            raise ValueError("every chunk must belong to the saved document")
        for existing in others:
            if existing.content_hash == document.content_hash:
                raise DuplicateContentError(existing)
        for existing in others:
            if existing.filename.lower() == document.filename.lower():
                raise FilenameConflictError(existing)

    def _insert(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        self._documents[document.id] = document
        self._entries.extend(
            _Entry(chunk, embedding, document)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        )

    async def vector_search(self, embedding: Embedding, k: int) -> list[RetrievedChunk]:
        by_distance = sorted(
            (
                (distance, e)
                for e in self._entries
                if (distance := _cosine_distance(embedding, e.embedding)) is not None
            ),
            key=lambda pair: (pair[0], pair[1].chunk.id),
        )
        return [
            RetrievedChunk(
                chunk=e.chunk,
                document=e.document,
                dense=StageRank(rank=rank, score=1 - distance),
            )
            for rank, (distance, e) in enumerate(by_distance[:k], start=1)
        ]

    async def keyword_search(self, query: str, k: int) -> list[RetrievedChunk]:
        by_score = sorted(
            (
                (score, e)
                for e in self._entries
                if (score := overlap(query, e.chunk.text)) > 0
            ),
            key=lambda pair: (-pair[0], pair[1].chunk.id),
        )
        return [
            RetrievedChunk(
                chunk=e.chunk,
                document=e.document,
                keyword=StageRank(rank=rank, score=score),
            )
            for rank, (score, e) in enumerate(by_score[:k], start=1)
        ]


# pgvector does not index zero vectors for cosine distance, so they are never found.
def _cosine_distance(a: Embedding, b: Embedding) -> float | None:
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    if not norms:
        return None
    return 1 - sum(x * y for x, y in zip(a, b, strict=True)) / norms


if TYPE_CHECKING:
    _repository: DocumentRepository = InMemoryStore()
    _vector_search: VectorSearch = InMemoryStore()
    _keyword_search: KeywordSearch = InMemoryStore()
