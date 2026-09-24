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
    Retriever,
)
from rag_anatomy.ports import DocumentRepository, KeywordSearch, VectorSearch
from tests.fakes._text import overlap


@dataclass(frozen=True, slots=True)
class _Entry:
    chunk: Chunk
    embedding: Embedding
    filename: str


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
        key = filename.casefold()
        return next(
            (d for d in self._documents.values() if d.filename.casefold() == key),
            None,
        )

    async def save(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if any(chunk.document_id != document.id for chunk in chunks):
            raise ValueError("every chunk must belong to the saved document")
        if existing := await self.find_by_hash(document.content_hash):
            raise DuplicateContentError(existing)
        if existing := await self.find_by_filename(document.filename):
            raise FilenameConflictError(existing)
        self._documents[document.id] = document
        self._entries.extend(
            _Entry(chunk, embedding, document.filename)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        )

    async def vector_search(self, embedding: Embedding, k: int) -> list[RetrievedChunk]:
        scored = [(_cosine(embedding, e.embedding), e) for e in self._entries]
        return _top_k(scored, k, Retriever.DENSE)

    async def keyword_search(self, query: str, k: int) -> list[RetrievedChunk]:
        scored = [(overlap(query, e.chunk.text), e) for e in self._entries]
        return _top_k([(s, e) for s, e in scored if s > 0], k, Retriever.KEYWORD)


def _cosine(a: Embedding, b: Embedding) -> float:
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    if not norms:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / norms


def _top_k(
    scored: list[tuple[float, _Entry]], k: int, retriever: Retriever
) -> list[RetrievedChunk]:
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        RetrievedChunk(
            chunk=entry.chunk, filename=entry.filename, score=score, retriever=retriever
        )
        for score, entry in scored[:k]
    ]


if TYPE_CHECKING:
    _repository: DocumentRepository = InMemoryStore()
    _vector_search: VectorSearch = InMemoryStore()
    _keyword_search: KeywordSearch = InMemoryStore()
