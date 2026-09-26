import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid7

from rag_anatomy.domain import (
    Document,
    DuplicateContentError,
    Embedding,
    EmptyDocumentError,
    FilenameConflictError,
)
from rag_anatomy.ports import Chunker, DocumentParser, DocumentRepository, Embedder
from rag_anatomy.services.filenames import numbered_filename


class ConflictPolicy(StrEnum):
    REJECT = "reject"
    REPLACE = "replace"
    KEEP_BOTH = "keep_both"


@dataclass(frozen=True, slots=True, kw_only=True)
class IngestionResult:
    document: Document
    chunk_count: int
    replaced: Document | None


class IngestionService:
    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        embedder: Embedder,
        repository: DocumentRepository,
        *,
        batch_size: int,
        max_concurrency: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._repository = repository
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency

    async def ingest(
        self,
        content: bytes,
        filename: str,
        media_type: str,
        policy: ConflictPolicy = ConflictPolicy.REJECT,
    ) -> IngestionResult:
        content_hash = await asyncio.to_thread(_sha256, content)
        if duplicate := await self._repository.find_by_hash(content_hash):
            raise DuplicateContentError(duplicate)
        replaced: Document | None = None
        if existing := await self._repository.find_by_filename(filename):
            match policy:
                case ConflictPolicy.REJECT:
                    raise FilenameConflictError(existing)
                case ConflictPolicy.REPLACE:
                    replaced = existing
                case ConflictPolicy.KEEP_BOTH:
                    filename = await self._first_free_filename(filename)

        document = Document(
            id=uuid7(),
            filename=filename,
            media_type=media_type,
            content_hash=content_hash,
            created_at=datetime.now(UTC),
        )
        pages = await self._parser.parse(content, media_type)
        chunks = await asyncio.to_thread(self._chunker.chunk, document.id, pages)
        if not chunks:
            raise EmptyDocumentError(filename)
        embeddings = await self._embed([chunk.text for chunk in chunks])
        if replaced is None:
            await self._repository.save(document, chunks, embeddings)
        else:
            await self._repository.replace(replaced.id, document, chunks, embeddings)
        return IngestionResult(
            document=document, chunk_count=len(chunks), replaced=replaced
        )

    async def _first_free_filename(self, filename: str) -> str:
        n = 1
        while await self._repository.find_by_filename(numbered_filename(filename, n)):
            n += 1
        return numbered_filename(filename, n)

    async def _embed(self, texts: list[str]) -> list[Embedding]:
        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]
        results: list[list[Embedding]] = [[] for _ in batches]
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def embed(index: int, batch: list[str]) -> None:
            async with semaphore:
                results[index] = await self._embedder.embed_documents(batch)

        async with asyncio.TaskGroup() as group:
            for index, batch in enumerate(batches):
                group.create_task(embed(index, batch))
        return [embedding for batch in results for embedding in batch]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
