import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx2

from rag_anatomy.adapters.driven.chunking import TokenChunker
from rag_anatomy.adapters.driven.openai import (
    MAX_INPUTS_PER_REQUEST,
    MAX_TOKENS_PER_REQUEST,
    OpenAIEmbedder,
    openai_client,
)
from rag_anatomy.adapters.driven.parsing import (
    DOCX,
    PDF,
    CompositeParser,
    DocxParser,
    PdfParser,
    TextParser,
)
from rag_anatomy.adapters.driven.postgres import (
    MIN_PGVECTOR_VERSION,
    EmbeddingSpace,
    PgVectorStore,
    connection_pool,
)
from rag_anatomy.config import DatabaseSettings, OpenAISettings
from rag_anatomy.ports import DocumentParser
from rag_anatomy.services import IngestionService, RetrievalService

_POOL_MAX_SIZE = 10
_DATABASE_TIMEOUT = 10.0
_CHUNK_SIZE = 400
_CHUNK_OVERLAP = 50
_EMBEDDING_BATCH_SIZE = min(
    256, MAX_INPUTS_PER_REQUEST, MAX_TOKENS_PER_REQUEST // _CHUNK_SIZE
)
_EMBEDDING_CONCURRENCY = 4


class ConfigurationError(Exception):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Container:
    ingestion: IngestionService
    retrieval: RetrievalService


@asynccontextmanager
async def open_container(
    *, openai_transport: httpx2.AsyncBaseTransport | None = None
) -> AsyncIterator[Container]:
    database = DatabaseSettings()
    openai = OpenAISettings()
    async with connection_pool(
        database.dsn, max_size=_POOL_MAX_SIZE, timeout=_DATABASE_TIMEOUT
    ) as pool:
        await pool.wait(_DATABASE_TIMEOUT)
        store = PgVectorStore(pool)
        await _check_pgvector_version(store)
        await _check_embedding_space(store, openai)
        chunker = await asyncio.to_thread(
            TokenChunker,
            model=openai.embedding_model,
            chunk_size=_CHUNK_SIZE,
            overlap=_CHUNK_OVERLAP,
        )
        async with openai_client(
            openai.api_key.get_secret_value(),
            timeout=openai.timeout,
            max_retries=openai.max_retries,
            transport=openai_transport,
        ) as client:
            embedder = OpenAIEmbedder(
                client,
                model=openai.embedding_model,
                dimensions=openai.embedding_dimensions,
            )
            yield Container(
                ingestion=IngestionService(
                    _parser(),
                    chunker,
                    embedder,
                    store,
                    batch_size=_EMBEDDING_BATCH_SIZE,
                    max_concurrency=_EMBEDDING_CONCURRENCY,
                ),
                retrieval=RetrievalService(embedder, store),
            )


def _parser() -> CompositeParser:
    text = TextParser()
    parsers: dict[str, DocumentParser] = dict.fromkeys(TextParser.MEDIA_TYPES, text)
    parsers[PDF] = PdfParser()
    parsers[DOCX] = DocxParser()
    return CompositeParser(parsers)


async def _check_pgvector_version(store: PgVectorStore) -> None:
    installed = await store.pgvector_version()
    if installed < MIN_PGVECTOR_VERSION:
        raise ConfigurationError(
            f"pgvector {_dotted(installed)} is installed, but vector search needs "
            f"{_dotted(MIN_PGVECTOR_VERSION)} or later for iterative index scans"
        )


def _dotted(version: tuple[int, ...]) -> str:
    return ".".join(map(str, version))


async def _check_embedding_space(store: PgVectorStore, openai: OpenAISettings) -> None:
    configured = EmbeddingSpace(
        model=openai.embedding_model, dimensions=openai.embedding_dimensions
    )
    column = await store.embedding_column_dimensions()
    if column != configured.dimensions:
        raise ConfigurationError(
            f"{configured.model} is configured with {configured.dimensions} dimensions, "
            f"but chunks.embedding is vector({column})"
        )
    stored = await store.claim_embedding_space(configured.model, configured.dimensions)
    if stored != configured:
        raise ConfigurationError(
            f"the database holds {stored.model} embeddings ({stored.dimensions} "
            f"dimensions), but {configured.model} ({configured.dimensions}) is configured"
        )
