import asyncio
import logging
import re
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
from rag_anatomy.adapters.driven.tei import (
    TeiInfo,
    TeiReranker,
    server_info,
    tei_client,
)
from rag_anatomy.config import (
    DatabaseSettings,
    OpenAISettings,
    RerankerServerSettings,
    RerankerSettings,
)
from rag_anatomy.ports import DocumentParser
from rag_anatomy.services import IngestionService, Reranking, RetrievalService

_POOL_MAX_SIZE = 10
_DATABASE_TIMEOUT = 10.0
_CHUNK_SIZE = 400
_CHUNK_OVERLAP = 50
_EMBEDDING_BATCH_SIZE = min(
    256, MAX_INPUTS_PER_REQUEST, MAX_TOKENS_PER_REQUEST // _CHUNK_SIZE
)
_EMBEDDING_CONCURRENCY = 4
_CANDIDATE_POOL = 50
_RERANK_CONCURRENCY = 2
_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")

_logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Container:
    ingestion: IngestionService
    retrieval: RetrievalService


@asynccontextmanager
async def open_container(
    *,
    openai_transport: httpx2.AsyncBaseTransport | None = None,
    reranker_transport: httpx2.AsyncBaseTransport | None = None,
) -> AsyncIterator[Container]:
    database = DatabaseSettings()
    openai = OpenAISettings()
    reranker = RerankerSettings()
    server = RerankerServerSettings() if reranker.enabled else None
    async with connection_pool(
        database.dsn, max_size=_POOL_MAX_SIZE, timeout=_DATABASE_TIMEOUT
    ) as pool:
        await pool.wait(_DATABASE_TIMEOUT)
        store = PgVectorStore(pool)
        check_pgvector_version(await store.pgvector_version())
        await _check_embedding_space(store, openai)
        chunker = await asyncio.to_thread(
            TokenChunker,
            model=openai.embedding_model,
            chunk_size=_CHUNK_SIZE,
            overlap=_CHUNK_OVERLAP,
        )
        async with (
            _open_reranking(reranker, server, reranker_transport) as reranking,
            openai_client(
                openai.api_key.get_secret_value(),
                timeout=openai.timeout,
                max_retries=openai.max_retries,
                transport=openai_transport,
            ) as client,
        ):
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
                retrieval=RetrievalService(
                    embedder,
                    store,
                    store,
                    candidate_pool=_CANDIDATE_POOL,
                    reranking=reranking,
                ),
            )


@asynccontextmanager
async def _open_reranking(
    settings: RerankerSettings,
    server: RerankerServerSettings | None,
    transport: httpx2.AsyncBaseTransport | None,
) -> AsyncIterator[Reranking | None]:
    if server is None:
        _logger.info("reranking disabled")
        yield None
        return
    async with tei_client(
        server.url, timeout=settings.timeout, transport=transport
    ) as client:
        info = await server_info(client)
        check_reranker(info, model=server.model, revision=server.revision)
        _logger.info(
            "reranking with %s@%s, pool %d",
            info.model_id,
            info.model_sha,
            settings.candidate_pool,
        )
        yield Reranking(
            reranker=TeiReranker(
                client,
                batch_size=info.max_client_batch_size,
                max_concurrency=_RERANK_CONCURRENCY,
            ),
            pool=settings.candidate_pool,
        )


def _parser() -> CompositeParser:
    text = TextParser()
    parsers: dict[str, DocumentParser] = dict.fromkeys(TextParser.MEDIA_TYPES, text)
    parsers[PDF] = PdfParser()
    parsers[DOCX] = DocxParser()
    return CompositeParser(parsers)


def check_pgvector_version(installed: str) -> None:
    required = ".".join(map(str, MIN_PGVECTOR_VERSION))
    match = _VERSION.fullmatch(installed)
    if match is None:
        raise ConfigurationError(
            f"cannot parse pgvector version {installed!r}; vector search needs "
            f"{required} or later"
        )
    if tuple(map(int, match.groups())) < MIN_PGVECTOR_VERSION:
        raise ConfigurationError(
            f"pgvector {installed} is installed, but vector search needs "
            f"{required} or later for iterative index scans"
        )


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


def check_reranker(info: TeiInfo, *, model: str, revision: str) -> None:
    if not info.reranker:
        raise ConfigurationError(f"{info.model_id} is served, but not as a reranker")
    if (info.model_id, info.model_sha) != (model, revision):
        raise ConfigurationError(
            f"the reranker serves {info.model_id}@{info.model_sha}, "
            f"but {model}@{revision} is configured"
        )
