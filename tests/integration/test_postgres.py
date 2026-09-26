import asyncio
from collections.abc import AsyncIterator

import pytest
from pgvector import Vector

from rag_anatomy.adapters.driven.postgres import (
    EMBEDDING_DIMENSIONS,
    PgVectorStore,
    connection_pool,
)
from rag_anatomy.adapters.driven.postgres.store import (
    KEYWORD_SEARCH_SQL,
    VECTOR_SEARCH_SQL,
    Pool,
)
from rag_anatomy.config import DatabaseSettings
from rag_anatomy.domain import DuplicateContentError, Embedding
from tests.builders import make_chunks, make_document
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

embedder = FakeEmbedder(dimensions=EMBEDDING_DIMENSIONS)


def _axis(index: int) -> Embedding:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = 1.0
    return vector


async def _plan(pool: Pool, sql: str, params: dict[str, object]) -> str:
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("SET LOCAL enable_seqscan = off")
        cursor = await conn.execute(f"EXPLAIN {sql}", params)
        return "\n".join(str(row[0]) for row in await cursor.fetchall())


@pytest.fixture
async def index_only_store(
    database: DatabaseSettings, pg_pool: Pool
) -> AsyncIterator[PgVectorStore]:
    async with pg_pool.connection() as conn:
        await conn.execute("ALTER ROLE CURRENT_USER SET enable_seqscan = off")
    try:
        async with connection_pool(database.dsn, max_size=1, timeout=10) as pool:
            yield PgVectorStore(pool)
    finally:
        async with pg_pool.connection() as conn:
            await conn.execute("ALTER ROLE CURRENT_USER RESET enable_seqscan")


async def test_vector_score_is_cosine_similarity(pg_store: PgVectorStore) -> None:
    document = make_document()
    await pg_store.save(
        document, make_chunks(document, "first", "second"), [_axis(0), _axis(1)]
    )
    results = await pg_store.vector_search(_axis(0), k=2)
    assert [r.chunk.text for r in results] == ["first", "second"]
    assert [r.score for r in results] == pytest.approx([1.0, 0.0])


async def test_searches_use_their_indexes(pg_pool: Pool) -> None:
    dense = await _plan(
        pg_pool, VECTOR_SEARCH_SQL, {"embedding": Vector(_axis(0)), "k": 5}
    )
    keyword = await _plan(
        pg_pool, KEYWORD_SEARCH_SQL, {"query": "hybrid search", "k": 5}
    )
    assert "chunks_embedding_idx" in dense
    assert "chunks_tsv_idx" in keyword


async def test_vector_search_returns_more_than_default_ef_search(
    index_only_store: PgVectorStore,
) -> None:
    texts = [f"chunk number {n}" for n in range(60)]
    document = make_document()
    await index_only_store.save(
        document, make_chunks(document, *texts), await embedder.embed_documents(texts)
    )
    results = await index_only_store.vector_search(
        await embedder.embed_query("chunk"), k=45
    )
    assert len(results) == 45


async def test_concurrent_saves_of_same_content_admit_one(
    pg_store: PgVectorStore,
) -> None:
    documents = [make_document(name, content=b"same") for name in ("a.txt", "b.txt")]

    async def attempt(index: int) -> DuplicateContentError | None:
        try:
            await pg_store.save(documents[index], [], [])
        except DuplicateContentError as error:
            return error
        return None

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(attempt(i)) for i in range(2)]
    outcomes = [task.result() for task in tasks]
    [winner] = [
        d for d, outcome in zip(documents, outcomes, strict=True) if not outcome
    ]
    [error] = [outcome for outcome in outcomes if outcome]
    assert error.existing == winner
