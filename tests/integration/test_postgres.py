import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from pgvector import Vector

from rag_anatomy.adapters.driven.postgres import (
    EMBEDDING_DIMENSIONS,
    MAX_SEARCH_K,
    EmbeddingSpace,
    PgVectorStore,
)
from rag_anatomy.adapters.driven.postgres.store import (
    KEYWORD_SEARCH_SQL,
    SEARCH_SETTINGS_SQL,
    VECTOR_SEARCH_SQL,
    Pool,
)
from rag_anatomy.domain import DuplicateContentError, Embedding
from tests.builders import make_chunks, make_document
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

embedder = FakeEmbedder(dimensions=EMBEDDING_DIMENSIONS)


def _axis(index: int) -> Embedding:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = 1.0
    return vector


type PlanNode = dict[str, Any]


async def _plan(pool: Pool, sql: str, params: dict[str, object]) -> PlanNode:
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(SEARCH_SETTINGS_SQL, {"ef_search": "40"})
        cursor = await conn.execute(f"EXPLAIN (FORMAT JSON) {sql}", params)
        row = await cursor.fetchone()
    assert row is not None
    plan: PlanNode = row[0][0]["Plan"]
    return plan


def _nodes(plan: PlanNode) -> Iterator[PlanNode]:
    yield plan
    for child in plan.get("Plans", []):
        yield from _nodes(child)


async def test_vector_score_is_cosine_similarity(pg_store: PgVectorStore) -> None:
    document = make_document()
    await pg_store.save(
        document, make_chunks(document, "first", "second"), [_axis(0), _axis(1)]
    )
    results = await pg_store.vector_search(_axis(0), k=2)
    assert [r.chunk.text for r in results] == ["first", "second"]
    assert [r.dense.score if r.dense else None for r in results] == pytest.approx(
        [1.0, 0.0]
    )


async def test_searches_use_their_indexes(pg_pool: Pool) -> None:
    dense = await _plan(
        pg_pool, VECTOR_SEARCH_SQL, {"embedding": Vector(_axis(0)), "k": 5}
    )
    keyword = await _plan(
        pg_pool, KEYWORD_SEARCH_SQL, {"query": "hybrid search", "k": 5}
    )
    [limit] = [node for node in _nodes(dense) if node["Node Type"] == "Limit"]
    [tie_break] = limit["Plans"]
    [scan] = tie_break["Plans"]
    assert tie_break["Node Type"] == "Incremental Sort"
    assert (scan["Node Type"], scan["Index Name"]) == (
        "Index Scan",
        "chunks_embedding_idx",
    )
    assert "chunks_tsv_idx" in {node.get("Index Name") for node in _nodes(keyword)}


@pytest.mark.parametrize("k", [0, MAX_SEARCH_K + 1])
async def test_vector_search_rejects_k_hnsw_cannot_serve(
    pg_store: PgVectorStore, k: int
) -> None:
    with pytest.raises(ValueError, match="k must be"):
        await pg_store.vector_search(_axis(0), k=k)


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


async def test_embedding_column_dimensions_match_the_migration(
    pg_store: PgVectorStore,
) -> None:
    assert await pg_store.embedding_column_dimensions() == EMBEDDING_DIMENSIONS


async def test_embedding_space_claim_replaces_it_while_no_chunks_exist(
    pg_store: PgVectorStore,
) -> None:
    await pg_store.claim_embedding_space("model-a", 1536)
    claimed = await pg_store.claim_embedding_space("model-b", 768)
    assert claimed == EmbeddingSpace(model="model-b", dimensions=768)


async def test_embedding_space_claim_keeps_it_once_chunks_exist(
    pg_store: PgVectorStore,
) -> None:
    await pg_store.claim_embedding_space("model-a", EMBEDDING_DIMENSIONS)
    document = make_document()
    await pg_store.save(document, make_chunks(document, "stored"), [_axis(0)])
    claimed = await pg_store.claim_embedding_space("model-b", EMBEDDING_DIMENSIONS)
    assert claimed == EmbeddingSpace(model="model-a", dimensions=EMBEDDING_DIMENSIONS)
