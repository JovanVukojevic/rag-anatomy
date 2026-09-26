import dataclasses
import math
from typing import Protocol
from uuid import UUID

import pytest

from rag_anatomy.adapters.driven.postgres import EMBEDDING_DIMENSIONS, PgVectorStore
from rag_anatomy.domain import (
    Chunk,
    DuplicateContentError,
    Embedding,
    FilenameConflictError,
    RetrievedChunk,
    StageRank,
)
from rag_anatomy.ports import DocumentRepository, KeywordSearch, VectorSearch
from tests.builders import make_chunks, make_document
from tests.fakes import FakeEmbedder, InMemoryStore


class Store(DocumentRepository, VectorSearch, KeywordSearch, Protocol): ...


@pytest.fixture(
    params=["memory", pytest.param("postgres", marks=pytest.mark.integration)]
)
def store(request: pytest.FixtureRequest) -> Store:
    if request.param == "memory":
        return InMemoryStore()
    postgres: PgVectorStore = request.getfixturevalue("pg_store")
    return postgres


embedder = FakeEmbedder(dimensions=EMBEDDING_DIMENSIONS)


async def _save(store: Store, *texts: str, filename: str = "guide.txt") -> None:
    document = make_document(filename)
    await store.save(
        document,
        make_chunks(document, *texts),
        await embedder.embed_documents(texts),
    )


def _texts(results: list[RetrievedChunk]) -> list[str]:
    return [r.chunk.text for r in results]


def _ids(results: list[RetrievedChunk]) -> list[UUID]:
    return [r.chunk.id for r in results]


def _assert_ranked(stages: list[StageRank | None]) -> list[float]:
    present = [stage for stage in stages if stage is not None]
    assert len(present) == len(stages)
    assert [stage.rank for stage in present] == list(range(1, len(present) + 1))
    scores = [stage.score for stage in present]
    assert scores == sorted(scores, reverse=True)
    return scores


# Neither this order nor its reverse is id order, so only an explicit tie-break by id
# can produce it; pgvector returns duplicate vectors in reverse insertion order.
def _scrambled(chunks: list[Chunk]) -> list[Chunk]:
    return [chunks[1], chunks[0], chunks[2], *chunks[3:]]


def _cosine_similarity(a: Embedding, b: Embedding) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / math.sqrt(
        math.fsum(x * x for x in a) * math.fsum(y * y for y in b)
    )


async def test_saved_document_is_found_by_hash_and_filename(store: Store) -> None:
    document = make_document("Guide.txt")
    await store.save(document, [], [])
    assert await store.find_by_hash(document.content_hash) == document
    assert await store.find_by_filename("guide.TXT") == document
    assert await store.find_by_filename("other.txt") is None


async def test_duplicate_content_is_rejected_with_existing_document(
    store: Store,
) -> None:
    original = make_document("a.txt", content=b"same")
    await store.save(original, [], [])
    with pytest.raises(DuplicateContentError) as raised:
        await store.save(make_document("b.txt", content=b"same"), [], [])
    assert raised.value.existing == original
    assert await store.find_by_filename("b.txt") is None


async def test_filename_conflict_is_rejected_case_insensitively(store: Store) -> None:
    original = make_document("Report.pdf", content=b"v1")
    await store.save(original, [], [])
    with pytest.raises(FilenameConflictError) as raised:
        await store.save(make_document("report.PDF", content=b"v2"), [], [])
    assert raised.value.existing == original


async def test_duplicate_content_takes_precedence_over_filename_conflict(
    store: Store,
) -> None:
    await store.save(make_document("a.txt", content=b"same"), [], [])
    with pytest.raises(DuplicateContentError):
        await store.save(make_document("a.txt", content=b"same"), [], [])


async def test_rejected_save_leaves_store_unchanged(store: Store) -> None:
    document = make_document()
    chunks = make_chunks(document, "orphaned text")
    with pytest.raises(ValueError):
        await store.save(document, chunks, [])
    assert await store.find_by_hash(document.content_hash) is None
    query = await embedder.embed_query("orphaned text")
    assert await store.vector_search(query, k=5) == []


async def test_empty_store_finds_nothing(store: Store) -> None:
    assert await store.vector_search(await embedder.embed_query("any"), k=5) == []
    assert await store.keyword_search("any", k=5) == []


async def test_vector_search_ranks_by_similarity_and_respects_k(store: Store) -> None:
    await _save(store, "dense retrieval", "keyword search", "reranking")
    results = await store.vector_search(
        await embedder.embed_query("keyword searching"), k=2
    )
    assert _texts(results)[0] == "keyword search"
    assert len(results) == 2
    _assert_ranked([r.dense for r in results])
    assert all((r.keyword, r.fusion, r.rerank) == (None, None, None) for r in results)
    assert {r.document.filename for r in results} == {"guide.txt"}


async def test_vector_search_orders_by_cosine_distance(store: Store) -> None:
    texts = [
        "chunk overlap",
        "chunking strategies",
        "cooking pasta",
        "chunks and pages",
        "reciprocal rank fusion",
        "chunk",
        "vector index",
        "overlapping chunks",
    ]
    await _save(store, *texts)
    query = await embedder.embed_query("chunking with overlap")
    vectors = await embedder.embed_documents(texts)
    similarity = {
        text: _cosine_similarity(query, vector)
        for text, vector in zip(texts, vectors, strict=True)
    }
    results = await store.vector_search(query, k=5)
    assert (
        _texts(results) == sorted(texts, key=similarity.__getitem__, reverse=True)[:5]
    )
    scores = _assert_ranked([r.dense for r in results])
    assert scores == pytest.approx(
        [similarity[text] for text in _texts(results)], abs=1e-6
    )


async def test_vector_search_breaks_ties_by_chunk_id(store: Store) -> None:
    document = make_document()
    chunks = _scrambled(make_chunks(document, *["same text"] * 3, "other text"))
    await store.save(
        document, chunks, await embedder.embed_documents([c.text for c in chunks])
    )
    results = await store.vector_search(await embedder.embed_query("same text"), k=2)
    tied = sorted(c.id for c in chunks if c.text == "same text")
    assert _ids(results) == tied[:2]


async def test_vector_search_returns_more_than_default_ef_search(
    store: Store,
) -> None:
    await _save(store, *[f"chunk number {n}" for n in range(60)])
    results = await store.vector_search(await embedder.embed_query("chunk"), k=45)
    assert len(results) == 45


async def test_vector_search_after_replace_returns_full_count(store: Store) -> None:
    old = make_document("report.pdf", content=b"v1")
    old_texts = [f"report section {n} of the old revision" for n in range(60)]
    await store.save(
        old, make_chunks(old, *old_texts), await embedder.embed_documents(old_texts)
    )
    new = make_document("report.pdf", content=b"v2")
    new_texts = [f"report section {n} of the new revision" for n in range(60)]
    await store.replace(
        old.id,
        new,
        make_chunks(new, *new_texts),
        await embedder.embed_documents(new_texts),
    )
    results = await store.vector_search(
        await embedder.embed_query("report section"), k=50
    )
    assert len(results) == 50
    assert {r.document for r in results} == {new}


async def test_retrieved_chunk_equals_saved_chunk(store: Store) -> None:
    document = make_document()
    [chunk] = make_chunks(document, "spans two pages")
    chunk = dataclasses.replace(chunk, position=3, page_start=2, page_end=3)
    await store.save(document, [chunk], await embedder.embed_documents([chunk.text]))
    [dense] = await store.vector_search(await embedder.embed_query(chunk.text), k=1)
    [keyword] = await store.keyword_search("pages", k=1)
    assert dense.chunk == keyword.chunk == chunk
    assert dense.document == keyword.document == document


async def test_keyword_search_returns_only_matching_chunks(store: Store) -> None:
    await _save(store, "hybrid search with fusion", "search engines", "unrelated text")
    results = await store.keyword_search("hybrid search", k=10)
    assert _texts(results) == ["hybrid search with fusion", "search engines"]
    assert all(score > 0 for score in _assert_ranked([r.keyword for r in results]))
    assert all((r.dense, r.fusion, r.rerank) == (None, None, None) for r in results)


async def test_keyword_search_breaks_ties_by_chunk_id(store: Store) -> None:
    document = make_document()
    chunks = _scrambled(make_chunks(document, *["tied words"] * 3, "unrelated"))
    await store.save(
        document, chunks, await embedder.embed_documents([c.text for c in chunks])
    )
    results = await store.keyword_search("tied", k=2)
    tied = sorted(c.id for c in chunks if c.text == "tied words")
    assert _ids(results) == tied[:2]


async def test_keyword_search_answers_natural_language_questions(
    store: Store,
) -> None:
    await _save(
        store,
        "Hybrid search combines keyword and dense retrieval.",
        "Notes about cooking pasta.",
    )
    results = await store.keyword_search("What is hybrid search?", k=10)
    assert _texts(results) == ["Hybrid search combines keyword and dense retrieval."]


async def test_keyword_search_prefers_breadth_over_repetition(store: Store) -> None:
    await _save(store, "search search search search", "hybrid search")
    results = await store.keyword_search("hybrid search", k=10)
    assert _texts(results) == ["hybrid search", "search search search search"]


@pytest.mark.parametrize(
    ("query", "text"),
    [
        ("sta je cena", "Šta je cena"),
        ("Šta je cena", "sta je cena"),
        ("dorde", "Đorđe"),
    ],
)
async def test_keyword_search_ignores_accents(
    store: Store, query: str, text: str
) -> None:
    await _save(store, text, "unrelated text")
    results = await store.keyword_search(query, k=10)
    assert _texts(results) == [text]


async def test_keyword_search_without_words_finds_nothing(store: Store) -> None:
    await _save(store, "some text")
    assert await store.keyword_search("?! ...", k=10) == []


async def test_replace_swaps_document_and_chunks_atomically(store: Store) -> None:
    old = make_document("report.pdf", content=b"v1")
    await store.save(
        old,
        make_chunks(old, "stale figures"),
        await embedder.embed_documents(["stale figures"]),
    )
    new = make_document("Report.pdf", content=b"v2")
    await store.replace(
        old.id,
        new,
        make_chunks(new, "fresh figures"),
        await embedder.embed_documents(["fresh figures"]),
    )
    assert await store.find_by_hash(old.content_hash) is None
    assert await store.find_by_filename("report.pdf") == new
    results = await store.keyword_search("figures", k=10)
    assert [(r.chunk.text, r.document.filename) for r in results] == [
        ("fresh figures", "Report.pdf")
    ]


async def test_replace_rejects_duplicate_content_and_leaves_store_unchanged(
    store: Store,
) -> None:
    old = make_document("report.pdf", content=b"v1")
    other = make_document("other.pdf", content=b"v2")
    await store.save(old, [], [])
    await store.save(other, [], [])
    with pytest.raises(DuplicateContentError) as raised:
        await store.replace(old.id, make_document("report.pdf", content=b"v2"), [], [])
    assert raised.value.existing == other
    assert await store.find_by_filename("report.pdf") == old


async def test_replace_rejects_filename_of_another_document(store: Store) -> None:
    old = make_document("report.pdf", content=b"v1")
    other = make_document("other.pdf", content=b"v2")
    await store.save(old, [], [])
    await store.save(other, [], [])
    with pytest.raises(FilenameConflictError) as raised:
        await store.replace(old.id, make_document("OTHER.pdf", content=b"v3"), [], [])
    assert raised.value.existing == other


async def test_replace_of_unknown_document_raises(store: Store) -> None:
    with pytest.raises(LookupError):
        await store.replace(make_document().id, make_document(), [], [])
