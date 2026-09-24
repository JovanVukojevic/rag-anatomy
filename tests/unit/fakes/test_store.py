import pytest

from rag_anatomy.domain import (
    DuplicateContentError,
    FilenameConflictError,
    Retriever,
)
from tests.builders import make_chunks, make_document
from tests.fakes import FakeEmbedder, InMemoryStore


async def _store_with(*texts: str) -> InMemoryStore:
    store = InMemoryStore()
    document = make_document()
    await store.save(
        document,
        make_chunks(document, *texts),
        await FakeEmbedder().embed_documents(texts),
    )
    return store


async def test_saved_document_is_found_by_hash_and_filename() -> None:
    store = InMemoryStore()
    document = make_document("Guide.txt")
    await store.save(document, [], [])
    assert await store.find_by_hash(document.content_hash) == document
    assert await store.find_by_filename("guide.TXT") == document
    assert await store.find_by_filename("other.txt") is None


async def test_duplicate_content_is_rejected_with_existing_document() -> None:
    store = InMemoryStore()
    original = make_document("a.txt", content=b"same")
    await store.save(original, [], [])
    with pytest.raises(DuplicateContentError) as raised:
        await store.save(make_document("b.txt", content=b"same"), [], [])
    assert raised.value.existing == original
    assert await store.find_by_filename("b.txt") is None


async def test_filename_conflict_is_rejected_case_insensitively() -> None:
    store = InMemoryStore()
    original = make_document("Report.pdf", content=b"v1")
    await store.save(original, [], [])
    with pytest.raises(FilenameConflictError) as raised:
        await store.save(make_document("report.PDF", content=b"v2"), [], [])
    assert raised.value.existing == original


async def test_duplicate_content_takes_precedence_over_filename_conflict() -> None:
    store = InMemoryStore()
    await store.save(make_document("a.txt", content=b"same"), [], [])
    with pytest.raises(DuplicateContentError):
        await store.save(make_document("a.txt", content=b"same"), [], [])


async def test_rejected_save_leaves_store_unchanged() -> None:
    store = InMemoryStore()
    document = make_document()
    chunks = make_chunks(document, "orphaned text")
    with pytest.raises(ValueError):
        await store.save(document, chunks, [])
    assert await store.find_by_hash(document.content_hash) is None
    assert await store.vector_search([1.0], k=5) == []


async def test_vector_search_ranks_by_similarity_and_respects_k() -> None:
    store = await _store_with("dense retrieval", "keyword search", "reranking")
    query = await FakeEmbedder().embed_query("keyword searching")
    results = await store.vector_search(query, k=2)
    assert results[0].chunk.text == "keyword search"
    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert {r.retriever for r in results} == {Retriever.DENSE}
    assert {r.filename for r in results} == {"guide.txt"}


async def test_keyword_search_returns_only_matching_chunks() -> None:
    store = await _store_with(
        "hybrid search with fusion", "search engines", "unrelated text"
    )
    results = await store.keyword_search("hybrid search", k=10)
    assert [(r.chunk.text, r.score) for r in results] == [
        ("hybrid search with fusion", 1.0),
        ("search engines", 0.5),
    ]
    assert {r.retriever for r in results} == {Retriever.KEYWORD}
