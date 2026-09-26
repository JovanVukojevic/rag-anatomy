import pytest

from rag_anatomy.domain import Document, EmptyQueryError
from rag_anatomy.services import MAX_TOP_K, RetrievalService
from tests.builders import make_chunks, make_document
from tests.fakes import FakeEmbedder, InMemoryStore


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def service(embedder: FakeEmbedder, store: InMemoryStore) -> RetrievalService:
    return RetrievalService(embedder, store)


async def _save(
    store: InMemoryStore, embedder: FakeEmbedder, document: Document, *texts: str
) -> None:
    await store.save(
        document, make_chunks(document, *texts), await embedder.embed_documents(texts)
    )


async def test_returns_the_nearest_chunks_with_their_document(
    service: RetrievalService, embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    document = make_document("retrieval.pdf")
    await _save(
        store,
        embedder,
        document,
        "cooking pasta at home",
        "dense retrieval embeds the query",
        "retrieval of dense vectors",
    )
    results = await service.retrieve("dense retrieval", top_k=2)
    assert [r.chunk.text for r in results] == [
        "retrieval of dense vectors",
        "dense retrieval embeds the query",
    ]
    assert [r.dense.rank if r.dense else None for r in results] == [1, 2]
    assert all(r.document == document for r in results)


async def test_returns_every_chunk_when_fewer_than_top_k_exist(
    service: RetrievalService, embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(store, embedder, make_document(), "one chunk", "two chunks")
    assert len(await service.retrieve("chunk", top_k=MAX_TOP_K)) == 2


async def test_empty_index_returns_nothing(service: RetrievalService) -> None:
    assert await service.retrieve("anything", top_k=5) == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "\x00"])
async def test_blank_query_is_rejected_before_embedding(
    service: RetrievalService, embedder: FakeEmbedder, query: str
) -> None:
    with pytest.raises(EmptyQueryError):
        await service.retrieve(query, top_k=5)
    assert embedder.queries == []


@pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1])
async def test_top_k_out_of_bounds_is_rejected(
    service: RetrievalService, embedder: FakeEmbedder, top_k: int
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        await service.retrieve("query", top_k=top_k)
    assert embedder.queries == []


async def test_query_is_normalized_like_documents(
    service: RetrievalService, embedder: FakeEmbedder
) -> None:
    await service.retrieve("  čemu\r\nsluži?  ", top_k=5)
    assert embedder.queries == ["čemu\nsluži?"]
