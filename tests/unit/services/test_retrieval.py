import pytest

from rag_anatomy.domain import Document, EmbeddingError, EmptyQueryError, RerankError
from rag_anatomy.services import MAX_TOP_K, Reranking, RetrievalMode, RetrievalService
from tests.builders import make_chunks, make_document
from tests.fakes import FailingSearch, FakeEmbedder, FakeReranker, InMemoryStore


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def service(embedder: FakeEmbedder, store: InMemoryStore) -> RetrievalService:
    return RetrievalService(embedder, store, store, candidate_pool=10, reranking=None)


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
    results = await service.retrieve(
        "dense retrieval", top_k=2, mode=RetrievalMode.DENSE
    )
    assert [r.chunk.text for r in results] == [
        "retrieval of dense vectors",
        "dense retrieval embeds the query",
    ]
    assert [r.dense.rank if r.dense else None for r in results] == [1, 2]
    assert all(r.document == document for r in results)


async def test_each_mode_records_its_own_stages(
    service: RetrievalService, embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(
        store, embedder, make_document(), "hybrid search", "search engines", "pasta"
    )
    dense = await service.retrieve("hybrid search", top_k=2, mode=RetrievalMode.DENSE)
    keyword = await service.retrieve(
        "hybrid search", top_k=2, mode=RetrievalMode.KEYWORD
    )
    hybrid = await service.retrieve("hybrid search", top_k=2, mode=RetrievalMode.HYBRID)
    assert all(r.dense and not (r.keyword or r.fusion) for r in dense)
    assert all(r.keyword and not (r.dense or r.fusion) for r in keyword)
    assert [r.fusion.rank if r.fusion else None for r in hybrid] == [1, 2]
    assert hybrid[0].chunk.text == "hybrid search"
    assert hybrid[0].dense and hybrid[0].keyword


async def test_hybrid_is_the_default_mode(
    service: RetrievalService, embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(store, embedder, make_document(), "hybrid search")
    [result] = await service.retrieve("hybrid search", top_k=1)
    assert result.fusion is not None


async def test_keyword_mode_never_embeds_the_query(
    embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(store, embedder, make_document(), "hybrid search")
    service = RetrievalService(
        embedder,
        FailingSearch(AssertionError("vector search was called")),
        store,
        candidate_pool=10,
        reranking=None,
    )
    results = await service.retrieve("hybrid", top_k=5, mode=RetrievalMode.KEYWORD)
    assert [r.chunk.text for r in results] == ["hybrid search"]
    assert embedder.queries == []


async def test_keyword_search_gets_the_normalized_query(
    service: RetrievalService, embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(store, embedder, make_document(), "hybrid search")
    results = await service.retrieve("hy\x00brid", top_k=5, mode=RetrievalMode.KEYWORD)
    assert [r.chunk.text for r in results] == ["hybrid search"]


async def test_dense_only_and_keyword_only_hits_both_survive_fusion(
    embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(
        store,
        embedder,
        make_document(),
        "ranking fusions",
        "reranking fusions ranked",
        "rank of a pasta recipe with tomatoes, garlic, basil and olive oil",
    )
    service = RetrievalService(embedder, store, store, candidate_pool=1, reranking=None)
    results = await service.retrieve("rank fusion", top_k=2)
    assert [r.chunk.text for r in results] == [
        "ranking fusions",
        "rank of a pasta recipe with tomatoes, garlic, basil and olive oil",
    ]
    dense_only, keyword_only = results
    assert dense_only.dense and dense_only.keyword is None
    assert keyword_only.keyword and keyword_only.dense is None


async def test_hybrid_fetches_at_least_top_k_candidates(
    embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(
        store,
        embedder,
        make_document(),
        *(f"search result {n}" for n in range(5)),
    )
    service = RetrievalService(embedder, store, store, candidate_pool=1, reranking=None)
    assert len(await service.retrieve("search", top_k=4)) == 4


async def test_hybrid_truncates_the_candidate_pool_to_top_k(
    service: RetrievalService, embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(
        store,
        embedder,
        make_document(),
        *(f"search result {n}" for n in range(5)),
    )
    assert len(await service.retrieve("search", top_k=2)) == 2


async def test_hybrid_embedding_failure_surfaces_as_itself(
    store: InMemoryStore,
) -> None:
    error = EmbeddingError("embeddings unavailable")
    service = RetrievalService(
        FakeEmbedder(query_failure=error),
        store,
        store,
        candidate_pool=10,
        reranking=None,
    )
    with pytest.raises(EmbeddingError) as raised:
        await service.retrieve("query", top_k=5)
    assert raised.value is error
    assert isinstance(raised.value.__cause__, ExceptionGroup)


async def test_hybrid_keyword_failure_surfaces_as_itself(
    embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    error = TimeoutError("statement timeout")
    service = RetrievalService(
        embedder, store, FailingSearch(error), candidate_pool=10, reranking=None
    )
    with pytest.raises(TimeoutError) as raised:
        await service.retrieve("query", top_k=5)
    assert raised.value is error
    assert isinstance(raised.value.__cause__, ExceptionGroup)


def test_candidate_pool_must_be_positive(
    embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    with pytest.raises(ValueError, match="candidate_pool"):
        RetrievalService(embedder, store, store, candidate_pool=0, reranking=None)


@pytest.mark.parametrize("mode", list(RetrievalMode))
async def test_returns_every_chunk_when_fewer_than_top_k_exist(
    service: RetrievalService,
    embedder: FakeEmbedder,
    store: InMemoryStore,
    mode: RetrievalMode,
) -> None:
    await _save(store, embedder, make_document(), "one chunk", "two chunk")
    assert len(await service.retrieve("chunk", top_k=MAX_TOP_K, mode=mode)) == 2


@pytest.mark.parametrize("mode", list(RetrievalMode))
async def test_empty_index_returns_nothing(
    service: RetrievalService, mode: RetrievalMode
) -> None:
    assert await service.retrieve("anything", top_k=5, mode=mode) == []


@pytest.mark.parametrize("mode", list(RetrievalMode))
@pytest.mark.parametrize("query", ["", "   ", "\n\t", "\x00"])
async def test_blank_query_is_rejected_before_searching(
    embedder: FakeEmbedder, query: str, mode: RetrievalMode
) -> None:
    failing = FailingSearch(AssertionError("search was called"))
    service = RetrievalService(
        embedder, failing, failing, candidate_pool=10, reranking=None
    )
    with pytest.raises(EmptyQueryError):
        await service.retrieve(query, top_k=5, mode=mode)
    assert embedder.queries == []


@pytest.mark.parametrize("mode", list(RetrievalMode))
@pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1])
async def test_top_k_out_of_bounds_is_rejected(
    embedder: FakeEmbedder, top_k: int, mode: RetrievalMode
) -> None:
    failing = FailingSearch(AssertionError("search was called"))
    service = RetrievalService(
        embedder, failing, failing, candidate_pool=10, reranking=None
    )
    with pytest.raises(ValueError, match="top_k"):
        await service.retrieve("query", top_k=top_k, mode=mode)
    assert embedder.queries == []


@pytest.mark.parametrize("mode", [RetrievalMode.DENSE, RetrievalMode.HYBRID])
async def test_query_is_normalized_like_documents(
    service: RetrievalService, embedder: FakeEmbedder, mode: RetrievalMode
) -> None:
    await service.retrieve("  čemu\r\nsluži?  ", top_k=5, mode=mode)
    assert embedder.queries == ["čemu\nsluži?"]


def _reranked(
    embedder: FakeEmbedder,
    store: InMemoryStore,
    reranker: FakeReranker,
    *,
    pool: int,
    candidate_pool: int = 10,
) -> RetrievalService:
    return RetrievalService(
        embedder,
        store,
        store,
        candidate_pool=candidate_pool,
        reranking=Reranking(reranker=reranker, pool=pool),
    )


@pytest.mark.parametrize("mode", list(RetrievalMode))
async def test_without_reranking_no_result_has_a_rerank_stage(
    service: RetrievalService,
    embedder: FakeEmbedder,
    store: InMemoryStore,
    mode: RetrievalMode,
) -> None:
    await _save(store, embedder, make_document(), "rank fusion", "fusion of ranks")
    results = await service.retrieve("rank fusion", top_k=2, mode=mode)
    assert results
    assert all(r.rerank is None for r in results)


async def test_reranking_orders_by_score_and_truncates_to_top_k(
    embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(
        store,
        embedder,
        make_document(),
        "fusion of ranked lists",
        "reciprocal rank fusion merges lists",
        "rank fusion",
        "a pasta recipe",
    )
    service = _reranked(embedder, store, FakeReranker(), pool=4)
    results = await service.retrieve("reciprocal rank fusion", top_k=2)
    assert [r.chunk.text for r in results] == [
        "reciprocal rank fusion merges lists",
        "rank fusion",
    ]
    assert [(r.rerank.rank, r.rerank.score) if r.rerank else None for r in results] == [
        (1, 1.0),
        (2, 2 / 3),
    ]
    assert all(r.fusion for r in results)


@pytest.mark.parametrize("mode", list(RetrievalMode))
async def test_every_mode_sends_the_rerank_pool(
    embedder: FakeEmbedder, store: InMemoryStore, mode: RetrievalMode
) -> None:
    await _save(
        store, embedder, make_document(), *(f"search result {n}" for n in range(30))
    )
    reranker = FakeReranker()
    service = _reranked(embedder, store, reranker, pool=20)
    results = await service.retrieve("search", top_k=5, mode=mode)
    [call] = reranker.calls
    assert len(call.texts) == 20
    assert len(results) == 5


async def test_rerank_pool_grows_to_top_k(
    embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(
        store, embedder, make_document(), *(f"search result {n}" for n in range(10))
    )
    reranker = FakeReranker()
    service = _reranked(embedder, store, reranker, pool=3)
    results = await service.retrieve("search", top_k=8)
    assert len(reranker.calls[0].texts) == len(results) == 8


async def test_hybrid_reranks_the_top_of_the_fused_list(
    service: RetrievalService, embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(
        store,
        embedder,
        make_document(),
        *(f"hybrid search result {n}" for n in range(8)),
        "search engines",
        "hybrid cars",
    )
    fused = await service.retrieve("hybrid search", top_k=3)
    reranker = FakeReranker()
    await _reranked(embedder, store, reranker, pool=3).retrieve(
        "hybrid search", top_k=1
    )
    assert reranker.calls[0].texts == [r.chunk.text for r in fused]


async def test_ties_keep_the_order_before_reranking(
    embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(
        store, embedder, make_document(), "retrieval", "retriever", "cooking pasta"
    )
    service = _reranked(embedder, store, FakeReranker(), pool=3)
    results = await service.retrieve("retrieve", top_k=3, mode=RetrievalMode.DENSE)
    assert {r.rerank.score if r.rerank else None for r in results} == {0.0}
    assert [r.dense.rank if r.dense else None for r in results] == [1, 2, 3]
    assert [r.rerank.rank if r.rerank else None for r in results] == [1, 2, 3]


@pytest.mark.parametrize("mode", list(RetrievalMode))
async def test_reranker_is_not_called_for_an_empty_index(
    embedder: FakeEmbedder, store: InMemoryStore, mode: RetrievalMode
) -> None:
    reranker = FakeReranker()
    service = _reranked(embedder, store, reranker, pool=5)
    assert await service.retrieve("anything", top_k=5, mode=mode) == []
    assert reranker.calls == []


async def test_reranker_gets_the_normalized_query(
    embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(store, embedder, make_document(), "čemu služi")
    reranker = FakeReranker()
    service = _reranked(embedder, store, reranker, pool=5)
    await service.retrieve("  čemu\r\nsluži?  ", top_k=1, mode=RetrievalMode.DENSE)
    assert reranker.calls[0].query == "čemu\nsluži?"


async def test_rerank_failure_surfaces_as_itself(
    embedder: FakeEmbedder, store: InMemoryStore
) -> None:
    await _save(store, embedder, make_document(), "hybrid search")
    error = RerankError("reranker unavailable")
    service = _reranked(embedder, store, FakeReranker(failure=error), pool=5)
    with pytest.raises(RerankError) as raised:
        await service.retrieve("hybrid search", top_k=1)
    assert raised.value is error


@pytest.mark.parametrize("pool", [0, -1, MAX_TOP_K + 1])
def test_rerank_pool_out_of_bounds_is_rejected(pool: int) -> None:
    with pytest.raises(ValueError, match="rerank pool"):
        Reranking(reranker=FakeReranker(), pool=pool)
