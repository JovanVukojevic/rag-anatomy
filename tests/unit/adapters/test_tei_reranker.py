from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
import pytest

from rag_anatomy.adapters.driven.tei import (
    TeiInfo,
    TeiReranker,
    server_info,
    tei_client,
)
from rag_anatomy.domain import RerankError
from tests.fakes import RERANKER_MODEL, RERANKER_REVISION, FakeRerankAPI


@asynccontextmanager
async def _client(api: FakeRerankAPI) -> AsyncIterator[httpx2.AsyncClient]:
    async with tei_client(
        "http://tei.test", timeout=5.0, transport=httpx2.MockTransport(api)
    ) as client:
        yield client


@asynccontextmanager
async def _reranker(
    api: FakeRerankAPI, *, batch_size: int = 32, max_concurrency: int = 2
) -> AsyncIterator[TeiReranker]:
    async with _client(api) as client:
        yield TeiReranker(
            client, batch_size=batch_size, max_concurrency=max_concurrency
        )


def _scores(*items: tuple[object, object]) -> httpx2.Response:
    return httpx2.Response(
        200, json=[{"index": index, "score": score} for index, score in items]
    )


async def test_scores_follow_input_order_not_response_order() -> None:
    texts = ["no match", "rank fusion", "fusion"]
    async with _reranker(FakeRerankAPI()) as reranker:
        assert await reranker.score("rank fusion", texts) == [0.0, 1.0, 0.5]


async def test_request_asks_for_raw_untruncated_scores() -> None:
    api = FakeRerankAPI()
    async with _reranker(api) as reranker:
        await reranker.score("query", ["text"])
    assert api.bodies == [
        {
            "query": "query",
            "texts": ["text"],
            "truncate": False,
            "raw_scores": True,
            "return_text": False,
        }
    ]


async def test_texts_are_split_into_batches_with_their_offsets() -> None:
    api = FakeRerankAPI(max_client_batch_size=2)
    texts = ["fusion", "a", "rank fusion", "b", "rank"]
    async with _reranker(api, batch_size=2, max_concurrency=3) as reranker:
        scores = await reranker.score("rank fusion", texts)
    assert scores == [0.5, 0.0, 1.0, 0.0, 0.5]
    assert [body["texts"] for body in api.bodies] == [
        ["fusion", "a"],
        ["rank fusion", "b"],
        ["rank"],
    ]
    assert api.max_in_flight == 3


async def test_concurrency_is_bounded() -> None:
    api = FakeRerankAPI(max_client_batch_size=1)
    async with _reranker(api, batch_size=1, max_concurrency=2) as reranker:
        await reranker.score("query", [f"text {n}" for n in range(6)])
    assert len(api.bodies) == 6
    assert api.max_in_flight == 2


async def test_empty_input_sends_no_request() -> None:
    api = FakeRerankAPI()
    async with _reranker(api) as reranker:
        assert await reranker.score("query", []) == []
    assert api.requests == []


@pytest.mark.parametrize(("query", "texts"), [("", ["text"]), ("query", ["a", ""])])
async def test_empty_query_or_text_is_rejected(query: str, texts: list[str]) -> None:
    api = FakeRerankAPI()
    async with _reranker(api) as reranker:
        with pytest.raises(ValueError, match="must not be empty"):
            await reranker.score(query, texts)
    assert api.requests == []


@pytest.mark.parametrize(
    "response",
    [
        _scores((0, 1.0)),
        _scores((0, 1.0), (0, 0.5)),
        _scores((0, 1.0), (2, 0.5)),
        _scores((0, 1.0), (1, 0.5), (2, 0.1)),
    ],
    ids=["missing", "duplicate", "out-of-range", "extra"],
)
async def test_indices_must_be_exactly_the_inputs(response: httpx2.Response) -> None:
    api = FakeRerankAPI()
    api.scripted.append(response)
    async with _reranker(api) as reranker:
        with pytest.raises(RerankError, match=r"expected indices 0\.\.1"):
            await reranker.score("query", ["a", "b"])


@pytest.mark.parametrize(
    "response",
    [
        _scores((0, None)),
        _scores((0, "0.5")),
        _scores((0.0, 0.5)),
        httpx2.Response(200, json={"scores": [0.5]}),
        httpx2.Response(200, content=b"not json"),
    ],
    ids=["null", "string", "float-index", "object", "not-json"],
)
async def test_malformed_body_is_a_rerank_error(response: httpx2.Response) -> None:
    api = FakeRerankAPI()
    api.scripted.append(response)
    async with _reranker(api) as reranker:
        with pytest.raises(RerankError, match="invalid rerank body"):
            await reranker.score("query", ["a"])


async def test_non_finite_score_is_a_rerank_error() -> None:
    api = FakeRerankAPI()
    api.scripted.append(httpx2.Response(200, content=b'[{"index": 0, "score": NaN}]'))
    async with _reranker(api) as reranker:
        with pytest.raises(RerankError):
            await reranker.score("query", ["a"])


async def test_error_status_carries_the_tei_message() -> None:
    api = FakeRerankAPI(max_client_batch_size=1)
    async with _reranker(api, batch_size=2) as reranker:
        with pytest.raises(RerankError, match=r"422: batch size 2 > maximum"):
            await reranker.score("query", ["a", "b"])


@pytest.mark.parametrize("status", [424, 429, 503])
async def test_errors_are_not_retried(status: int) -> None:
    api = FakeRerankAPI()
    api.scripted.append(
        httpx2.Response(status, json={"error": "busy", "error_type": "Overloaded"})
    )
    async with _reranker(api) as reranker:
        with pytest.raises(RerankError, match=rf"{status}: busy"):
            await reranker.score("query", ["a"])
    assert len(api.bodies) == 1


@pytest.mark.parametrize(
    "error",
    [httpx2.ConnectError("refused"), httpx2.ReadTimeout("slow")],
    ids=["connect", "timeout"],
)
async def test_transport_failure_is_a_rerank_error(
    error: httpx2.TransportError,
) -> None:
    api = FakeRerankAPI()
    api.scripted.append(error)
    async with _reranker(api) as reranker:
        with pytest.raises(RerankError, match=r"tei\.test") as raised:
            await reranker.score("query", ["a"])
    assert raised.value.__cause__ is error


async def test_a_failed_batch_surfaces_as_itself() -> None:
    api = FakeRerankAPI(max_client_batch_size=1)
    api.scripted.append(httpx2.Response(503, json={"error": "unhealthy"}))
    async with _reranker(api, batch_size=1, max_concurrency=1) as reranker:
        with pytest.raises(RerankError, match="503") as raised:
            await reranker.score("query", ["a", "b", "c"])
    assert isinstance(raised.value.__context__, ExceptionGroup)


@pytest.mark.parametrize(("batch_size", "max_concurrency"), [(0, 1), (1, 0)])
async def test_limits_must_be_positive(batch_size: int, max_concurrency: int) -> None:
    async with _client(FakeRerankAPI()) as client:
        with pytest.raises(ValueError, match="must be >= 1"):
            TeiReranker(client, batch_size=batch_size, max_concurrency=max_concurrency)


async def test_server_info_reports_the_served_model() -> None:
    async with _client(FakeRerankAPI(max_client_batch_size=8)) as client:
        assert await server_info(client) == TeiInfo(
            model_id=RERANKER_MODEL,
            model_sha=RERANKER_REVISION,
            reranker=True,
            max_client_batch_size=8,
        )


async def test_server_info_recognises_an_embedding_model() -> None:
    async with _client(FakeRerankAPI(reranker=False)) as client:
        assert not (await server_info(client)).reranker


async def test_unreachable_server_is_a_rerank_error() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    async with tei_client(
        "http://tei.test", timeout=5.0, transport=httpx2.MockTransport(refuse)
    ) as client:
        with pytest.raises(RerankError, match=r"tei\.test"):
            await server_info(client)


async def test_invalid_info_is_a_rerank_error() -> None:
    def invalid(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"model_id": "x"})

    async with tei_client(
        "http://tei.test", timeout=5.0, transport=httpx2.MockTransport(invalid)
    ) as client:
        with pytest.raises(RerankError, match="invalid /info"):
            await server_info(client)
