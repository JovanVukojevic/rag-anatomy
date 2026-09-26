from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
import openai
import pytest

from rag_anatomy.adapters.driven.openai import (
    MAX_INPUTS_PER_REQUEST,
    OpenAIEmbedder,
    openai_client,
)
from rag_anatomy.domain import EmbeddingError
from tests.fakes import FakeEmbeddingsAPI, trigram_embedding

_MODEL = "text-embedding-3-small"
_DIMENSIONS = 8
_TIMEOUT = 5.0


@asynccontextmanager
async def _embedder(
    api: FakeEmbeddingsAPI, *, max_retries: int = 0
) -> AsyncIterator[OpenAIEmbedder]:
    async with openai_client(
        "test-key",
        timeout=_TIMEOUT,
        max_retries=max_retries,
        transport=httpx2.MockTransport(api),
    ) as client:
        yield OpenAIEmbedder(client, model=_MODEL, dimensions=_DIMENSIONS)


def _error(status: int) -> httpx2.Response:
    return httpx2.Response(
        status,
        headers={"retry-after-ms": "1"},
        json={"error": {"message": f"status {status}", "type": "test"}},
    )


def _embeddings(*indices: int) -> httpx2.Response:
    data = [
        {"object": "embedding", "index": i, "embedding": [0.0] * _DIMENSIONS}
        for i in indices
    ]
    return httpx2.Response(
        200,
        json={
            "object": "list",
            "data": data,
            "model": _MODEL,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        },
    )


async def test_results_follow_input_order_not_response_order() -> None:
    texts = ["dense retrieval", "keyword search", "rank fusion"]
    async with _embedder(FakeEmbeddingsAPI(reverse=True)) as embedder:
        embeddings = await embedder.embed_documents(texts)
    for embedding, text in zip(embeddings, texts, strict=True):
        assert embedding == pytest.approx(
            trigram_embedding(text, _DIMENSIONS), abs=1e-6
        )


async def test_one_request_carries_model_dimensions_and_every_input() -> None:
    api = FakeEmbeddingsAPI()
    async with _embedder(api) as embedder:
        await embedder.embed_documents(["first", "second"])
    [body] = api.bodies
    assert body["model"] == _MODEL
    assert body["dimensions"] == _DIMENSIONS
    assert body["input"] == ["first", "second"]


async def test_query_is_embedded_like_a_document() -> None:
    async with _embedder(FakeEmbeddingsAPI()) as embedder:
        query = await embedder.embed_query("hybrid search")
        [document] = await embedder.embed_documents(["hybrid search"])
    assert query == document


async def test_dimension_mismatch_is_rejected() -> None:
    async with _embedder(FakeEmbeddingsAPI(dimensions=4)) as embedder:
        with pytest.raises(EmbeddingError, match="4 dimensions"):
            await embedder.embed_documents(["text"])


@pytest.mark.parametrize("indices", [(0,), (0, 0), (0, 2)], ids=str)
async def test_missing_or_duplicate_indices_are_rejected(
    indices: tuple[int, ...],
) -> None:
    api = FakeEmbeddingsAPI()
    api.scripted.append(_embeddings(*indices))
    async with _embedder(api) as embedder:
        with pytest.raises(EmbeddingError, match="indices"):
            await embedder.embed_documents(["a", "b"])


@pytest.mark.parametrize("status", [400, 401])
async def test_permanent_failure_is_not_retried(status: int) -> None:
    api = FakeEmbeddingsAPI()
    api.scripted.append(_error(status))
    async with _embedder(api, max_retries=2) as embedder:
        with pytest.raises(EmbeddingError) as raised:
            await embedder.embed_documents(["text"])
    assert isinstance(raised.value.__cause__, openai.APIStatusError)
    assert len(api.requests) == 1


async def test_rate_limit_is_retried_by_the_sdk() -> None:
    api = FakeEmbeddingsAPI()
    api.scripted.append(_error(429))
    async with _embedder(api, max_retries=1) as embedder:
        [embedding] = await embedder.embed_documents(["text"])
    assert len(embedding) == _DIMENSIONS
    assert len(api.requests) == 2


async def test_exhausted_retries_raise_embedding_error() -> None:
    api = FakeEmbeddingsAPI()
    api.scripted.extend(_error(500) for _ in range(3))
    async with _embedder(api, max_retries=2) as embedder:
        with pytest.raises(EmbeddingError):
            await embedder.embed_documents(["text"])
    assert len(api.requests) == 3


async def test_timeout_is_configured_and_mapped() -> None:
    api = FakeEmbeddingsAPI()
    api.scripted.append(httpx2.ReadTimeout("read timed out"))
    async with _embedder(api) as embedder:
        with pytest.raises(EmbeddingError) as raised:
            await embedder.embed_documents(["text"])
    assert isinstance(raised.value.__cause__, openai.APITimeoutError)
    [request] = api.requests
    assert request.extensions["timeout"] == dict.fromkeys(
        ("connect", "read", "write", "pool"), _TIMEOUT
    )


async def test_no_inputs_send_no_request() -> None:
    api = FakeEmbeddingsAPI()
    async with _embedder(api) as embedder:
        assert await embedder.embed_documents([]) == []
    assert api.requests == []


async def test_inputs_beyond_api_limits_are_rejected_before_sending() -> None:
    api = FakeEmbeddingsAPI()
    async with _embedder(api) as embedder:
        with pytest.raises(ValueError, match="limit"):
            await embedder.embed_documents(["x"] * (MAX_INPUTS_PER_REQUEST + 1))
        with pytest.raises(ValueError, match="empty"):
            await embedder.embed_query("")
    assert api.requests == []


def test_dimensions_must_be_positive() -> None:
    with pytest.raises(ValueError):
        OpenAIEmbedder(
            openai.AsyncOpenAI(api_key="test-key"), model=_MODEL, dimensions=0
        )
