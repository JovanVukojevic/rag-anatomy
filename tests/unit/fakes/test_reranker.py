import pytest

from rag_anatomy.domain import RerankError
from tests.fakes import FakeReranker, RerankCall


async def test_scores_query_overlap_in_input_order() -> None:
    reranker = FakeReranker()
    scores = await reranker.score(
        "cross encoder", ["no match", "cross encoder", "encoder"]
    )
    assert scores == [0.0, 1.0, 0.5]
    assert reranker.calls == [
        RerankCall(
            query="cross encoder", texts=["no match", "cross encoder", "encoder"]
        )
    ]


async def test_failure_is_raised_after_recording_the_call() -> None:
    error = RerankError("reranker unavailable")
    reranker = FakeReranker(failure=error)
    with pytest.raises(RerankError) as raised:
        await reranker.score("query", ["text"])
    assert raised.value is error
    assert len(reranker.calls) == 1
