import math

import pytest

from rag_anatomy.domain import RetrievedChunk, StageRank
from tests.builders import make_chunks, make_document


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_stage_rank_rejects_non_finite_score(score: float) -> None:
    with pytest.raises(ValueError):
        StageRank(rank=1, score=score)


def test_stage_rank_allows_negative_score() -> None:
    assert StageRank(rank=1, score=-0.2).score == -0.2


def test_stage_rank_is_one_based() -> None:
    with pytest.raises(ValueError):
        StageRank(rank=0, score=1.0)


def test_retrieved_chunk_rejects_chunk_of_another_document() -> None:
    [chunk] = make_chunks(make_document("a.txt"), "text")
    with pytest.raises(ValueError, match="belongs to document"):
        RetrievedChunk(
            chunk=chunk,
            document=make_document("b.txt"),
            dense=StageRank(rank=1, score=1.0),
        )


def test_retrieved_chunk_needs_a_retriever_stage() -> None:
    document = make_document()
    [chunk] = make_chunks(document, "text")
    with pytest.raises(ValueError, match="dense or keyword"):
        RetrievedChunk(
            chunk=chunk,
            document=document,
            fusion=StageRank(rank=1, score=0.5),
            rerank=StageRank(rank=1, score=0.9),
        )


def test_retrieved_chunk_found_by_keyword_search_alone_is_valid() -> None:
    document = make_document()
    [chunk] = make_chunks(document, "text")
    retrieved = RetrievedChunk(
        chunk=chunk, document=document, keyword=StageRank(rank=3, score=0.1)
    )
    assert retrieved.dense is None
