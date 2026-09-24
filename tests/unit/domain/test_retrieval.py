import math

import pytest

from rag_anatomy.domain import RetrievedChunk, Retriever
from tests.builders import make_chunks, make_document


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_retrieved_chunk_rejects_non_finite_score(score: float) -> None:
    [chunk] = make_chunks(make_document(), "text")
    with pytest.raises(ValueError):
        RetrievedChunk(
            chunk=chunk, filename="guide.txt", score=score, retriever=Retriever.DENSE
        )


def test_retrieved_chunk_allows_negative_score() -> None:
    [chunk] = make_chunks(make_document(), "text")
    retrieved = RetrievedChunk(
        chunk=chunk, filename="guide.txt", score=-0.2, retriever=Retriever.DENSE
    )
    assert retrieved.score == -0.2
