from uuid import uuid7

import pytest

from rag_anatomy.domain import Answer, Citation, RetrievedChunk
from tests.builders import make_document, make_retrieved


def _context() -> tuple[RetrievedChunk, ...]:
    return tuple(make_retrieved(make_document(), "first", "second"))


def test_answer_accepts_citations_from_context() -> None:
    context = _context()
    citation = Citation(
        chunk_id=context[1].chunk.id, filename="guide.txt", page_start=1, page_end=1
    )
    answer = Answer(text="second", citations=(citation,), context=context)
    assert answer.citations == (citation,)


def test_answer_rejects_citation_of_unseen_chunk() -> None:
    citation = Citation(
        chunk_id=uuid7(), filename="guide.txt", page_start=1, page_end=1
    )
    with pytest.raises(ValueError, match="not in context"):
        Answer(text="made up", citations=(citation,), context=_context())


@pytest.mark.parametrize(("page_start", "page_end"), [(0, 1), (2, 1)])
def test_citation_rejects_invalid_page_range(page_start: int, page_end: int) -> None:
    with pytest.raises(ValueError):
        Citation(
            chunk_id=uuid7(),
            filename="guide.txt",
            page_start=page_start,
            page_end=page_end,
        )
