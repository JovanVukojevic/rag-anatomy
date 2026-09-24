from dataclasses import replace
from typing import Any

import pytest

from tests.builders import make_chunks, make_document


def test_chunk_may_span_pages() -> None:
    [chunk] = make_chunks(make_document(), "spans two pages")
    assert replace(chunk, page_start=2, page_end=3).page_end == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"text": " \n"},
        {"position": -1},
        {"page_start": 0, "page_end": 0},
        {"page_start": 3, "page_end": 2},
    ],
)
def test_chunk_rejects_invalid_fields(changes: dict[str, Any]) -> None:
    [chunk] = make_chunks(make_document(), "text")
    with pytest.raises(ValueError):
        replace(chunk, **changes)
