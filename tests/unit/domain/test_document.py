from dataclasses import replace
from datetime import datetime
from typing import Any

import pytest

from rag_anatomy.domain import Page
from tests.builders import make_document


def test_valid_document_is_created() -> None:
    assert make_document("notes.txt").filename == "notes.txt"


@pytest.mark.parametrize(
    "changes",
    [
        {"filename": "  "},
        {"media_type": ""},
        {"content_hash": "abc"},
        {"content_hash": "A" * 64},
        {"created_at": datetime(2026, 1, 1)},
    ],
)
def test_document_rejects_invalid_fields(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(make_document(), **changes)


def test_page_numbers_start_at_one() -> None:
    with pytest.raises(ValueError):
        Page(number=0, text="cover")


def test_page_text_may_be_empty() -> None:
    assert Page(number=1, text="").text == ""
