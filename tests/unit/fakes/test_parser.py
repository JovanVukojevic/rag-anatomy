import pytest

from rag_anatomy.domain import Page, UnsupportedMediaTypeError
from tests.fakes import FakeParser


async def test_form_feeds_split_numbered_pages() -> None:
    pages = await FakeParser().parse(b"one\ftwo\f", "text/plain")
    assert pages == [
        Page(number=1, text="one"),
        Page(number=2, text="two"),
        Page(number=3, text=""),
    ]


async def test_other_media_types_are_unsupported() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        await FakeParser().parse(b"%PDF", "application/pdf")
