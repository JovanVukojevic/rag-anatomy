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


async def test_calls_are_recorded_even_when_rejected() -> None:
    parser = FakeParser()
    await parser.parse(b"text", "text/plain")
    with pytest.raises(UnsupportedMediaTypeError):
        await parser.parse(b"%PDF", "application/pdf")
    assert parser.calls == ["text/plain", "application/pdf"]
