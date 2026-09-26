import pytest

from rag_anatomy.adapters.driven.parsing import CompositeParser, TextParser
from rag_anatomy.domain import InvalidEncodingError, Page, UnsupportedMediaTypeError
from tests.fakes import FakeParser


@pytest.mark.parametrize(
    "media_type",
    ["text/plain", "text/markdown", "text/plain; charset=utf-8", " Text/Markdown "],
)
async def test_text_is_one_page(media_type: str) -> None:
    pages = await TextParser().parse("# Título\n\nbody".encode(), media_type)
    assert pages == [Page(number=1, text="# Título\n\nbody")]


async def test_byte_order_mark_is_stripped() -> None:
    pages = await TextParser().parse(b"\xef\xbb\xbfhello", "text/plain")
    assert pages == [Page(number=1, text="hello")]


async def test_invalid_utf8_is_rejected() -> None:
    with pytest.raises(InvalidEncodingError):
        await TextParser().parse("café".encode("latin-1"), "text/plain")


async def test_text_parser_rejects_other_media_types() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        await TextParser().parse(b"%PDF", "application/pdf")


async def test_composite_dispatches_by_media_type_essence() -> None:
    fake = FakeParser()
    parser = CompositeParser({"Text/Plain": fake, "text/markdown": TextParser()})
    plain = await parser.parse(b"one\ftwo", "text/plain")
    markdown = await parser.parse(b"# Title", "TEXT/markdown; charset=utf-8")
    assert [page.text for page in plain] == ["one", "two"]
    assert markdown == [Page(number=1, text="# Title")]
    assert fake.calls == ["text/plain"]


async def test_composite_rejects_unregistered_media_types() -> None:
    parser = CompositeParser({"text/plain": TextParser()})
    with pytest.raises(UnsupportedMediaTypeError) as raised:
        await parser.parse(b"%PDF", "application/pdf")
    assert raised.value.media_type == "application/pdf"
