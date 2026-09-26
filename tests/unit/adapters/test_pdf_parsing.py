import asyncio

import pytest

from rag_anatomy.adapters.driven.chunking import TokenChunker
from rag_anatomy.adapters.driven.parsing import PdfParser
from rag_anatomy.domain import (
    CorruptDocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
    Page,
    UnsupportedMediaTypeError,
)
from rag_anatomy.services import IngestionService
from tests.builders import make_pdf
from tests.fakes import FakeEmbedder, InMemoryStore

_LATIN = "Šta je hibridna pretraga? Čačak, ćevapi, žaba i đak."
_CYRILLIC = "Шта је хибридна претрага? Ђак, Љубав, Њива, Џеп."


async def test_pages_are_numbered_from_one() -> None:
    pdf = make_pdf("Dense retrieval.", "Keyword search.", "Reciprocal rank fusion.")
    assert await PdfParser().parse(pdf, "application/pdf") == [
        Page(number=1, text="Dense retrieval."),
        Page(number=2, text="Keyword search."),
        Page(number=3, text="Reciprocal rank fusion."),
    ]


async def test_page_without_text_is_kept_empty() -> None:
    pages = await PdfParser().parse(make_pdf("one", "", "three"), "application/pdf")
    assert pages == [
        Page(number=1, text="one"),
        Page(number=2, text=""),
        Page(number=3, text="three"),
    ]


async def test_serbian_latin_and_cyrillic_survive() -> None:
    pages = await PdfParser().parse(make_pdf(_LATIN, _CYRILLIC), "application/pdf")
    assert [page.text for page in pages] == [_LATIN, _CYRILLIC]


async def test_line_breaks_are_normalized() -> None:
    pages = await PdfParser().parse(
        make_pdf("first line\nsecond line"), "application/pdf"
    )
    assert pages == [Page(number=1, text="first line\nsecond line")]


async def test_text_is_normalized_to_nfc() -> None:
    decomposed = "Šta je c\u030cudno"
    pages = await PdfParser().parse(make_pdf(decomposed), "application/pdf")
    assert pages == [Page(number=1, text="Šta je čudno")]


async def test_media_type_parameters_are_ignored() -> None:
    pages = await PdfParser().parse(make_pdf("text"), "Application/PDF; version=1.7")
    assert pages == [Page(number=1, text="text")]


async def test_password_protected_pdf_is_rejected() -> None:
    with pytest.raises(EncryptedDocumentError) as raised:
        await PdfParser().parse(make_pdf("secret", encrypted=True), "application/pdf")
    assert raised.value.media_type == "application/pdf"


@pytest.mark.parametrize(
    "content",
    [b"%PDF-1.7\n" + bytes(range(256)) * 4, b"just some text", b""],
    ids=["corrupt", "not-pdf", "empty"],
)
async def test_unreadable_pdf_is_rejected(content: bytes) -> None:
    with pytest.raises(CorruptDocumentError):
        await PdfParser().parse(content, "application/pdf")


async def test_other_media_types_are_rejected() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        await PdfParser().parse(make_pdf("text"), "text/plain")


async def test_concurrent_parses_are_serialized_correctly() -> None:
    parser = PdfParser()
    documents = [make_pdf(f"document {n}", f"page two of {n}") for n in range(8)]
    async with asyncio.TaskGroup() as group:
        tasks = [
            group.create_task(parser.parse(pdf, "application/pdf")) for pdf in documents
        ]
    for n, task in enumerate(tasks):
        assert [page.text for page in task.result()] == [
            f"document {n}",
            f"page two of {n}",
        ]


async def test_scanned_pdf_without_text_layer_is_an_empty_document() -> None:
    service = IngestionService(
        PdfParser(),
        TokenChunker(),
        FakeEmbedder(),
        InMemoryStore(),
        batch_size=16,
        max_concurrency=1,
    )
    with pytest.raises(EmptyDocumentError):
        await service.ingest(make_pdf("", ""), "scan.pdf", "application/pdf")
