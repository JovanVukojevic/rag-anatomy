import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pypdfium2
import pypdfium2.raw

from rag_anatomy.adapters.driven.parsing._media_type import media_type_essence
from rag_anatomy.domain import (
    CorruptDocumentError,
    EncryptedDocumentError,
    Page,
    UnsupportedMediaTypeError,
    normalized,
)
from rag_anatomy.ports import DocumentParser

PDF = "application/pdf"

# PDFium is process-global and not thread-safe, so every call into it, including the
# closing of its objects, runs on this one thread (D43).
_PDFIUM = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdfium")
_PASSWORD_ERRORS = frozenset(
    {pypdfium2.raw.FPDF_ERR_PASSWORD, pypdfium2.raw.FPDF_ERR_SECURITY}
)


class PdfParser:
    MEDIA_TYPES = frozenset({PDF})

    async def parse(self, content: bytes, media_type: str) -> list[Page]:
        if media_type_essence(media_type) not in self.MEDIA_TYPES:
            raise UnsupportedMediaTypeError(media_type)
        loop = asyncio.get_running_loop()
        texts = await loop.run_in_executor(_PDFIUM, _page_texts, content)
        return [
            Page(number=number, text=text) for number, text in enumerate(texts, start=1)
        ]


def _page_texts(content: bytes) -> list[str]:
    try:
        document = pypdfium2.PdfDocument(content)
    except pypdfium2.PdfiumError as error:
        if error.err_code in _PASSWORD_ERRORS:
            raise EncryptedDocumentError(PDF) from error
        raise CorruptDocumentError(PDF) from error
    try:
        return [normalized(_page_text(document, i)) for i in range(len(document))]
    except pypdfium2.PdfiumError as error:
        raise CorruptDocumentError(PDF) from error
    finally:
        document.close()


def _page_text(document: pypdfium2.PdfDocument, index: int) -> str:
    page = document[index]
    try:
        text_page = page.get_textpage()
        try:
            return text_page.get_text_bounded()
        finally:
            text_page.close()
    finally:
        page.close()


if TYPE_CHECKING:
    _: DocumentParser = PdfParser()
