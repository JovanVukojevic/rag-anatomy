import asyncio
import io
import zipfile
from typing import TYPE_CHECKING

import docx
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table, _Cell, _Row
from docx.text.paragraph import Paragraph

from rag_anatomy.adapters.driven.parsing._normalize import media_type_essence
from rag_anatomy.domain import (
    CorruptDocumentError,
    EncryptedDocumentError,
    Page,
    UnsupportedMediaTypeError,
    normalized,
)
from rag_anatomy.ports import DocumentParser

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# Office encrypts OOXML by wrapping it in a Compound File with an "EncryptedPackage" stream.
_COMPOUND_FILE_MAGIC = bytes.fromhex("d0cf11e0a1b11ae1")
_ENCRYPTED_PACKAGE = "EncryptedPackage".encode("utf-16-le")


class DocxParser:
    MEDIA_TYPES = frozenset({DOCX})

    async def parse(self, content: bytes, media_type: str) -> list[Page]:
        if media_type_essence(media_type) not in self.MEDIA_TYPES:
            raise UnsupportedMediaTypeError(media_type)
        text = await asyncio.to_thread(_body_text, content)
        return [Page(number=1, text=text)]


def _body_text(content: bytes) -> str:
    if content.startswith(_COMPOUND_FILE_MAGIC) and _ENCRYPTED_PACKAGE in content:
        raise EncryptedDocumentError(DOCX)
    try:
        document = docx.Document(io.BytesIO(content))
    except (
        zipfile.BadZipFile,
        KeyError,
        ValueError,
        PackageNotFoundError,
        # lxml's XMLSyntaxError; lxml itself ships no type information.
        SyntaxError,
    ) as error:
        raise CorruptDocumentError(DOCX) from error
    blocks = (_block_text(block) for block in document.iter_inner_content())
    return normalized("\n\n".join(block for block in blocks if block.strip()))


def _block_text(block: Paragraph | Table) -> str:
    if isinstance(block, Paragraph):
        return block.text
    rows = (_row_text(row) for row in block.rows)
    return "\n".join(row for row in rows if row)


def _row_text(row: _Row) -> str:
    cells = row.cells
    texts: list[str] = []
    index = 0
    while index < len(cells):
        texts.append(_cell_text(cells[index]))
        index += cells[index].grid_span
    return " | ".join(texts) if any(texts) else ""


def _cell_text(cell: _Cell) -> str:
    return " ".join(
        word
        for block in cell.iter_inner_content()
        for word in _block_text(block).split()
    )


if TYPE_CHECKING:
    _: DocumentParser = DocxParser()
