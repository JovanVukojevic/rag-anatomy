import asyncio
from typing import TYPE_CHECKING

from rag_anatomy.adapters.driven.parsing._media_type import media_type_essence
from rag_anatomy.domain import (
    InvalidEncodingError,
    Page,
    UnsupportedMediaTypeError,
    normalized,
)
from rag_anatomy.ports import DocumentParser


class TextParser:
    MEDIA_TYPES = frozenset({"text/plain", "text/markdown"})

    async def parse(self, content: bytes, media_type: str) -> list[Page]:
        if media_type_essence(media_type) not in self.MEDIA_TYPES:
            raise UnsupportedMediaTypeError(media_type)
        text = await asyncio.to_thread(_decode_utf8, content)
        return [Page(number=1, text=text)]


def _decode_utf8(content: bytes) -> str:
    try:
        return normalized(content.decode("utf-8-sig"))
    except UnicodeDecodeError as error:
        raise InvalidEncodingError("UTF-8") from error


if TYPE_CHECKING:
    _: DocumentParser = TextParser()
