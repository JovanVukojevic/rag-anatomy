import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING

from rag_anatomy.domain import InvalidEncodingError, Page, UnsupportedMediaTypeError
from rag_anatomy.ports import DocumentParser


def media_type_essence(media_type: str) -> str:
    return media_type.partition(";")[0].strip().lower()


class TextParser:
    MEDIA_TYPES = frozenset({"text/plain", "text/markdown"})

    async def parse(self, content: bytes, media_type: str) -> list[Page]:
        if media_type_essence(media_type) not in self.MEDIA_TYPES:
            raise UnsupportedMediaTypeError(media_type)
        text = await asyncio.to_thread(_decode_utf8, content)
        return [Page(number=1, text=text)]


class CompositeParser:
    def __init__(self, parsers: Mapping[str, DocumentParser]) -> None:
        self._parsers = {media_type_essence(k): p for k, p in parsers.items()}

    async def parse(self, content: bytes, media_type: str) -> list[Page]:
        parser = self._parsers.get(media_type_essence(media_type))
        if parser is None:
            raise UnsupportedMediaTypeError(media_type)
        return await parser.parse(content, media_type)


def _decode_utf8(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise InvalidEncodingError("UTF-8") from error


if TYPE_CHECKING:
    _text: DocumentParser = TextParser()
    _composite: DocumentParser = CompositeParser({})
