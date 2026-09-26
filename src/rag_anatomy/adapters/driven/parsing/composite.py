from collections.abc import Mapping
from typing import TYPE_CHECKING

from rag_anatomy.adapters.driven.parsing._media_type import media_type_essence
from rag_anatomy.domain import Page, UnsupportedMediaTypeError
from rag_anatomy.ports import DocumentParser


class CompositeParser:
    def __init__(self, parsers: Mapping[str, DocumentParser]) -> None:
        self._parsers = {media_type_essence(k): p for k, p in parsers.items()}

    async def parse(self, content: bytes, media_type: str) -> list[Page]:
        parser = self._parsers.get(media_type_essence(media_type))
        if parser is None:
            raise UnsupportedMediaTypeError(media_type)
        return await parser.parse(content, media_type)


if TYPE_CHECKING:
    _: DocumentParser = CompositeParser({})
