from typing import TYPE_CHECKING

from rag_anatomy.domain import Page, UnsupportedMediaTypeError
from rag_anatomy.ports import DocumentParser


class FakeParser:
    async def parse(self, content: bytes, media_type: str) -> list[Page]:
        if media_type != "text/plain":
            raise UnsupportedMediaTypeError(media_type)
        return [
            Page(number=number, text=text)
            for number, text in enumerate(content.decode().split("\f"), start=1)
        ]


if TYPE_CHECKING:
    _: DocumentParser = FakeParser()
