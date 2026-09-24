from typing import Protocol

from rag_anatomy.domain import Page


class DocumentParser(Protocol):
    """Extracts numbered pages from raw bytes; raises UnsupportedMediaTypeError."""

    async def parse(self, content: bytes, media_type: str) -> list[Page]: ...
