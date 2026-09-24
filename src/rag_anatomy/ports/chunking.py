from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from rag_anatomy.domain import Chunk, Page


class Chunker(Protocol):
    """Splits a document's pages into ordered chunks; pure and CPU-bound."""

    def chunk(self, document_id: UUID, pages: Sequence[Page]) -> list[Chunk]: ...
