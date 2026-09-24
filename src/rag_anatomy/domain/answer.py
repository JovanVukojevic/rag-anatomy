from dataclasses import dataclass
from uuid import UUID

from rag_anatomy.domain._validation import check_page_range
from rag_anatomy.domain.retrieval import RetrievedChunk


@dataclass(frozen=True, slots=True, kw_only=True)
class Citation:
    chunk_id: UUID
    filename: str
    page_start: int
    page_end: int

    def __post_init__(self) -> None:
        check_page_range(self.page_start, self.page_end)


@dataclass(frozen=True, slots=True, kw_only=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    context: tuple[RetrievedChunk, ...]

    def __post_init__(self) -> None:
        seen = {retrieved.chunk.id for retrieved in self.context}
        unseen = [c.chunk_id for c in self.citations if c.chunk_id not in seen]
        if unseen:
            raise ValueError(f"citations reference chunks not in context: {unseen}")
