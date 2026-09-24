from dataclasses import dataclass
from uuid import UUID

from rag_anatomy.domain._validation import check_page_range


@dataclass(frozen=True, slots=True, kw_only=True)
class Chunk:
    id: UUID
    document_id: UUID
    text: str
    position: int
    page_start: int
    page_end: int

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("chunk text must not be blank")
        if self.position < 0:
            raise ValueError(f"position must be >= 0, got {self.position}")
        check_page_range(self.page_start, self.page_end)
