import math
from dataclasses import dataclass

from rag_anatomy.domain.chunk import Chunk
from rag_anatomy.domain.document import Document

type Embedding = list[float]


@dataclass(frozen=True, slots=True, kw_only=True)
class StageRank:
    rank: int
    score: float

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if not math.isfinite(self.score):
            raise ValueError(f"score must be finite, got {self.score}")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievedChunk:
    chunk: Chunk
    document: Document
    dense: StageRank | None = None
    keyword: StageRank | None = None
    fusion: StageRank | None = None
    rerank: StageRank | None = None

    def __post_init__(self) -> None:
        if self.chunk.document_id != self.document.id:
            raise ValueError(
                f"chunk {self.chunk.id} belongs to document {self.chunk.document_id}, "
                f"not {self.document.id}"
            )
        if self.dense is None and self.keyword is None:
            raise ValueError(
                "a retrieved chunk must be found by a dense or keyword search"
            )
