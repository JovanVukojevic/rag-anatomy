import math
from dataclasses import dataclass
from enum import StrEnum

from rag_anatomy.domain.chunk import Chunk

type Embedding = list[float]


class Retriever(StrEnum):
    DENSE = "dense"
    KEYWORD = "keyword"
    FUSION = "fusion"
    RERANK = "rerank"


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievedChunk:
    chunk: Chunk
    filename: str
    score: float
    retriever: Retriever

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError(f"score must be finite, got {self.score}")
