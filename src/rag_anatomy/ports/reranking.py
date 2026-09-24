from collections.abc import Sequence
from typing import Protocol

from rag_anatomy.domain import RetrievedChunk


class Reranker(Protocol):
    """Rescores candidates against the query and returns the best k, best first."""

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]: ...
