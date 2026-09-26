from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from rag_anatomy.domain import RetrievedChunk, StageRank
from rag_anatomy.ports import Reranker
from tests.fakes._text import overlap


class FakeReranker:
    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]:
        scored = sorted(
            ((overlap(query, c.chunk.text), c) for c in candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [
            replace(candidate, rerank=StageRank(rank=rank, score=score))
            for rank, (score, candidate) in enumerate(scored[:k], start=1)
        ]


if TYPE_CHECKING:
    _: Reranker = FakeReranker()
