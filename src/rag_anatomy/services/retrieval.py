import asyncio
from dataclasses import dataclass, replace
from enum import StrEnum

from rag_anatomy.domain import EmptyQueryError, RetrievedChunk, StageRank, normalized
from rag_anatomy.ports import Embedder, KeywordSearch, Reranker, VectorSearch
from rag_anatomy.services.fusion import reciprocal_rank_fusion

MAX_TOP_K = 100


class RetrievalMode(StrEnum):
    DENSE = "dense"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True, kw_only=True)
class Reranking:
    reranker: Reranker
    pool: int

    def __post_init__(self) -> None:
        if not 1 <= self.pool <= MAX_TOP_K:
            raise ValueError(
                f"rerank pool must be between 1 and {MAX_TOP_K}, got {self.pool}"
            )


class RetrievalService:
    def __init__(
        self,
        embedder: Embedder,
        vector_search: VectorSearch,
        keyword_search: KeywordSearch,
        *,
        candidate_pool: int,
        reranking: Reranking | None,
    ) -> None:
        if candidate_pool < 1:
            raise ValueError(f"candidate_pool must be >= 1, got {candidate_pool}")
        self._embedder = embedder
        self._vector_search = vector_search
        self._keyword_search = keyword_search
        self._candidate_pool = candidate_pool
        self._reranking = reranking

    async def retrieve(
        self, query: str, top_k: int, mode: RetrievalMode = RetrievalMode.HYBRID
    ) -> list[RetrievedChunk]:
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
        query = normalized(query).strip()
        if not query:
            raise EmptyQueryError()
        if self._reranking is None:
            return await self._candidates(query, top_k, mode)
        pool = max(self._reranking.pool, top_k)
        candidates = await self._candidates(query, pool, mode)
        reranked = await self._rerank(self._reranking.reranker, query, candidates)
        return reranked[:top_k]

    async def _candidates(
        self, query: str, k: int, mode: RetrievalMode
    ) -> list[RetrievedChunk]:
        match mode:
            case RetrievalMode.DENSE:
                return await self._dense(query, k)
            case RetrievalMode.KEYWORD:
                return await self._keyword_search.keyword_search(query, k)
            case RetrievalMode.HYBRID:
                pool = max(self._candidate_pool, k)
                return (await self._hybrid(query, pool))[:k]

    async def _dense(self, query: str, k: int) -> list[RetrievedChunk]:
        embedding = await self._embedder.embed_query(query)
        return await self._vector_search.vector_search(embedding, k)

    async def _hybrid(self, query: str, k: int) -> list[RetrievedChunk]:
        try:
            async with asyncio.TaskGroup() as group:
                dense = group.create_task(self._dense(query, k))
                keyword = group.create_task(
                    self._keyword_search.keyword_search(query, k)
                )
        except ExceptionGroup as eg:
            # The first recorded failure is the one that cancelled the other search (D41).
            raise eg.exceptions[0] from eg
        return reciprocal_rank_fusion(dense.result(), keyword.result())

    async def _rerank(
        self, reranker: Reranker, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = await reranker.score(query, [hit.chunk.text for hit in candidates])
        # Ties keep the order of the stage before reranking (D53, D56).
        order = sorted(
            zip(scores, range(len(candidates)), candidates, strict=True),
            key=lambda item: (-item[0], item[1]),
        )
        return [
            replace(hit, rerank=StageRank(rank=rank, score=score))
            for rank, (score, _, hit) in enumerate(order, start=1)
        ]
