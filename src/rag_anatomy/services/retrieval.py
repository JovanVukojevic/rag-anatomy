import asyncio
from enum import StrEnum

from rag_anatomy.domain import EmptyQueryError, RetrievedChunk, normalized
from rag_anatomy.ports import Embedder, KeywordSearch, VectorSearch
from rag_anatomy.services.fusion import reciprocal_rank_fusion

MAX_TOP_K = 100


class RetrievalMode(StrEnum):
    DENSE = "dense"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class RetrievalService:
    def __init__(
        self,
        embedder: Embedder,
        vector_search: VectorSearch,
        keyword_search: KeywordSearch,
        *,
        candidate_pool: int,
    ) -> None:
        if candidate_pool < 1:
            raise ValueError(f"candidate_pool must be >= 1, got {candidate_pool}")
        self._embedder = embedder
        self._vector_search = vector_search
        self._keyword_search = keyword_search
        self._candidate_pool = candidate_pool

    async def retrieve(
        self, query: str, top_k: int, mode: RetrievalMode = RetrievalMode.HYBRID
    ) -> list[RetrievedChunk]:
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
        query = normalized(query).strip()
        if not query:
            raise EmptyQueryError()
        match mode:
            case RetrievalMode.DENSE:
                return await self._dense(query, top_k)
            case RetrievalMode.KEYWORD:
                return await self._keyword_search.keyword_search(query, top_k)
            case RetrievalMode.HYBRID:
                pool = max(self._candidate_pool, top_k)
                return (await self._hybrid(query, pool))[:top_k]

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
