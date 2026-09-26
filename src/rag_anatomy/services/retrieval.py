from rag_anatomy.domain import EmptyQueryError, RetrievedChunk, normalized
from rag_anatomy.ports import Embedder, VectorSearch

MAX_TOP_K = 100


class RetrievalService:
    def __init__(self, embedder: Embedder, vector_search: VectorSearch) -> None:
        self._embedder = embedder
        self._vector_search = vector_search

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
        query = normalized(query).strip()
        if not query:
            raise EmptyQueryError()
        embedding = await self._embedder.embed_query(query)
        return await self._vector_search.vector_search(embedding, top_k)
