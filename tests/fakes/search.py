from typing import TYPE_CHECKING

from rag_anatomy.domain import Embedding, RetrievedChunk
from rag_anatomy.ports import KeywordSearch, VectorSearch


class FailingSearch:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def vector_search(self, embedding: Embedding, k: int) -> list[RetrievedChunk]:
        raise self._error

    async def keyword_search(self, query: str, k: int) -> list[RetrievedChunk]:
        raise self._error


if TYPE_CHECKING:
    _vector_search: VectorSearch = FailingSearch(RuntimeError())
    _keyword_search: KeywordSearch = FailingSearch(RuntimeError())
