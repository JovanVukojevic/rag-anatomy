from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_anatomy.ports import Reranker
from tests.fakes._text import overlap


@dataclass(frozen=True, slots=True)
class RerankCall:
    query: str
    texts: list[str]


class FakeReranker:
    def __init__(self, failure: Exception | None = None) -> None:
        self.calls: list[RerankCall] = []
        self._failure = failure

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append(RerankCall(query=query, texts=list(texts)))
        if self._failure is not None:
            raise self._failure
        return [overlap(query, text) for text in texts]


if TYPE_CHECKING:
    _: Reranker = FakeReranker()
