import asyncio
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from rag_anatomy.domain import Embedding
from rag_anatomy.ports import Embedder
from tests.fakes._text import words


def trigram_embedding(text: str, dimensions: int) -> Embedding:
    vector = [0.0] * dimensions
    for word in words(text):
        padded = f"#{word}#"
        for start in range(len(padded) - 2):
            trigram = padded[start : start + 3].encode()
            value = int.from_bytes(hashlib.blake2b(trigram, digest_size=8).digest())
            index, sign_bit = divmod(value, 2)
            vector[index % dimensions] += 1.0 if sign_bit else -1.0
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


class FakeEmbedder:
    def __init__(
        self,
        dimensions: int = 256,
        failures: Mapping[int, Exception] | None = None,
        query_failure: Exception | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.batches: list[list[str]] = []
        self.queries: list[str] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._calls = 0
        self._failures = dict(failures or {})
        self._query_failure = query_failure

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        call = self._calls
        self._calls += 1
        if call in self._failures:
            raise self._failures[call]
        self.batches.append(list(texts))
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(0)
            return [trigram_embedding(text, self.dimensions) for text in texts]
        finally:
            self._in_flight -= 1

    async def embed_query(self, text: str) -> Embedding:
        self.queries.append(text)
        if self._query_failure is not None:
            raise self._query_failure
        return trigram_embedding(text, self.dimensions)


if TYPE_CHECKING:
    _: Embedder = FakeEmbedder()
