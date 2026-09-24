import hashlib
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rag_anatomy.domain import Embedding
from rag_anatomy.ports import Embedder
from tests.fakes._text import words


class FakeEmbedder:
    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> Embedding:
        return self._embed(text)

    def _embed(self, text: str) -> Embedding:
        vector = [0.0] * self.dimensions
        for word in words(text):
            padded = f"#{word}#"
            for start in range(len(padded) - 2):
                trigram = padded[start : start + 3].encode()
                value = int.from_bytes(hashlib.blake2b(trigram, digest_size=8).digest())
                index, sign_bit = divmod(value, 2)
                vector[index % self.dimensions] += 1.0 if sign_bit else -1.0
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector] if norm else vector


if TYPE_CHECKING:
    _: Embedder = FakeEmbedder()
