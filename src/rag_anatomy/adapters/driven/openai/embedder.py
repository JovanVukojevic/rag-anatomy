from collections.abc import Sequence
from typing import TYPE_CHECKING

from openai import APIError, AsyncOpenAI
from openai.types import Embedding as OpenAIEmbedding

from rag_anatomy.domain import Embedding, EmbeddingError
from rag_anatomy.ports import Embedder

MAX_INPUTS_PER_REQUEST = 2048
MAX_TOKENS_PER_INPUT = 8192
MAX_TOKENS_PER_REQUEST = 300_000


class OpenAIEmbedder:
    def __init__(self, client: AsyncOpenAI, *, model: str, dimensions: int) -> None:
        if dimensions < 1:
            raise ValueError(f"dimensions must be >= 1, got {dimensions}")
        self._client = client
        self._model = model
        self._dimensions = dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        if not texts:
            return []
        if len(texts) > MAX_INPUTS_PER_REQUEST:
            raise ValueError(
                f"{len(texts)} inputs exceed the limit of {MAX_INPUTS_PER_REQUEST}"
            )
        if not all(texts):
            raise ValueError("inputs must not be empty")
        try:
            response = await self._client.embeddings.create(
                model=self._model, input=list(texts), dimensions=self._dimensions
            )
        except APIError as error:
            raise EmbeddingError(f"{self._model}: {error}") from error
        return self._in_input_order(response.data, len(texts))

    async def embed_query(self, text: str) -> Embedding:
        [embedding] = await self.embed_documents([text])
        return embedding

    def _in_input_order(
        self, data: Sequence[OpenAIEmbedding], count: int
    ) -> list[Embedding]:
        by_index = {item.index: item.embedding for item in data}
        if len(data) != count or sorted(by_index) != list(range(count)):
            raise EmbeddingError(
                f"{self._model}: expected indices 0..{count - 1}, "
                f"got {sorted(item.index for item in data)}"
            )
        for index, embedding in by_index.items():
            if len(embedding) != self._dimensions:
                raise EmbeddingError(
                    f"{self._model}: input {index} has {len(embedding)} dimensions, "
                    f"expected {self._dimensions}"
                )
        return [by_index[index] for index in range(count)]


if TYPE_CHECKING:
    _: Embedder = OpenAIEmbedder(AsyncOpenAI(api_key=""), model="", dimensions=1)
