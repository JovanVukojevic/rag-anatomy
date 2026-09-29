import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

import httpx2
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from rag_anatomy.adapters.driven.tei.client import error_message
from rag_anatomy.domain import RerankError
from rag_anatomy.ports import Reranker


class _Score(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    index: int
    score: float


_SCORES = TypeAdapter(list[_Score])


class TeiReranker:
    def __init__(
        self, client: httpx2.AsyncClient, *, batch_size: int, max_concurrency: int
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self._client = client
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        if not query or not all(texts):
            raise ValueError("query and texts must not be empty")
        batches = [
            texts[start : start + self._batch_size]
            for start in range(0, len(texts), self._batch_size)
        ]
        results: list[list[float]] = [[] for _ in batches]
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def score_batch(slot: int, batch: Sequence[str]) -> None:
            async with semaphore:
                results[slot] = await self._score_batch(query, batch)

        try:
            async with asyncio.TaskGroup() as group:
                for slot, batch in enumerate(batches):
                    group.create_task(score_batch(slot, batch))
        except ExceptionGroup as eg:
            # Without "from eg", the error keeps its transport cause; the group stays in
            # __context__.
            raise eg.exceptions[0]  # noqa: B904
        return [score for batch in results for score in batch]

    async def _score_batch(self, query: str, texts: Sequence[str]) -> list[float]:
        try:
            response = await self._client.post(
                "/rerank",
                json={
                    "query": query,
                    "texts": list(texts),
                    "truncate": False,
                    "raw_scores": True,
                    "return_text": False,
                },
            )
        except httpx2.HTTPError as error:
            raise RerankError(f"TEI at {self._client.base_url}: {error!r}") from error
        if response.is_error:
            raise RerankError(
                f"TEI at {self._client.base_url} returned {response.status_code}: "
                f"{error_message(response)}"
            )
        try:
            items = _SCORES.validate_json(response.content)
        except ValidationError as error:
            raise RerankError(
                f"TEI returned an invalid rerank body: {error}"
            ) from error
        by_index = {item.index: item.score for item in items}
        if len(items) != len(texts) or sorted(by_index) != list(range(len(texts))):
            raise RerankError(
                f"TEI: expected indices 0..{len(texts) - 1}, "
                f"got {sorted(item.index for item in items)}"
            )
        return [by_index[index] for index in range(len(texts))]


if TYPE_CHECKING:
    _: Reranker = TeiReranker(httpx2.AsyncClient(), batch_size=1, max_concurrency=1)
