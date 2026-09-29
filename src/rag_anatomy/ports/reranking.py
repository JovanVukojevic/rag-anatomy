from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    """Scores each text against the query, in input order; higher is better, within one call."""

    async def score(self, query: str, texts: Sequence[str]) -> list[float]: ...
