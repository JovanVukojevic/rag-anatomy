import asyncio
import json
from typing import TYPE_CHECKING, Any

import httpx2

from tests.fakes._text import overlap

RERANKER_MODEL = "onnx-community/bge-reranker-v2-m3-ONNX"
RERANKER_REVISION = "6f5ff65298512715a1e669753bc754d2bc8f367b"


class FakeRerankAPI:
    def __init__(
        self,
        *,
        model_id: str = RERANKER_MODEL,
        model_sha: str | None = RERANKER_REVISION,
        reranker: bool = True,
        max_client_batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.model_sha = model_sha
        self.reranker = reranker
        self.max_client_batch_size = max_client_batch_size
        self.scripted: list[httpx2.Response | httpx2.TransportError] = []
        self.requests: list[httpx2.Request] = []
        self.max_in_flight = 0
        self._in_flight = 0

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.method == "POST"]

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(0)
            if request.url.path == "/info":
                return self._info()
            if self.scripted:
                outcome = self.scripted.pop(0)
                if isinstance(outcome, httpx2.TransportError):
                    raise outcome
                return outcome
            return self._rerank(json.loads(request.content))
        finally:
            self._in_flight -= 1

    def _info(self) -> httpx2.Response:
        model_type: dict[str, Any] = (
            {"reranker": {"id2label": {"0": "LABEL_0"}}}
            if self.reranker
            else {"embedding": {"pooling": "cls"}}
        )
        return httpx2.Response(
            200,
            json={
                "model_id": self.model_id,
                "model_sha": self.model_sha,
                "model_type": model_type,
                "max_client_batch_size": self.max_client_batch_size,
            },
        )

    def _rerank(self, body: dict[str, Any]) -> httpx2.Response:
        texts: list[str] = body["texts"]
        if len(texts) > self.max_client_batch_size:
            return httpx2.Response(
                422,
                json={
                    "error": f"batch size {len(texts)} > maximum allowed batch size "
                    f"{self.max_client_batch_size}",
                    "error_type": "Validation",
                },
            )
        scores = [
            {"index": index, "score": overlap(body["query"], text)}
            for index, text in enumerate(texts)
        ]
        scores.sort(key=lambda item: -item["score"])
        return httpx2.Response(200, json=scores)


if TYPE_CHECKING:
    _ = httpx2.MockTransport(FakeRerankAPI())
