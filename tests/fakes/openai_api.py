import base64
import json
import struct
from typing import TYPE_CHECKING, Any

import httpx2

from tests.fakes.embedder import trigram_embedding


class FakeEmbeddingsAPI:
    def __init__(self, *, dimensions: int | None = None, reverse: bool = False) -> None:
        self.dimensions = dimensions
        self.reverse = reverse
        self.scripted: list[httpx2.Response | httpx2.TransportError] = []
        self.requests: list[httpx2.Request] = []

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.scripted:
            outcome = self.scripted.pop(0)
            if isinstance(outcome, httpx2.TransportError):
                raise outcome
            return outcome
        body = json.loads(request.content)
        dimensions = self.dimensions or body["dimensions"]
        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": _encoded(
                    trigram_embedding(text, dimensions), body["encoding_format"]
                ),
            }
            for index, text in enumerate(body["input"])
        ]
        if self.reverse:
            data.reverse()
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": body["model"],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            },
        )


def _encoded(vector: list[float], encoding_format: str) -> list[float] | str:
    if encoding_format == "float":
        return vector
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()


if TYPE_CHECKING:
    _ = httpx2.MockTransport(FakeEmbeddingsAPI())
