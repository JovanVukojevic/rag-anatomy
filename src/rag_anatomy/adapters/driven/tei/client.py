from dataclasses import dataclass
from typing import Any

import httpx2
from pydantic import BaseModel, ConfigDict, ValidationError

from rag_anatomy.domain import RerankError


@dataclass(frozen=True, slots=True, kw_only=True)
class TeiInfo:
    model_id: str
    model_sha: str | None
    reranker: bool
    max_client_batch_size: int


class _Info(BaseModel):
    model_config = ConfigDict(strict=True, protected_namespaces=())

    model_id: str
    model_sha: str | None
    model_type: dict[str, Any]
    max_client_batch_size: int


def tei_client(
    base_url: str,
    *,
    timeout: float,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)


async def server_info(client: httpx2.AsyncClient) -> TeiInfo:
    try:
        response = await client.get("/info")
        response.raise_for_status()
        info = _Info.model_validate_json(response.content)
    except httpx2.HTTPError as error:
        raise RerankError(f"TEI at {client.base_url}: {error!r}") from error
    except ValidationError as error:
        raise RerankError(
            f"TEI at {client.base_url} returned an invalid /info: {error}"
        ) from error
    return TeiInfo(
        model_id=info.model_id,
        model_sha=info.model_sha,
        reranker="reranker" in info.model_type,
        max_client_batch_size=info.max_client_batch_size,
    )


def error_message(response: httpx2.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    return str(body.get("error", body)) if isinstance(body, dict) else str(body)
