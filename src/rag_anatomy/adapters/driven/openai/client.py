import httpx2
from openai import AsyncOpenAI, DefaultAsyncHttpx2Client


def openai_client(
    api_key: str,
    *,
    timeout: float,
    max_retries: int,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> AsyncOpenAI:
    http_client = (
        None if transport is None else DefaultAsyncHttpx2Client(transport=transport)
    )
    return AsyncOpenAI(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        http_client=http_client,
    )
