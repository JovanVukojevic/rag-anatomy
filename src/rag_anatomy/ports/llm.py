from typing import Protocol


class LLMClient(Protocol):
    """Returns the model's completion for a system and a user message."""

    async def complete(self, system: str, user: str) -> str: ...
