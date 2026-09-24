from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_anatomy.ports import LLMClient


@dataclass(frozen=True, slots=True)
class Prompt:
    system: str
    user: str


class FakeLLMClient:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.prompts: list[Prompt] = []

    async def complete(self, system: str, user: str) -> str:
        self.prompts.append(Prompt(system=system, user=user))
        return self.response


if TYPE_CHECKING:
    _: LLMClient = FakeLLMClient()
