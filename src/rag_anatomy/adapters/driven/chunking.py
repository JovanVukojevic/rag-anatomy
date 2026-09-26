import re
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID, uuid7

import tiktoken

from rag_anatomy.domain import Chunk, Page
from rag_anatomy.ports import Chunker

_PAGE_SEPARATOR = "\n\n"
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_BREAKS = (_PARAGRAPH_BREAK, re.compile(r"(?<=[.!?])\s+"), re.compile(r"\s+"))
_WORD = re.compile(r"\S+")

type _Span = tuple[int, int]


@dataclass(frozen=True, slots=True)
class _Words:
    starts: list[int]
    sentence_starts: list[int]

    @classmethod
    def of(cls, text: str) -> _Words:
        starts: list[int] = []
        sentence_starts: list[int] = []
        previous_end: int | None = None
        for match in _WORD.finditer(text):
            starts.append(match.start())
            if (
                previous_end is None
                or text[previous_end - 1] in ".!?"
                or _PARAGRAPH_BREAK.search(text, previous_end, match.start())
            ):
                sentence_starts.append(match.start())
            previous_end = match.end()
        return cls(starts, sentence_starts)


class TokenChunker:
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if not 0 <= overlap < chunk_size:
            raise ValueError(f"overlap must be in [0, {chunk_size}), got {overlap}")
        self._encoding = tiktoken.encoding_for_model(model)
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, document_id: UUID, pages: Sequence[Page]) -> list[Chunk]:
        text = _PAGE_SEPARATOR.join(page.text for page in pages)
        page_offsets: list[int] = []
        offset = 0
        for page in pages:
            page_offsets.append(offset)
            offset += len(page.text) + len(_PAGE_SEPARATOR)

        spans = self._split(text, 0, len(text), level=0)
        chunks: list[Chunk] = []
        for position, window in enumerate(self._merge(text, spans)):
            start, end = _trim(text, *window)
            chunks.append(
                Chunk(
                    id=uuid7(),
                    document_id=document_id,
                    text=text[start:end],
                    position=position,
                    page_start=pages[bisect_right(page_offsets, start) - 1].number,
                    page_end=pages[bisect_right(page_offsets, end - 1) - 1].number,
                )
            )
        return chunks

    def _count(self, text: str) -> int:
        return len(self._encoding.encode_ordinary(text))

    def _split(self, text: str, start: int, end: int, *, level: int) -> list[_Span]:
        start, end = _trim(text, start, end)
        if start == end:
            return []
        limit = self._chunk_size - self._overlap
        if self._count(text[start:end]) <= limit:
            return [(start, end)]
        if level == len(_BREAKS):
            return self._split_characters(text, start, end, limit)
        spans: list[_Span] = []
        position = start
        for match in _BREAKS[level].finditer(text, start, end):
            spans += self._split(text, position, match.start(), level=level + 1)
            position = match.end()
        spans += self._split(text, position, end, level=level + 1)
        return spans

    def _split_characters(
        self, text: str, start: int, end: int, limit: int
    ) -> list[_Span]:
        spans: list[_Span] = []
        while start < end:
            low, high = start + 1, end
            while low < high:
                middle = (low + high + 1) // 2
                if self._count(text[start:middle]) <= limit:
                    low = middle
                else:
                    high = middle - 1
            spans.append((start, low))
            start = low
        return spans

    def _merge(self, text: str, spans: list[_Span]) -> list[_Span]:
        words = _Words.of(text)
        windows: list[_Span] = []
        i = 0
        while i < len(spans):
            first_start, first_end = spans[i]
            start = first_start
            if windows and self._overlap:
                start = self._overlap_start(text, windows[-1], spans[i], words)
            tokens = self._count(text[start:first_end])
            j = i + 1
            while j < len(spans):
                step = self._count(text[spans[j - 1][1] : spans[j][1]])
                if tokens + step > self._chunk_size:
                    break
                tokens += step
                j += 1
            end = spans[j - 1][1]
            # BPE counts are not additive across span boundaries, so verify exactly.
            while j > i + 1 and self._count(text[start:end]) > self._chunk_size:
                j -= 1
                end = spans[j - 1][1]
            if j == i + 1 and self._count(text[start:end]) > self._chunk_size:
                start = first_start
            windows.append((start, end))
            i = j
        return windows

    def _overlap_start(
        self, text: str, previous: _Span, following: _Span, words: _Words
    ) -> int:
        previous_start, previous_end = previous
        next_start, next_end = following
        budget = min(
            self._overlap,
            self._chunk_size - self._count(text[previous_end:next_end]),
        )
        candidates = words.starts[
            bisect_right(words.starts, previous_start) : bisect_left(
                words.starts, previous_end
            )
        ]
        low, high = 0, len(candidates)
        while low < high:
            middle = (low + high) // 2
            if self._count(text[candidates[middle] : previous_end]) <= budget:
                high = middle
            else:
                low = middle + 1
        if low == len(candidates):
            return next_start
        first_fit = candidates[low]
        k = bisect_left(words.sentence_starts, first_fit)
        if k < len(words.sentence_starts) and words.sentence_starts[k] < previous_end:
            return words.sentence_starts[k]
        return first_fit


def _trim(text: str, start: int, end: int) -> _Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


if TYPE_CHECKING:
    _: Chunker = TokenChunker()
