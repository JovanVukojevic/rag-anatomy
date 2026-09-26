from itertools import pairwise
from uuid import uuid7

import pytest
import tiktoken

from rag_anatomy.adapters.driven.chunking import TokenChunker
from rag_anatomy.domain import Chunk, Page

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> int:
    return len(_ENCODING.encode_ordinary(text))


def _pages(*texts: str) -> list[Page]:
    return [Page(number=n, text=t) for n, t in enumerate(texts, start=1)]


def _sentences(topic: str, count: int) -> str:
    return " ".join(
        f"Note {i} on {topic} explains how stage {i % 7} ranks passages."
        for i in range(count)
    )


def _shared(previous: Chunk, current: Chunk) -> str:
    return max(
        (
            current.text[:k]
            for k in range(1, len(current.text) + 1)
            if previous.text.endswith(current.text[:k])
        ),
        key=len,
        default="",
    )


def test_single_short_page_is_one_chunk() -> None:
    document_id = uuid7()
    [chunk] = TokenChunker().chunk(document_id, _pages("  A short page.\n"))
    assert chunk.text == "A short page."
    assert chunk.document_id == document_id
    assert (chunk.position, chunk.page_start, chunk.page_end) == (0, 1, 1)


def test_chunk_spans_pages() -> None:
    [chunk] = TokenChunker().chunk(uuid7(), _pages("First page.", "Second page."))
    assert chunk.text == "First page.\n\nSecond page."
    assert (chunk.page_start, chunk.page_end) == (1, 2)


def test_empty_pages_are_skipped_but_stay_inside_the_range() -> None:
    [chunk] = TokenChunker().chunk(uuid7(), _pages("", "Alpha.", "", " ", "Omega."))
    assert (chunk.page_start, chunk.page_end) == (2, 5)


@pytest.mark.parametrize("texts", [(), ("",), ("", " \n", "\t")])
def test_pages_without_text_yield_no_chunks(texts: tuple[str, ...]) -> None:
    assert TokenChunker().chunk(uuid7(), _pages(*texts)) == []


def test_chunk_beginning_right_after_a_page_break_starts_on_that_page() -> None:
    first = "The first page describes dense retrieval with embeddings."
    second = "The second page describes keyword search with ts_rank."
    chunker = TokenChunker(chunk_size=_tokens(first) + 2, overlap=0)
    chunks = chunker.chunk(uuid7(), _pages(f"\n{first}\n\n  ", f" \n{second}\n"))
    assert [(c.text, c.page_start, c.page_end) for c in chunks] == [
        (first, 1, 1),
        (second, 2, 2),
    ]


def test_page_ranges_follow_the_text_across_pages() -> None:
    texts = [_sentences(f"page {n}", 12) for n in range(1, 5)]
    chunks = TokenChunker(chunk_size=60, overlap=10).chunk(uuid7(), _pages(*texts))
    assert len(chunks) > len(texts)
    for chunk in chunks:
        first_word, last_word = chunk.text.split()[0], chunk.text.split()[-1]
        assert first_word in texts[chunk.page_start - 1].split()
        assert last_word in texts[chunk.page_end - 1].split()
    starts = [chunk.page_start for chunk in chunks]
    assert starts == sorted(starts)
    assert (chunks[0].page_start, chunks[-1].page_end) == (1, 4)


def test_chunks_respect_the_token_budget() -> None:
    text = "\n\n".join(_sentences(f"topic {n}", 5 + 9 * n) for n in range(6))
    chunks = TokenChunker(chunk_size=50, overlap=10).chunk(uuid7(), _pages(text))
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert all(_tokens(chunk.text) <= 50 for chunk in chunks)


def test_long_paragraphs_overlap_at_realistic_sizes() -> None:
    paragraphs = [_sentences(f"topic {n}", 20) for n in range(5)]
    assert all(250 < _tokens(p) < 350 for p in paragraphs)
    chunks = TokenChunker(chunk_size=400, overlap=50).chunk(
        uuid7(), _pages("\n\n".join(paragraphs))
    )
    assert len(chunks) == len(paragraphs)
    for previous, current in pairwise(chunks):
        shared = _shared(previous, current)
        assert shared.startswith("Note ")
        assert 20 <= _tokens(shared) <= 50
        assert _tokens(current.text) <= 400


def test_overlap_falls_back_to_a_word_boundary_without_sentences() -> None:
    text = " ".join(f"token{i}" for i in range(300))
    chunks = TokenChunker(chunk_size=100, overlap=20).chunk(uuid7(), _pages(text))
    for previous, current in pairwise(chunks):
        shared = _shared(previous, current)
        assert shared == current.text[: len(shared)]
        assert shared.split()[0] in previous.text.split()
        assert 0 < _tokens(shared) <= 20


def test_zero_overlap_shares_no_text() -> None:
    text = " ".join(f"token{i}" for i in range(300))
    chunks = TokenChunker(chunk_size=100, overlap=0).chunk(uuid7(), _pages(text))
    assert " ".join(c.text for c in chunks) == text


def test_oversized_paragraphs_are_split_between_sentences() -> None:
    text = f"Short paragraph one.\n\n{_sentences('fusion', 6)}"
    chunks = TokenChunker(chunk_size=40, overlap=0).chunk(uuid7(), _pages(text))
    assert len(chunks) > 1
    assert all(c.text.startswith(("Short", "Note")) for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_oversized_word_is_split_to_fit() -> None:
    blob = "x" * 5000
    chunks = TokenChunker(chunk_size=30, overlap=5).chunk(uuid7(), _pages(blob))
    assert "".join(c.text for c in chunks) == blob
    assert all(_tokens(c.text) <= 30 for c in chunks)


def test_special_token_text_is_treated_as_plain_text() -> None:
    [chunk] = TokenChunker().chunk(uuid7(), _pages("before <|endoftext|> after"))
    assert chunk.text == "before <|endoftext|> after"


def test_chunking_is_deterministic() -> None:
    pages = _pages(*(_sentences(f"page {n}", 10) for n in range(3)))
    chunker = TokenChunker(chunk_size=40, overlap=8)
    first = chunker.chunk(uuid7(), pages)
    second = chunker.chunk(uuid7(), pages)
    assert [(c.text, c.page_start, c.page_end) for c in first] == [
        (c.text, c.page_start, c.page_end) for c in second
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"), [(0, 0), (10, 10), (10, -1), (10, 11)]
)
def test_invalid_sizes_are_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        TokenChunker(chunk_size=chunk_size, overlap=overlap)


@pytest.mark.parametrize(
    ("text", "chunk_size", "overlap"),
    [
        (" é- ?\t3日本!日本's  ??.x's!3's12", 7, 6),
        (
            "3..ab\t-\tx?!\t!\n-.?\téb-日本?  's    ababab--\nb.日本xéba?\n"
            "-ab12\t3\nabx's\né..\tx",
            15,
            14,
        ),
    ],
)
def test_budget_holds_when_token_counts_are_not_additive(
    text: str, chunk_size: int, overlap: int
) -> None:
    chunker = TokenChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk(uuid7(), _pages(text))
    assert chunks
    assert all(_tokens(c.text) <= chunk_size for c in chunks)
