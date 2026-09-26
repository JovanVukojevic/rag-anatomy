import hashlib
from dataclasses import dataclass

import pytest

from rag_anatomy.adapters.driven.chunking import TokenChunker
from rag_anatomy.domain import (
    DuplicateContentError,
    EmptyDocumentError,
    FilenameConflictError,
    UnsupportedMediaTypeError,
)
from rag_anatomy.services import ConflictPolicy, IngestionService
from tests.builders import make_document
from tests.fakes import FakeEmbedder, FakeParser, InMemoryStore

_TOPICS = (
    "Dense retrieval embeds the query and ranks passages by cosine similarity.",
    "Keyword search matches exact terms such as product codes and names.",
    "Reciprocal rank fusion merges several ranked lists without tuning weights.",
    "A cross-encoder reranker rescores the top candidates against the query.",
    "Chunk overlap keeps a sentence that straddles a boundary retrievable.",
    "Citations point to the chunk and the page range that support a claim.",
    "The evaluation harness measures recall at k for every configuration.",
)
_ARTICLE = "\n\n".join(_TOPICS).encode()


@dataclass
class _Harness:
    service: IngestionService
    parser: FakeParser
    embedder: FakeEmbedder
    store: InMemoryStore


def _harness(batch_size: int = 16, max_concurrency: int = 4) -> _Harness:
    parser, embedder, store = FakeParser(), FakeEmbedder(), InMemoryStore()
    service = IngestionService(
        parser,
        TokenChunker(chunk_size=20, overlap=5),
        embedder,
        store,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )
    return _Harness(service, parser, embedder, store)


async def test_ingest_saves_document_with_chunks() -> None:
    h = _harness()
    result = await h.service.ingest(_ARTICLE, "guide.txt", "text/plain")
    assert result.document.filename == "guide.txt"
    assert result.document.content_hash == hashlib.sha256(_ARTICLE).hexdigest()
    assert result.chunk_count == len(_TOPICS)
    assert result.replaced is None
    assert await h.store.find_by_filename("guide.txt") == result.document
    hits = await h.store.keyword_search("reranker", k=1)
    assert hits[0].chunk.document_id == result.document.id


async def test_duplicate_content_never_reaches_parser_or_embedder() -> None:
    h = _harness()
    existing = make_document("original.txt", content=_ARTICLE)
    await h.store.save(existing, [], [])
    with pytest.raises(DuplicateContentError) as raised:
        await h.service.ingest(_ARTICLE, "copy.txt", "text/plain")
    assert raised.value.existing == existing
    assert h.parser.calls == []
    assert h.embedder.batches == []


async def test_filename_conflict_is_rejected_by_default() -> None:
    h = _harness()
    existing = make_document("Guide.txt", content=b"older version")
    await h.store.save(existing, [], [])
    with pytest.raises(FilenameConflictError) as raised:
        await h.service.ingest(_ARTICLE, "guide.TXT", "text/plain")
    assert raised.value.existing == existing
    assert h.parser.calls == []
    assert h.embedder.batches == []


async def test_replace_swaps_out_the_existing_document() -> None:
    h = _harness()
    old = await h.service.ingest(b"Obsolete pricing table.", "prices.txt", "text/plain")
    new = await h.service.ingest(
        b"Current pricing table.", "Prices.txt", "text/plain", ConflictPolicy.REPLACE
    )
    assert new.replaced == old.document
    assert await h.store.find_by_hash(old.document.content_hash) is None
    assert await h.store.find_by_filename("prices.txt") == new.document
    assert await h.store.keyword_search("obsolete", k=5) == []


@pytest.mark.parametrize(
    ("existing", "uploaded", "expected"),
    [
        (["report.txt"], "report.txt", "report (1).txt"),
        (["report.txt", "Report (1).txt"], "report.txt", "report (2).txt"),
        (["report (1).txt"], "report (1).txt", "report (2).txt"),
    ],
)
async def test_keep_both_picks_first_free_numbered_filename(
    existing: list[str], uploaded: str, expected: str
) -> None:
    h = _harness()
    for filename in existing:
        await h.store.save(make_document(filename), [], [])
    result = await h.service.ingest(
        _ARTICLE, uploaded, "text/plain", ConflictPolicy.KEEP_BOTH
    )
    assert result.document.filename == expected
    assert result.replaced is None
    assert await h.store.find_by_filename(expected) == result.document


@pytest.mark.parametrize("content", [b"", b"  \n\n\t ", b"\f\f"])
async def test_document_without_text_is_rejected(content: bytes) -> None:
    h = _harness()
    with pytest.raises(EmptyDocumentError):
        await h.service.ingest(content, "blank.txt", "text/plain")
    assert h.embedder.batches == []
    assert await h.store.find_by_filename("blank.txt") is None


@pytest.mark.parametrize("max_concurrency", [1, 2])
async def test_embeddings_are_batched_with_bounded_concurrency(
    max_concurrency: int,
) -> None:
    h = _harness(batch_size=3, max_concurrency=max_concurrency)
    await h.service.ingest(_ARTICLE, "guide.txt", "text/plain")
    assert [len(batch) for batch in h.embedder.batches] == [3, 3, 1]
    assert h.embedder.max_in_flight == max_concurrency


async def test_each_chunk_is_saved_with_its_own_embedding() -> None:
    h = _harness(batch_size=2, max_concurrency=3)
    await h.service.ingest(_ARTICLE, "guide.txt", "text/plain")
    stored = await h.store.keyword_search(" ".join(_TOPICS), k=len(_TOPICS))
    chunks = sorted((hit.chunk for hit in stored), key=lambda c: c.position)
    assert [c.text for c in chunks] == [t for b in h.embedder.batches for t in b]
    for chunk in chunks:
        query = await h.embedder.embed_query(chunk.text)
        [nearest] = await h.store.vector_search(query, k=1)
        assert nearest.chunk == chunk


async def test_unsupported_media_type_saves_nothing() -> None:
    h = _harness()
    with pytest.raises(UnsupportedMediaTypeError):
        await h.service.ingest(b"%PDF-1.7", "scan.pdf", "application/pdf")
    assert await h.store.find_by_filename("scan.pdf") is None


@pytest.mark.parametrize(
    "limits",
    [{"batch_size": 0, "max_concurrency": 1}, {"batch_size": 1, "max_concurrency": 0}],
)
def test_invalid_limits_are_rejected(limits: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        IngestionService(
            FakeParser(), TokenChunker(), FakeEmbedder(), InMemoryStore(), **limits
        )
