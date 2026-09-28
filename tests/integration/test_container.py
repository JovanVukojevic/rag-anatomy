import httpx2
import pytest
from pgvector import Vector

from rag_anatomy.adapters.driven.parsing import DOCX, PDF
from rag_anatomy.adapters.driven.postgres import EMBEDDING_DIMENSIONS, PgVectorStore
from rag_anatomy.adapters.driven.postgres.store import Pool
from rag_anatomy.config import DatabaseSettings
from rag_anatomy.container import (
    ConfigurationError,
    check_pgvector_version,
    open_container,
)
from rag_anatomy.services import RetrievalMode
from tests.builders import make_docx, make_pdf
from tests.fakes import FakeEmbeddingsAPI, trigram_embedding

pytestmark = pytest.mark.integration

DENSE = RetrievalMode.DENSE

_LATIN_MARKER = "Šta je hibridna pretraga? Čačak, ćevapi, žaba i đak."
_CYRILLIC_MARKER = "Шта је хибридна претрага? Ђак, Љубав, Њива, Џеп."
_CYRILLIC_QUERY = "Шта је хибридна претрага?"
_DOCX_TEXT = "Upload limits protect the parser from zip bombs."


def _prose(page: int, topic: str) -> str:
    return "\n".join(
        f"Sentence {n} of page {page} explains {topic}." for n in range(40)
    )


_PAGES = (
    _prose(1, "hybrid retrieval"),
    "",
    _prose(3, "rank fusion"),
    _prose(4, "chunk overlap"),
    _LATIN_MARKER,
    _CYRILLIC_MARKER,
)


@pytest.fixture
def openai_env(
    database: DatabaseSettings, pg_pool: Pool, monkeypatch: pytest.MonkeyPatch
) -> pytest.MonkeyPatch:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", str(EMBEDDING_DIMENSIONS))
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "0")
    return monkeypatch


async def test_pdf_is_ingested_end_to_end(
    openai_env: pytest.MonkeyPatch, pg_pool: Pool
) -> None:
    api = FakeEmbeddingsAPI()
    async with open_container(openai_transport=httpx2.MockTransport(api)) as container:
        result = await container.ingestion.ingest(
            make_pdf(*_PAGES), "hybrid.pdf", "application/pdf"
        )

    async with pg_pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT position, text, page_start, page_end
            FROM chunks WHERE document_id = %s ORDER BY position
            """,
            (result.document.id,),
        )
        chunks = await cursor.fetchall()
    positions = [position for position, *_ in chunks]
    texts = [text for _, text, *_ in chunks]
    assert result.chunk_count == len(chunks) > 3
    assert positions == list(range(len(chunks)))
    assert chunks[0][2] == 1
    assert chunks[-1][3] == len(_PAGES)
    assert all(2 not in (start, end) for *_, start, end in chunks)
    [(*_, latin_end)] = [c for c in chunks if _LATIN_MARKER in c[1]]
    [(*_, cyrillic_end)] = [c for c in chunks if _CYRILLIC_MARKER in c[1]]
    assert latin_end >= 5
    assert cyrillic_end == 6
    assert [text for body in api.bodies for text in body["input"]] == texts

    async with pg_pool.connection() as conn:
        for position, text in zip(positions, texts, strict=True):
            expected = Vector(trigram_embedding(text, EMBEDDING_DIMENSIONS))
            cursor = await conn.execute(
                """
                SELECT embedding <=> %s FROM chunks
                WHERE document_id = %s AND position = %s
                """,
                (expected, result.document.id, position),
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == pytest.approx(0.0, abs=1e-6)


async def test_ingested_documents_are_retrieved_end_to_end(
    openai_env: pytest.MonkeyPatch,
) -> None:
    api = FakeEmbeddingsAPI()
    async with open_container(openai_transport=httpx2.MockTransport(api)) as container:
        await container.ingestion.ingest(make_pdf(*_PAGES), "hybrid.pdf", PDF)
        await container.ingestion.ingest(
            make_docx(lambda d: d.add_paragraph(_DOCX_TEXT)), "limits.docx", DOCX
        )
        fusion = await container.retrieval.retrieve("rank fusion", top_k=3, mode=DENSE)
        cyrillic = await container.retrieval.retrieve(
            _CYRILLIC_QUERY, top_k=3, mode=DENSE
        )
        docx = await container.retrieval.retrieve(
            "zip bombs in uploads", top_k=3, mode=DENSE
        )

    top = fusion[0]
    assert "rank fusion" in top.chunk.text
    assert top.chunk.page_start <= 3 <= top.chunk.page_end
    assert (top.document.filename, top.document.media_type) == ("hybrid.pdf", PDF)

    top = cyrillic[0]
    assert _CYRILLIC_MARKER in top.chunk.text
    assert top.chunk.page_end == len(_PAGES)
    assert top.document.filename == "hybrid.pdf"

    top = docx[0]
    assert top.chunk.text == _DOCX_TEXT
    assert (top.document.filename, top.document.media_type) == ("limits.docx", DOCX)
    assert (top.chunk.page_start, top.chunk.page_end) == (1, 1)

    assert all(
        [r.dense.rank if r.dense else None for r in results] == [1, 2, 3]
        for results in (fusion, cyrillic, docx)
    )
    assert api.bodies[-1]["input"] == ["zip bombs in uploads"]


async def test_hybrid_query_fuses_dense_and_keyword_hits_end_to_end(
    openai_env: pytest.MonkeyPatch,
) -> None:
    api = FakeEmbeddingsAPI()
    async with open_container(openai_transport=httpx2.MockTransport(api)) as container:
        await container.ingestion.ingest(make_pdf(*_PAGES), "hybrid.pdf", PDF)
        results = await container.retrieval.retrieve("rank fusion", top_k=3)

    assert [r.fusion.rank if r.fusion else None for r in results] == [1, 2, 3]
    top = results[0]
    assert top.dense and top.keyword
    assert "rank fusion" in top.chunk.text


async def test_latin_query_without_diacritics_matches_through_keyword_search(
    openai_env: pytest.MonkeyPatch,
) -> None:
    api = FakeEmbeddingsAPI()
    async with open_container(openai_transport=httpx2.MockTransport(api)) as container:
        await container.ingestion.ingest(make_pdf(*_PAGES), "hybrid.pdf", PDF)
        requests = len(api.requests)
        results = await container.retrieval.retrieve(
            "sta je hibridna pretraga cacak", top_k=3, mode=RetrievalMode.KEYWORD
        )

    top = results[0]
    assert _LATIN_MARKER in top.chunk.text
    assert top.keyword and top.keyword.rank == 1
    assert top.dense is None
    assert len(api.requests) == requests


async def test_installed_pgvector_passes_the_version_check(
    pg_store: PgVectorStore,
) -> None:
    check_pgvector_version(await pg_store.pgvector_version())


async def test_dimension_mismatch_fails_before_any_api_request(
    openai_env: pytest.MonkeyPatch,
) -> None:
    openai_env.setenv("OPENAI_EMBEDDING_DIMENSIONS", "768")
    api = FakeEmbeddingsAPI()
    with pytest.raises(ConfigurationError, match=r"vector\(1536\)"):
        async with open_container(openai_transport=httpx2.MockTransport(api)):
            pass
    assert api.requests == []


async def test_model_switch_is_adopted_while_no_chunks_exist(
    openai_env: pytest.MonkeyPatch, pg_pool: Pool
) -> None:
    api = FakeEmbeddingsAPI()
    async with open_container(openai_transport=httpx2.MockTransport(api)):
        pass
    openai_env.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    async with open_container(openai_transport=httpx2.MockTransport(api)):
        pass
    async with pg_pool.connection() as conn:
        cursor = await conn.execute("SELECT model, dimensions FROM embedding_space")
        rows = await cursor.fetchall()
    assert rows == [("text-embedding-3-large", EMBEDDING_DIMENSIONS)]
    assert api.requests == []


async def test_model_switch_with_stored_chunks_fails_before_any_api_request(
    openai_env: pytest.MonkeyPatch,
) -> None:
    async with open_container(
        openai_transport=httpx2.MockTransport(FakeEmbeddingsAPI())
    ) as container:
        await container.ingestion.ingest(
            make_pdf(_LATIN_MARKER), "stored.pdf", "application/pdf"
        )
    openai_env.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    api = FakeEmbeddingsAPI()
    with pytest.raises(ConfigurationError, match="text-embedding-3-small"):
        async with open_container(openai_transport=httpx2.MockTransport(api)):
            pass
    assert api.requests == []
