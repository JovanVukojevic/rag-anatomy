import math
from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import UUID

from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

from rag_anatomy.domain import (
    Chunk,
    Document,
    DuplicateContentError,
    Embedding,
    FilenameConflictError,
    RetrievedChunk,
    Retriever,
)
from rag_anatomy.ports import DocumentRepository, KeywordSearch, VectorSearch

EMBEDDING_DIMENSIONS = 1536

type Pool = AsyncConnectionPool[AsyncConnection[TupleRow]]

_DEFAULT_EF_SEARCH = 40
_MAX_EF_SEARCH = 1000
_INSERT_ATTEMPTS = 3

_DOCUMENT_COLUMNS = "id, filename, media_type, content_hash, created_at"
_FIND_BY_HASH = f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE content_hash = %s"
_FIND_BY_FILENAME = (
    f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE lower(filename) = lower(%s)"
)
_INSERT_DOCUMENT = f"""
    INSERT INTO documents ({_DOCUMENT_COLUMNS}) VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT DO NOTHING
    RETURNING id
"""
_DELETE_DOCUMENT = "DELETE FROM documents WHERE id = %s RETURNING id"
_COPY_CHUNKS = """
    COPY chunks (id, document_id, position, text, page_start, page_end, embedding)
    FROM STDIN (FORMAT BINARY)
"""
_CHUNK_TYPES = ["uuid", "uuid", "int4", "text", "int4", "int4", "vector"]

VECTOR_SEARCH_SQL = """
    WITH nearest AS (
        SELECT id, document_id, text, position, page_start, page_end,
               embedding <=> %(embedding)s AS distance
        FROM chunks
        ORDER BY embedding <=> %(embedding)s
        LIMIT %(k)s
    )
    SELECT n.id, n.document_id, n.text, n.position, n.page_start, n.page_end,
           d.filename, 1 - n.distance
    FROM nearest n
    JOIN documents d ON d.id = n.document_id
    ORDER BY n.distance, n.id
"""

KEYWORD_SEARCH_SQL = """
    WITH q AS (
        SELECT replace(
            plainto_tsquery('simple_unaccent', %(query)s)::text, ' & ', ' | '
        )::tsquery AS query
    )
    SELECT c.id, c.document_id, c.text, c.position, c.page_start, c.page_end,
           d.filename, ts_rank(c.tsv, q.query) AS score
    FROM q
    JOIN chunks c ON c.tsv @@ q.query
    JOIN documents d ON d.id = c.document_id
    ORDER BY score DESC, c.id
    LIMIT %(k)s
"""


def connection_pool(conninfo: str, *, max_size: int, timeout: float) -> Pool:
    return AsyncConnectionPool(
        conninfo,
        min_size=1,
        max_size=max_size,
        open=False,
        timeout=timeout,
        configure=register_vector_async,
        kwargs={
            "autocommit": True,
            "connect_timeout": math.ceil(timeout),
            "options": f"-c statement_timeout={math.ceil(timeout * 1000)}",
        },
    )


class PgVectorStore:
    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def find_by_hash(self, content_hash: str) -> Document | None:
        async with self._pool.connection() as conn:
            return await _find(conn, _FIND_BY_HASH, content_hash)

    async def find_by_filename(self, filename: str) -> Document | None:
        async with self._pool.connection() as conn:
            return await _find(conn, _FIND_BY_FILENAME, filename)

    async def save(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        _check(document, chunks, embeddings)
        async with self._pool.connection() as conn, conn.transaction():
            await _insert(conn, document, chunks, embeddings)

    async def replace(
        self,
        existing_id: UUID,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        _check(document, chunks, embeddings)
        async with self._pool.connection() as conn, conn.transaction():
            cursor = await conn.execute(_DELETE_DOCUMENT, (existing_id,))
            if await cursor.fetchone() is None:
                raise LookupError(f"no document with id {existing_id}")
            await _insert(conn, document, chunks, embeddings)

    async def vector_search(self, embedding: Embedding, k: int) -> list[RetrievedChunk]:
        # HNSW returns at most ef_search rows, so a larger k would be cut short silently.
        ef_search = min(max(k, _DEFAULT_EF_SEARCH), _MAX_EF_SEARCH)
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),)
            )
            cursor = await conn.execute(
                VECTOR_SEARCH_SQL, {"embedding": Vector(embedding), "k": k}
            )
            rows = await cursor.fetchall()
        return [_retrieved(row, Retriever.DENSE) for row in rows]

    async def keyword_search(self, query: str, k: int) -> list[RetrievedChunk]:
        async with self._pool.connection() as conn:
            cursor = await conn.execute(KEYWORD_SEARCH_SQL, {"query": query, "k": k})
            rows = await cursor.fetchall()
        return [_retrieved(row, Retriever.KEYWORD) for row in rows]


def _check(
    document: Document, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    if any(chunk.document_id != document.id for chunk in chunks):
        raise ValueError("every chunk must belong to the saved document")


async def _insert(
    conn: AsyncConnection[TupleRow],
    document: Document,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Embedding],
) -> None:
    for _ in range(_INSERT_ATTEMPTS):
        cursor = await conn.execute(
            _INSERT_DOCUMENT,
            (
                document.id,
                document.filename,
                document.media_type,
                document.content_hash,
                document.created_at,
            ),
        )
        if await cursor.fetchone() is not None:
            break
        if existing := await _find(conn, _FIND_BY_HASH, document.content_hash):
            raise DuplicateContentError(existing)
        if existing := await _find(conn, _FIND_BY_FILENAME, document.filename):
            raise FilenameConflictError(existing)
    else:
        raise RuntimeError(
            f"{document.filename!r} kept conflicting with concurrently deleted documents"
        )
    async with conn.cursor() as cursor, cursor.copy(_COPY_CHUNKS) as copy:
        copy.set_types(_CHUNK_TYPES)
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            await copy.write_row(
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.position,
                    chunk.text,
                    chunk.page_start,
                    chunk.page_end,
                    Vector(embedding),
                )
            )


async def _find(
    conn: AsyncConnection[TupleRow], query: str, value: str
) -> Document | None:
    cursor = await conn.execute(query, (value,))
    row = await cursor.fetchone()
    if row is None:
        return None
    id_, filename, media_type, content_hash, created_at = row
    return Document(
        id=id_,
        filename=filename,
        media_type=media_type,
        content_hash=content_hash,
        created_at=created_at,
    )


def _retrieved(row: TupleRow, retriever: Retriever) -> RetrievedChunk:
    id_, document_id, text, position, page_start, page_end, filename, score = row
    return RetrievedChunk(
        chunk=Chunk(
            id=id_,
            document_id=document_id,
            text=text,
            position=position,
            page_start=page_start,
            page_end=page_end,
        ),
        filename=filename,
        score=score,
        retriever=retriever,
    )


if TYPE_CHECKING:
    _store = PgVectorStore(connection_pool("", max_size=1, timeout=1))
    _repository: DocumentRepository = _store
    _vector_search: VectorSearch = _store
    _keyword_search: KeywordSearch = _store
