import hashlib
from datetime import UTC, datetime
from uuid import uuid7

from rag_anatomy.domain import Chunk, Document


def make_document(
    filename: str = "guide.txt", content: bytes | None = None
) -> Document:
    return Document(
        id=uuid7(),
        filename=filename,
        media_type="text/plain",
        content_hash=hashlib.sha256(content or filename.encode()).hexdigest(),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def make_chunks(document: Document, *texts: str) -> list[Chunk]:
    return [
        Chunk(
            id=uuid7(),
            document_id=document.id,
            text=text,
            position=position,
            page_start=1,
            page_end=1,
        )
        for position, text in enumerate(texts)
    ]
