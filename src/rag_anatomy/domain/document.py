import re
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Document:
    id: UUID
    filename: str
    media_type: str
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.filename.strip():
            raise ValueError("filename must not be blank")
        if not self.media_type.strip():
            raise ValueError("media_type must not be blank")
        if not _SHA256_HEX.fullmatch(self.content_hash):
            raise ValueError("content_hash must be a lowercase sha256 hex digest")
        if self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")


@dataclass(frozen=True, slots=True, kw_only=True)
class Page:
    number: int
    text: str

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError(f"page number must be >= 1, got {self.number}")
