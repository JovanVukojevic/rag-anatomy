from rag_anatomy.domain.document import Document


class DuplicateContentError(Exception):
    def __init__(self, existing: Document) -> None:
        super().__init__(f"content already ingested as {existing.filename!r}")
        self.existing = existing


class FilenameConflictError(Exception):
    def __init__(self, existing: Document) -> None:
        super().__init__(f"a document named {existing.filename!r} already exists")
        self.existing = existing


class UnsupportedMediaTypeError(Exception):
    def __init__(self, media_type: str) -> None:
        super().__init__(f"unsupported media type {media_type!r}")
        self.media_type = media_type
