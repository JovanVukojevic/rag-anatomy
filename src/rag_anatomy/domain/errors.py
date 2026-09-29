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


class EmptyDocumentError(Exception):
    def __init__(self, filename: str) -> None:
        super().__init__(f"{filename!r} contains no text to index")
        self.filename = filename


class InvalidEncodingError(Exception):
    def __init__(self, encoding: str) -> None:
        super().__init__(f"content is not valid {encoding}")
        self.encoding = encoding


class EncryptedDocumentError(Exception):
    def __init__(self, media_type: str) -> None:
        super().__init__(
            f"{media_type} content is password-protected; remove the password and retry"
        )
        self.media_type = media_type


class CorruptDocumentError(Exception):
    def __init__(self, media_type: str) -> None:
        super().__init__(f"content is not a valid {media_type} file")
        self.media_type = media_type


class EmptyQueryError(Exception):
    def __init__(self) -> None:
        super().__init__("query contains no text to search for")


class EmbeddingError(Exception):
    pass


class RerankError(Exception):
    pass
