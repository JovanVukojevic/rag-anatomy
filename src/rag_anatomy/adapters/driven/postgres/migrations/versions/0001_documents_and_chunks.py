"""documents and chunks

Revision ID: 0001
Revises:
Create Date: 2026-09-26
"""

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
    # unaccent() is not IMMUTABLE, so a generated column cannot call it directly;
    # the unaccent dictionary inside a text search configuration can.
    op.execute("CREATE TEXT SEARCH CONFIGURATION simple_unaccent (COPY = simple)")
    op.execute(
        """
        ALTER TEXT SEARCH CONFIGURATION simple_unaccent
            ALTER MAPPING FOR hword, hword_part, word WITH unaccent, simple
        """
    )
    op.execute(
        """
        CREATE TABLE documents (
            id uuid PRIMARY KEY,
            filename text NOT NULL,
            media_type text NOT NULL,
            content_hash text NOT NULL UNIQUE,
            created_at timestamptz NOT NULL
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX documents_filename_key ON documents (lower(filename))"
    )
    op.execute(
        """
        CREATE TABLE chunks (
            id uuid PRIMARY KEY,
            document_id uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
            position integer NOT NULL CHECK (position >= 0),
            text text NOT NULL,
            page_start integer NOT NULL CHECK (page_start >= 1),
            page_end integer NOT NULL CHECK (page_end >= page_start),
            embedding vector(1536) NOT NULL,
            tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple_unaccent', text))
                STORED,
            UNIQUE (document_id, position)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX chunks_embedding_idx ON chunks
            USING hnsw (embedding vector_cosine_ops)
        """
    )
    op.execute("CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv)")


def downgrade() -> None:
    op.execute("DROP TABLE chunks")
    op.execute("DROP TABLE documents")
    op.execute("DROP TEXT SEARCH CONFIGURATION simple_unaccent")
