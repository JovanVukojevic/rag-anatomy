"""embedding space

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-26
"""

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE embedding_space (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            model text NOT NULL CHECK (model <> ''),
            dimensions integer NOT NULL CHECK (dimensions > 0)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE embedding_space")
