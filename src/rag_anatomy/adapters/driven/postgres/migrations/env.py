from alembic import context
from sqlalchemy import NullPool, create_engine, make_url

from rag_anatomy.config import DatabaseSettings

url = make_url(DatabaseSettings().dsn).set(drivername="postgresql+psycopg")
engine = create_engine(url, poolclass=NullPool)

with engine.connect() as connection:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()
