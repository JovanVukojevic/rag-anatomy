# rag-anatomy

Retrieval-augmented generation built from first principles, with every retrieval
and generation choice measured.

rag-anatomy builds a RAG pipeline step by step (ingestion, dense and hybrid retrieval,
reranking, cited generation, and agentic RAG) and evaluates each stage with the
same harness, so every design decision is backed by numbers.

**Status:** work in progress. Nothing is usable yet.

**Stack:** Python 3.14, FastAPI, Postgres + pgvector, OpenAI.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```sh
uv sync
cp .env.example .env
make db-up migrate
make check        # includes integration tests against a throwaway Postgres
make test-unit    # fast loop, no Docker
```
