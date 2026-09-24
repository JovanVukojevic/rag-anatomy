# rag-anatomy

Retrieval-augmented generation built from first principles, with every retrieval
and generation choice measured.

rag-anatomy builds a RAG pipeline step by step (ingestion, dense and hybrid retrieval,
reranking, cited generation, and agentic RAG) and evaluates each stage with the
same harness, so every design decision is backed by numbers.

**Status:** work in progress. Nothing is usable yet.

**Stack:** Python 3.14, FastAPI, Postgres + pgvector, OpenAI.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
make check
```
