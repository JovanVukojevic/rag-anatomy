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

### Reranking (optional, off by default)

Retrieval can rerank its candidates with
[bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) served by
Hugging Face text-embeddings-inference. On a laptop CPU this takes tens of seconds per
query, so it is meant for offline evaluation, not interactive use.

```sh
make reranker-up          # first start downloads 2.3 GB of weights
# set RERANKER_ENABLED=true in .env
make test-reranker        # tests against the real model
```

Data leaves the machine only for embeddings: chunk text and queries are sent to OpenAI.
The reranker runs locally.
