.PHONY: lint format typecheck encodings test test-unit test-reranker check db-up db-down reranker-up reranker-down migrate

export TIKTOKEN_CACHE_DIR ?= $(CURDIR)/.cache/tiktoken

lint:
	uv run ruff check
	uv run ruff format --check
	uv run lint-imports

format:
	uv run ruff check --fix
	uv run ruff format

typecheck:
	uv run mypy

encodings:
	uv run python -c "import tiktoken; tiktoken.encoding_for_model('text-embedding-3-small')"

test: encodings
	uv run pytest --cov

test-unit: encodings
	uv run pytest -m "not integration"

test-reranker: encodings
	uv run pytest -m reranker

check: lint typecheck test

db-up:
	docker compose up -d --wait

db-down:
	docker compose down

reranker-up:
	docker compose --profile rerank up -d --wait reranker

reranker-down:
	docker compose --profile rerank rm -sf reranker

migrate:
	uv run alembic upgrade head
