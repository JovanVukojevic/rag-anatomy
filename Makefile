.PHONY: lint format typecheck encodings test check

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

check: lint typecheck test
