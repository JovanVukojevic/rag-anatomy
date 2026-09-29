from dataclasses import replace

import pytest

from rag_anatomy.adapters.driven.tei import TeiInfo
from rag_anatomy.container import (
    ConfigurationError,
    check_pgvector_version,
    check_reranker,
)
from tests.fakes import RERANKER_MODEL, RERANKER_REVISION

_PINNED = TeiInfo(
    model_id=RERANKER_MODEL,
    model_sha=RERANKER_REVISION,
    reranker=True,
    max_client_batch_size=32,
)


@pytest.mark.parametrize("installed", ["0.8.0", "0.8.6", "0.10.0", "1.0.0"])
def test_pgvector_with_iterative_scans_is_accepted(installed: str) -> None:
    check_pgvector_version(installed)


@pytest.mark.parametrize("installed", ["0.7.4", "0.5.1"])
def test_pgvector_without_iterative_scans_is_rejected(installed: str) -> None:
    with pytest.raises(ConfigurationError, match=rf"pgvector {installed} .* 0\.8\.0"):
        check_pgvector_version(installed)


@pytest.mark.parametrize("installed", ["", "0.8", "0.8.0-dev", "v0.8.0", "latest"])
def test_unparsable_pgvector_version_is_rejected(installed: str) -> None:
    with pytest.raises(ConfigurationError, match="cannot parse"):
        check_pgvector_version(installed)


def test_pinned_reranker_is_accepted() -> None:
    check_reranker(_PINNED, model=RERANKER_MODEL, revision=RERANKER_REVISION)


@pytest.mark.parametrize(
    "served",
    [
        replace(_PINNED, model_id="BAAI/bge-reranker-v2-m3"),
        replace(_PINNED, model_sha="main"),
        replace(_PINNED, model_sha=None),
    ],
    ids=["other-model", "other-revision", "no-revision"],
)
def test_reranker_serving_something_else_is_rejected(served: TeiInfo) -> None:
    with pytest.raises(
        ConfigurationError, match=rf"{RERANKER_MODEL}@{RERANKER_REVISION} is configured"
    ):
        check_reranker(served, model=RERANKER_MODEL, revision=RERANKER_REVISION)


def test_embedding_model_is_not_a_reranker() -> None:
    with pytest.raises(ConfigurationError, match="not as a reranker"):
        check_reranker(
            replace(_PINNED, reranker=False),
            model=RERANKER_MODEL,
            revision=RERANKER_REVISION,
        )
