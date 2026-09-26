import pytest

from rag_anatomy.container import ConfigurationError, check_pgvector_version


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
