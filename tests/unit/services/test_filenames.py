import pytest

from rag_anatomy.services import numbered_filename


@pytest.mark.parametrize(
    ("filename", "n", "expected"),
    [
        ("report.pdf", 1, "report (1).pdf"),
        ("report (1).pdf", 2, "report (2).pdf"),
        ("report (12).pdf", 3, "report (3).pdf"),
        ("README", 1, "README (1)"),
        ("notes (4)", 5, "notes (5)"),
        (".env", 1, ".env (1)"),
        ("archive.tar.gz", 1, "archive.tar (1).gz"),
        ("draft (x).txt", 1, "draft (x) (1).txt"),
        ("scan(1).png", 2, "scan(1) (2).png"),
    ],
)
def test_numbered_filename(filename: str, n: int, expected: str) -> None:
    assert numbered_filename(filename, n) == expected


def test_numbered_filename_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError):
        numbered_filename("report.pdf", 0)
