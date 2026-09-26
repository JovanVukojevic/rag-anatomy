import pytest

from rag_anatomy.domain import normalized


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("a\r\nb\rc", "a\nb\nc"),
        ("keeps\ttabs\nand newlines", "keeps\ttabs\nand newlines"),
        ("nul\x00 bell\x07 escape\x1b", "nul bell escape"),
        ("lone \ud800high \udfffand low", "lone high and low"),
        ("čudno", "čudno"),
    ],
    ids=["line-breaks", "whitespace", "controls", "lone-surrogates", "nfc"],
)
def test_normalized_text_is_storable(raw: str, clean: str) -> None:
    text = normalized(raw)
    assert text == clean
    text.encode("utf-8")
