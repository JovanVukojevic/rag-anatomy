import re
import unicodedata

_LINE_BREAK = re.compile(r"\r\n?")
# Postgres text rejects U+0000 and psycopg cannot encode lone surrogates; the other C0
# controls are extraction artefacts.
_UNSTORABLE = re.compile(r"[\x00-\x08\x0b-\x1f\ud800-\udfff]")


def normalized(text: str) -> str:
    text = _UNSTORABLE.sub("", _LINE_BREAK.sub("\n", text))
    return unicodedata.normalize("NFC", text)
