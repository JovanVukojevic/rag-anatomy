import re
import unicodedata

_WORD = re.compile(r"\w+")
# đ has no Unicode decomposition, but Postgres unaccent folds it to d.
_UNDECOMPOSABLE = str.maketrans({"đ": "d", "Đ": "D"})


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.translate(_UNDECOMPOSABLE))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def overlap(query: str, text: str) -> float:
    query_terms = set(words(fold(query)))
    if not query_terms:
        return 0.0
    return len(query_terms & set(words(fold(text)))) / len(query_terms)
