import re

_WORD = re.compile(r"\w+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def overlap(query: str, text: str) -> float:
    query_terms = set(words(query))
    if not query_terms:
        return 0.0
    return len(query_terms & set(words(text))) / len(query_terms)
