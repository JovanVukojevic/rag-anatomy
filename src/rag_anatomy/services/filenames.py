import re

_NUMBER_SUFFIX = re.compile(r" \(\d+\)$")


def numbered_filename(filename: str, n: int) -> str:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    dot = filename.rfind(".")
    if dot <= 0:
        stem, extension = filename, ""
    else:
        stem, extension = filename[:dot], filename[dot:]
    return f"{_NUMBER_SUFFIX.sub('', stem)} ({n}){extension}"
