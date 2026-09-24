def check_page_range(page_start: int, page_end: int) -> None:
    if page_start < 1:
        raise ValueError(f"page_start must be >= 1, got {page_start}")
    if page_start > page_end:
        raise ValueError(f"page_start {page_start} is after page_end {page_end}")
