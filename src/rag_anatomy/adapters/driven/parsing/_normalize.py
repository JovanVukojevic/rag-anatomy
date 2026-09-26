def media_type_essence(media_type: str) -> str:
    return media_type.partition(";")[0].strip().lower()
