import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from uuid import UUID

from rag_anatomy.domain import RetrievedChunk, StageRank


def reciprocal_rank_fusion(
    dense: Sequence[RetrievedChunk],
    keyword: Sequence[RetrievedChunk],
    *,
    c: int = 60,
) -> list[RetrievedChunk]:
    if c < 0:
        raise ValueError(f"c must be >= 0, got {c}")
    merged = _by_chunk_id(dense, "dense", lambda hit: hit.dense)
    keyword_hits = _by_chunk_id(keyword, "keyword", lambda hit: hit.keyword)
    for chunk_id, hit in keyword_hits.items():
        found = merged.get(chunk_id)
        merged[chunk_id] = hit if found is None else replace(found, keyword=hit.keyword)

    # The dense term is always added first, so symmetric rank pairs tie exactly.
    scored = [
        (_term(hit.dense, c) + _term(hit.keyword, c), hit) for hit in merged.values()
    ]
    # Ranks are unique within each list, so the better dense rank settles every tie (D53).
    scored.sort(key=lambda pair: (-pair[0], _rank(pair[1].dense)))
    return [
        replace(hit, fusion=StageRank(rank=rank, score=score))
        for rank, (score, hit) in enumerate(scored, start=1)
    ]


def _by_chunk_id(
    ranking: Sequence[RetrievedChunk],
    stage: str,
    stage_rank: Callable[[RetrievedChunk], StageRank | None],
) -> dict[UUID, RetrievedChunk]:
    hits: dict[UUID, RetrievedChunk] = {}
    for position, hit in enumerate(ranking, start=1):
        rank = stage_rank(hit)
        if rank is None or rank.rank != position:
            raise ValueError(
                f"hit {position} of the {stage} ranking must have {stage} rank {position}"
            )
        if hit.chunk.id in hits:
            raise ValueError(
                f"chunk {hit.chunk.id} appears twice in the {stage} ranking"
            )
        hits[hit.chunk.id] = hit
    return hits


def _term(rank: StageRank | None, c: int) -> float:
    return 0.0 if rank is None else 1 / (c + rank.rank)


def _rank(rank: StageRank | None) -> float:
    return math.inf if rank is None else rank.rank
