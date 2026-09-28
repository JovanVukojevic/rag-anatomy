import pytest

from rag_anatomy.domain import Chunk, RetrievedChunk
from rag_anatomy.services import reciprocal_rank_fusion
from tests.builders import make_chunks, make_document, make_ranking

document = make_document()
a, b, c, d = make_chunks(document, "a", "b", "c", "d")


def _dense(*chunks: Chunk) -> list[RetrievedChunk]:
    return make_ranking(document, chunks, stage="dense")


def _keyword(*chunks: Chunk) -> list[RetrievedChunk]:
    return make_ranking(document, chunks, stage="keyword")


def _texts(results: list[RetrievedChunk]) -> list[str]:
    return [r.chunk.text for r in results]


def _scores(results: list[RetrievedChunk]) -> list[float]:
    return [r.fusion.score if r.fusion else float("nan") for r in results]


def _ranks(results: list[RetrievedChunk]) -> list[tuple[int | None, int | None]]:
    return [
        (r.dense.rank if r.dense else None, r.keyword.rank if r.keyword else None)
        for r in results
    ]


def test_hand_computed_example() -> None:
    results = reciprocal_rank_fusion(_dense(a, b, c), _keyword(c, a, d))
    assert _texts(results) == ["a", "c", "b", "d"]
    assert _scores(results) == pytest.approx(
        [1 / 61 + 1 / 62, 1 / 61 + 1 / 63, 1 / 62, 1 / 63]
    )
    assert [r.fusion.rank if r.fusion else None for r in results] == [1, 2, 3, 4]
    assert _ranks(results) == [(1, 2), (3, 1), (2, None), (None, 3)]
    assert all(r.document == document for r in results)


def test_c_controls_how_fast_rank_weight_decays() -> None:
    results = reciprocal_rank_fusion(_dense(a, b, c), _keyword(c, a, d), c=0)
    assert _texts(results) == ["a", "c", "b", "d"]
    assert _scores(results) == pytest.approx([1 + 1 / 2, 1 / 3 + 1, 1 / 2, 1 / 3])


def test_symmetric_ranks_tie_exactly_and_the_better_dense_rank_wins() -> None:
    results = reciprocal_rank_fusion(_dense(a, b, c), _keyword(c, d, a))
    assert _texts(results) == ["a", "c", "b", "d"]
    first, second, third, fourth = _scores(results)
    assert first == second
    assert third == fourth
    assert _ranks(results) == [(1, 3), (3, 1), (2, None), (None, 2)]


def test_one_list_rank_tying_a_both_list_rank_goes_to_the_better_dense_rank() -> None:
    results = reciprocal_rank_fusion(_dense(a, b), _keyword(c, b), c=0)
    assert _scores(results) == [1.0, 1.0, 1.0]
    assert _ranks(results) == [(1, None), (2, 2), (None, 1)]


def test_disjoint_lists_interleave() -> None:
    results = reciprocal_rank_fusion(_dense(a, b), _keyword(c, d))
    assert _texts(results) == ["a", "c", "b", "d"]
    assert _ranks(results) == [(1, None), (None, 1), (2, None), (None, 2)]


def test_overlapping_lists_merge_each_chunk_once() -> None:
    results = reciprocal_rank_fusion(_dense(a, b, c), _keyword(b, c, a))
    assert sorted(_texts(results)) == ["a", "b", "c"]
    assert all(r.dense and r.keyword for r in results)


@pytest.mark.parametrize("empty", ["dense", "keyword"])
def test_one_empty_list_keeps_the_other_order(empty: str) -> None:
    if empty == "dense":
        results = reciprocal_rank_fusion([], _keyword(a, b, c))
        assert _ranks(results) == [(None, 1), (None, 2), (None, 3)]
    else:
        results = reciprocal_rank_fusion(_dense(a, b, c), [])
        assert _ranks(results) == [(1, None), (2, None), (3, None)]
    assert _texts(results) == ["a", "b", "c"]
    assert _scores(results) == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_both_lists_empty_fuse_to_nothing() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_negative_c_is_rejected() -> None:
    with pytest.raises(ValueError, match="c must be"):
        reciprocal_rank_fusion(_dense(a), _keyword(b), c=-1)


def test_list_without_its_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="dense rank 1"):
        reciprocal_rank_fusion(_keyword(a), _keyword(b))


def test_rank_must_match_position() -> None:
    with pytest.raises(ValueError, match="keyword rank 1"):
        reciprocal_rank_fusion(_dense(a), list(reversed(_keyword(b, c))))


def test_duplicate_chunk_in_one_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="twice"):
        reciprocal_rank_fusion(_dense(a, a), [])
