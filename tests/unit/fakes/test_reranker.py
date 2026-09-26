from rag_anatomy.domain import StageRank
from tests.builders import make_document, make_retrieved
from tests.fakes import FakeReranker


async def test_rerank_orders_by_query_overlap_and_truncates() -> None:
    candidates = make_retrieved(make_document(), "no match", "cross encoder", "encoder")
    results = await FakeReranker().rerank("cross encoder", candidates, k=2)
    assert [(r.chunk.text, r.rerank) for r in results] == [
        ("cross encoder", StageRank(rank=1, score=1.0)),
        ("encoder", StageRank(rank=2, score=0.5)),
    ]
    assert [r.dense for r in results] == [candidates[1].dense, candidates[2].dense]
