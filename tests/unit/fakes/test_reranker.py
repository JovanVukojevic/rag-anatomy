from rag_anatomy.domain import RetrievedChunk, Retriever
from tests.builders import make_chunks, make_document
from tests.fakes import FakeReranker


async def test_rerank_orders_by_query_overlap_and_truncates() -> None:
    chunks = make_chunks(make_document(), "no match", "cross encoder", "encoder")
    candidates = [
        RetrievedChunk(
            chunk=c, filename="guide.txt", score=0.0, retriever=Retriever.DENSE
        )
        for c in chunks
    ]
    results = await FakeReranker().rerank("cross encoder", candidates, k=2)
    assert [(r.chunk.text, r.score) for r in results] == [
        ("cross encoder", 1.0),
        ("encoder", 0.5),
    ]
    assert {r.retriever for r in results} == {Retriever.RERANK}
