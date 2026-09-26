import math

from tests.fakes import FakeEmbedder


def _similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_embeddings_are_deterministic_across_instances() -> None:
    first = await FakeEmbedder().embed_query("hybrid retrieval")
    second = await FakeEmbedder().embed_documents(["hybrid retrieval"])
    assert second == [first]


async def test_embeddings_are_unit_length() -> None:
    [vector] = await FakeEmbedder().embed_documents(["reciprocal rank fusion"])
    assert math.isclose(math.fsum(x * x for x in vector), 1.0)


async def test_similar_words_have_similar_embeddings() -> None:
    embedder = FakeEmbedder()
    chunk, chunking, database = await embedder.embed_documents(
        ["chunk", "chunking", "database"]
    )
    assert _similarity(chunk, chunking) > _similarity(chunk, database) + 0.3


async def test_empty_text_embeds_to_zero_vector() -> None:
    assert await FakeEmbedder(dimensions=8).embed_query("") == [0.0] * 8


async def test_batches_are_recorded_in_call_order() -> None:
    embedder = FakeEmbedder()
    await embedder.embed_documents(["a", "b"])
    await embedder.embed_documents(["c"])
    assert embedder.batches == [["a", "b"], ["c"]]
