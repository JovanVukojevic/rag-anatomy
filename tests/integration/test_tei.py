import math

import httpx2
import pytest

from rag_anatomy.adapters.driven.tei import TeiReranker, server_info, tei_client
from rag_anatomy.config import RerankerServerSettings
from rag_anatomy.container import check_reranker, open_container
from tests.fakes import FakeEmbeddingsAPI

pytestmark = [pytest.mark.integration, pytest.mark.reranker]

_QUESTION = "Колико дуго важи гаранција на лаптоп рачунаре?"
_ANSWER = (
    "Proizvođač daje dvogodišnje jemstvo za prenosne kompjutere, "
    "računajući od dana kupovine."
)
_DISTRACTORS = (
    "Garancija ne pokriva oštećenja nastala padom laptopa ili prolivanjem tečnosti.",
    "Laptop torbe i punjači prodaju se odvojeno od računara.",
    "Servis za mobilne telefone radi radnim danima od 9 do 17 časova.",
    "Reklamacija se podnosi uz fiskalni račun i popunjen garantni list.",
    "Cena laptopa uključuje PDV i besplatnu dostavu na teritoriji Srbije.",
    "Baterija laptopa obično izdrži između četiri i osam sati rada.",
    "Za monitore i štampače važi garancija od šest meseci.",
    "Kupci mogu vratiti proizvod u roku od 14 dana bez navođenja razloga.",
    "Novi modeli laptopova stižu u prodavnice početkom oktobra.",
)


async def test_served_model_matches_the_pin(
    reranker_server: RerankerServerSettings,
) -> None:
    async with tei_client(reranker_server.url, timeout=60) as client:
        info = await server_info(client)
    check_reranker(info, model=reranker_server.model, revision=reranker_server.revision)
    assert info.max_client_batch_size == 4


async def test_batched_scores_match_scores_of_single_texts(
    reranker_server: RerankerServerSettings,
) -> None:
    texts = [_ANSWER, *_DISTRACTORS[:8]]
    async with tei_client(reranker_server.url, timeout=60) as client:
        batched = await TeiReranker(client, batch_size=4, max_concurrency=2).score(
            _QUESTION, texts
        )
        single = TeiReranker(client, batch_size=1, max_concurrency=1)
        one_by_one = [(await single.score(_QUESTION, [text]))[0] for text in texts]
    assert all(math.isfinite(score) for score in batched)
    assert batched == pytest.approx(one_by_one, abs=1e-3)


async def test_reranking_ranks_a_cyrillic_questions_latin_paraphrase_first(
    reranker_env: pytest.MonkeyPatch, reranker_server: RerankerServerSettings
) -> None:
    reranker_env.setenv("RERANKER_HOST", reranker_server.host)
    reranker_env.setenv("RERANKER_PORT", str(reranker_server.port))
    async with open_container(
        openai_transport=httpx2.MockTransport(FakeEmbeddingsAPI())
    ) as container:
        for n, text in enumerate((_ANSWER, *_DISTRACTORS)):
            await container.ingestion.ingest(
                text.encode(), f"warranty-{n}.txt", "text/plain"
            )
        results = await container.retrieval.retrieve(_QUESTION, top_k=3)

    top = results[0]
    assert top.chunk.text == _ANSWER
    assert top.rerank and top.rerank.rank == 1
    assert top.fusion and top.fusion.rank > 1
