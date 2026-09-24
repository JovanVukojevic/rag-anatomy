from tests.fakes import FakeLLMClient, Prompt


async def test_llm_returns_canned_response_and_records_prompts() -> None:
    llm = FakeLLMClient(response="42 [1]")
    assert await llm.complete("be terse", "what is it?") == "42 [1]"
    assert llm.prompts == [Prompt(system="be terse", user="what is it?")]
