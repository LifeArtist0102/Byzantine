import pytest

from byzantine.generation.deepseek import (
    GroundingError,
    build_user_prompt,
    generate_grounded_answer,
    make_sources,
    summarize_research_chat,
    validate_citations,
)


def evidence() -> dict:
    return {
        "hits": [
            {
                "section_path": ["PART I", "INTRODUCTION"],
                "page_start": 298,
                "page_end": 299,
                "text": "Basil II exercised real power after 976.",
            }
        ]
    }


def test_sources_keep_book_citations() -> None:
    sources = make_sources(evidence(), max_characters=200)
    prompt = build_user_prompt("Who ruled?", sources)

    assert sources[0].label == "S1"
    assert sources[0].pages == "PDF pp. 298-299"
    assert "<SOURCE label=\"[S1]\">" in prompt


def test_validation_rejects_missing_or_unknown_citations() -> None:
    sources = make_sources(evidence(), max_characters=200)

    with pytest.raises(GroundingError, match="did not include"):
        validate_citations("Basil II ruled.", sources)
    with pytest.raises(GroundingError, match="unknown"):
        validate_citations("Basil II ruled. [S9]", sources)


def test_generation_uses_fake_client_and_returns_sources() -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            message = type("Message", (), {"content": "Basil II held real power after 976. [S1]"})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    completions = FakeCompletions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    result = generate_grounded_answer(
        "Who held power?",
        evidence(),
        api_key=None,
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        max_output_tokens=500,
        temperature=0.1,
        max_evidence_characters=200,
        client=client,
    )

    assert result["answer"].endswith("[S1]")
    assert result["sources"][0]["label"] == "S1"
    assert completions.kwargs["model"] == "deepseek-v4-flash"


def test_research_summary_is_cited_and_structured() -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            content = '{"title":"Basil II","tags":["rulership"],"summary":"Basil II held real power after 976. [S1]"}'
            message = type("Message", (), {"content": content})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    completions = FakeCompletions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    summary = summarize_research_chat(
        [{"role": "user", "content": "Who held power?"}],
        evidence(),
        api_key=None,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        max_evidence_characters=200,
        client=client,
    )

    assert summary["tags"] == ["rulership"]
    assert "[S1]" in summary["summary"]
