"""
Tests for real-provider answer generation (mocked LLM provider — no network).

Covers Stage 5: AnswerGenerator uses an injected provider, records
provider/model/usage in metadata, and falls back safely on provider errors.
"""

import json
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generation.answer import AnswerGenerator, GroundedAnswer, create_answer_generator
from providers.llm import MockLLMProvider


def make_docs():
    return [
        {"content": "Remote work is allowed up to 3 days per week with manager approval.",
         "metadata": {"source": "data/raw/hr_remote_work_policy.md", "chunk_index": 0, "total_chunks": 1}},
        {"content": "Employees are eligible after a 6-month introductory period.",
         "metadata": {"source": "data/raw/hr_remote_work_policy.md", "chunk_index": 1, "total_chunks": 2}},
    ]


class FakeProvider:
    """Minimal real-provider stand-in that returns a fixed answer + usage."""

    def __init__(self, text="Employees can work remotely 3 days per week."):
        self.text = text
        self.calls = []
        from providers.llm_nvidia import Usage, ChatResult
        self._usage = Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        self.last_result = None
        self._ChatResult = ChatResult

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        self.last_result = self._ChatResult(
            text=self.text, model="fake-model", usage=self._usage, latency_ms=12.0
        )
        return self.text

    def get_model_info(self):
        return {"provider": "nvidia", "model_name": "fake-model"}


def test_generator_uses_real_provider():
    provider = FakeProvider()
    gen = AnswerGenerator(model_name="fake-model", llm_provider=provider)
    result = gen.generate_answer_with_context("How many remote days?", make_docs())
    assert result.answer == "Employees can work remotely 3 days per week."
    assert len(provider.calls) == 1
    # prompt is the grounded template with context + question
    prompt_lower = provider.calls[0].lower()
    assert "context:" in prompt_lower and "question:" in prompt_lower
    # metadata records the real provider/model and provider-reported usage
    assert result.metadata["provider"] == "nvidia"
    assert result.metadata["model_used"] == "fake-model"
    assert result.metadata["usage"]["total_tokens"] == 120
    assert result.metadata["usage"]["source"] == "provider-reported"
    # sources preserved
    assert result.sources[0]["source_filename"] == "hr_remote_work_policy.md"


def test_generator_falls_back_on_provider_error():
    class BadProvider(FakeProvider):
        def generate(self, prompt, **kwargs):
            raise RuntimeError("boom")

    gen = AnswerGenerator(model_name="fake-model", llm_provider=BadProvider())
    result = gen.generate_answer_with_context("q", make_docs())
    assert "don't have enough information" in result.answer


def test_generator_mock_when_no_provider():
    gen = AnswerGenerator(model_name="mock", llm_provider=None)
    result = gen.generate_answer_with_context("What is the remote work policy?", make_docs())
    assert result.metadata["provider"] == "mock"
    assert isinstance(result.answer, str) and len(result.answer) > 0


def test_factory_uses_env_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    gen = create_answer_generator()
    assert gen.llm_provider is not None  # mock provider from env
    assert gen.llm_provider.get_model_info()["provider"] == "mock"


def test_grounded_answer_schema():
    a = GroundedAnswer.model_validate({
        "answer": "3 days", "citations": ["hr_remote_work_policy.md"], "insufficient_evidence": False
    })
    assert a.answer == "3 days" and a.citations == ["hr_remote_work_policy.md"]
    with pytest.raises(Exception):
        GroundedAnswer.model_validate({"answer": ""})  # min_length=1
