"""Reasoning-layer tests with a stubbed provider.

No API key required. These cover request construction, provider selection, and
response handling. They do not verify that the model's written output obeys the
system prompt -- that needs a live call and an eval set, and is the one part of
this layer that stays unverified offline.
"""

import json

import pytest

from evaluator.llm import evaluator as llm
from evaluator.llm.provider import (
    AnthropicProvider,
    GeminiProvider,
    LLMResponse,
    LLMUnavailable,
    get_provider,
)

PAYLOAD = {
    "ticker": "TEST",
    "as_of": "2026-09-08",
    "gbm": {"available": True, "targets": {"direction_5d": {"probabilities": {"DROP": 0.1}}}},
    "sentiment": {"available": False, "reason": "feed empty"},
    "model_quality": {"interpretation": "This model does NOT beat base rates."},
}


class StubProvider:
    name = "stub"

    def __init__(self, text="## Assessment\nNo skill.", refused=False):
        self.text = text
        self.refused = refused
        self.calls = []

    def available(self):
        return True

    def complete(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user, **kwargs})
        return LLMResponse(
            "" if self.refused else self.text,
            "stub-model",
            self.name,
            refused=self.refused,
            refusal_reason="SAFETY" if self.refused else None,
        )


def test_payload_is_sent_verbatim_as_json():
    provider = StubProvider()
    llm.evaluate(PAYLOAD, provider=provider)

    user = provider.calls[0]["user"]
    embedded = json.loads(user.split("```json")[1].split("```")[0])
    assert embedded == PAYLOAD


def test_system_prompt_defers_to_the_measured_skill_verdict():
    provider = StubProvider()
    llm.evaluate(PAYLOAD, provider=provider)

    system = provider.calls[0]["system"]
    assert "model_quality.interpretation" in system
    assert "must not be presented" in system.lower() or "must NOT present" in system


def test_system_prompt_forbids_reasoning_about_absent_signals():
    system = llm.SYSTEM_PROMPT
    assert "`available` field is false" in system
    assert "not evidence of calm" in system


def test_system_prompt_frames_output_as_research_not_advice():
    assert "never phrase output as advice" in llm.SYSTEM_PROMPT.lower()


def test_returns_extracted_text():
    result = llm.evaluate(PAYLOAD, provider=StubProvider())
    assert result["refused"] is False
    assert result["evaluation"] == "## Assessment\nNo skill."
    assert result["provider"] == "stub"


def test_refusal_is_reported_not_raised():
    result = llm.evaluate(PAYLOAD, provider=StubProvider(refused=True))
    assert result["refused"] is True
    assert result["evaluation"] is None
    assert result["refusal_reason"] == "SAFETY"


class TestProviderSelection:
    def test_explicit_kind_wins(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        assert isinstance(get_provider("anthropic"), AnthropicProvider)
        assert isinstance(get_provider("gemini"), GeminiProvider)

    def test_env_var_selects_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        assert isinstance(get_provider(), AnthropicProvider)

    def test_falls_back_to_whichever_has_credentials(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        assert isinstance(get_provider(), AnthropicProvider)

    def test_gemini_preferred_when_both_present(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
        assert isinstance(get_provider(), GeminiProvider)


class TestGeminiProvider:
    def test_missing_key_raises_a_clear_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        provider = GeminiProvider()
        assert provider.available() is False
        with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
            provider.complete("sys", "user")

    def test_api_key_is_read_from_environment_only(self, monkeypatch):
        """The key must never be a default, a constant, or a committed value."""
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        assert GeminiProvider().api_key == "from-env"

    def test_no_credential_literal_is_committed(self):
        """Guard against a key being pasted into the provider module.

        Checks for the shapes real credentials take rather than any specific
        value, so the test itself never carries key material.
        """
        import inspect
        import re

        from evaluator.llm import provider as provider_module

        source = inspect.getsource(provider_module)
        credential_shapes = [
            r"AIza[0-9A-Za-z_\-]{30,}",       # Google API key
            r"sk-ant-[0-9A-Za-z_\-]{20,}",    # Anthropic key
            r"AQ\.[0-9A-Za-z_\-]{30,}",       # Google OAuth-style token
        ]
        for pattern in credential_shapes:
            assert not re.search(pattern, source), f"credential literal matching {pattern}"

    def test_missing_key_is_not_silently_treated_as_available(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert GeminiProvider().available() is False
