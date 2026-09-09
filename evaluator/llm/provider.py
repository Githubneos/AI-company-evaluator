"""LLM provider abstraction.

Two implementations behind one call: Gemini (default, free tier) and Anthropic.
The reasoning-layer system prompt is provider-independent and lives in
`evaluator/llm/evaluator.py`; only transport differs here.

Credentials come from the environment only -- `GEMINI_API_KEY` or
`ANTHROPIC_API_KEY`. No key is ever written to source, config, artifacts, or the
prediction database.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

log = logging.getLogger(__name__)

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

#: Reasoning model for evaluations, and a cheap one for bulk post-mortem
#: tagging where volume matters more than nuance.
GEMINI_EVALUATOR_MODEL = "gemini-3.1-pro-preview"
GEMINI_TAGGER_MODEL = "gemini-3.5-flash-lite"
ANTHROPIC_EVALUATOR_MODEL = "claude-opus-5"


class LLMUnavailable(RuntimeError):
    """No credentials, or the provider could not be reached."""


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    refused: bool = False
    refusal_reason: str | None = None


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str = GEMINI_EVALUATOR_MODEL, timeout: int = 120):
        self.model = model
        self.timeout = timeout
        self.api_key = os.environ.get("GEMINI_API_KEY", "")

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, *, max_output_tokens: int = 8192) -> LLMResponse:
        if not self.available():
            raise LLMUnavailable(
                "GEMINI_API_KEY is not set. Export it, or configure the Anthropic provider."
            )

        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_output_tokens},
        }
        request = urllib.request.Request(
            f"{GEMINI_ENDPOINT.format(model=self.model)}?key={self.api_key}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:400]
            raise LLMUnavailable(f"Gemini HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMUnavailable(f"Gemini unreachable: {exc}") from exc

        candidates = body.get("candidates") or []
        if not candidates:
            blocked = (body.get("promptFeedback") or {}).get("blockReason")
            return LLMResponse("", self.model, self.name, refused=True, refusal_reason=blocked)

        candidate = candidates[0]
        finish = candidate.get("finishReason")
        if finish in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"):
            return LLMResponse("", self.model, self.name, refused=True, refusal_reason=finish)

        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        return LLMResponse(text, self.model, self.name)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_EVALUATOR_MODEL):
        self.model = model

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def complete(self, system: str, user: str, *, max_output_tokens: int = 8192) -> LLMResponse:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("anthropic SDK is not installed") from exc

        client = anthropic.Anthropic()
        try:
            response = client.beta.messages.create(
                model=self.model,
                max_tokens=max_output_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"Anthropic request failed: {exc}") from exc

        if response.stop_reason == "refusal":
            return LLMResponse(
                "", response.model, self.name,
                refused=True,
                refusal_reason=getattr(response.stop_details, "category", None),
            )
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(text, response.model, self.name)


def get_provider(kind: str | None = None, model: str | None = None):
    """Pick a provider: explicit `kind`, else `LLM_PROVIDER`, else whatever has keys."""
    kind = (kind or os.environ.get("LLM_PROVIDER") or "").lower()

    if kind == "anthropic":
        return AnthropicProvider(model or ANTHROPIC_EVALUATOR_MODEL)
    if kind == "gemini":
        return GeminiProvider(model or GEMINI_EVALUATOR_MODEL)

    gemini = GeminiProvider(model or GEMINI_EVALUATOR_MODEL)
    if gemini.available():
        return gemini
    anthropic_provider = AnthropicProvider(model or ANTHROPIC_EVALUATOR_MODEL)
    if anthropic_provider.available():
        return anthropic_provider
    return gemini  # raises a clear LLMUnavailable when actually called
