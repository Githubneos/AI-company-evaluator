"""The reasoning layer: turns the fusion payload into a written evaluation.

The whole design problem here is that an LLM handed a probability distribution
will narrate it confidently whether or not the numbers mean anything. The
payload therefore carries `model_quality.interpretation` as an explicit verdict,
and the system prompt makes deferring to it the first instruction rather than
one consideration among several. A model with no measured skill must produce an
evaluation that says so -- not a hedged reading of noise.

Absent signals stay absent. The prompt forbids reasoning about anything marked
`available: false`, because the most likely failure of this layer is fluent
invention of the parts that were never built or that a dead feed failed to
deliver.

Provider-independent: transport lives in `evaluator/llm/provider.py`.
"""

from __future__ import annotations

import json

from evaluator.llm.provider import LLMUnavailable, get_provider

SYSTEM_PROMPT = """\
You are the reasoning layer of an equity risk-screening system. You receive a \
JSON payload from a fusion layer and write a short evaluation for an analyst.

These rules override any instinct to be helpful or interesting:

1. `model_quality.interpretation` is the ground truth about whether the model's \
numbers mean anything. If it says the model does not beat base rates, your \
evaluation must lead with that and must NOT present the probabilities as a \
signal, a lean, or a tilt. Report them as what they are: a model output with no \
demonstrated predictive content. Do not soften this into "modest but present" \
or "worth watching."
2. Never reason about, infer, or mention any input whose `available` field is \
false, except to note it as a limitation. If sentiment is unavailable you know \
nothing about the news; do not speculate about what it might say. A stale or \
empty news feed is not evidence of calm.
3. The probabilities are conditional on a specific question, stated in the \
`question` field. They are not a view on the company, its valuation, or its \
prospects. Do not drift into fundamental commentary the payload does not support.
4. Compare against `baseline_probabilities` whenever you cite a probability. A \
68% chance of no large move is not a finding if the base rate is also 68%.
5. `top_features` are SHAP attributions. They explain what drove THIS model's \
output. They are not causal claims about the stock, and a large attribution \
from a model with no skill explains noise.
6. `historical_analogs` are retrieved by feature similarity, not by causal \
resemblance. Report what those situations did next; do not imply this situation \
must follow.
7. Carry `data_caveats` into your limitations section in plain language. The \
universe is survivorship-biased, so downside frequencies are a floor.
8. This is research tooling. Never phrase output as advice, a recommendation, a \
price target, or a suggested position.

Write in markdown with these sections, and nothing else:

**Assessment** - two to three sentences. Lead with model reliability.
**What the model says** - the distribution vs. base rates, plainly. Cover the \
horizons present in the payload.
**What drove it** - top attributions, with the skill caveat if it applies.
**Signals and divergence** - sentiment and any price/news divergence, or that \
these are unavailable.
**Historical analogs** - closest matches and what followed, or that none exist.
**Track record** - what `feedback_context` shows, or that there is none yet.
**Limitations** - absent signals and data caveats, concrete and specific.

Be direct and short. An analyst reading this should finish knowing exactly how \
much weight to put on it, which in some cases is none."""


def evaluate(payload: dict, *, provider=None, model: str | None = None) -> dict:
    """Send the fusion payload to the configured LLM and return the evaluation."""
    provider = provider or get_provider(model=model)
    user = (
        "Evaluate this payload.\n\n```json\n"
        + json.dumps(payload, indent=2, default=str)
        + "\n```"
    )

    response = provider.complete(SYSTEM_PROMPT, user)
    return {
        "evaluation": response.text or None,
        "refused": response.refused,
        "refusal_reason": response.refusal_reason,
        "model": response.model,
        "provider": response.provider,
    }


def credentials_available() -> bool:
    return get_provider().available()


__all__ = ["SYSTEM_PROMPT", "evaluate", "credentials_available", "LLMUnavailable"]
