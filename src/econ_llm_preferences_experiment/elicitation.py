from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass

from econ_llm_preferences_experiment.models import DIMENSIONS, AgentInferred, AgentTruth
from econ_llm_preferences_experiment.openai_client import OpenAIClient, OpenAIUsage


def _level(x: float) -> str:
    if x >= 0.30:
        return "high"
    if x >= 0.18:
        return "medium"
    if x >= 0.08:
        return "low"
    return "none"


def standard_form_text(agent: AgentTruth) -> str:
    """
    A plausible "standard form" baseline: it captures the top-3 dimensions.
    """
    return standard_form_text_topk(agent, top_k=3)


def standard_form_text_topk(agent: AgentTruth, *, top_k: int) -> str:
    k = len(DIMENSIONS)
    weights = agent.weights
    top_k = max(1, min(k, int(top_k)))
    top = sorted(range(k), key=lambda idx: weights[idx], reverse=True)[:top_k]
    lines = [
        f"Side: {agent.side}",
        f"Category: {agent.category}",
        f"Stated priorities (top-{top_k}):",
    ]
    for idx in top:
        lines.append(f"- {DIMENSIONS[idx]} is {_level(weights[idx])} importance")
    lines.append("Other dimensions are not specified.")
    return "\n".join(lines)


def standard_long_form_text(agent: AgentTruth, *, rng: random.Random) -> str:
    """
    A more complete baseline: many platforms collect richer free-text + multiple-choice
    answers, but without an adaptive agent. This is intentionally "messy" (still synthetic).
    """
    synonyms: dict[str, list[str]] = {
        "price_sensitivity": ["price", "cost", "budget"],
        "quality_focus": ["quality", "craftsmanship"],
        "speed_urgency": ["speed", "turnaround", "asap"],
        "communication_fit": ["communication", "updates", "responsiveness"],
        "weirdness_tolerance": ["weird stuff", "edge cases", "surprises"],
        "schedule_flexibility": ["schedule", "timing", "availability"],
    }

    lines = [
        f"Side: {agent.side}",
        f"Category: {agent.category}",
        "",
        "Free-response preferences (not all fields filled):",
    ]
    for dim, w in zip(DIMENSIONS, agent.weights, strict=True):
        if rng.random() < 0.25:
            continue
        lvl = _level(w)
        mention = rng.choice(synonyms.get(dim, [dim]))
        if rng.random() < 0.35:
            lines.append(f"- {mention}: {lvl} priority ({dim})")
        else:
            lines.append(f"- {mention}: {lvl} priority")
        if rng.random() < 0.10:
            lines[-1] = lines[-1].replace("priority", "prioirty")  # typo

    if rng.random() < 0.50:
        lines.append("")
        lines.append("Other notes:")
        if agent.weights[0] >= 0.30:
            lines.append("- Please keep it affordable.")
        if agent.weights[1] >= 0.30:
            lines.append("- I care a lot about quality.")
        if agent.weights[3] >= 0.18:
            lines.append("- Good communication matters.")
        if agent.weights[4] >= 0.30:
            lines.append("- Not into unusual/weird jobs.")

    return "\n".join(lines)


def ai_conversation_transcript(agent: AgentTruth, *, rng: random.Random) -> str:
    """
    A lightweight "conversational" transcript that surfaces all dimensions.
    """
    lines = [
        "Agent: I’ll ask a few quick questions to understand your preferences.",
        f"User: (side={agent.side}, category={agent.category})",
    ]
    for dim, w in zip(DIMENSIONS, agent.weights, strict=True):
        lines.append(f"Agent: How important is `{dim}` for you?")
        lvl = _level(w)
        if rng.random() < 0.12:
            lines.append(f"User: {dim}… {lvl}, I think.")
        else:
            lines.append(f"User: {dim} is {lvl} importance.")
    tags = ["budget_sensitive"] if agent.weights[0] >= 0.30 else []
    tags += ["quality_first"] if agent.weights[1] >= 0.30 else []
    if not tags:
        tags = ["balanced"]
    lines.append("Agent: Any keywords/tags that describe you best?")
    lines.append(f"User: {', '.join(tags)}")
    return "\n".join(lines)


@dataclass(frozen=True)
class ParsedBatch:
    inferred: tuple[AgentInferred, ...]
    raw_text: str
    usage: OpenAIUsage


def _prompt_for_batch(items: Iterable[tuple[str, str]]) -> str:
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["agent_id", "weights", "tags"],
            "properties": {
                "agent_id": {"type": "string"},
                "weights": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": len(DIMENSIONS),
                    "maxItems": len(DIMENSIONS),
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    }

    return "\n".join(
        [
            "You are extracting structured preference weights from marketplace intake text.",
            "",
            "Rules:",
            "- Output MUST be valid JSON only (no markdown, no prose).",
            "- Output MUST be compact/minified JSON (no pretty-print, no extra whitespace).",
            "- For any dimension not mentioned (directly or via a clear synonym), "
            "set its weight to 0.0.",
            "- Map importance levels to numeric weights:",
            "  high=0.35, medium=0.20, low=0.10, none=0.00.",
            "- After mapping, renormalize weights to sum to 1.0 (if all zero, set uniform).",
            "- tags: extract any tags/keywords mentioned; else []",
            f"- weights MUST be an array in this exact order: {list(DIMENSIONS)}",
            "- Treat these common-sense synonym mentions as the corresponding dimensions:",
            "  price/cost/budget -> price_sensitivity",
            "  quality/craftsmanship -> quality_focus",
            "  speed/turnaround/asap/urgent -> speed_urgency",
            "  communication/updates/responsiveness -> communication_fit",
            "  weird/unusual/edge/surprises -> weirdness_tolerance",
            "  schedule/timing/availability -> schedule_flexibility",
            "",
            f"JSON Schema:\n{json.dumps(schema, indent=2)}",
            "",
            "Items to parse (agent_id -> text):",
            *[f"\n[{agent_id}]\n{text}" for agent_id, text in items],
        ]
    )


def _renormalize_list(weights: list[float]) -> tuple[float, ...]:
    vals = [max(0.0, float(x)) for x in weights[: len(DIMENSIONS)]]
    total = sum(vals)
    if total <= 0:
        return tuple(1.0 / len(vals) for _ in vals)
    return tuple(v / total for v in vals)


def parse_batch_with_gpt(
    *,
    client: OpenAIClient,
    texts_by_agent_id: dict[str, str],
    truth_by_agent_id: dict[str, AgentTruth],
    max_output_tokens: int = 6000,
) -> ParsedBatch:
    prompt = _prompt_for_batch(texts_by_agent_id.items())
    resp = client.responses_create(input_text=prompt, max_output_tokens=max_output_tokens)
    raw = resp.text.strip()

    parsed = json.loads(raw)
    inferred: list[AgentInferred] = []
    for item in parsed:
        agent_id = str(item["agent_id"])
        truth = truth_by_agent_id[agent_id]
        weights = _renormalize_list(item["weights"])
        tags = tuple(str(t) for t in item.get("tags", []))
        inferred.append(
            AgentInferred(
                agent_id=agent_id,
                side=truth.side,
                category=truth.category,
                weights=weights,
                tags=tags,
            )
        )
    inferred.sort(key=lambda a: a.agent_id)
    return ParsedBatch(inferred=tuple(inferred), raw_text=raw, usage=resp.usage)


# --- Functions for field_sim_v2 integration (simpler interface) ---


def weights_to_ai_text(
    *,
    agent_id: str,
    weights: tuple[float, ...],
    side: str,
    category: str,
    rng: random.Random,
) -> str:
    """
    Generate an AI conversation transcript from raw weights.
    Simpler interface for field_sim_v2 integration.
    """
    lines = [
        "Agent: I'll ask a few quick questions to understand your preferences.",
        f"User: (side={side}, category={category})",
    ]
    for dim, w in zip(DIMENSIONS, weights, strict=True):
        lines.append(f"Agent: How important is `{dim}` for you?")
        lvl = _level(w)
        if rng.random() < 0.12:
            lines.append(f"User: {dim}… {lvl}, I think.")
        else:
            lines.append(f"User: {dim} is {lvl} importance.")
    tags = ["budget_sensitive"] if weights[0] >= 0.30 else []
    tags += ["quality_first"] if weights[1] >= 0.30 else []
    if not tags:
        tags = ["balanced"]
    lines.append("Agent: Any keywords/tags that describe you best?")
    lines.append(f"User: {', '.join(tags)}")
    return "\n".join(lines)


def weights_to_standard_text(
    *,
    agent_id: str,
    weights: tuple[float, ...],
    side: str,
    category: str,
) -> str:
    """
    Generate a standard form text from raw weights.
    Simpler interface for field_sim_v2 integration.
    """
    k = len(DIMENSIONS)
    top_k = min(3, k)
    top = sorted(range(k), key=lambda idx: weights[idx], reverse=True)[:top_k]
    lines = [
        f"Side: {side}",
        f"Category: {category}",
        f"Stated priorities (top-{top_k}):",
    ]
    for idx in top:
        lines.append(f"- {DIMENSIONS[idx]} is {_level(weights[idx])} importance")
    lines.append("Other dimensions are not specified.")
    return "\n".join(lines)


@dataclass(frozen=True)
class ParsedWeightsBatch:
    """Simpler result type for field_sim_v2 integration."""

    weights_by_id: dict[str, tuple[float, ...]]
    usage: OpenAIUsage


def parse_weights_batch_with_gpt(
    *,
    client: OpenAIClient,
    texts_by_id: dict[str, str],
    max_output_tokens: int = 6000,
) -> ParsedWeightsBatch:
    """
    Parse preference texts and return just the weights.
    Simpler interface for field_sim_v2 integration.
    """
    prompt = _prompt_for_batch(texts_by_id.items())
    resp = client.responses_create(input_text=prompt, max_output_tokens=max_output_tokens)
    raw = resp.text.strip()

    parsed = json.loads(raw)
    weights_by_id: dict[str, tuple[float, ...]] = {}
    for item in parsed:
        agent_id = str(item["agent_id"])
        weights = _renormalize_list(item["weights"])
        weights_by_id[agent_id] = weights

    return ParsedWeightsBatch(weights_by_id=weights_by_id, usage=resp.usage)
