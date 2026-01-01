from __future__ import annotations

import json

from econ_llm_preferences_experiment.heterogeneity_llm import run_once_with_llm
from econ_llm_preferences_experiment.models import DIMENSIONS
from econ_llm_preferences_experiment.openai_client import OpenAIUsage
from econ_llm_preferences_experiment.simulation import MarketParams


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage = OpenAIUsage()


class FakeOpenAIClient:
    def responses_create(self, *, input_text: str, **_kwargs):
        agent_ids: list[str] = []
        for line in input_text.splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]") and len(line) > 2:
                agent_ids.append(line[1:-1])
        items = []
        for agent_id in agent_ids:
            items.append({"agent_id": agent_id, "weights": [1.0 for _ in DIMENSIONS], "tags": []})
        return _FakeResp(json.dumps(items))


def test_run_once_with_llm_uses_fake_client() -> None:
    client = FakeOpenAIClient()
    market_params = MarketParams(n_customers=6, n_providers=6, idiosyncratic_noise_sd=0.08)
    out = run_once_with_llm(
        category="hard",
        client=client,
        market_params=market_params,
        replications=3,
        seed=2,
        attention_cost=0.01,
    )
    assert {"net_welfare_did", "net_welfare_did_se", "ai_central_mean", "std_central_mean"} <= set(
        out.keys()
    )
