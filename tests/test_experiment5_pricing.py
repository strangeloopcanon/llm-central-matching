from __future__ import annotations

import json
from pathlib import Path

from econ_llm_preferences_experiment.experiment5_pricing import run_pricing_experiment
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


def test_run_pricing_experiment_smoke(tmp_path: Path) -> None:
    client = FakeOpenAIClient()
    params = MarketParams(idiosyncratic_noise_sd=0.08)
    rows = run_pricing_experiment(
        category="hard",
        out_dir=tmp_path,
        seed=1,
        n_cells=1,
        market_params=params,
        client=client,
        n_customers=10,
        n_providers=6,
    )
    assert {str(r["pricing_mode"]) for r in rows} == {"none", "fixed", "dynamic"}
    assert all(str(r["category"]) == "hard" for r in rows)
