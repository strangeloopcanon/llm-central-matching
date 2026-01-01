from __future__ import annotations

import json
from pathlib import Path

from econ_llm_preferences_experiment.models import DIMENSIONS
from econ_llm_preferences_experiment.openai_client import OpenAIUsage
from econ_llm_preferences_experiment.roth_da_sweep import run_sweep
from econ_llm_preferences_experiment.simulation import MarketParams


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage = OpenAIUsage()


class _FakeEnv:
    model = "fake"
    base_url = "https://example.invalid"


class FakeOpenAIClient:
    env = _FakeEnv()

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


def test_roth_da_sweep_writes_outputs(tmp_path: Path) -> None:
    client = FakeOpenAIClient()
    run_sweep(
        out_dir=tmp_path,
        replications=2,
        seed=1,
        attention_cost=0.01,
        da_proposer_side="customer",
        da_list_ks=[1, 3],
        standard_form_top_k=1,
        market_params=MarketParams(n_customers=6, n_providers=6, idiosyncratic_noise_sd=0.08),
        client=client,  # type: ignore[arg-type]
    )
    assert (tmp_path / "summary_table.csv").exists()
    assert (tmp_path / "effects_table.csv").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "fig_easy_net_welfare_by_k.svg").exists()
    assert (tmp_path / "fig_hard_net_welfare_by_k.svg").exists()
