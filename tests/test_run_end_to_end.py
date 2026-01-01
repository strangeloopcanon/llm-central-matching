from __future__ import annotations

import json

from econ_llm_preferences_experiment.elicitation import parse_batch_with_gpt
from econ_llm_preferences_experiment.models import DIMENSIONS, AgentTruth
from econ_llm_preferences_experiment.openai_client import OpenAIUsage
from econ_llm_preferences_experiment.run import _write_report, run_once
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
            weights = [1.0 for _ in DIMENSIONS]
            items.append(
                {
                    "agent_id": agent_id,
                    "side": "customer",
                    "category": "easy",
                    "weights": weights,
                    "tags": [],
                }
            )
        return _FakeResp(json.dumps(items))


def test_parse_batch_with_fake_client() -> None:
    client = FakeOpenAIClient()
    truth = {
        "c000": AgentTruth(
            agent_id="c000",
            side="customer",
            category="easy",
            weights=(1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6),
        )
    }
    texts = {
        "c000": (
            "Side: customer\nCategory: easy\nStated priorities (top-1 only):\n"
            "- quality_focus is high importance"
        )
    }
    parsed = parse_batch_with_gpt(client=client, texts_by_agent_id=texts, truth_by_agent_id=truth)
    assert parsed.inferred[0].agent_id == "c000"
    assert abs(sum(parsed.inferred[0].weights) - 1.0) < 1e-9


def test_run_once_and_write_report(tmp_path) -> None:
    client = FakeOpenAIClient()
    params = MarketParams(n_customers=6, n_providers=6)
    rows, meta = run_once(
        category="easy",
        client=client,
        market_params=params,
        replications=3,
        seed=1,
        attention_cost=0.01,
    )
    assert len(rows) == 4

    out_dir = tmp_path / "report"
    _write_report(out_dir=out_dir, summary_rows=rows, effects_rows=None, metadata=meta)
    assert (out_dir / "summary_table.csv").exists()
    assert (out_dir / "summary_table.md").exists()
    assert (out_dir / "README.md").exists()
    assert (out_dir / "fig_easy_match_rate.svg").exists()


def test_run_once_with_deferred_acceptance_central(tmp_path) -> None:
    client = FakeOpenAIClient()
    params = MarketParams(n_customers=6, n_providers=6)
    rows, meta = run_once(
        category="easy",
        client=client,
        market_params=params,
        replications=3,
        seed=1,
        attention_cost=0.01,
        central_mechanism="da",
        da_proposer_side="customer",
    )
    assert len(rows) == 4

    out_dir = tmp_path / "report_da"
    _write_report(out_dir=out_dir, summary_rows=rows, effects_rows=None, metadata=meta)
    assert (out_dir / "summary_table.csv").exists()
