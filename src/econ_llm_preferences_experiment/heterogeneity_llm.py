"""
LLM-backed heterogeneity sweep using real GPT calls.

Unlike field_sim_v2_heterogeneity.py which uses synthetic AI simulation,
this module uses real LLM calls via parse_batch_with_gpt() to parse
preference descriptions into weights.

Sweeps: idiosyncratic_noise_sd (utility randomness)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, TypeAlias

from econ_llm_preferences_experiment.elicitation import (
    ai_conversation_transcript,
    parse_batch_with_gpt,
    standard_form_text,
)
from econ_llm_preferences_experiment.logging_utils import get_logger, log
from econ_llm_preferences_experiment.mechanisms import (
    CentralizedParams,
    SearchParams,
    centralized_recommendations,
    decentralized_search,
)
from econ_llm_preferences_experiment.models import Category
from econ_llm_preferences_experiment.openai_client import OpenAIClient
from econ_llm_preferences_experiment.plotting import LineSeries, write_line_chart_svg
from econ_llm_preferences_experiment.simulation import (
    MarketParams,
    generate_market_instance,
    generate_population,
    inferred_value_matrix,
)

logger = get_logger(__name__)

RowValue: TypeAlias = str | float | int


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _se(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _arm_label(elicitation: str, mechanism: str) -> str:
    return f"{elicitation}_{mechanism}"


def run_once_with_llm(
    *,
    category: Category,
    client: OpenAIClient,
    market_params: MarketParams,
    replications: int,
    seed: int,
    attention_cost: float,
) -> dict[str, float]:
    """Run a single configuration with real LLM parsing."""
    rng = random.Random(seed)  # nosec B311
    customers, providers = generate_population(
        rng=rng,
        category=category,
        n_customers=market_params.n_customers,
        n_providers=market_params.n_providers,
    )

    truth = {a.agent_id: a for a in customers + providers}
    std_texts = {a.agent_id: standard_form_text(a) for a in customers + providers}
    ai_texts = {a.agent_id: ai_conversation_transcript(a, rng=rng) for a in customers + providers}

    log(logger, 20, "llm_parsing_start", category=category, agents=len(truth))
    parsed_std = parse_batch_with_gpt(
        client=client, texts_by_agent_id=std_texts, truth_by_agent_id=truth
    )
    parsed_ai = parse_batch_with_gpt(
        client=client, texts_by_agent_id=ai_texts, truth_by_agent_id=truth
    )
    log(logger, 20, "llm_parsing_done", category=category)

    inferred_std = {a.agent_id: a for a in parsed_std.inferred}
    inferred_ai = {a.agent_id: a for a in parsed_ai.inferred}

    def weights(side: str, which: str) -> tuple[tuple[float, ...], ...]:
        src = inferred_std if which == "standard" else inferred_ai
        agents = customers if side == "customer" else providers
        return tuple(src[a.agent_id].weights for a in agents)

    weights_customer = {
        "standard": weights("customer", "standard"),
        "ai": weights("customer", "ai"),
    }
    weights_provider = {
        "standard": weights("provider", "standard"),
        "ai": weights("provider", "ai"),
    }

    n_customers = len(customers)
    net_welfare: dict[str, list[float]] = {
        _arm_label("standard", "search"): [],
        _arm_label("standard", "central"): [],
        _arm_label("ai", "search"): [],
        _arm_label("ai", "central"): [],
    }

    for r in range(replications):
        rep_rng = random.Random(seed * 10_000 + r)  # nosec B311
        market = generate_market_instance(
            rng=rep_rng,
            customers=customers,
            providers=providers,
            idiosyncratic_noise_sd=market_params.idiosyncratic_noise_sd,
        )

        for elicitation in ("standard", "ai"):
            vhat_c = inferred_value_matrix(
                weights_by_agent=weights_customer[elicitation],
                partner_attributes=market.provider_attributes,
            )
            vhat_p = inferred_value_matrix(
                weights_by_agent=weights_provider[elicitation],
                partner_attributes=market.customer_attributes,
            )

            outcome_search = decentralized_search(
                v_customer_true=market.v_customer,
                v_provider_true=market.v_provider,
                v_customer_hat=vhat_c,
                accept_threshold=market_params.accept_threshold,
                params=SearchParams(max_rounds=30),
            )
            tv_sum = sum(
                market.v_customer[i][j] + market.v_provider[j][i] for i, j in outcome_search.matches
            )
            attn_total = outcome_search.proposals + outcome_search.accept_decisions
            net_welfare[_arm_label(elicitation, "search")].append(
                (tv_sum / n_customers) - attention_cost * (attn_total / n_customers)
            )

            outcome_central = centralized_recommendations(
                v_customer_true=market.v_customer,
                v_provider_true=market.v_provider,
                v_customer_hat=vhat_c,
                v_provider_hat=vhat_p,
                accept_threshold=market_params.accept_threshold,
                params=CentralizedParams(rec_k=3),
            )
            tv_sum = sum(
                market.v_customer[i][j] + market.v_provider[j][i]
                for i, j in outcome_central.matches
            )
            attn_total = outcome_central.proposals + outcome_central.accept_decisions
            net_welfare[_arm_label(elicitation, "central")].append(
                (tv_sum / n_customers) - attention_cost * (attn_total / n_customers)
            )

    # Compute DiD: (ai_central - ai_search) - (std_central - std_search)
    did = [
        (net_welfare["ai_central"][t] - net_welfare["ai_search"][t])
        - (net_welfare["standard_central"][t] - net_welfare["standard_search"][t])
        for t in range(replications)
    ]

    return {
        "net_welfare_did": _mean(did),
        "net_welfare_did_se": _se(did),
        "ai_central_mean": _mean(net_welfare["ai_central"]),
        "std_central_mean": _mean(net_welfare["standard_central"]),
    }


def _write_csv(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-backed heterogeneity sweep (real GPT calls)")
    parser.add_argument("--out", default="reports/heterogeneity_llm_latest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--attention-cost", type=float, default=0.01)
    parser.add_argument(
        "--utility-noise-sds",
        type=str,
        default="0.04,0.08,0.15",
        help="Idiosyncratic utility noise levels to sweep.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    utility_noise_sds = [float(x.strip()) for x in args.utility_noise_sds.split(",")]
    client = OpenAIClient(max_calls=50)  # Allow more calls for sweep

    rows: list[dict[str, RowValue]] = []

    for un_sd in utility_noise_sds:
        for category in ("easy", "hard"):
            log(
                logger,
                20,
                "llm_sweep_run",
                utility_noise_sd=un_sd,
                category=category,
            )
            market_params = MarketParams(idiosyncratic_noise_sd=un_sd)
            result = run_once_with_llm(
                category=category,
                client=client,
                market_params=market_params,
                replications=args.replications,
                seed=args.seed,
                attention_cost=args.attention_cost,
            )
            rows.append(
                {
                    "utility_noise_sd": un_sd,
                    "category": category,
                    "net_welfare_did": round(result["net_welfare_did"], 4),
                    "net_welfare_did_se": round(result["net_welfare_did_se"], 4),
                    "ai_central_mean": round(result["ai_central_mean"], 4),
                    "std_central_mean": round(result["std_central_mean"], 4),
                }
            )

    _write_csv(rows, out_dir / "runs.csv")

    # Summary by utility noise
    by_noise: dict[float, list[float]] = {}
    for r in rows:
        if r["category"] == "hard":  # Focus on hard category
            by_noise.setdefault(float(r["utility_noise_sd"]), []).append(
                _as_float(r["net_welfare_did"])
            )

    summary: list[dict[str, RowValue]] = []
    for un in sorted(by_noise.keys()):
        vals = by_noise[un]
        summary.append(
            {
                "utility_noise_sd": un,
                "mean_did": round(_mean(vals), 4),
                "se_did": round(_se(vals), 4),
            }
        )
    _write_csv(summary, out_dir / "summary_by_utility_noise.csv")

    # Plot
    if summary:
        series = [
            LineSeries(
                label="AI×Central DiD (hard)",
                points=[(float(r["utility_noise_sd"]), _as_float(r["mean_did"])) for r in summary],
            )
        ]
        write_line_chart_svg(
            out_path=out_dir / "fig_did_vs_utility_noise.svg",
            title="AI×Central advantage vs utility noise (LLM-backed)",
            x_label="Utility noise (SD)",
            y_label="DiD: (AI Central - AI Search) - (Std Central - Std Search)",
            series=series,
        )

    # README
    readme_lines = [
        "# LLM-Backed Heterogeneity Sweep",
        "",
        "**Uses real GPT calls** to parse preference descriptions, unlike the",
        "synthetic simulation in `field_sim_v2_heterogeneity.py`.",
        "",
        "## Utility Noise Sweep Results",
        "",
        "| utility_noise_sd | mean_did | se |",
        "|------------------|----------|-----|",
    ]
    for r in summary:
        readme_lines.append(f"| {r['utility_noise_sd']} | {r['mean_did']} | {r['se_did']} |")

    readme_lines.extend(
        [
            "",
            "## Run",
            "",
            "```bash",
            "make heterogeneity-llm",
            "```",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    # Summary JSON
    summary_json = {
        "utility_noise_sds": utility_noise_sds,
        "replications": args.replications,
        "seed": args.seed,
        "model": client.env.model,
        "n_runs": len(rows),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2) + "\n", encoding="utf-8"
    )

    log(logger, 20, "llm_sweep_done", out_dir=str(out_dir), n_runs=len(rows))


if __name__ == "__main__":
    main()
