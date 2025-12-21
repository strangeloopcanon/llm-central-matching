"""
Experiment 5: Congestion with Pricing

Tests whether introducing attention pricing fixes the congestion externality.
Compares: no pricing (baseline) vs fixed pricing vs dynamic pricing.

Key insight from barter_to_money: Money/Exchange achieves 100% success vs 62.5% for barter
by collapsing O(N²) coordination to O(N) hub-facing interactions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from econ_llm_preferences_experiment.elicitation import (
    ai_conversation_transcript,
    parse_batch_with_gpt,
    standard_form_text,
)
from econ_llm_preferences_experiment.logging_utils import get_logger, log
from econ_llm_preferences_experiment.models import AgentTruth, Category
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


@dataclass(frozen=True)
class PricingParams:
    """Parameters for the pricing mechanism."""

    # Base congestion params
    horizon_days: int = 7
    k_manual_per_day: int = 2
    k_agent_per_day: int = 10
    provider_daily_response_cap: int = 5
    provider_weekly_capacity: int = 4
    customer_accept_threshold: float = 0.25
    provider_accept_threshold: float = 0.25
    provider_threshold_uplift_per_extra_cap: float = 0.10
    provider_threshold_cap: float = 0.55
    manual_ranking_noise_sd: float = 0.18
    agent_ranking_noise_sd: float = 0.06
    provider_eval_noise_sd: float = 0.04
    message_quality_bonus: float = 0.03
    attention_cost: float = 0.01

    # Pricing mechanism params
    enable_pricing: bool = False
    base_attention_price: float = 0.05  # Cost per message to provider
    agent_budget: float = 0.50  # Per-customer budget for outreach
    dynamic_pricing: bool = False
    price_sensitivity: float = 0.02  # Price increase per message above threshold


@dataclass(frozen=True)
class PricingOutcomes:
    saturation: float
    pricing_mode: str  # "none", "fixed", "dynamic"
    match_rate_all: float
    match_rate_treated: float
    match_rate_control: float
    mean_days_to_match_all: float
    messages_sent_per_customer: float
    inbox_per_provider_per_day: float
    provider_response_rate: float
    net_welfare_per_customer: float
    avg_price_paid: float
    budget_utilization: float


def _simulate_with_pricing(
    *,
    rng: random.Random,
    saturation: float,
    params: PricingParams,
    customers: tuple[AgentTruth, ...],
    providers: tuple[AgentTruth, ...],
    v_customer_true: tuple[tuple[float, ...], ...],
    v_provider_true: tuple[tuple[float, ...], ...],
    vhat_customer_standard: tuple[tuple[float, ...], ...],
    vhat_customer_ai: tuple[tuple[float, ...], ...],
) -> PricingOutcomes:
    n_c = len(customers)
    n_p = len(providers)
    n_treated = int(round(saturation * n_c))
    treated = set(rng.sample(list(range(n_c)), k=n_treated)) if n_treated > 0 else set()

    matched_provider: list[int | None] = [None for _ in range(n_c)]
    match_day: list[int | None] = [None for _ in range(n_c)]
    contacted: list[set[int]] = [set() for _ in range(n_c)]

    provider_capacity = [params.provider_weekly_capacity for _ in range(n_p)]

    # Pricing state
    customer_budget_remaining = [params.agent_budget if i in treated else 0.0 for i in range(n_c)]
    provider_prices = [params.base_attention_price for _ in range(n_p)]
    provider_inbox_today = [0 for _ in range(n_p)]

    messages_sent = 0
    responses_sent = 0
    provider_inbox_total = 0
    total_price_paid = 0.0

    for day in range(1, params.horizon_days + 1):
        # Reset daily inbox counters
        provider_inbox_today = [0 for _ in range(n_p)]

        # Dynamic pricing: adjust prices based on yesterday's load
        if params.dynamic_pricing and day > 1:
            for j in range(n_p):
                excess = max(0, provider_inbox_today[j] - params.provider_daily_response_cap)
                provider_prices[j] = params.base_attention_price + params.price_sensitivity * excess

        inbox: list[list[tuple[int, float]]] = [[] for _ in range(n_p)]  # (customer_idx, bid)

        for i in range(n_c):
            if matched_provider[i] is not None:
                continue

            is_treated = i in treated
            k = params.k_agent_per_day if is_treated else params.k_manual_per_day
            if k <= 0:
                continue

            base_scores = vhat_customer_ai[i] if is_treated else vhat_customer_standard[i]
            noise_sd = (
                params.agent_ranking_noise_sd if is_treated
                else params.manual_ranking_noise_sd
            )

            # Score and rank providers
            scored = []
            for j in range(n_p):
                if provider_capacity[j] <= 0 or j in contacted[i]:
                    continue
                score = base_scores[j] + rng.gauss(0.0, noise_sd)
                scored.append((score, j))
            scored.sort(reverse=True)

            # Send messages (with pricing for treated agents)
            messages_this_round = 0
            for _score, j in scored:
                if messages_this_round >= k:
                    break

                price = provider_prices[j] if params.enable_pricing else 0.0

                # Budget constraint for treated agents with pricing
                if params.enable_pricing and is_treated:
                    if customer_budget_remaining[i] < price:
                        continue  # Can't afford this provider
                    customer_budget_remaining[i] -= price
                    total_price_paid += price

                inbox[j].append((i, price))
                contacted[i].add(j)
                messages_sent += 1
                messages_this_round += 1

        # Provider receives and responds
        responses_by_customer: list[list[int]] = [[] for _ in range(n_c)]
        for j in range(n_p):
            incoming = inbox[j]
            if not incoming:
                continue

            provider_inbox_total += len(incoming)
            provider_inbox_today[j] = len(incoming)

            if provider_capacity[j] <= 0:
                continue

            load_ratio = len(incoming) / max(1, params.provider_daily_response_cap)
            threshold = (
                params.provider_accept_threshold
                + params.provider_threshold_uplift_per_extra_cap * max(0.0, load_ratio - 1.0)
            )
            threshold = min(params.provider_threshold_cap, threshold)

            candidates = []
            for i, _bid in incoming:
                observed = v_provider_true[j][i]
                if i in treated:
                    observed += params.message_quality_bonus
                observed += rng.gauss(0.0, params.provider_eval_noise_sd)
                observed = max(0.0, min(1.0, observed))
                if observed < threshold:
                    continue
                candidates.append((observed, i))

            candidates.sort(reverse=True)
            for _obs, i in candidates[: params.provider_daily_response_cap]:
                responses_by_customer[i].append(j)
                responses_sent += 1

        # Customer accepts best offer
        accepts_by_provider: list[list[int]] = [[] for _ in range(n_p)]
        for i in range(n_c):
            if matched_provider[i] is not None:
                continue
            offers = responses_by_customer[i]
            if not offers:
                continue
            best_j = max(offers, key=lambda j: v_customer_true[i][j])
            if v_customer_true[i][best_j] >= params.customer_accept_threshold:
                accepts_by_provider[best_j].append(i)

        # Provider finalizes matches
        for j in range(n_p):
            if provider_capacity[j] <= 0:
                continue
            accepters = accepts_by_provider[j]
            if not accepters:
                continue
            accepters.sort(key=lambda i: v_provider_true[j][i], reverse=True)
            for i in accepters:
                if provider_capacity[j] <= 0 or matched_provider[i] is not None:
                    continue
                provider_capacity[j] -= 1
                matched_provider[i] = j
                match_day[i] = day

    # Calculate outcomes
    matched_indices = [i for i in range(n_c) if matched_provider[i] is not None]
    match_rate_all = len(matched_indices) / n_c if n_c else 0.0

    treated_indices = list(treated)
    control_indices = [i for i in range(n_c) if i not in treated]

    def rate(indices: list[int]) -> float:
        if not indices:
            return 0.0
        return sum(1 for i in indices if matched_provider[i] is not None) / len(indices)

    def mean_days(indices: list[int]) -> float:
        days = [match_day[i] for i in indices if match_day[i] is not None]
        return sum(days) / len(days) if days else float(params.horizon_days)

    # Welfare calculation
    total_value = 0.0
    for i in matched_indices:
        j = matched_provider[i]
        if j is not None:
            total_value += v_customer_true[i][j] + v_provider_true[j][i]

    total_actions = messages_sent + provider_inbox_total + responses_sent
    net_welfare = (
        (total_value / n_c) - params.attention_cost * (total_actions / n_c)
        if n_c else 0.0
    )

    # Pricing mode label
    if not params.enable_pricing:
        pricing_mode = "none"
    elif params.dynamic_pricing:
        pricing_mode = "dynamic"
    else:
        pricing_mode = "fixed"

    # Budget utilization
    total_budget = sum(params.agent_budget for i in treated)
    budget_util = total_price_paid / total_budget if total_budget > 0 else 0.0

    inbox_per_day = (
        provider_inbox_total / (n_p * params.horizon_days) if n_p else 0.0
    )
    response_rate = (
        responses_sent / provider_inbox_total if provider_inbox_total else 0.0
    )

    return PricingOutcomes(
        saturation=saturation,
        pricing_mode=pricing_mode,
        match_rate_all=match_rate_all,
        match_rate_treated=rate(treated_indices),
        match_rate_control=rate(control_indices),
        mean_days_to_match_all=mean_days(list(range(n_c))),
        messages_sent_per_customer=messages_sent / n_c if n_c else 0.0,
        inbox_per_provider_per_day=inbox_per_day,
        provider_response_rate=response_rate,
        net_welfare_per_customer=net_welfare,
        avg_price_paid=total_price_paid / messages_sent if messages_sent else 0.0,
        budget_utilization=budget_util,
    )


def run_pricing_experiment(
    *,
    category: Category,
    out_dir: Path,
    seed: int,
    n_cells: int,
    market_params: MarketParams,
    client: OpenAIClient,
    n_customers: int,
    n_providers: int,
) -> list[dict[str, RowValue]]:
    rng = random.Random(seed)  # nosec B311
    customers, providers = generate_population(
        rng=rng, category=category, n_customers=n_customers, n_providers=n_providers
    )
    truth_by_id = {a.agent_id: a for a in customers + providers}
    std_texts = {a.agent_id: standard_form_text(a) for a in customers + providers}
    ai_texts = {a.agent_id: ai_conversation_transcript(a, rng=rng) for a in customers + providers}

    log(logger, 20, "gpt_parse_start", category=category, agents=len(truth_by_id))
    parsed_std = parse_batch_with_gpt(
        client=client, texts_by_agent_id=std_texts, truth_by_agent_id=truth_by_id
    )
    parsed_ai = parse_batch_with_gpt(
        client=client, texts_by_agent_id=ai_texts, truth_by_agent_id=truth_by_id
    )
    log(logger, 20, "gpt_parse_done", category=category)

    std_by_id = {a.agent_id: a for a in parsed_std.inferred}
    ai_by_id = {a.agent_id: a for a in parsed_ai.inferred}
    w_customer_std = tuple(std_by_id[a.agent_id].weights for a in customers)
    w_customer_ai = tuple(ai_by_id[a.agent_id].weights for a in customers)

    rows: list[dict[str, RowValue]] = []

    # Test at 100% saturation (worst case for congestion)
    saturation = 1.0

    conditions = [
        ("none", PricingParams(enable_pricing=False)),
        ("fixed", PricingParams(enable_pricing=True, dynamic_pricing=False)),
        ("dynamic", PricingParams(enable_pricing=True, dynamic_pricing=True)),
    ]

    for pricing_mode, params in conditions:
        outcomes: list[PricingOutcomes] = []
        for c in range(n_cells):
            cell_seed = seed * 1_000_000 + hash(pricing_mode) % 10_000 + c
            cell_rng = random.Random(cell_seed)  # nosec B311
            market = generate_market_instance(
                rng=cell_rng,
                customers=customers,
                providers=providers,
                idiosyncratic_noise_sd=market_params.idiosyncratic_noise_sd,
            )
            vhat_std = inferred_value_matrix(
                weights_by_agent=w_customer_std, partner_attributes=market.provider_attributes
            )
            vhat_ai = inferred_value_matrix(
                weights_by_agent=w_customer_ai, partner_attributes=market.provider_attributes
            )
            out = _simulate_with_pricing(
                rng=cell_rng,
                saturation=saturation,
                params=params,
                customers=customers,
                providers=providers,
                v_customer_true=market.v_customer,
                v_provider_true=market.v_provider,
                vhat_customer_standard=vhat_std,
                vhat_customer_ai=vhat_ai,
            )
            outcomes.append(out)

        rows.append(
            {
                "category": category,
                "pricing_mode": pricing_mode,
                "saturation": saturation,
                "match_rate_all": round(_mean([o.match_rate_all for o in outcomes]), 3),
                "match_rate_all_se": round(_se([o.match_rate_all for o in outcomes]), 3),
                "messages_sent_per_customer": round(
                    _mean([o.messages_sent_per_customer for o in outcomes]), 3
                ),
                "inbox_per_provider_per_day": round(
                    _mean([o.inbox_per_provider_per_day for o in outcomes]), 3
                ),
                "provider_response_rate": round(
                    _mean([o.provider_response_rate for o in outcomes]), 3
                ),
                "net_welfare_per_customer": round(
                    _mean([o.net_welfare_per_customer for o in outcomes]), 3
                ),
                "net_welfare_per_customer_se": round(
                    _se([o.net_welfare_per_customer for o in outcomes]), 3
                ),
                "avg_price_paid": round(_mean([o.avg_price_paid for o in outcomes]), 4),
                "budget_utilization": round(_mean([o.budget_utilization for o in outcomes]), 3),
            }
        )

    return rows


def _write_csv(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _md_table(rows: list[dict[str, RowValue]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Does pricing fix the congestion externality?"
    )
    parser.add_argument("--out", default="reports/experiment5_pricing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cells", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    market_params = MarketParams()
    client = OpenAIClient(max_calls=20)

    all_rows: list[dict[str, RowValue]] = []
    for category in ("easy", "hard"):
        log(logger, 20, "pricing_experiment_start", category=category)
        rows = run_pricing_experiment(
            category=category,
            out_dir=out_dir,
            seed=args.seed,
            n_cells=args.cells,
            market_params=market_params,
            client=client,
            n_customers=80,
            n_providers=40,
        )
        all_rows.extend(rows)

    _write_csv(all_rows, out_dir / "pricing_results.csv")

    # Generate plots
    for category in ("easy", "hard"):
        cat_rows = [r for r in all_rows if r["category"] == category]
        modes = ["none", "fixed", "dynamic"]
        welfare_pts = []
        for i, m in enumerate(modes):
            val = next(r["net_welfare_per_customer"] for r in cat_rows if r["pricing_mode"] == m)
            welfare_pts.append((i, float(val)))
        write_line_chart_svg(
            out_path=out_dir / f"fig_{category}_welfare_by_pricing.svg",
            title=f"Net welfare by pricing mode ({category}, saturation=100%)",
            x_label="Pricing mode (0=none, 1=fixed, 2=dynamic)",
            y_label="Net welfare per customer",
            series=[LineSeries(label="welfare", points=welfare_pts)],
        )

    # Generate README
    hard_rows = [r for r in all_rows if r["category"] == "hard"]
    baseline_welfare = next(
        r["net_welfare_per_customer"] for r in hard_rows if r["pricing_mode"] == "none"
    )
    fixed_welfare = next(
        r["net_welfare_per_customer"] for r in hard_rows if r["pricing_mode"] == "fixed"
    )
    dynamic_welfare = next(
        r["net_welfare_per_customer"] for r in hard_rows if r["pricing_mode"] == "dynamic"
    )

    if baseline_welfare:
        fixed_recovery = ((fixed_welfare - baseline_welfare) / abs(baseline_welfare)) * 100
        dynamic_recovery = ((dynamic_welfare - baseline_welfare) / abs(baseline_welfare)) * 100
    else:
        fixed_recovery = 0
        dynamic_recovery = 0

    readme = f"""# Experiment 5: Congestion with Pricing

**Question**: Does introducing attention pricing fix the congestion externality?

## Results (hard category, 100% saturation)

| Pricing Mode | Net Welfare | Recovery vs None |
|--------------|-------------|------------------|
| None (baseline) | {baseline_welfare} | — |
| Fixed pricing | {fixed_welfare} | {fixed_recovery:+.1f}% |
| Dynamic pricing | {dynamic_welfare} | {dynamic_recovery:+.1f}% |

## Key Finding

{
"Pricing significantly recovers welfare lost to congestion." if dynamic_recovery > 50 else
"Pricing partially recovers welfare, but congestion effects persist." if dynamic_recovery > 10 else
"Pricing has limited effect on the congestion externality."
}

## Full Results

{_md_table(all_rows)}

## Run

```bash
make experiment5
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    summary = {
        "seed": args.seed,
        "n_cells": args.cells,
        "baseline_welfare_hard": baseline_welfare,
        "fixed_welfare_hard": fixed_welfare,
        "dynamic_welfare_hard": dynamic_welfare,
        "fixed_recovery_pct": round(fixed_recovery, 1),
        "dynamic_recovery_pct": round(dynamic_recovery, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    log(logger, 20, "pricing_experiment_done", out_dir=str(out_dir))


if __name__ == "__main__":
    main()
