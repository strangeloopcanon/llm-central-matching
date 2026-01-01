from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Literal, TypeAlias, cast

from econ_llm_preferences_experiment.elicitation import (
    ai_conversation_transcript,
    parse_batch_with_gpt,
    standard_form_text_topk,
)
from econ_llm_preferences_experiment.logging_utils import get_logger, log
from econ_llm_preferences_experiment.mechanisms import (
    DeferredAcceptanceParams,
    SearchParams,
    decentralized_search,
    deferred_acceptance,
)
from econ_llm_preferences_experiment.openai_client import OpenAIClient
from econ_llm_preferences_experiment.plotting import LineSeries, write_line_chart_svg
from econ_llm_preferences_experiment.simulation import (
    MarketInstance,
    MarketParams,
    generate_market_instance,
    generate_population,
    inferred_value_matrix,
    preference_density_proxy,
)

logger = get_logger(__name__)

RowValue: TypeAlias = str | int | float | None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _se(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    variance = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(variance / n)


def _write_markdown_table(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append(
            "| " + " | ".join("" if row[h] is None else str(row[h]) for h in headers) + " |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_k_list(arg: str) -> list[int]:
    vals: list[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("Expected at least one k value")
    return sorted(set(vals))


def _arm_label(elicitation: str, mechanism: str) -> str:
    return f"{elicitation}_{mechanism}"


def _report_actions_total(
    *,
    market: MarketInstance,
    accept_threshold: float,
    proposer_side: Literal["customer", "provider"],
    k: int,
) -> int:
    n_c = len(market.v_customer)
    n_p = len(market.v_provider)

    customer_accept_counts = [
        sum(1 for j in range(n_p) if market.v_customer[i][j] >= accept_threshold)
        for i in range(n_c)
    ]
    provider_accept_counts = [
        sum(1 for i in range(n_c) if market.v_provider[j][i] >= accept_threshold)
        for j in range(n_p)
    ]

    k = max(0, int(k))
    if proposer_side == "customer":
        # Customers (proposers) submit top-k providers.
        # Providers (receivers) submit top-k customers.
        return sum(min(k, c) for c in customer_accept_counts) + sum(
            min(k, c) for c in provider_accept_counts
        )
    # Providers (proposers) submit top-k customers.
    # Customers (receivers) submit top-k providers.
    return sum(min(k, c) for c in provider_accept_counts) + sum(
        min(k, c) for c in customer_accept_counts
    )


def _lambda_star(
    *,
    value_search: float,
    attention_search: float,
    value_central: float,
    attention_central: float,
) -> float | None:
    denom = attention_search - attention_central
    if denom <= 1e-12:
        return None
    return (value_search - value_central) / denom


def run_sweep(
    *,
    out_dir: Path,
    replications: int,
    seed: int,
    attention_cost: float,
    da_proposer_side: Literal["customer", "provider"],
    da_list_ks: list[int],
    standard_form_top_k: int,
    market_params: MarketParams,
    client: OpenAIClient,
) -> tuple[list[dict[str, RowValue]], list[dict[str, RowValue]], dict[str, object]]:
    all_summary_rows: list[dict[str, RowValue]] = []
    all_effects_rows: list[dict[str, RowValue]] = []

    metadata: dict[str, object] = {
        "replications": replications,
        "seed": seed,
        "attention_cost": attention_cost,
        "market_params": asdict(market_params),
        "da_proposer_side": da_proposer_side,
        "da_list_ks": da_list_ks,
        "standard_form_top_k": standard_form_top_k,
        "model": client.env.model,
        "base_url": client.env.base_url,
    }

    for category in ("easy", "hard"):
        rng = random.Random(seed)  # nosec B311
        customers, providers = generate_population(
            rng=rng,
            category=category,
            n_customers=market_params.n_customers,
            n_providers=market_params.n_providers,
        )
        truth = {a.agent_id: a for a in customers + providers}
        std_texts = {
            a.agent_id: standard_form_text_topk(a, top_k=standard_form_top_k)
            for a in customers + providers
        }
        ai_texts = {
            a.agent_id: ai_conversation_transcript(a, rng=rng) for a in customers + providers
        }

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

        weights_customer = {
            "standard": tuple(inferred_std[a.agent_id].weights for a in customers),
            "ai": tuple(inferred_ai[a.agent_id].weights for a in customers),
        }
        weights_provider = {
            "standard": tuple(inferred_std[a.agent_id].weights for a in providers),
            "ai": tuple(inferred_ai[a.agent_id].weights for a in providers),
        }

        n_customers = len(customers)
        denom = min(len(customers), len(providers))

        d_hat_i: dict[str, list[float]] = {"standard": [], "ai": []}
        d_hat_j: dict[str, list[float]] = {"standard": [], "ai": []}

        search_match_rate: dict[str, list[float]] = {"standard": [], "ai": []}
        search_value_pc: dict[str, list[float]] = {"standard": [], "ai": []}
        search_attention_pc: dict[str, list[float]] = {"standard": [], "ai": []}
        search_welfare_pc: dict[str, list[float]] = {"standard": [], "ai": []}

        da_match_rate: dict[tuple[str, int], list[float]] = {
            (e, k): [] for e in ("standard", "ai") for k in da_list_ks
        }
        da_value_pc: dict[tuple[str, int], list[float]] = {
            (e, k): [] for e in ("standard", "ai") for k in da_list_ks
        }
        da_report_attention_pc: dict[tuple[str, int], list[float]] = {
            (e, k): [] for e in ("standard", "ai") for k in da_list_ks
        }
        da_welfare_pc: dict[tuple[str, int], list[float]] = {
            (e, k): [] for e in ("standard", "ai") for k in da_list_ks
        }

        for r in range(replications):
            rep_rng = random.Random(seed * 10_000 + r)  # nosec B311
            market = generate_market_instance(
                rng=rep_rng,
                customers=customers,
                providers=providers,
                idiosyncratic_noise_sd=market_params.idiosyncratic_noise_sd,
            )

            report_actions_by_k = {
                k: _report_actions_total(
                    market=market,
                    accept_threshold=market_params.accept_threshold,
                    proposer_side=da_proposer_side,
                    k=k,
                )
                for k in da_list_ks
            }

            for elicitation in ("standard", "ai"):
                vhat_c = inferred_value_matrix(
                    weights_by_agent=weights_customer[elicitation],
                    partner_attributes=market.provider_attributes,
                )
                vhat_p = inferred_value_matrix(
                    weights_by_agent=weights_provider[elicitation],
                    partner_attributes=market.customer_attributes,
                )

                d_hat_i[elicitation].append(
                    preference_density_proxy(
                        v_true=market.v_customer, v_hat=vhat_c, epsilon=market_params.epsilon
                    )
                )
                d_hat_j[elicitation].append(
                    preference_density_proxy(
                        v_true=market.v_provider, v_hat=vhat_p, epsilon=market_params.epsilon
                    )
                )

                outcome_search = decentralized_search(
                    v_customer_true=market.v_customer,
                    v_provider_true=market.v_provider,
                    v_customer_hat=vhat_c,
                    accept_threshold=market_params.accept_threshold,
                    params=SearchParams(max_rounds=30),
                )
                tv_sum = sum(
                    market.v_customer[i][j] + market.v_provider[j][i]
                    for i, j in outcome_search.matches
                )
                attn_total = outcome_search.proposals + outcome_search.accept_decisions
                search_match_rate[elicitation].append(
                    len(outcome_search.matches) / denom if denom else 0.0
                )
                search_value_pc[elicitation].append(tv_sum / n_customers if n_customers else 0.0)
                search_attention_pc[elicitation].append(
                    attn_total / n_customers if n_customers else 0.0
                )
                search_welfare_pc[elicitation].append(
                    (tv_sum / n_customers if n_customers else 0.0)
                    - attention_cost * (attn_total / n_customers if n_customers else 0.0)
                )

                for k in da_list_ks:
                    outcome_da = deferred_acceptance(
                        v_customer_true=market.v_customer,
                        v_provider_true=market.v_provider,
                        v_customer_hat=vhat_c,
                        v_provider_hat=vhat_p,
                        accept_threshold=market_params.accept_threshold,
                        params=DeferredAcceptanceParams(
                            proposer_side=da_proposer_side,
                            proposer_list_k=k,
                            receiver_list_k=k,
                        ),
                    )
                    tv_sum_da = sum(
                        market.v_customer[i][j] + market.v_provider[j][i]
                        for i, j in outcome_da.matches
                    )
                    report_actions = report_actions_by_k[k]
                    report_attention_pc = report_actions / n_customers if n_customers else 0.0
                    da_match_rate[(elicitation, k)].append(
                        len(outcome_da.matches) / denom if denom else 0.0
                    )
                    da_value_pc[(elicitation, k)].append(
                        tv_sum_da / n_customers if n_customers else 0.0
                    )
                    da_report_attention_pc[(elicitation, k)].append(report_attention_pc)
                    da_welfare_pc[(elicitation, k)].append(
                        (tv_sum_da / n_customers if n_customers else 0.0)
                        - attention_cost * report_attention_pc
                    )

        for k in da_list_ks:
            for elicitation in ("standard", "ai"):
                arm_da = _arm_label(elicitation, "da")
                arm_search = _arm_label(elicitation, "search")

                all_summary_rows.append(
                    {
                        "category": category,
                        "k": k,
                        "arm": arm_search,
                        "elicitation": elicitation,
                        "mechanism": "search",
                        "d_hat_I": round(_mean(d_hat_i[elicitation]), 3),
                        "d_hat_I_se": round(_se(d_hat_i[elicitation]), 3),
                        "d_hat_J": round(_mean(d_hat_j[elicitation]), 3),
                        "d_hat_J_se": round(_se(d_hat_j[elicitation]), 3),
                        "match_rate": round(_mean(search_match_rate[elicitation]), 3),
                        "match_rate_se": round(_se(search_match_rate[elicitation]), 3),
                        "total_value_per_customer": round(_mean(search_value_pc[elicitation]), 3),
                        "total_value_per_customer_se": round(_se(search_value_pc[elicitation]), 3),
                        "attention_per_customer": round(_mean(search_attention_pc[elicitation]), 3),
                        "attention_per_customer_se": round(
                            _se(search_attention_pc[elicitation]), 3
                        ),
                        "net_welfare_per_customer": round(_mean(search_welfare_pc[elicitation]), 3),
                        "net_welfare_per_customer_se": round(
                            _se(search_welfare_pc[elicitation]), 3
                        ),
                    }
                )

                all_summary_rows.append(
                    {
                        "category": category,
                        "k": k,
                        "arm": arm_da,
                        "elicitation": elicitation,
                        "mechanism": "deferred_acceptance",
                        "d_hat_I": round(_mean(d_hat_i[elicitation]), 3),
                        "d_hat_I_se": round(_se(d_hat_i[elicitation]), 3),
                        "d_hat_J": round(_mean(d_hat_j[elicitation]), 3),
                        "d_hat_J_se": round(_se(d_hat_j[elicitation]), 3),
                        "match_rate": round(_mean(da_match_rate[(elicitation, k)]), 3),
                        "match_rate_se": round(_se(da_match_rate[(elicitation, k)]), 3),
                        "total_value_per_customer": round(_mean(da_value_pc[(elicitation, k)]), 3),
                        "total_value_per_customer_se": round(_se(da_value_pc[(elicitation, k)]), 3),
                        "attention_per_customer": round(
                            _mean(da_report_attention_pc[(elicitation, k)]), 3
                        ),
                        "attention_per_customer_se": round(
                            _se(da_report_attention_pc[(elicitation, k)]), 3
                        ),
                        "net_welfare_per_customer": round(
                            _mean(da_welfare_pc[(elicitation, k)]), 3
                        ),
                        "net_welfare_per_customer_se": round(
                            _se(da_welfare_pc[(elicitation, k)]), 3
                        ),
                    }
                )

            did_match_rate = [
                (da_match_rate[("ai", k)][t] - da_match_rate[("standard", k)][t])
                - (search_match_rate["ai"][t] - search_match_rate["standard"][t])
                for t in range(replications)
            ]
            did_net_welfare = [
                (da_welfare_pc[("ai", k)][t] - da_welfare_pc[("standard", k)][t])
                - (search_welfare_pc["ai"][t] - search_welfare_pc["standard"][t])
                for t in range(replications)
            ]

            ls_std = _lambda_star(
                value_search=_mean(search_value_pc["standard"]),
                attention_search=_mean(search_attention_pc["standard"]),
                value_central=_mean(da_value_pc[("standard", k)]),
                attention_central=_mean(da_report_attention_pc[("standard", k)]),
            )
            ls_ai = _lambda_star(
                value_search=_mean(search_value_pc["ai"]),
                attention_search=_mean(search_attention_pc["ai"]),
                value_central=_mean(da_value_pc[("ai", k)]),
                attention_central=_mean(da_report_attention_pc[("ai", k)]),
            )

            all_effects_rows.append(
                {
                    "category": category,
                    "k": k,
                    "match_rate_did": round(_mean(did_match_rate), 3),
                    "match_rate_did_se": round(_se(did_match_rate), 3),
                    "net_welfare_did": round(_mean(did_net_welfare), 3),
                    "net_welfare_did_se": round(_se(did_net_welfare), 3),
                    "lambda_star_standard": None if ls_std is None else round(ls_std, 4),
                    "lambda_star_ai": None if ls_ai is None else round(ls_ai, 4),
                }
            )

    _write_report(
        out_dir=out_dir,
        summary_rows=all_summary_rows,
        effects_rows=all_effects_rows,
        metadata=metadata,
    )

    return all_summary_rows, all_effects_rows, metadata


def _write_report(
    *,
    out_dir: Path,
    summary_rows: list[dict[str, RowValue]],
    effects_rows: list[dict[str, RowValue]],
    metadata: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "summary_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    _write_markdown_table(summary_rows, out_dir / "summary_table.md")

    effects_csv = out_dir / "effects_table.csv"
    with effects_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(effects_rows[0].keys()))
        writer.writeheader()
        writer.writerows(effects_rows)
    _write_markdown_table(effects_rows, out_dir / "effects_table.md")

    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def _get_row(category: str, k: int, arm: str) -> dict[str, RowValue]:
        for row in summary_rows:
            if (
                str(row["category"]) == category
                and int(row["k"] or 0) == k
                and str(row["arm"]) == arm
            ):
                return row
        raise KeyError((category, k, arm))

    categories = sorted({str(r["category"]) for r in summary_rows})
    ks = sorted({int(r["k"] or 0) for r in summary_rows})
    for category in categories:
        pts: dict[str, list[tuple[float, float]]] = {
            "standard_da": [],
            "ai_da": [],
            "standard_search": [],
            "ai_search": [],
            "lambda_star_standard": [],
            "lambda_star_ai": [],
        }

        for k in ks:
            std_da = _get_row(category, k, "standard_da")
            ai_da = _get_row(category, k, "ai_da")
            std_s = _get_row(category, k, "standard_search")
            ai_s = _get_row(category, k, "ai_search")

            pts["standard_da"].append((float(k), float(std_da["net_welfare_per_customer"] or 0.0)))
            pts["ai_da"].append((float(k), float(ai_da["net_welfare_per_customer"] or 0.0)))
            pts["standard_search"].append(
                (float(k), float(std_s["net_welfare_per_customer"] or 0.0))
            )
            pts["ai_search"].append((float(k), float(ai_s["net_welfare_per_customer"] or 0.0)))

            for eff in effects_rows:
                if str(eff["category"]) != category or int(eff["k"] or 0) != k:
                    continue
                ls_std = eff.get("lambda_star_standard")
                ls_ai = eff.get("lambda_star_ai")
                if isinstance(ls_std, (int, float)):
                    pts["lambda_star_standard"].append((float(k), float(ls_std)))
                if isinstance(ls_ai, (int, float)):
                    pts["lambda_star_ai"].append((float(k), float(ls_ai)))

        write_line_chart_svg(
            out_path=out_dir / f"fig_{category}_net_welfare_by_k.svg",
            title=f"Net welfare per customer vs ranked-list length k ({category})",
            series=[
                LineSeries(label="DA (standard)", points=pts["standard_da"]),
                LineSeries(label="DA (AI)", points=pts["ai_da"]),
                LineSeries(label="Search (standard)", points=pts["standard_search"]),
                LineSeries(label="Search (AI)", points=pts["ai_search"]),
            ],
            x_label="k (ranked partners per side)",
            y_label="net_welfare_per_customer",
        )
        if pts["lambda_star_standard"] and pts["lambda_star_ai"]:
            write_line_chart_svg(
                out_path=out_dir / f"fig_{category}_lambda_star_by_k.svg",
                title=f"ROI boundary λ* vs ranked-list length k ({category})",
                series=[
                    LineSeries(label="λ* (standard)", points=pts["lambda_star_standard"]),
                    LineSeries(label="λ* (AI)", points=pts["lambda_star_ai"]),
                ],
                x_label="k (ranked partners per side)",
                y_label="lambda_star",
            )

    # README: summarize best-k and crossover-k.
    readme_lines: list[str] = []
    readme_lines.extend(
        [
            "# Roth-style DA sweep (rank-list length k)",
            "",
            "So what: DA is powerful but requires agents to submit ranked preference lists.",
            "Here we treat each ranked partner as an attention action.",
            "As k grows, match quality can improve but reporting costs rise.",
            "AI can reduce the k needed for DA to beat decentralized search.",
            "(Especially in hard categories.)",
            "",
            "Attention accounting:",
            "- Search: attention = proposals (+ accept decisions, if any).",
            "- DA: attention = reported list entries (customers + providers).",
            "  (Does not count internal DA rounds.)",
            "",
            "Key outputs:",
            "- `summary_table.csv` / `summary_table.md`: outcomes by category × k × arm",
            "- `effects_table.csv` / `effects_table.md`: DiD and λ* by category × k",
            "- `fig_*_net_welfare_by_k.svg`: welfare-vs-k curves",
            "",
        ]
    )

    for category in categories:
        ks_cat = [
            k
            for k in ks
            if any(str(r["category"]) == category and int(r["k"] or 0) == k for r in summary_rows)
        ]
        if not ks_cat:
            continue
        search_std = float(
            _get_row(category, ks_cat[0], "standard_search")["net_welfare_per_customer"] or 0.0
        )
        search_ai = float(
            _get_row(category, ks_cat[0], "ai_search")["net_welfare_per_customer"] or 0.0
        )

        da_std_by_k = {
            k: float(_get_row(category, k, "standard_da")["net_welfare_per_customer"] or 0.0)
            for k in ks_cat
        }
        da_ai_by_k = {
            k: float(_get_row(category, k, "ai_da")["net_welfare_per_customer"] or 0.0)
            for k in ks_cat
        }

        best_k_std = max(ks_cat, key=lambda k: da_std_by_k[k])
        best_k_ai = max(ks_cat, key=lambda k: da_ai_by_k[k])

        crossover_std = next((k for k in ks_cat if da_std_by_k[k] >= search_std), None)
        crossover_ai = next((k for k in ks_cat if da_ai_by_k[k] >= search_ai), None)
        crossover_std_label = str(crossover_std) if crossover_std is not None else "none"
        crossover_ai_label = str(crossover_ai) if crossover_ai is not None else "none"

        readme_lines.append(f"## {category}")
        readme_lines.append(f"- search welfare (standard): {search_std:.3f}")
        readme_lines.append(f"- search welfare (AI): {search_ai:.3f}")
        readme_lines.append(
            f"- best-k DA welfare (standard): k={best_k_std}, welfare={da_std_by_k[best_k_std]:.3f}"
        )
        readme_lines.append(
            f"- best-k DA welfare (AI): k={best_k_ai}, welfare={da_ai_by_k[best_k_ai]:.3f}"
        )
        readme_lines.append(f"- smallest k where DA ≥ search (standard): {crossover_std_label}")
        readme_lines.append(f"- smallest k where DA ≥ search (AI): {crossover_ai_label}")
        readme_lines.append("")

    (out_dir / "README.md").write_text("\n".join(readme_lines).rstrip() + "\n", encoding="utf-8")

    log(logger, 20, "roth_da_sweep_written", out_dir=str(out_dir), n_rows=len(summary_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Roth-style DA sweep over rank-list length k.")
    parser.add_argument("--out", default="reports/roth_da_latest")
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attention-cost", type=float, default=0.01)
    parser.add_argument("--da-proposer-side", choices=["customer", "provider"], default="customer")
    parser.add_argument("--da-list-k", default="1,2,3,5,10,15,20,30")
    parser.add_argument(
        "--standard-form-top-k",
        type=int,
        default=3,
        help="How many dimensions the standard baseline form reveals (default: 3).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    market_params = MarketParams()
    client = OpenAIClient(max_calls=9)

    run_sweep(
        out_dir=out_dir,
        replications=args.replications,
        seed=args.seed,
        attention_cost=args.attention_cost,
        da_proposer_side=cast(Literal["customer", "provider"], args.da_proposer_side),
        da_list_ks=_parse_k_list(args.da_list_k),
        standard_form_top_k=args.standard_form_top_k,
        market_params=market_params,
        client=client,
    )


if __name__ == "__main__":
    main()
